"""Batch landing eval — sweep DX, measure DETERMINISTIC distance/hit across many envs.

Answers "how far can the policy actually jump, and does it TRACK the command dx?" without the
noise of a 3-jump play. Drives the env exactly like training/play (normal jump flow, Step H
comp_torque fed, PD replayed at the ckpt's own general_scale), but headless with NUM_ENVS envs,
DETERMINISTIC actions, and NO obs-noise / domain-rand (isolates the policy). For each DX it fixes
the command range to [DX,DX], resets, runs, and over every REAL landing (peak>=min_peak) records:
  - land_err = |landing_xy - landing_target|  -> hit if <= tol
  - squat->land = |landing_xy - squat_root_xy|   (the anti-cheat distance = what the reward scores)
  - air         = |landing_xy - takeoff_root_xy| (pure airborne)
If squat->land is ~flat across DX -> the policy jumps a FIXED distance (doesn't track the command);
if it rises with DX -> it tracks. Run:
  python legged_gym/scripts/eval_landing_sweep.py --task=go2_omnijump_landing_torque [--checkpoint=N]
"""
import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math

DX_LIST = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_ENVS = 256
STEPS_PER_DX = 1500
# EVAL_TRAIN_COND=1 → 复现训练条件(obs噪声 + friction/mass domain-rand + 随机采样动作), 证明 eval 忠实、
# train/eval gap 纯粹是条件差异(而非 eval bug): 这时 far-band hit 应跳回训练那样的高值.
# 默认(0) = 干净确定性标称参数 = 部署时的真实能力.
TRAIN_COND = os.environ.get("EVAL_TRAIN_COND", "0") == "1"


def train_step_count_at_iter(target_iter, W, X, sf, mf, dt, nspe):
    """TRAINING's NON-LINEAR step_count at iter N (freq ramp -> step/iter 96->48). See play_landing.py."""
    r0 = nspe / (dt * sf)
    iw = W / r0
    if target_iter <= iw:
        return r0 * target_iter
    Delta = X - W
    a = (mf - sf) / (2.0 * Delta)
    b = sf
    iter_fade_end = iw + (dt / nspe) * (a * Delta * Delta + b * Delta)
    if target_iter <= iter_fade_end:
        c = (nspe / dt) * (target_iter - iw)
        u = (-b + math.sqrt(b * b + 4.0 * a * c)) / (2.0 * a)
        return W + u
    return X + nspe / (dt * mf) * (target_iter - iter_fade_end)


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.env.episode_length_s = 3           # short episodes -> more jumps per DX
    env_cfg.commands.resampling_time = 20.0     # hold command for the whole episode
    env_cfg.commands.landing_dx_curriculum = False   # we drive command_ranges ourselves per DX
    env_cfg.commands.jump_command_range = [1.0, 1.0] # every episode jumps
    # Conditions: default = clean/deterministic/nominal (= deployable capability). TRAIN_COND -> reproduce
    # TRAINING conditions (obs noise + friction/mass domain-rand + stochastic actions below) to PROVE the eval
    # is faithful and the train/eval gap is CONDITIONS, not a bug (far-band hit should jump to ~training levels).
    env_cfg.noise.add_noise = TRAIN_COND
    env_cfg.domain_rand.randomize_friction = TRAIN_COND
    env_cfg.domain_rand.randomize_base_mass = TRAIN_COND
    env_cfg.domain_rand.push_robots = False        # training also disables push DURING the jump
    env_cfg.rewards.landing_tilt_terminate = 0.0
    env_cfg.rewards.rsi_prob = 0.0                 # NO RSI air-drops during eval (isolate the policy; RSI is a
                                                   # training-only exploration mechanism that would randomly spawn
                                                   # envs mid-flight and corrupt the distance measurement)
    train_cfg.runner.resume = True

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    _load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    _checkpoint = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    _resume_path = get_load_path(log_root, load_run=_load_run, checkpoint=_checkpoint)
    ckpt_iter = int(os.path.basename(_resume_path).replace('model_', '').replace('.pt', ''))

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic

    REPLAY_STEP = int(round(train_step_count_at_iter(
        ckpt_iter,
        float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
        float(env.start_freq), float(env.max_freq),
        float(env.dt), int(train_cfg.runner.num_steps_per_env))))
    gs = min(1.0, max(0.0, (REPLAY_STEP - float(env.cfg.growth.warmup_steps)) /
             max(1.0, float(env.cfg.growth.x0) - float(env.cfg.growth.warmup_steps))))
    pd_a = float(env.cfg.control.pd_prior_weight) * (1.0 - gs)

    tol = float(getattr(env.cfg.commands, "landing_dx_hit_tol", 0.10))
    min_peak = float(getattr(env.cfg.rewards, "landing_real_jump_min_peak", 0.40))

    mode = "TRAIN-COND (obs噪声+domain rand+随机采样)" if TRAIN_COND else "deterministic, no-noise, nominal"
    print(f"\n[eval_sweep] {os.path.basename(_resume_path)} | iter {ckpt_iter} | step_count={REPLAY_STEP} "
          f"pd_alpha={pd_a:.3f} | {NUM_ENVS} envs | {mode} | hit tol={tol}", flush=True)
    print(f"{'DX(cmd)':>8} {'n':>6} {'hit@'+str(tol):>9} {'hit@0.15':>9} "
          f"{'squat->land':>12} {'air(flight)':>12} {'peak':>7}", flush=True)

    for DX in DX_LIST:
        env.command_ranges["lin_vel_x"] = [DX, DX]
        env.reset()
        obs = env.get_observations()
        n = hits = hits15 = 0
        sqld = air = pk = 0.0
        for _ in range(STEPS_PER_DX):
            env.step_count = REPLAY_STEP
            env.common_step_counter = REPLAY_STEP
            with torch.no_grad():
                comp = actor_critic.comp_forward(obs)
                if comp is not None:
                    env.comp_torque = comp.detach()
                if TRAIN_COND:
                    actions = actor_critic.act(obs.detach())   # 随机采样(带策略探索噪声)= 训练时的动作
                else:
                    actions = policy(obs.detach())             # 确定性 mean = 部署时的动作
            obs, _, _, _, _ = env.step(actions.detach())
            jl = env.just_landed & (env.peak_base_height >= min_peak)
            if torch.any(jl):
                land = env.landing_root_xy[jl]
                err = torch.norm(land - env.landing_target[jl], dim=1)
                hits += int((err <= tol).sum().item())
                hits15 += int((err <= 0.15).sum().item())
                n += int(jl.sum().item())
                sqld += float(torch.norm(land - env.squat_root_xy[jl], dim=1).sum().item())
                air += float(torch.norm(land - env.takeoff_root_xy[jl], dim=1).sum().item())
                pk += float(env.peak_base_height[jl].sum().item())
        if n > 0:
            print(f"{DX:>8.2f} {n:>6d} {hits/n:>9.2f} {hits15/n:>9.2f} "
                  f"{sqld/n:>12.3f} {air/n:>12.3f} {pk/n:>7.3f}", flush=True)
        else:
            print(f"{DX:>8.2f} {0:>6d}  (no real jumps peak>={min_peak})", flush=True)


if __name__ == '__main__':
    main()
