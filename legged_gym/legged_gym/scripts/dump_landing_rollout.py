"""Dump an Isaac-Gym stand->jump rollout (obs / action / comp / torque) as .npz for SIM2SIM cross-check.

Companion to deploy_mujoco/sim2sim_landing_torque.py. It runs the SAME deploy protocol as the
validated play_landing_deploy.py -- reset -> PURE-PD stand at the default pose -> hand to the RL
policy with cmd4=1 -> jump -> land -- but headless, and records every quantity the MuJoCo port
needs to be checked against:

    obs    (T, 69)  the observation actually fed to the policy
    action (T, 12)  actor output
    comp   (T, 12)  comp_head output (carries 0.5 of the torque at the pure-torque endpoint)
    tau    (T, 12)  torque applied by the env that step
    q/dq   (T, 12)  joint state, base (T, 3)

Then:
    python deploy_mujoco/sim2sim_landing_torque.py --ckpt <same ckpt> --replay <this .npz>
feeds these observations through the MuJoCo script's own network copy and compares the actions.
A match proves the checkpoint loading and the 69-dim observation ORDER are right, so any
difference in how far it jumps in MuJoCo is physics, not wiring.

RUN (headless, no viewer):
  python legged_gym/scripts/dump_landing_rollout.py --task=go2_omnijump_landing_torque \
      --load_run=Jul16_12-07-41_stage1_landing --checkpoint=4600 --headless
Knobs (env vars, same names/meanings as play_landing_deploy.py):
  DUMP_OUT (default /tmp/isaac_rollout.npz), PLAY_DEPLOY_DX/HEIGHT/STAND_HOLD/STAND_MAX/JUMP_MAX/
  STAND_PD_WEIGHT
"""
import isaacgym  # noqa: F401  (must precede torch)
import torch
import numpy as np
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math


def _envf(name, default):
    v = os.environ.get(name)
    try:
        return float(v) if v is not None else float(default)
    except ValueError:
        return float(default)


def _envi(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else int(default)


def train_step_count_at_iter(target_iter, W, X, sf, mf, dt, nspe):
    """TRAINING's non-linear step_count at iter N (identical closed form to play_landing*.py)."""
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
    r_end = nspe / (dt * mf)
    return X + r_end * (target_iter - iter_fade_end)


OUT = os.environ.get("DUMP_OUT", "/tmp/isaac_rollout.npz")
DX = _envf("PLAY_DEPLOY_DX", 1.0)
DY = _envf("PLAY_DEPLOY_DY", 0.0)
HEIGHT = _envf("PLAY_DEPLOY_HEIGHT", 0.5)
STAND_PD_WEIGHT = _envf("PLAY_DEPLOY_STAND_PD_WEIGHT", 1.0)
STAND_HOLD = _envi("PLAY_DEPLOY_STAND_HOLD", 60)
STAND_MAX = _envi("PLAY_DEPLOY_STAND_MAX", 400)
JUMP_MAX = _envi("PLAY_DEPLOY_JUMP_MAX", 400)


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    # Same clean, real-robot-like conditions as play_landing_deploy.py.
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 12
    env_cfg.commands.resampling_time = 40.0
    env_cfg.commands.landing_dx_curriculum = False
    env_cfg.commands.jump_command_range = [0.0, 0.0]
    env_cfg.commands.landing_disp_x_stage2 = [0.0, 0.0]
    env_cfg.commands.landing_disp_y_stage2 = [0.0, 0.0]
    env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
    env_cfg.commands.ranges.jump_command = [1.0, 1.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.rewards.rsi_prob = 0.0
    env_cfg.rewards.landing_tilt_terminate = 0.0
    train_cfg.runner.resume = True
    # A checkpoint must be evaluated under the joint limits it was TRAINED with -- the official-range
    # override changes the physics, so scoring a pre-override policy with it (or vice versa) is not an
    # A/B, it is a different robot. DUMP_OFFICIAL_LIMITS=0/1 forces it; unset keeps the task default.
    _ol = os.environ.get("DUMP_OFFICIAL_LIMITS")
    if _ol is not None:
        env_cfg.asset.official_dof_pos_limits = _ol.strip() in ("1", "true", "yes", "on")

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    _load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    _checkpoint = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    resume_path = get_load_path(log_root, load_run=_load_run, checkpoint=_checkpoint)
    ckpt_iter = int(os.path.basename(resume_path).replace('model_', '').replace('.pt', ''))

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic

    REPLAY_STEP = int(round(train_step_count_at_iter(
        ckpt_iter,
        float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
        float(env.start_freq), float(env.max_freq),
        float(env.dt), int(train_cfg.runner.num_steps_per_env))))
    PD_W_ORIG = float(env.cfg.control.pd_prior_weight)
    _zero_act = torch.zeros((env.num_envs, env.num_actions), device=env.device)

    rec = dict(obs=[], action=[], comp=[], tau=[], q=[], dq=[], base=[], done=[])

    def act_step():
        """One faithful pure-torque RL step, recording what the MuJoCo port must match."""
        env.cfg.control.pd_prior_weight = PD_W_ORIG
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        obs = env.get_observations().detach()
        with torch.no_grad():
            actions = policy(obs)
            comp = actor_critic.comp_forward(obs)
            if comp is not None:
                env.comp_torque = comp
        rec["obs"].append(obs[0].cpu().numpy().copy())
        rec["action"].append(actions[0].cpu().numpy().copy())
        rec["comp"].append((comp[0].cpu().numpy().copy() if comp is not None else np.zeros(12)))
        _, _, _, dones, _ = env.step(actions.detach())
        rec["tau"].append(env.torques[0].cpu().numpy().copy())
        rec["q"].append(env.dof_pos[0].cpu().numpy().copy())
        rec["dq"].append(env.dof_vel[0].cpu().numpy().copy())
        rec["base"].append(env.root_states[0, :3].cpu().numpy().copy())
        # env.step() resets a terminated env INSIDE the step, so the state recorded on a done step is
        # already back at the spawn -- flag it so trajectory analysis drops that frame instead of
        # reading the teleport as motion.
        rec["done"].append(np.array([1.0 if bool(dones[0]) else 0.0]))
        return bool(dones[0])

    def pd_hold_step(weight):
        """One PURE-PD step holding the default pose (general_scale pinned to 0)."""
        env.cfg.control.pd_prior_weight = float(weight)
        env.step_count = 0
        env.common_step_counter = 0
        _, _, _, dones, _ = env.step(_zero_act)
        return bool(dones[0])

    print(f"\n[dump] {os.path.basename(resume_path)} -> iter {ckpt_iter}, step_count={REPLAY_STEP} "
          f"(pure torque), dx={DX} h={HEIGHT}", flush=True)

    env.reset()
    env.commands[:, 0] = 0.0
    env.commands[:, 1] = 0.0
    env.commands[:, 4] = 0.0
    spawn_xy = env.root_states[0, :2].clone()
    env.landing_target[:, 0] = spawn_xy[0]
    env.landing_target[:, 1] = spawn_xy[1]
    env.compute_observations()

    # ---- PD stand ----------------------------------------------------------
    # DUMP_PD_STAND=0 skips it: reset -> command the jump straight away, i.e. the condition the policy
    # was actually TRAINED in (episodes reset and the jump fires at first_jump_delay_steps). Use it to
    # tell "the policy cannot jump" apart from "the PD-stand handover is out of distribution for it".
    settled, st = False, -1
    if os.environ.get("DUMP_PD_STAND", "1").strip() in ("0", "false", "no", "off"):
        print("[dump] PD stand SKIPPED -- jumping straight from reset (training-like condition)", flush=True)
        settled = True
    else:
        for st in range(STAND_MAX):
            env.commands[:, 4] = 0.0
            if pd_hold_step(STAND_PD_WEIGHT):
                print("[dump] FELL while standing -- aborting", flush=True)
                return
            if int(env.stand_step_counter[0]) >= STAND_HOLD:
                settled = True
                break
        print(f"[dump] stand: {'settled' if settled else 'NOT settled'} after {st + 1} steps, "
              f"h={float(env.root_states[0, 2]):.3f}m", flush=True)

    # ---- hand to RL and jump ----------------------------------------------
    env.commands[:, 0] = DX
    env.commands[:, 1] = DY
    env.commands[:, 4] = 1.0
    env.landing_target[:, 0] = env.root_states[0, 0] + DX
    env.landing_target[:, 1] = env.root_states[0, 1] + DY
    env.compute_observations()

    took_off = landed = False
    creep = flight = float("nan")
    post = 0
    for jt in range(JUMP_MAX):
        if act_step():
            print("[dump] episode terminated (fell)", flush=True)
            break
        # Capture AT THE EVENT. Reading these anchors after the loop is wrong: a termination inside
        # env.step() resets the jump buffers, so a fallen/timed-out episode reports creep=flight=0.
        if not took_off and bool(env.has_taken_off[0]):
            took_off = True
            creep = float(torch.norm(env.takeoff_root_xy[0] - env.jump_start_root_xy[0]))
        if took_off and not landed and bool(env.has_landed[0]):
            landed = True
            flight = float(torch.norm(env.landing_root_xy[0] - env.takeoff_root_xy[0]))
            peak_at_land = float(env.peak_base_height[0])
        if landed:
            post += 1
            if post >= 60:
                break

    # Same-protocol run-up accounting, read straight off the env's own anchors, so two checkpoints can
    # be compared apples-to-apples (the TRAINING numbers are averages over the whole command range with
    # noise/DR and failed attempts -- not comparable to a deterministic dx=1.0 deploy jump).
    if took_off:
        total = creep + (flight if flight == flight else 0.0)
        print(f"[dump] official_dof_pos_limits={env.cfg.asset.official_dof_pos_limits} | "
              f"peak={peak_at_land if landed else float('nan'):.3f}m  run_up_creep={creep:.3f}m  "
              f"clean_flight={flight:.3f}m  total={total:.3f}m  "
              f"run_up_share={100.0 * creep / max(total, 1e-3):.0f}%", flush=True)
    else:
        print(f"[dump] official_dof_pos_limits={env.cfg.asset.official_dof_pos_limits} | "
              f"NEVER LEFT THE GROUND in {JUMP_MAX} steps", flush=True)

    n = len(rec["obs"])
    np.savez(OUT, **{k: np.stack(v).astype(np.float32) for k, v in rec.items()})
    print(f"[dump] took_off={took_off} landed={landed} | {n} RL steps -> {OUT}", flush=True)
    print(f"[dump] peak={float(env.peak_base_height[0]):.3f}m  "
          f"cross-check with:\n"
          f"       python deploy_mujoco/sim2sim_landing_torque.py --ckpt {resume_path} --replay {OUT}",
          flush=True)


if __name__ == '__main__':
    main()
