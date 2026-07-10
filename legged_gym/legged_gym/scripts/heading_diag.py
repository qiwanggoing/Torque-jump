"""Headless HEADING diagnostic — does the policy jump SIDEWAYS (横着跳)?

Same deterministic/no-noise/nominal-mass driving as eval_landing_sweep, but per REAL landing it records
world-frame takeoff->land displacement (dx, dy), the body YAW (heading) at landing, and the WORST |yaw|
and WORST |roll| seen during the jump. A clean forward jump = dy~0, yaw~0. Sideways = large |dy| and/or
large |yaw|. Run:
  TQ_DX=0.8 python legged_gym/scripts/heading_diag.py --task=go2_omnijump_landing_torque --load_run=RUN --checkpoint=N --headless
"""
import isaacgym  # noqa: F401
import torch
from isaacgym.torch_utils import get_euler_xyz

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
import os, math

DX = float(os.environ.get("TQ_DX", "0.8"))
NUM_ENVS = 256
STEPS = 1500


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.env.episode_length_s = 3
    env_cfg.commands.resampling_time = 20.0
    env_cfg.commands.landing_dx_curriculum = False
    env_cfg.commands.jump_command_range = [1.0, 1.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.rewards.landing_tilt_terminate = 0.0
    env_cfg.rewards.rsi_prob = 0.0
    train_cfg.runner.resume = True

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    _load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    _checkpoint = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    _resume_path = get_load_path(log_root, load_run=_load_run, checkpoint=_checkpoint)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic

    # REPLAY_STEP: past x0 all these ckpts are pd_alpha=0 (pure torque); just pin a large step_count.
    REPLAY_STEP = 200000
    min_peak = float(getattr(env.cfg.rewards, "landing_real_jump_min_peak", 0.40))

    env.command_ranges["lin_vel_x"] = [DX, DX]
    env.reset()
    obs = env.get_observations()

    # per-env worst |yaw| / |roll| during current jump (reset when a new jump's flight begins is hard to
    # track cleanly; instead we just snapshot yaw/roll AT landing, which is what "lands sideways" means).
    n = 0
    sum_dx = sum_dy = sum_absdy = 0.0
    sum_absyaw = sum_absroll = 0.0
    side_dy = 0        # landings with |dy| > 0.15
    side_yaw = 0       # landings with |yaw| > 20 deg
    for _ in range(STEPS):
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        with torch.no_grad():
            comp = actor_critic.comp_forward(obs)
            if comp is not None:
                env.comp_torque = comp.detach()
            actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        jl = env.just_landed & (env.peak_base_height >= min_peak)
        if torch.any(jl):
            land = env.root_states[jl, :2]
            tk = env.takeoff_root_xy[jl]
            dxv = land[:, 0] - tk[:, 0]
            dyv = land[:, 1] - tk[:, 1]
            roll, pitch, yaw = get_euler_xyz(env.base_quat[jl])
            yaw = wrap_pi(yaw)
            roll = wrap_pi(roll)
            k = int(jl.sum().item())
            n += k
            sum_dx += float(dxv.sum().item())
            sum_dy += float(dyv.sum().item())
            sum_absdy += float(dyv.abs().sum().item())
            sum_absyaw += float(yaw.abs().sum().item())
            sum_absroll += float(roll.abs().sum().item())
            side_dy += int((dyv.abs() > 0.15).sum().item())
            side_yaw += int((yaw.abs() > math.radians(20)).sum().item())

    print(f"\n[heading_diag] {os.path.basename(_resume_path)} | cmd dx={DX} | {NUM_ENVS} envs | deterministic nominal", flush=True)
    if n == 0:
        print("  no real jumps", flush=True)
        return
    print(f"  n_landings        = {n}", flush=True)
    print(f"  mean world dx     = {sum_dx/n:+.3f} m   (forward)", flush=True)
    print(f"  mean world dy     = {sum_dy/n:+.3f} m   (signed lateral)", flush=True)
    print(f"  mean |dy|         = {sum_absdy/n:.3f} m   (lateral magnitude)", flush=True)
    print(f"  mean |yaw|        = {math.degrees(sum_absyaw/n):.1f} deg  (heading rotation)", flush=True)
    print(f"  mean |roll|       = {math.degrees(sum_absroll/n):.1f} deg", flush=True)
    print(f"  |dy|>0.15 frac    = {side_dy/n:.2f}   (sideways-displacement)", flush=True)
    print(f"  |yaw|>20deg frac  = {side_yaw/n:.2f}   (turned-sideways)", flush=True)
    print(f"  lateral ratio |dy|/|dx| = {sum_absdy/max(1e-6,abs(sum_dx)):.2f}", flush=True)


if __name__ == '__main__':
    main()
