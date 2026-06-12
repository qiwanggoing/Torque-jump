"""Clean landing-point play — mirrors the VALIDATED batch eval.

Background: the legacy play.py drives the jump task through a TEST-mode state machine
(forces commands[:] = test.vel every step, KILLS cmd[4] at takeoff, single_jump_play
re-arm, etc.). That path DIVERGES from how the policy was trained -> in play the robot
lands short / topples, even though the SAME checkpoint scores landing_hit_rate ~0.98 at
dx=0.40 in a faithful, normal-mode rollout (256-env batch eval).

This script drives the env exactly the way training / the batch eval do:
  - NORMAL command flow (use_test = False), env fires the jump itself (first_jump_delay,
    single-jump logic) and holds cmd[4] through the jump,
  - fixed forward command via the command ranges,
  - PD prior fully faded (pd_alpha = 0, the iter-10000 trained condition),
  - deterministic (mean) policy.
plus a viewer + per-landing readout (fwd displacement, landing error vs target, hit/miss).

Edit DX / DY / HEIGHT below to test different commands. Run:
  python legged_gym/scripts/play_landing.py --task=go2_omnijump_landing_torque
"""
import isaacgym
from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

# === command to visualise (training ranges: dx in [0,0.40], height in [0.40,0.70]) ===
DX = 0.6       # forward landing displacement (m)
DY = 0.0        # lateral landing displacement (m)
HEIGHT = 0.70   # jump-height command
# ====================================================================================


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 4          # ~1 jump per episode, paced for viewing

    # Fixed forward landing command — fed through the env's NORMAL command ranges so the
    # jump fires and evolves exactly as in training (no test-mode override / state machine).
    env_cfg.commands.landing_dx_curriculum = False
    env_cfg.commands.landing_disp_x_stage2 = [DX, DX]
    env_cfg.commands.landing_disp_y_stage2 = [DY, DY]
    env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
    env_cfg.commands.ranges.jump_command = [1.0, 1.0]    # always > jump_command_threshold (0.5) -> always jump
    # Clean conditions (the eval that scored 0.98 used these; isolates the policy).
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    train_cfg.runner.resume = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    BIG = 500000   # force the PD prior fully faded (general_scale=1 -> pd_alpha=0), matching iter 10000
    init = torch.tensor([float(env.cfg.init_state.pos[0]), float(env.cfg.init_state.pos[1])], device=env.device)
    spawn = env.env_origins[:, :2] + init

    # Side view of the spawn + landing strip.
    if getattr(env, "viewer", None) is not None:
        sx, sy = float(spawn[0, 0]), float(spawn[0, 1])
        env.gym.viewer_camera_look_at(
            env.viewer, None,
            gymapi.Vec3(sx - 1.6, sy - 1.6, 1.1),
            gymapi.Vec3(sx + 0.4, sy, 0.25),
        )

    obs = env.get_observations()
    print(f"[play_landing] cmd dx={DX} dy={DY} height={HEIGHT} | clean env, normal jump flow, pd_alpha=0", flush=True)

    n_jump = n_hit = 0
    for _ in range(100000):
        env.step_count = BIG
        env.common_step_counter = BIG
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, infos = env.step(actions.detach())
        if hasattr(env, "just_landed") and bool(env.just_landed.any()):
            for idx in env.just_landed.nonzero(as_tuple=False).flatten().tolist():
                land = env.root_states[idx, :2]
                err = torch.norm(land - env.landing_target[idx, :2]).item()
                fwd = float(land[0] - spawn[idx, 0])
                lat = float(land[1] - spawn[idx, 1])
                peak = float(env.peak_base_height[idx])
                hit = (err <= 0.10) and (peak >= 0.40)
                n_jump += 1
                n_hit += int(hit)
                print(f"[land #{n_jump}] peak={peak:.3f} fwd={fwd:+.3f}m lat={lat:+.3f}m "
                      f"land_err={err:.3f}m {'HIT ' if hit else 'miss'} | hit_rate={n_hit/n_jump:.2f}", flush=True)


if __name__ == "__main__":
    main()
