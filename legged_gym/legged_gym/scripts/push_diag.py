"""Headless PUSH-PHASE mechanism diagnostic — WHY does four_leg_push make it jump sideways?

Over the exact four_leg_push PUSH window (jumping & ~taken_off & squat_deep & total_fz>floor), accumulates
population means of:
  n_feet_contact   : how many feet are still on the ground (4 = "keeping all legs loaded"; <2 = staggered liftoff)
  fz_mean_grounded : per-grounded-foot vertical GRF (N)
  fz_min_grounded  : the LEAST-loaded still-grounded foot (four_leg maxes when this is high)
  |yaw_rate|       : body yaw angular velocity (rad/s) -> is a yaw SPIN being built at takeoff?
  |pitch_rate|     : body pitch angular velocity (the NORMAL forward-launch rotation)
  shear_ratio      : horizontal GRF / vertical GRF per grounded foot (forward push needs backward shear)
Compare four_leg=10 (sideways) vs baseline (clean) to see the mechanism. Run:
  TQ_DX=0.8 python legged_gym/scripts/push_diag.py --task=go2_omnijump_landing_torque --load_run=RUN --checkpoint=N --headless
"""
import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
import os, math

DX = float(os.environ.get("TQ_DX", "0.8"))
NUM_ENVS = 256
STEPS = 1500


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
    REPLAY_STEP = 200000

    floor = float(getattr(env.cfg.rewards, "four_leg_push_force_floor", 200.0))

    env.command_ranges["lin_vel_x"] = [DX, DX]
    env.reset()
    obs = env.get_observations()

    acc = {k: 0.0 for k in ["nfeet", "fzmean", "fzmin", "yawrate", "pitchrate", "shear"]}
    nsamp = 0
    for _ in range(STEPS):
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        with torch.no_grad():
            comp = actor_critic.comp_forward(obs)
            if comp is not None:
                env.comp_torque = comp.detach()
            actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

        fz = torch.clamp(env.contact_forces[:, env.feet_indices, 2], min=0.0)   # (N,4)
        horiz = torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2)  # (N,4)
        total = fz.sum(dim=1)
        contact = fz > 1.0
        try:
            squat = env._squat_deep_enough()
        except Exception:
            squat = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        push = env.jumping_state & (~env.has_taken_off) & squat & (total > floor)
        if not torch.any(push):
            continue
        p = push
        ncontact = contact.float().sum(dim=1)                       # (N,)
        cf = contact.float()
        fz_mean_g = (fz * cf).sum(dim=1) / cf.sum(dim=1).clamp(min=1.0)
        fz_masked = fz.clone()
        fz_masked[~contact] = 1e9
        fz_min_g = fz_masked.min(dim=1).values
        shear = ((horiz / fz.clamp(min=1.0)) * cf).sum(dim=1) / cf.sum(dim=1).clamp(min=1.0)
        yaw_rate = env.base_ang_vel[:, 2].abs()
        pitch_rate = env.base_ang_vel[:, 1].abs()

        k = int(p.sum().item())
        nsamp += k
        acc["nfeet"] += float(ncontact[p].sum().item())
        acc["fzmean"] += float(fz_mean_g[p].sum().item())
        acc["fzmin"] += float(fz_min_g[p].clamp(max=1e6).sum().item())
        acc["yawrate"] += float(yaw_rate[p].sum().item())
        acc["pitchrate"] += float(pitch_rate[p].sum().item())
        acc["shear"] += float(shear[p].sum().item())

    print(f"\n[push_diag] {os.path.basename(_resume_path)} | cmd dx={DX} | push-window samples={nsamp}", flush=True)
    if nsamp == 0:
        print("  no push samples", flush=True)
        return
    n = nsamp
    print(f"  n_feet_contact   = {acc['nfeet']/n:.2f}   (4=all loaded, staggered<2)", flush=True)
    print(f"  fz_mean_grounded = {acc['fzmean']/n:.1f} N", flush=True)
    print(f"  fz_min_grounded  = {acc['fzmin']/n:.1f} N   (least-loaded down foot; four_leg maxes this)", flush=True)
    print(f"  |yaw_rate|       = {acc['yawrate']/n:.2f} rad/s   (yaw SPIN at takeoff)", flush=True)
    print(f"  |pitch_rate|     = {acc['pitchrate']/n:.2f} rad/s   (normal forward-launch pitch)", flush=True)
    print(f"  shear_ratio      = {acc['shear']/n:.2f}   (horiz/vert GRF; forward push needs this)", flush=True)


if __name__ == '__main__':
    main()
