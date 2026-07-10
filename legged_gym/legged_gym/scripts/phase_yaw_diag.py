"""When does the 90-degree yaw happen — ON THE GROUND before takeoff, or as an in-flight spin?

Latches the body YAW at three per-jump events and reports population means of |yaw|:
  yaw@jump_start : first step jumping_state is True and not yet taken off (BEFORE the squat/push)
  yaw@takeoff    : the step has_taken_off flips False->True (leaves the ground)
  yaw@landing    : just_landed
If |yaw@takeoff| is already large -> it TURNED ON THE GROUND (pre-takeoff), and flight only carries it.
If |yaw@takeoff| ~0 but |yaw@landing| large -> it's an in-flight spin. Run:
  TQ_DX=0.8 python legged_gym/scripts/phase_yaw_diag.py --task=go2_omnijump_landing_torque --load_run=RUN --checkpoint=N --headless
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


def absdeg(y):
    y = (y + math.pi) % (2 * math.pi) - math.pi
    return y.abs()


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
    min_peak = float(getattr(env.cfg.rewards, "landing_real_jump_min_peak", 0.40))

    env.command_ranges["lin_vel_x"] = [DX, DX]
    env.reset()
    obs = env.get_observations()

    dev = env.device
    prev_jump = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    prev_takeoff = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    yaw_start = torch.zeros(env.num_envs, device=dev)   # latched at jump_start
    yaw_takeoff = torch.zeros(env.num_envs, device=dev) # latched at takeoff

    sum_start = sum_to = sum_land = 0.0
    n_start = n_to = n_land = 0
    # also: how much yaw is GAINED start->takeoff (ground turn) vs takeoff->landing (flight)?
    sum_ground_gain = sum_flight_gain = 0.0

    for _ in range(STEPS):
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        with torch.no_grad():
            comp = actor_critic.comp_forward(obs)
            if comp is not None:
                env.comp_torque = comp.detach()
            actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

        _, _, yaw = get_euler_xyz(env.base_quat)
        yaw_abs = absdeg(yaw)

        jump = env.jumping_state & (~env.has_taken_off)
        just_start = jump & (~prev_jump)
        if torch.any(just_start):
            yaw_start[just_start] = yaw_abs[just_start]
            sum_start += float(yaw_abs[just_start].sum().item()); n_start += int(just_start.sum().item())

        just_takeoff = env.has_taken_off & (~prev_takeoff)
        if torch.any(just_takeoff):
            yaw_takeoff[just_takeoff] = yaw_abs[just_takeoff]
            sum_to += float(yaw_abs[just_takeoff].sum().item()); n_to += int(just_takeoff.sum().item())
            sum_ground_gain += float((yaw_abs[just_takeoff] - yaw_start[just_takeoff]).abs().sum().item())

        jl = env.just_landed & (env.peak_base_height >= min_peak)
        if torch.any(jl):
            sum_land += float(yaw_abs[jl].sum().item()); n_land += int(jl.sum().item())
            sum_flight_gain += float((yaw_abs[jl] - yaw_takeoff[jl]).abs().sum().item())

        prev_jump = jump.clone()
        prev_takeoff = env.has_taken_off.clone()

    d = math.degrees
    print(f"\n[phase_yaw] {os.path.basename(_resume_path)} | cmd dx={DX}", flush=True)
    if n_to:
        print(f"  |yaw| @ jump_start (pre-squat)   = {d(sum_start/max(1,n_start)):5.1f} deg  (n={n_start})", flush=True)
        print(f"  |yaw| @ TAKEOFF (leaves ground)  = {d(sum_to/n_to):5.1f} deg  (n={n_to})", flush=True)
        print(f"  |yaw| @ landing                  = {d(sum_land/max(1,n_land)):5.1f} deg  (n={n_land})", flush=True)
        print(f"  --- yaw GAINED on GROUND (start->takeoff) = {d(sum_ground_gain/n_to):5.1f} deg", flush=True)
        print(f"  --- yaw GAINED in FLIGHT (takeoff->land)  = {d(sum_flight_gain/max(1,n_land)):5.1f} deg", flush=True)
    else:
        print("  no takeoffs", flush=True)


if __name__ == '__main__':
    main()
