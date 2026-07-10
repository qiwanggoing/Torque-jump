"""Launch-ANGLE diagnostic — is the jump launched at the range-optimal ~45deg, or too steep?

At takeoff (has_taken_off False->True) latches CoM velocity (vx forward, vz up), and reports population means:
  launch speed |v|, forward vx, up vz, launch angle atan2(vz,vx), peak height, and the range you'd get at the
  SAME speed if launched at 45deg vs the current angle. If angle >> 45 -> too steep -> flattening trades height
  for FREE distance (no need to push harder / beat the knee wall). Run:
  TQ_DX=0.8 python legged_gym/scripts/launch_diag.py --task=go2_omnijump_landing_torque --load_run=RUN --checkpoint=N --headless
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
G = 9.81


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

    env.command_ranges["lin_vel_x"] = [DX, DX]
    env.reset()
    obs = env.get_observations()

    prev_takeoff = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    n = 0
    s_v = s_vx = s_vy = s_vz = s_ang = s_r_cur = s_r_45 = 0.0
    for _ in range(STEPS):
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        with torch.no_grad():
            comp = actor_critic.comp_forward(obs)
            if comp is not None:
                env.comp_torque = comp.detach()
            actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        just_to = env.has_taken_off & (~prev_takeoff)
        if torch.any(just_to):
            vx = env.root_states[just_to, 7]
            vy = env.root_states[just_to, 8]
            vz = env.root_states[just_to, 9]
            vh = torch.sqrt(vx * vx + vy * vy)              # horizontal speed
            v = torch.sqrt(vh * vh + vz * vz)
            ang = torch.atan2(vz, vh)                        # launch angle from horizontal
            # ballistic range on flat ground from this launch (point mass): R = (vh/g)*(vz+sqrt(vz^2)) ~ 2*vh*vz/g
            r_cur = 2.0 * vh * vz / G
            r_45 = v * v / G                                # same speed at 45deg
            k = int(just_to.sum().item()); n += k
            s_v += float(v.sum().item()); s_vx += float(vx.sum().item()); s_vy += float(vy.abs().sum().item())
            s_vz += float(vz.sum().item()); s_ang += float(ang.sum().item())
            s_r_cur += float(r_cur.sum().item()); s_r_45 += float(r_45.sum().item())
        prev_takeoff = env.has_taken_off.clone()

    print(f"\n[launch_diag] {os.path.basename(_resume_path)} | cmd dx={DX} | n_takeoffs={n}", flush=True)
    if n == 0:
        print("  no takeoffs", flush=True); return
    print(f"  launch speed |v|   = {s_v/n:.2f} m/s", flush=True)
    print(f"  forward vx         = {s_vx/n:.2f} m/s", flush=True)
    print(f"  |vy| lateral       = {s_vy/n:.2f} m/s", flush=True)
    print(f"  up vz              = {s_vz/n:.2f} m/s", flush=True)
    print(f"  launch angle       = {math.degrees(s_ang/n):.1f} deg   (45=range-optimal)", flush=True)
    print(f"  ballistic range now= {s_r_cur/n:.2f} m   (point-mass, this angle)", flush=True)
    print(f"  range @ 45deg      = {s_r_45/n:.2f} m   (SAME speed, optimal angle)", flush=True)
    print(f"  --> headroom from angle alone = {(s_r_45 - s_r_cur)/n:+.2f} m", flush=True)


if __name__ == '__main__':
    main()
