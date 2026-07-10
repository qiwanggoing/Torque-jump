"""Freeze the robot in chosen leg POSES in the VIEWER (NOT headless) so you can eyeball the 'straight leg'.

Shows 3 envs side by side, each held at a fixed pose (gravity off), so you can compare knee extension:
  env0 = squat bottom (calf -2.40)
  env1 = current takeoff (calf -1.16, the 'tucks early' pose)
  env2 = straightest possible (calf -0.84 = the hardware extension limit = baseline's takeoff)
All use the SAME thigh angle (default 0.75) so only the KNEE differs. Override angles with env vars:
  TQ_THIGH (thigh, default 0.75) ; TQ_CALF0/1/2 (the three calf angles).
Run (interactive, viewer):
  cd ~/torque_jump2/SATA/legged_gym
  python legged_gym/scripts/show_pose.py --task=go2_omnijump_landing_torque --num_envs=3
Press V to toggle sync, ESC/close window to quit.
"""
import isaacgym  # noqa: F401
import torch
from isaacgym import gymtorch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry, get_args
import os

THIGH = float(os.environ.get("TQ_THIGH", "0.75"))
CALFS = [float(os.environ.get("TQ_CALF0", "-2.40")),
         float(os.environ.get("TQ_CALF1", "-1.16")),
         float(os.environ.get("TQ_CALF2", "-0.84"))]


def main():
    args = get_args()
    args.headless = False                      # force the viewer on
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 3
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # DOF order per leg = [hip, thigh, calf] x 4 (FL,FR,RL,RR). Build the 3 target poses (one per env).
    def pose(calf):
        p = torch.zeros(12, device=env.device)
        for leg in range(4):
            p[3 * leg + 0] = 0.0      # hip (no abduction)
            p[3 * leg + 1] = THIGH    # thigh
            p[3 * leg + 2] = calf     # calf
        return p
    targets = torch.stack([pose(c) for c in CALFS], dim=0)   # (3,12)

    # a fixed, floating, upright base pose so the leg is clearly visible above the ground
    base_h = 0.55
    root = env.root_states.clone()
    root[:, 2] = env.env_origins[:, 2] + base_h
    root[:, 3:7] = torch.tensor([0., 0., 0., 1.], device=env.device)  # upright quat
    root[:, 7:13] = 0.0

    print(f"\n[show_pose] thigh={THIGH} | env0 calf={CALFS[0]} (squat)  env1 calf={CALFS[1]} (current takeoff)  "
          f"env2 calf={CALFS[2]} (straightest = -0.84 limit)")
    print("  近处=env0. 看膝盖:从左折→右直. calf 到 -0.84 就是最直(硬件极限,永远弯~48°).\n")

    while not env.gym.query_viewer_has_closed(env.viewer):
        # hold the pose: overwrite dof + root state every frame (gravity/contact would otherwise move it)
        env.dof_pos[:] = targets
        env.dof_vel[:] = 0.0
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.root_states[:] = root
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)
        env.gym.sync_frame_time(env.sim)


if __name__ == '__main__':
    main()
