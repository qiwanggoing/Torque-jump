from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch


def play_zero(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    num_commands = env_cfg.commands.num_commands
    is_jump_task = (
        "jump" in args.task
        or hasattr(env_cfg.commands.ranges, "landing_dx")
        or hasattr(env_cfg.commands.ranges, "jump_height")
        or hasattr(env_cfg.commands.ranges, "jump_toggle")
    )
    test_checkpoint = 0 if args.checkpoint is None else args.checkpoint

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_length_s = 20
    if not is_jump_task:
        env_cfg.control.control_type = 'T'
    env_cfg.test.use_test = True
    env_cfg.test.checkpoint = test_checkpoint
    env_cfg.test.vel = torch.zeros(num_commands, dtype=torch.float32)

    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.heading_command = False

    if is_jump_task and num_commands >= 1:
        env_cfg.test.vel = torch.zeros(num_commands, dtype=torch.float32)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.common_step_counter = 0
    if hasattr(env, "step_count"):
        env.step_count = 0

    print(f"[PlayZero] task={args.task}")
    print(f"[PlayZero] control_type={env_cfg.control.control_type}")
    print(f"[PlayZero] zero-action standing test command={env_cfg.test.vel.tolist()}")
    print(f"[PlayZero] growth checkpoint={env_cfg.test.checkpoint}")

    obs = env.get_observations()
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    for _ in range(10 * int(env.max_episode_length)):
        obs, _, _, _, _ = env.step(zero_actions)
        if not args.headless:
            robot_pos = env.root_states[0, :3].cpu().numpy()
            camera_position = robot_pos + np.array([1.0, 1.0, 0.8])
            env.set_camera(camera_position, robot_pos)


if __name__ == '__main__':
    args = get_args()
    play_zero(args)
