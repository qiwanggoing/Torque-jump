# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import time

from isaacgym import gymtorch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch


POSE_HOLD_STEPS = 180
DEFAULT_TASK = "go2_omnijump_curriculum_torque"
RESET_TO_DEFAULT_EACH_POSE = True
FREEZE_BASE = False


def _target_tensor(env, name):
    if name == "default":
        return env.default_dof_pos.squeeze(0)
    if name == "ground":
        return env.q_ground_target
    if name == "air":
        return env.q_air_target
    if name == "prelanding":
        return env.q_pre_target
    raise ValueError(f"Unknown pose target: {name}")


def _print_pose_targets(env, names):
    targets = [(name, _target_tensor(env, name).detach().cpu().numpy()) for name in names]
    default = targets[0][1]

    print("\n[PosePreview] Joint target angles in radians")
    header = f"{'joint':<28}" + "".join(f"{name:>13}" for name, _ in targets)
    print(header)
    print("-" * len(header))
    for joint_idx, joint_name in enumerate(env.dof_names):
        row = f"{joint_name:<28}"
        for _, target in targets:
            row += f"{target[joint_idx]:>13.4f}"
        print(row)

    print("\n[PosePreview] Delta from default in radians")
    header = f"{'joint':<28}" + "".join(f"{name:>13}" for name, _ in targets[1:])
    print(header)
    print("-" * len(header))
    for joint_idx, joint_name in enumerate(env.dof_names):
        row = f"{joint_name:<28}"
        for _, target in targets[1:]:
            row += f"{target[joint_idx] - default[joint_idx]:>13.4f}"
        print(row)

    print(
        "\n[PosePreview] Note: these targets are solved from foot height."
        " Hip ab/ad joints stay at the default value; thigh/calf joints change."
    )


def _reset_preview_state(env, base_height):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env_ids_int32 = env_ids.to(dtype=torch.int32)

    env.dof_pos[:] = env.default_dof_pos.expand(env.num_envs, -1)
    env.dof_vel[:] = 0.0

    env.root_states[:, :3] = env.env_origins
    env.root_states[:, 2] += base_height
    env.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    env.root_states[:, 7:13] = 0.0

    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)


def _step_pd_target(env, target, base_height):
    target = target.to(env.device).unsqueeze(0).expand(env.num_envs, -1)
    torques = env.p_gains * (target - env.dof_pos) - env.d_gains * env.dof_vel
    torques = torch.clip(torques, -env.torque_limits, env.torque_limits)
    env.torques[:] = torques

    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
    env.gym.simulate(env.sim)
    if env.device == "cpu":
        env.gym.fetch_results(env.sim, True)

    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    if FREEZE_BASE:
        env.root_states[:, :3] = env.env_origins
        env.root_states[:, 2] += base_height
        env.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        env.root_states[:, 7:13] = 0.0
        env_ids_int32 = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        env.gym.refresh_actor_root_state_tensor(env.sim)


def preview(args):
    if args.task == "go2":
        args.task = DEFAULT_TASK

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 60
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    names = ("default", "ground", "air", "prelanding")
    _print_pose_targets(env, names)

    base_height = float(getattr(env_cfg.rewards, "base_height_target", env.root_states[0, 2].item()))
    camera_position = np.array([1.4, -1.2, 0.8])
    camera_target = np.array([0.0, 0.0, 0.25])
    if env.viewer is not None:
        env.set_camera(camera_position, camera_target)
        print("\n[PosePreview] Viewer cycling: default -> ground -> air -> prelanding")
        print("[PosePreview] Close the viewer or press Esc to exit.")
    else:
        print("\n[PosePreview] Headless mode: printed q targets only.")
        return

    step = 0
    while True:
        pose_idx = (step // POSE_HOLD_STEPS) % len(names)
        pose_name = names[pose_idx]
        if step % POSE_HOLD_STEPS == 0:
            if RESET_TO_DEFAULT_EACH_POSE:
                _reset_preview_state(env, base_height)
            print(f"[PosePreview] PD tracking pose: {pose_name}")
        _step_pd_target(env, _target_tensor(env, pose_name), base_height)
        env.render(sync_frame_time=True)
        time.sleep(0.001)
        step += 1


if __name__ == "__main__":
    args = get_args()
    preview(args)
