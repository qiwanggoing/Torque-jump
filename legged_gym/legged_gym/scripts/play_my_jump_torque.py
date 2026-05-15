import os

import isaacgym
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import export_policy_as_jit, get_args, task_registry


EXPORT_POLICY = True
RECORD_FRAMES = False
MOVE_CAMERA = True


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def play(args):
    if args.task == "anymal_c_flat":
        args.task = "go2_my_jump_torque"

    debug_dump_path = os.environ.get("GO2_PLAY_DEBUG_NPZ")
    debug_dump_steps = int(os.environ.get("GO2_PLAY_DEBUG_STEPS", "5"))
    debug_records = []

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    num_commands = env_cfg.commands.num_commands

    target_command = torch.zeros(num_commands, dtype=torch.float32)
    if num_commands > 0:
        target_command[0] = _env_float("GO2_PLAY_VX", 0.0)
    if num_commands > 1:
        target_command[1] = _env_float("GO2_PLAY_VY", 0.0)
    if num_commands > 2:
        target_command[2] = _env_float("GO2_PLAY_WZ", 0.0)

    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.commands.resampling_time = 1e9
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.loss_rate = 0.0
    env_cfg.test.use_test = True
    env_cfg.test.vel = target_command

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if args.checkpoint != -1:
        try:
            checkpoint_iter = int(args.checkpoint)
            curriculum_step = checkpoint_iter * train_cfg.runner.num_steps_per_env
            env.common_step_counter = curriculum_step
            if hasattr(env, "step_count"):
                env.step_count = curriculum_step
            print(f"[Play my-jump] Synced curriculum step to {curriculum_step}")
        except ValueError:
            print("[Play my-jump] Warning: checkpoint is not an integer; curriculum was not synced.")

    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    robot_index = 0
    img_idx = 0
    print(f"[Play my-jump] task={args.task} command={target_command.tolist()}")

    for i in range(10 * int(env.max_episode_length)):
        env.commands[:, :num_commands] = target_command.to(env.device)

        obs_in = obs.detach().clone()
        actions = policy(obs_in)
        obs, _, rews, dones, infos = env.step(actions.detach())

        if debug_dump_path and len(debug_records) < debug_dump_steps:
            debug_records.append(
                {
                    "step": i,
                    "common_step_counter": int(env.common_step_counter),
                    "command": env.commands[robot_index, :num_commands].detach().cpu().numpy().astype(np.float32),
                    "obs_in": obs_in[robot_index].detach().cpu().numpy().astype(np.float32),
                    "obs_out": obs[robot_index].detach().cpu().numpy().astype(np.float32),
                    "action": actions[robot_index].detach().cpu().numpy().astype(np.float32),
                    "tau": env.torques[robot_index].detach().cpu().numpy().astype(np.float32),
                    "q": env.dof_pos[robot_index].detach().cpu().numpy().astype(np.float32),
                    "dq": env.dof_vel[robot_index].detach().cpu().numpy().astype(np.float32),
                    "base_lin_vel": env.base_lin_vel[robot_index].detach().cpu().numpy().astype(np.float32),
                    "base_ang_vel": env.base_ang_vel[robot_index].detach().cpu().numpy().astype(np.float32),
                    "growth": np.array([float(getattr(env, "general_scale", 0.0))], dtype=np.float32),
                }
            )
            if len(debug_records) == debug_dump_steps:
                os.makedirs(os.path.dirname(debug_dump_path) or ".", exist_ok=True)
                np.savez(
                    debug_dump_path,
                    step=np.array([r["step"] for r in debug_records], dtype=np.int32),
                    common_step_counter=np.array(
                        [r["common_step_counter"] for r in debug_records],
                        dtype=np.int32,
                    ),
                    command=np.stack([r["command"] for r in debug_records], axis=0),
                    obs_in=np.stack([r["obs_in"] for r in debug_records], axis=0),
                    obs_out=np.stack([r["obs_out"] for r in debug_records], axis=0),
                    action=np.stack([r["action"] for r in debug_records], axis=0),
                    tau=np.stack([r["tau"] for r in debug_records], axis=0),
                    q=np.stack([r["q"] for r in debug_records], axis=0),
                    dq=np.stack([r["dq"] for r in debug_records], axis=0),
                    base_lin_vel=np.stack([r["base_lin_vel"] for r in debug_records], axis=0),
                    base_ang_vel=np.stack([r["base_ang_vel"] for r in debug_records], axis=0),
                    growth=np.stack([r["growth"] for r in debug_records], axis=0),
                )
                print(f"[Play my-jump debug] Saved first {debug_dump_steps} records to: {debug_dump_path}")

        if MOVE_CAMERA and not env.headless:
            robot_pos = env.root_states[robot_index, :3].cpu().numpy()
            camera_position = robot_pos + np.array([-2.5, 1.5, 1.0])
            env.set_camera(camera_position, robot_pos)

        if RECORD_FRAMES and i % 2:
            filename = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "frames",
                f"{img_idx}.png",
            )
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1

        if i % 100 == 0:
            actual_vel = env.base_lin_vel[robot_index, 0].item()
            max_torque = torch.max(torch.abs(env.torques[robot_index])).item()
            height = env.root_states[robot_index, 2].item()
            print(
                f"Step {i:5d} | Cmd vx: {target_command[0].item():.2f} | "
                f"Actual vx: {actual_vel:.3f} | Height: {height:.3f} | "
                f"Max torque: {max_torque:.2f} Nm | Reward: {rews[robot_index].item():.3f}"
            )


if __name__ == "__main__":
    args = get_args()
    play(args)
