# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import re

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_load_path

import numpy as np
import torch
from isaacgym.torch_utils import get_euler_xyz
from legged_gym.utils.math import wrap_to_pi


def infer_checkpoint_iter(load_path):
    match = re.search(r"model_(\d+)\.pt", os.path.basename(load_path))
    if match is None:
        return None
    return int(match.group(1))


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    num_commands = env_cfg.commands.num_commands
    is_jump_task = (
        "jump" in args.task
        or hasattr(env_cfg.commands.ranges, "landing_dx")
        or hasattr(env_cfg.commands.ranges, "jump_height")
        or hasattr(env_cfg.commands.ranges, "jump_toggle")
        or hasattr(env_cfg.commands.ranges, "jump_command")
    )
    supports_heading_command = (
        (not is_jump_task)
        and num_commands > 3
        and hasattr(env_cfg.commands.ranges, "ang_vel_yaw")
    )
    experiment_name = train_cfg.runner.experiment_name if args.experiment_name is None else args.experiment_name
    load_run = train_cfg.runner.load_run if args.load_run is None else args.load_run
    checkpoint = train_cfg.runner.checkpoint if args.checkpoint is None else args.checkpoint

    checkpoint_iter = None
    try:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', experiment_name)
        load_path = get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)
        checkpoint_iter = infer_checkpoint_iter(load_path)
        if checkpoint_iter is not None:
            print(f"[Play] Resolved checkpoint: {os.path.basename(load_path)} -> iter {checkpoint_iter}")
    except Exception as exc:
        print(f"[Play] Warning: could not resolve checkpoint path for curriculum sync: {exc}")

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_length_s = 20
    # Keep the jump task on the same control stack used during training.
    if not is_jump_task:
        env_cfg.control.control_type = 'T'
    env_cfg.test.use_test = True
    if is_jump_task:
        env_cfg.test.use_test = True
        env_cfg.test.single_jump_play = True   # mirror training: one jump per episode, then disable jump_command for post-landing stand
        if (
            hasattr(env_cfg, "curriculum")
            and getattr(env_cfg.curriculum, "enabled", False)
            and hasattr(env_cfg.curriculum, "play_stage")
        ):
            env_cfg.curriculum.force_stage = int(env_cfg.curriculum.play_stage)
            if hasattr(env_cfg.curriculum, "play_stand_command_prob"):
                env_cfg.curriculum.stand_command_prob_after_takeoff = float(
                    env_cfg.curriculum.play_stand_command_prob
                )
            print(
                f"[Play] Forcing metric curriculum stage={env_cfg.curriculum.force_stage},"
                f" stand_prob={getattr(env_cfg.curriculum, 'stand_command_prob_after_takeoff', 'n/a')}"
            )
    env_cfg.test.checkpoint = 3000
    default_test_vel = getattr(env_cfg.test, "vel", None)
    if default_test_vel is None or len(default_test_vel) != num_commands:
        env_cfg.test.vel = torch.zeros(num_commands, dtype=torch.float32)
    else:
        env_cfg.test.vel = default_test_vel.clone().float()

    # === Play-time command override (edit the number to test different jump heights) ===
    # cmd layout: [lin_vel_x, lin_vel_y, ang_vel_yaw, jump_height, jump_command]
    # 修改下面这行即可改变跳跃高度（保持在训练范围 [0.40, 0.70] 内）
    env_cfg.test.vel[3] = 0.7
    print(f"[Play] cmd[3] (jump_height) = {float(env_cfg.test.vel[3]):.2f}")
    # ============================================================================

    if checkpoint_iter is not None:
        policy_steps_per_iter = train_cfg.runner.num_steps_per_env
        physics_steps_per_policy_step = max(1, int(round(1.0 / (env_cfg.sim.dt * env_cfg.growth.start_freq))))
        approx_physics_steps = checkpoint_iter * policy_steps_per_iter * physics_steps_per_policy_step
        env_cfg.test.checkpoint = checkpoint_iter * physics_steps_per_policy_step
    else:
        approx_physics_steps = None

    env_cfg.control.activation_process = True
    env_cfg.control.hill_model = True
    env_cfg.control.motor_fatigue = True
    env_cfg.commands.heading_command = supports_heading_command
    print(f"[Play] Using control_type={env_cfg.control.control_type}")

    env_cfg.terrain.mesh_type = 'plane'  # 'trimesh'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.terrain.terrain_proportions = [0, 1, 0, 0, 0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if approx_physics_steps is not None:
        env.common_step_counter = approx_physics_steps
        if hasattr(env, "step_count"):
            env.step_count = approx_physics_steps
        print(
            f"[Play] Syncing play-time curriculum approximately:"
            f" common_step_counter={env.common_step_counter},"
            f" step_count={getattr(env, 'step_count', -1)},"
            f" test.checkpoint={env_cfg.test.checkpoint}"
        )
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 10  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    img_idx = 0
    vel_x = 1.0
    if num_commands > 0 and not is_jump_task:
        env_cfg.test.vel[0] = vel_x
    if is_jump_task and env_cfg.test.use_test:
        print(f"[Play] Continuous jump command = {env_cfg.test.vel.tolist()}")
    elif is_jump_task:
        print("[Play] Jump task uses train-style command resampling.")
    change_vel = 0.2
    single_jump_play = bool(getattr(env_cfg.test, "single_jump_play", False))
    single_jump_initial_command = env_cfg.test.vel.clone()
    target_jump_cmd = float(single_jump_initial_command[4].item()) if env.commands.shape[1] > 4 else 0.0

    # Play state machine:
    #   pre_idle  → cmd[4]=0 for PRE_JUMP_IDLE_SECONDS (let robot settle from reset)
    #   jumping   → cmd[4]=target until just_landed fires
    #   post_stand→ cmd[4]=0 for POST_JUMP_STAND_SECONDS (verify landing stability)
    #   then manual reset → back to pre_idle
    PRE_JUMP_IDLE_SECONDS = 2.0
    POST_JUMP_STAND_SECONDS = 2.0   # short stand window; play state machine fires manual reset before atanassov landing instability triggers env collision/roll cutoff
    CONTINUOUS_JUMP = False         # mirror training: one jump per episode → manual reset → init pose each cycle
    PRE_JUMP_IDLE_STEPS = max(int(round(PRE_JUMP_IDLE_SECONDS / env.dt)), 1)
    POST_JUMP_STAND_STEPS = max(int(round(POST_JUMP_STAND_SECONDS / env.dt)), 1)

    # Suppress env's one_jump auto-reset (push env's post_jump_stand_steps beyond play's window)
    # so the play state machine alone controls cycle timing. Env can still reset on real
    # termination (collision/roll/time_out) — that path is intentional, indicates actual fall.
    if single_jump_play and hasattr(env.cfg.rewards, "post_jump_stand_steps"):
        env.cfg.rewards.post_jump_stand_steps = max(int(env.cfg.rewards.post_jump_stand_steps), POST_JUMP_STAND_STEPS + 200)

    # Disable Reference State Init (RSI). RSI is a training bootstrap that resets envs mid-air
    # with upward velocity + jumping_state=True. At play time it causes apparent "stuck" cycles:
    # env auto-resets via RSI → robot crashes (collision/roll) → reset → RSI → loop. Play should
    # always start from default standing pose.
    if hasattr(env.cfg.rewards, "rsi_prob"):
        env.cfg.rewards.rsi_prob = 0.0
    if hasattr(env.cfg.rewards, "atanassov_rsi_prob"):
        env.cfg.rewards.atanassov_rsi_prob = 0.0

    # Disable atanassov's too_low termination (base_z < threshold). At play time the policy
    # autonomously drifts into a low squat under cmd[4]=0, dropping base below the training
    # threshold (0.12m) and resetting. Removing the floor lets the cycle continue regardless
    # of how low the robot crouches.
    if hasattr(env.cfg.rewards, "atanassov_terminate_base_height"):
        env.cfg.rewards.atanassov_terminate_base_height = -1.0

    # Diagnostic: wrap check_termination so we know which condition triggered each env auto-reset.
    _orig_check_termination = env.check_termination
    def _check_term_with_log(*args, **kwargs):
        _orig_check_termination(*args, **kwargs)
        if env.reset_buf[robot_index].item():
            roll_all, pitch_all, _ = get_euler_xyz(env.base_quat)
            roll_v = wrap_to_pi(roll_all)[robot_index].item()
            pitch_v = wrap_to_pi(pitch_all)[robot_index].item()
            coll_force = torch.sum(
                torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1),
                dim=-1,
            )[robot_index].item()
            base_z = env.root_states[robot_index, 2].item()
            too_low_thresh = float(getattr(env.cfg.rewards, "atanassov_terminate_base_height", 0.0))
            ep_len = env.episode_length_buf[robot_index].item()
            max_ep = env.max_episode_length
            jumping = bool(env.jumping_state[robot_index].item())
            reasons = []
            if abs(roll_v) > 2.4: reasons.append(f"roll_cutoff(|{roll_v:.2f}|>2.4)")
            if coll_force > 0.2: reasons.append(f"collision({coll_force:.2f}N>0.2)")
            if ep_len > max_ep: reasons.append(f"timeout({ep_len}>{max_ep})")
            if base_z < too_low_thresh and too_low_thresh > 0: reasons.append(f"too_low({base_z:.3f}<{too_low_thresh:.2f})")
            print(
                f"[TermDebug] ep_step={ep_len}: reasons={reasons or 'UNKNOWN'} | "
                f"base_z={base_z:.3f}, roll={roll_v:.2f}, pitch={pitch_v:.2f}, "
                f"contact_force={coll_force:.2f}N, jumping_state={jumping}"
            )
    env.check_termination = _check_term_with_log

    play_phase = "pre_idle"
    play_phase_step = 0

    # Initialize: cmd[4]=0, robot stays idle until pre_idle timer elapses.
    if single_jump_play and env.commands.shape[1] > 4:
        env_cfg.test.vel[4] = 0.0
        env.commands[:, 4] = 0.0
        if hasattr(env, "single_jump_play_done"):
            env.single_jump_play_done[:] = True   # double-lock: cfg.test.vel[4]=0 AND play_done flag → cmd absolutely 0
        print(
            f"[Play] State machine: pre_idle ({PRE_JUMP_IDLE_SECONDS}s = {PRE_JUMP_IDLE_STEPS} steps)"
            f" → jump → post_stand ({POST_JUMP_STAND_SECONDS}s = {POST_JUMP_STAND_STEPS} steps) → reset"
        )

    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # Diagnostic state print every 50 steps + on every play_phase transition (caller prints).
        if i % 50 == 0:
            cmd4 = float(env.commands[robot_index, 4])
            test_vel4 = float(env_cfg.test.vel[4])
            base_z = float(env.root_states[robot_index, 2])
            js = bool(env.jumping_state[robot_index].item())
            hto = bool(env.has_taken_off[robot_index].item())
            jcc = int(env.jump_step_counter[robot_index].item()) if hasattr(env, "jump_step_counter") else -1
            print(
                f"[State] step={i} phase={play_phase}(ps={play_phase_step}): "
                f"cmd[4]={cmd4:.2f} (test.vel[4]={test_vel4:.2f}), "
                f"base_z={base_z:.3f}, jumping={js}, taken_off={hto}, jump_step={jcc}"
            )
        # === Play state machine ===
        if single_jump_play:
            # Auto-reset from env (termination/fall or timeout) → back to pre_idle
            if bool(dones[robot_index].item()):
                play_phase = "pre_idle"
                play_phase_step = 0
                env_cfg.test.vel[4] = 0.0
                if env.commands.shape[1] > 4:
                    env.commands[:, 4] = 0.0
                if hasattr(env, "single_jump_play_done"):
                    env.single_jump_play_done[:] = True
                print(f"[Play] Step {i}: env auto-reset → pre_idle ({PRE_JUMP_IDLE_SECONDS}s)")
            else:
                play_phase_step += 1
                if play_phase == "pre_idle":
                    # Enforce cmd[4]=0 throughout idle (defense against any stray state)
                    env_cfg.test.vel[4] = 0.0
                    if env.commands.shape[1] > 4:
                        env.commands[:, 4] = 0.0
                    if hasattr(env, "single_jump_play_done"):
                        env.single_jump_play_done[:] = True
                    if play_phase_step >= PRE_JUMP_IDLE_STEPS:
                        # Transition to jumping: enable cmd[4]=target
                        play_phase = "jumping"
                        play_phase_step = 0
                        env_cfg.test.vel[4] = target_jump_cmd
                        if env.commands.shape[1] > 4:
                            env.commands[:, 4] = target_jump_cmd
                        if hasattr(env, "single_jump_play_done"):
                            env.single_jump_play_done[:] = False  # unlock jump command in env
                        print(f"[Play] Step {i}: pre_idle done → jumping (cmd[4]={target_jump_cmd})")
                elif play_phase == "jumping":
                    # Kill cmd[4] as soon as robot is airborne. Otherwise cmd[4]=1 stays active
                    # throughout the jumping phase, and if env-internal jumping_state ever clears
                    # before play detects just_landed, `ready_to_jump` re-fires → second jump
                    # (visually: "continuous jumping").
                    if hasattr(env, "has_taken_off") and bool(env.has_taken_off[robot_index].item()):
                        env_cfg.test.vel[4] = 0.0
                        if env.commands.shape[1] > 4:
                            env.commands[:, 4] = 0.0
                        if hasattr(env, "single_jump_play_done"):
                            env.single_jump_play_done[:] = True
                    # Detect jump cycle completion. Three exit signals:
                    #   1. just_landed: clean landing (success path)
                    #   2. single_jump_command_done: env says jump cycle ended (success OR takeoff_timeout fail)
                    #   3. play_phase_step timeout: safety net if neither signal arrives
                    single_jump_landed = hasattr(env, "just_landed") and bool(env.just_landed[robot_index].item())
                    cmd_done_in_env = (
                        hasattr(env, "single_jump_command_done")
                        and bool(env.single_jump_command_done[robot_index].item())
                    )
                    JUMPING_PHASE_TIMEOUT_STEPS = int(round(4.0 / env.dt))   # 4s hard cap
                    timeout_exit = play_phase_step >= JUMPING_PHASE_TIMEOUT_STEPS

                    exit_reason = None
                    if single_jump_landed:
                        exit_reason = "just_landed"
                    elif cmd_done_in_env:
                        exit_reason = "env_cmd_done (likely takeoff_timeout fail)"
                    elif timeout_exit:
                        exit_reason = "play timeout fallback"

                    if exit_reason is not None:
                        play_phase = "post_stand"
                        play_phase_step = 0
                        env_cfg.test.vel[4] = 0.0
                        if env.commands.shape[1] > 4:
                            env.commands[:, 4] = 0.0
                        if hasattr(env, "single_jump_play_done"):
                            env.single_jump_play_done[:] = True   # lock cmd[4]=0 in env
                        print(
                            f"[Play] Step {i}: jump cycle done ({exit_reason})"
                            f" peak={env.peak_base_height[robot_index].item():.3f}"
                            f" base={env.root_states[robot_index, 2].item():.3f}"
                            f" → post_stand ({POST_JUMP_STAND_SECONDS}s)"
                        )
                elif play_phase == "post_stand":
                    # Enforce cmd[4]=0 throughout stand
                    env_cfg.test.vel[4] = 0.0
                    if env.commands.shape[1] > 4:
                        env.commands[:, 4] = 0.0
                    if hasattr(env, "single_jump_play_done"):
                        env.single_jump_play_done[:] = True
                    if play_phase_step >= POST_JUMP_STAND_STEPS:
                        env_ids = torch.arange(env.num_envs, device=env.device)
                        if CONTINUOUS_JUMP:
                            # Re-arm for next jump without resetting robot pose.
                            # Clear the env's jump-tracking flags so the next cmd[4]=1
                            # is recognised as a new jump cycle, but keep the robot
                            # standing where it landed.
                            if hasattr(env, "_reset_jump_buffers"):
                                env._reset_jump_buffers(env_ids)
                            # Update landing target to current xy so the robot is
                            # rewarded for landing back where it currently is.
                            if hasattr(env, "atan_p_des"):
                                env.atan_p_des[:, 0] = env.root_states[:, 0]
                                env.atan_p_des[:, 1] = env.root_states[:, 1]
                            env_cfg.test.vel[4] = target_jump_cmd
                            if env.commands.shape[1] > 4:
                                env.commands[:, 4] = target_jump_cmd
                            if hasattr(env, "single_jump_play_done"):
                                env.single_jump_play_done[:] = False
                            play_phase = "jumping"
                            play_phase_step = 0
                            env.compute_observations()
                            obs = env.get_observations()
                            print(f"[Play] Step {i}: post_stand done → next jump (continuous)")
                        else:
                            # Manual reset → back to pre_idle
                            env.reset_idx(env_ids)
                            play_phase = "pre_idle"
                            play_phase_step = 0
                            env_cfg.test.vel[4] = 0.0
                            if env.commands.shape[1] > 4:
                                env.commands[:, 4] = 0.0
                            if hasattr(env, "single_jump_play_done"):
                                env.single_jump_play_done[:] = True
                            env.compute_observations()
                            obs = env.get_observations()
                            print(f"[Play] Step {i}: post_stand done → manual reset → pre_idle")
        foot_z = env.rigid_body_states[0, env.feet_indices, 2].cpu().numpy()
        if CHANGE_VEL and supports_heading_command and not is_jump_task:
            if i % 100 == 0:
                if vel_x > 1.5 or vel_x < -0.0:
                    # change_vel = -change_vel
                    change_vel = 0
                vel_x += change_vel
                # vel_x = 0.5
                env_cfg.test.vel[0] = vel_x
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            robot_pos = env.root_states[0, :3].cpu().numpy()
            camera_position = robot_pos + np.array([1, 1, 1])
            env.set_camera(camera_position, robot_pos)
        if i < stop_state_log:
            command_0 = env.commands[robot_index, 0].item() if env.commands.shape[1] > 0 else 0.0
            command_1 = env.commands[robot_index, 1].item() if env.commands.shape[1] > 1 else 0.0
            command_2 = env.commands[robot_index, 2].item() if env.commands.shape[1] > 2 else 0.0
            command_3 = env.commands[robot_index, 3].item() if env.commands.shape[1] > 3 else 0.0
            command_4 = env.commands[robot_index, 4].item() if env.commands.shape[1] > 4 else 0.0
            logger.log_states(
                {
                    'actions': actions[robot_index].detach().cpu().numpy(),
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': command_0,
                    'command_y': command_1,
                    'command_yaw': command_2,
                    'command_jump_height': command_3,
                    'command_jump': command_4,
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'base_height': env.root_states[robot_index, 2].item(),
                    'torques': env.torques[robot_index].cpu().numpy(),
                    'dof_vels': env.dof_vel[robot_index].cpu().numpy(),
                    'foot_z': foot_z,
                    'reward': rews[robot_index].cpu().numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()
        # (Old post-landing reset block removed; play state machine above handles reset timing.)


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    CHANGE_VEL = True
    args = get_args()
    play(args)
