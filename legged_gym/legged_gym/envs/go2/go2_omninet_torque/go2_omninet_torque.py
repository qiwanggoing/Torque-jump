import numpy as np
import torch
from isaacgym.torch_utils import *

from legged_gym.envs.go2.go2_omninet_torque.go2_omninet_torque_config import GO2OmniNetTorqueCfg
from legged_gym.envs.go2.go2_torque.go2_torque import GO2Torque


class GO2OmniNetTorque(GO2Torque):
    """OmniNet-style jumping task with history + estimator targets on top of SATA torque actuation."""

    cfg: GO2OmniNetTorqueCfg

    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        super().post_physics_step()

    def check_termination(self):
        super().check_termination()
        if hasattr(self, "non_feet_indices") and self.non_feet_indices.numel() > 0:
            non_foot_contact = torch.any(
                torch.norm(self.contact_forces[:, self.non_feet_indices, :], dim=-1) > 1.0,
                dim=1,
            )
            self.reset_buf |= non_foot_contact

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self._log_jump_episode_stats(env_ids)
        self._reset_omninet_buffers(env_ids)
        self._prime_history(env_ids)

    def _init_buffers(self):
        super()._init_buffers()
        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.commands_scale = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float, device=self.device)

        self.non_feet_indices = self._build_non_feet_indices()
        self.command_input = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.residual_torques_action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.pd_prior_torques = torch.zeros_like(self.residual_torques_action)
        self.pd_prior_alpha = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)

        history_length = self.cfg.env.history_length
        single_obs_dim = self.cfg.env.num_single_obs
        terrain_obs_dim = self.cfg.env.num_terrain_obs
        self.obs_history = torch.zeros(
            self.num_envs, history_length, single_obs_dim, dtype=torch.float, device=self.device
        )
        self.terrain_obs_buf = torch.zeros(self.num_envs, terrain_obs_dim, dtype=torch.float, device=self.device)
        self.estimator_target_buf = torch.zeros(
            self.num_envs, self.cfg.env.num_estimator_targets, dtype=torch.float, device=self.device
        )
        self.clean_single_obs_buf = torch.zeros(self.num_envs, single_obs_dim, dtype=torch.float, device=self.device)

        self.jumping_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_taken_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prelanding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.landing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_took_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_jump_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.jump_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stand_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.landing_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.airborne_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.peak_base_height = self.root_states[:, 2].clone()
        self.landing_min_height = self.root_states[:, 2].clone()

        self.jump_starts = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_flights = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_landings = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_completed_cycles = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.successful_jumps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.peak_height_error_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.peak_height_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_evaluations = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.q_ground_target = self._solve_pose_from_foot_height(self.cfg.rewards.ground_foot_height)
        self.q_air_target = self._solve_pose_from_foot_height(self.cfg.rewards.air_foot_height)
        self.q_pre_target = self._solve_pose_from_foot_height(self.cfg.rewards.prelanding_foot_height)
        self.default_joint_pd_target = self.q_ground_target.unsqueeze(0).repeat(self.num_envs, 1)

        self.previous_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        self._prime_history(torch.arange(self.num_envs, device=self.device))

    def _build_non_feet_indices(self):
        all_body_indices = torch.arange(self.num_bodies, dtype=torch.long, device=self.device)
        feet_mask = torch.zeros(self.num_bodies, dtype=torch.bool, device=self.device)
        feet_mask[self.feet_indices.long()] = True
        return all_body_indices[~feet_mask]

    def _solve_pose_from_foot_height(self, foot_height):
        l1 = self.cfg.rewards.ik_thigh_length
        l2 = self.cfg.rewards.ik_calf_length
        x = self.cfg.rewards.ik_nominal_foot_x
        z = -foot_height
        reach = np.clip(np.sqrt(x * x + z * z), 1e-4, l1 + l2 - 1e-4)
        cos_knee = np.clip((reach * reach - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
        calf = -np.arccos(cos_knee)
        thigh = np.arctan2(z, x) - np.arctan2(l2 * np.sin(calf), l1 + l2 * np.cos(calf))
        thigh += np.pi / 2.0

        target = self.default_joint_pd_target[0].clone()
        target[[1, 4, 7, 10]] = thigh
        target[[2, 5, 8, 11]] = calf
        return target

    def _reset_omninet_buffers(self, env_ids):
        self.jumping_state[env_ids] = False
        self.has_taken_off[env_ids] = False
        self.has_landed[env_ids] = False
        self.airborne[env_ids] = False
        self.prelanding[env_ids] = False
        self.landing[env_ids] = False
        self.just_took_off[env_ids] = False
        self.just_landed[env_ids] = False
        self.last_jump_success[env_ids] = False

        self.jump_step_counter[env_ids] = 0
        self.stand_step_counter[env_ids] = 0
        self.landing_step_counter[env_ids] = 0
        self.airborne_time[env_ids] = 0.0
        self.peak_base_height[env_ids] = self.root_states[env_ids, 2]
        self.landing_min_height[env_ids] = self.root_states[env_ids, 2]

        self.jump_starts[env_ids] = 0.0
        self.jump_flights[env_ids] = 0.0
        self.jump_landings[env_ids] = 0.0
        self.jump_completed_cycles[env_ids] = 0.0
        self.successful_jumps[env_ids] = 0.0
        self.peak_height_error_sum[env_ids] = 0.0
        self.peak_height_sum[env_ids] = 0.0
        self.jump_evaluations[env_ids] = 0.0

    def _prime_history(self, env_ids):
        if len(env_ids) == 0:
            return
        single_obs = self._get_single_observation()
        tiled_obs = single_obs[env_ids].unsqueeze(1).repeat(1, self.cfg.env.history_length, 1)
        self.obs_history[env_ids] = tiled_obs
        self.clean_single_obs_buf[env_ids] = single_obs[env_ids]

    def _log_jump_episode_stats(self, env_ids):
        if "episode" not in self.extras:
            self.extras["episode"] = {}
        jump_den = torch.clamp(self.jump_starts[env_ids], min=1.0)
        eval_den = torch.clamp(self.jump_evaluations[env_ids], min=1.0)
        self.extras["episode"]["jump_flight_rate"] = torch.mean(self.jump_flights[env_ids] / jump_den)
        self.extras["episode"]["jump_landing_rate"] = torch.mean(self.jump_landings[env_ids] / jump_den)
        self.extras["episode"]["jump_completed_cycles"] = torch.mean(self.jump_completed_cycles[env_ids])
        self.extras["episode"]["successful_jump_rate"] = torch.mean(self.successful_jumps[env_ids] / jump_den)
        self.extras["episode"]["peak_height_error"] = torch.mean(self.peak_height_error_sum[env_ids] / eval_den)
        self.extras["episode"]["mean_peak_height"] = torch.mean(self.peak_height_sum[env_ids] / eval_den)

    def _post_physics_step_callback(self):
        resample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (
            (self.episode_length_buf % max(resample_interval, 1) == 0)
            & (~self.jumping_state)
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.test.use_test:
            self.commands[:] = self.cfg.test.vel.to(self.device)
        self._update_jump_state()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_ranges["jump_height"][0],
            self.command_ranges["jump_height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

    def _start_jump(self, env_ids):
        if len(env_ids) == 0:
            return
        self.jumping_state[env_ids] = True
        self.has_taken_off[env_ids] = False
        self.has_landed[env_ids] = False
        self.airborne[env_ids] = False
        self.prelanding[env_ids] = False
        self.landing[env_ids] = False
        self.just_took_off[env_ids] = False
        self.just_landed[env_ids] = False
        self.last_jump_success[env_ids] = False
        self.jump_step_counter[env_ids] = 0
        self.landing_step_counter[env_ids] = 0
        self.airborne_time[env_ids] = 0.0
        self.peak_base_height[env_ids] = self.root_states[env_ids, 2]
        self.landing_min_height[env_ids] = self.root_states[env_ids, 2]
        self.jump_starts[env_ids] += 1.0

    def _finish_jump(self, env_ids, completed=True):
        if len(env_ids) == 0:
            return
        self.jumping_state[env_ids] = False
        self.has_taken_off[env_ids] = False
        self.has_landed[env_ids] = False
        self.airborne[env_ids] = False
        self.prelanding[env_ids] = False
        self.landing[env_ids] = False
        self.just_took_off[env_ids] = False
        self.just_landed[env_ids] = False
        self.jump_step_counter[env_ids] = 0
        self.landing_step_counter[env_ids] = 0
        self.airborne_time[env_ids] = 0.0
        self.stand_step_counter[env_ids] = 0
        self.landing_min_height[env_ids] = self.root_states[env_ids, 2]
        if completed:
            self.jump_completed_cycles[env_ids] += 1.0

    def _update_jump_state(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        any_foot_contact = torch.any(contact, dim=1)
        all_feet_contact = torch.all(contact, dim=1)

        stable_stand = (
            all_feet_contact
            & (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.35)
            & (torch.norm(self.base_ang_vel[:, :2], dim=1) < 3.0)
            & (torch.abs(self.base_lin_vel[:, 2]) < 0.35)
        )
        self.stand_step_counter = torch.where(
            (~self.jumping_state) & stable_stand,
            self.stand_step_counter + 1,
            torch.zeros_like(self.stand_step_counter),
        )

        ready_to_jump = (~self.jumping_state) & (self.stand_step_counter >= self.cfg.rewards.stand_rearm_steps)
        start_ids = ready_to_jump.nonzero(as_tuple=False).flatten()
        self._start_jump(start_ids)

        self.just_took_off = self.jumping_state & (~self.has_taken_off) & (~any_foot_contact)
        self.has_taken_off |= self.just_took_off
        self.airborne = self.jumping_state & self.has_taken_off & (~self.has_landed) & (~any_foot_contact)
        self.airborne_time += self.airborne.float() * self.dt
        self.peak_base_height = torch.where(
            self.jumping_state,
            torch.maximum(self.peak_base_height, self.root_states[:, 2]),
            self.peak_base_height,
        )
        descending = self.base_lin_vel[:, 2] < -0.05
        prelanding_height = torch.maximum(
            self.commands[:, 3] - self.cfg.rewards.prelanding_height_margin,
            torch.full_like(self.commands[:, 3], self.cfg.rewards.base_height_target + 0.04),
        )
        self.prelanding = self.airborne & descending & (self.root_states[:, 2] <= prelanding_height)

        self.just_landed = self.jumping_state & self.has_taken_off & (~self.has_landed) & any_foot_contact
        self.has_landed |= self.just_landed
        self.landing = self.jumping_state & self.has_landed
        self.jump_step_counter = torch.where(
            self.jumping_state,
            self.jump_step_counter + 1,
            self.jump_step_counter,
        )
        self.landing_step_counter = torch.where(
            self.landing,
            self.landing_step_counter + 1,
            torch.zeros_like(self.landing_step_counter),
        )
        self.landing_min_height = torch.where(
            self.landing,
            torch.minimum(self.landing_min_height, self.root_states[:, 2]),
            self.landing_min_height,
        )

        self.last_jump_success[:] = False
        if torch.any(self.just_took_off):
            self.jump_flights[self.just_took_off] += 1.0
        if torch.any(self.just_landed):
            self.jump_landings[self.just_landed] += 1.0
            peak_err = torch.abs(self.peak_base_height - self.commands[:, 3])
            success = self.just_landed & (peak_err <= self.cfg.rewards.success_height_tolerance)
            self.last_jump_success = success
            self.successful_jumps += success.float()
            self.jump_evaluations += self.just_landed.float()
            self.peak_height_error_sum += peak_err * self.just_landed.float()
            self.peak_height_sum += self.peak_base_height * self.just_landed.float()

        takeoff_timeout = self.jumping_state & (~self.has_taken_off) & (
            self.jump_step_counter > self.cfg.rewards.takeoff_timeout_steps
        )
        if torch.any(takeoff_timeout):
            timeout_ids = takeoff_timeout.nonzero(as_tuple=False).flatten()
            self._finish_jump(timeout_ids, completed=False)

        switch_zone = (
            (self.jump_step_counter >= self.cfg.rewards.state_switch_window_start)
            & (self.jump_step_counter <= self.cfg.rewards.state_switch_window_end)
        ) | (self.jump_step_counter > self.cfg.rewards.state_switch_window_end)
        height_rebounded = self.root_states[:, 2] >= (self.landing_min_height + 0.005)
        can_finish = (
            self.jumping_state
            & self.has_landed
            & switch_zone
            & height_rebounded
        )
        finish_ids = can_finish.nonzero(as_tuple=False).flatten()
        self._finish_jump(finish_ids, completed=True)

    def _get_single_observation(self):
        self.command_input = self.commands[:, :4] * self.commands_scale
        return torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,       # 3
                self.projected_gravity,                            # 3
                self.command_input,                                # 4
                (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,            # 12
                self.last_actions,                                 # 12
            ),
            dim=-1,
        )

    def _get_estimator_targets(self):
        local_base_pos = self.root_states[:, :3] - self.env_origins
        p_com = torch.stack(
            (
                local_base_pos[:, 2],
                local_base_pos[:, 0],
                local_base_pos[:, 1],
            ),
            dim=1,
        )
        feet_heights = self.rigid_body_states[:, self.feet_indices, 2] - self.env_origins[:, 2:3]
        return torch.cat((p_com, feet_heights, self.base_lin_vel), dim=1)

    def _get_terrain_observations(self):
        if self.cfg.terrain.measure_heights:
            return torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1.0,
                1.0,
            ) * self.obs_scales.height_measurements
        return self.terrain_obs_buf

    def compute_observations(self):
        clean_single_obs = self._get_single_observation()
        noisy_single_obs = clean_single_obs
        if self.add_noise:
            noise = (2 * torch.rand_like(clean_single_obs) - 1.0) * self.single_obs_noise_scale_vec
            noisy_single_obs = clean_single_obs + noise

        self.clean_single_obs_buf = clean_single_obs
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = noisy_single_obs
        self.obs_buf = self.obs_history.reshape(self.num_envs, -1)

        self.estimator_target_buf = self._get_estimator_targets()
        terrain_obs = self._get_terrain_observations()
        self.privileged_obs_buf = torch.cat(
            (
                self.estimator_target_buf,
                clean_single_obs,
                terrain_obs,
            ),
            dim=-1,
        )

    def _get_noise_scale_vec(self, cfg):
        single_noise = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # ang_vel [0:3]
        single_noise[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # projected_gravity [3:6]
        single_noise[3:6] = noise_scales.gravity * noise_level
        # command [6:10]
        single_noise[6:10] = 0.0
        # dof_pos [10:22]
        single_noise[10:22] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # dof_vel [22:34]
        single_noise[22:34] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # last_actions [34:46]
        single_noise[34:46] = 0.0
        self.single_obs_noise_scale_vec = single_noise
        return single_noise.repeat(self.cfg.env.history_length)

    def _compute_torques(self, actions):
        self._update_growth_scale()
        residual_torques = actions[:, :12] * self.cfg.control.action_scale
        self.residual_torques_action = residual_torques

        pd_alpha = self.cfg.control.pd_prior_end + (
            self.cfg.control.pd_prior_start - self.cfg.control.pd_prior_end
        ) * (1.0 - float(getattr(self, "general_scale", 0.0)))
        self.pd_prior_alpha[:] = pd_alpha
        self.pd_prior_torques = (
            self.p_gains * (self.default_joint_pd_target - self.dof_pos)
            - self.d_gains * self.dof_vel
        ) * pd_alpha

        self.torques_action = residual_torques + self.pd_prior_torques
        torques_limits = torch.clamp(self.current_torque_limit_scale * self.torque_limits, min=1e-6)
        torques_limits = torques_limits.clone()
        torques_limits[6:] = torques_limits[6:] * self.r_leg_scaled

        if self.cfg.control.activation_process:
            current_activation_sign = torch.tanh(self.torques_action / torques_limits)
            activation_sign = (current_activation_sign - self.activation_sign) * 0.6 + self.activation_sign
        else:
            activation_sign = self.torques_action / torques_limits
        self.activation_sign = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            activation_sign,
            self.activation_sign,
        )

        if self.cfg.control.hill_model:
            self.torques = self.activation_sign * torques_limits * (
                1 - torch.sign(self.activation_sign) * self.dof_vel / self.dof_vel_limits
            )
        else:
            self.torques = self.activation_sign * torques_limits

        if self.cfg.control.motor_fatigue:
            self.motor_fatigue += torch.abs(self.torques) * self.dt
            self.motor_fatigue *= 0.9
        else:
            self.motor_fatigue = torch.zeros_like(self.motor_fatigue)

        if self.low_torque:
            self.torques[:, :3] = self.torques[:, :3] * 0.2

        self.previous_torques = self.torques.clone()
        return self.torques

    def _reward_height_tracking(self):
        active = (self.jumping_state & (~self.has_landed)).float()
        height_error = torch.square(self.root_states[:, 2] - self.commands[:, 3])
        return active * torch.exp(-height_error / self.cfg.rewards.height_tracking_sigma)

    def _reward_successful_jump(self):
        return self.last_jump_success.float()

    def _reward_tracking_linear_velocity(self):
        active = (self.jumping_state & (~self.has_landed)).float()
        vel_error = torch.sum(torch.square(self.base_lin_vel[:, :2] - self.commands[:, :2]), dim=1)
        return active * torch.exp(-vel_error / self.cfg.rewards.velocity_tracking_sigma)

    def _reward_tracking_angular_velocity(self):
        active = (self.jumping_state & (~self.has_landed)).float()
        yaw_error = torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2])
        return active * torch.exp(-yaw_error / self.cfg.rewards.angular_tracking_sigma)

    def _reward_pose(self):
        active = (self.airborne & (~self.prelanding)).float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_air_target.unsqueeze(0)), dim=1)
        return active * pose_error

    def _reward_prelanding(self):
        active = self.prelanding.float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_pre_target.unsqueeze(0)), dim=1)
        return active * pose_error

    def _reward_landing(self):
        active = self.landing.float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_ground_target.unsqueeze(0)), dim=1)
        return active * pose_error

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf
