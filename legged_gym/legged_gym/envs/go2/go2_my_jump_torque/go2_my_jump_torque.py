import numpy as np
from isaacgym.torch_utils import *
import torch

from legged_gym.envs.go2.go2_my_jump_torque.go2_my_jump_torque_config import GO2MyJumpTorqueCfg
from legged_gym.envs.go2.go2_torque.go2_torque import GO2Torque


def get_euler_xyz_tensor(quat):
    roll, pitch, yaw = get_euler_xyz(quat)
    euler_xyz = torch.stack((roll, pitch, yaw), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class GO2MyJumpTorque(GO2Torque):
    """Commanded triggered standing long-jump task using SATA's torque actuator model."""

    cfg: GO2MyJumpTorqueCfg

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
        super().reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "peak_base_height"):
            return
        self._reset_jump_buffers(env_ids)

    def _init_buffers(self):
        super()._init_buffers()
        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.commands_scale = torch.tensor(
            [
                1.0, # distance scale
                1.0, # y scale
                1.0, # yaw scale
                1.0, # trigger scale
            ],
            dtype=torch.float,
            device=self.device,
        )

        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.non_feet_indices = self._build_non_feet_indices()

        if hasattr(self, "friction_coeffs"):
            self.env_frictions = self.friction_coeffs.to(self.device).view(self.num_envs, 1).float()
        else:
            self.env_frictions = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        self.command_input = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.residual_torques_action = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.pd_prior_torques = torch.zeros_like(self.residual_torques_action)
        self.pd_prior_alpha = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)

        self.had_ground_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_taken_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_liftoff = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.airborne_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.landing_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.peak_base_height = self.root_states[:, 2].clone()
        self.liftoff_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.liftoff_pos_x = self.root_states[:, 0].clone()
        self._reset_jump_buffers(torch.arange(self.num_envs, device=self.device))

    def _build_non_feet_indices(self):
        all_body_indices = torch.arange(self.num_bodies, dtype=torch.long, device=self.device)
        feet_mask = torch.zeros(self.num_bodies, dtype=torch.bool, device=self.device)
        feet_mask[self.feet_indices.long()] = True
        return all_body_indices[~feet_mask]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.cfg.test.use_test:
            command_dim = min(self.commands.shape[1], len(self.cfg.test.vel))
            self.commands[:, :command_dim] = self.cfg.test.vel[:command_dim].to(self.device)
        self._update_jump_state()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        # Target distance
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["jump_distance"][0],
            self.command_ranges["jump_distance"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # Lateral and Yaw
        self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 2] = 0.0

        # Trigger logic: 0 or 1. Most time jumping for efficient learning.
        self.commands[env_ids, 3] = (torch.rand(len(env_ids), device=self.device) > 0.2).float()

    def _reset_jump_buffers(self, env_ids):
        self.had_ground_contact[env_ids] = False
        self.has_taken_off[env_ids] = False
        self.has_landed[env_ids] = False
        self.airborne[env_ids] = False
        self.just_liftoff[env_ids] = False
        self.just_landed[env_ids] = False
        self.airborne_time[env_ids] = 0.0
        self.landing_time[env_ids] = 0.0
        self.peak_base_height[env_ids] = self.root_states[env_ids, 2]
        self.liftoff_lin_vel[env_ids] = 0.0
        self.liftoff_pos_x[env_ids] = self.root_states[env_ids, 0]

    def _update_jump_state(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        all_feet_contact = torch.all(contact, dim=1)
        any_foot_contact = torch.any(contact, dim=1)
        all_feet_air = ~any_foot_contact

        jump_requested = self.commands[:, 3] > self.cfg.commands.jump_trigger_min

        self.had_ground_contact |= all_feet_contact
        # Trigger liftoff only if jump is requested
        self.just_liftoff = (~self.has_taken_off) & self.had_ground_contact & all_feet_air & jump_requested
        self.just_landed = self.has_taken_off & (~self.has_landed) & any_foot_contact

        self.liftoff_lin_vel = torch.where(
            self.just_liftoff.unsqueeze(1),
            self.root_states[:, 7:10],
            self.liftoff_lin_vel,
        )
        self.liftoff_pos_x = torch.where(
            self.just_liftoff,
            self.root_states[:, 0],
            self.liftoff_pos_x,
        )

        self.has_taken_off |= self.just_liftoff
        self.has_landed |= self.just_landed
        self.airborne = self.has_taken_off & (~self.has_landed) & all_feet_air
        self.airborne_time += self.airborne.float() * self.dt
        self.landing_time += self.has_landed.float() * self.dt
        self.peak_base_height = torch.where(
            self.has_taken_off & (~self.has_landed),
            torch.maximum(self.peak_base_height, self.root_states[:, 2]),
            self.peak_base_height,
        )

        # Reset jump state after landing and staying on ground for a while to allow continuous jumping
        can_reset = self.has_landed & (self.landing_time > self.cfg.rewards.landing_window_s) & any_foot_contact
        if torch.any(can_reset):
            reset_env_ids = can_reset.nonzero(as_tuple=False).flatten()
            self._reset_jump_buffers(reset_env_ids)
            self._resample_commands(reset_env_ids)

    def compute_observations(self):
        contact_mask = (self.contact_forces[:, self.feet_indices, 2] > 5.0).float()
        feet_height = self.rigid_body_states[:, self.feet_indices, 2] - 0.02
        self.command_input = self.commands[:, :4] * self.commands_scale

        q = (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.command_input,
                q,
                dq,
                self.actions,
            ),
            dim=-1,
        )

        if self.add_noise:
            self.obs_buf = obs_buf + (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            self.obs_buf = obs_buf

        self.privileged_obs_buf = torch.cat(
            (
                self.obs_buf,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.root_states[:, 2:3],
                feet_height,
                contact_mask,
                self.has_taken_off.float().unsqueeze(1),
                self.airborne.float().unsqueeze(1),
                self.has_landed.float().unsqueeze(1),
                self.peak_base_height.unsqueeze(1),
                self.liftoff_lin_vel,
                self.env_frictions,
                self.body_mass / 10.0,
            ),
            dim=-1,
        )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity
        noise_vec[6:10] = 0.0
        noise_vec[10:22] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[22:34] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[34:46] = 0.0
        return noise_vec

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

        return self.torques

    def _reward_liftoff_momentum(self):
        # Standing Long Jump Momentum. Target d = 2 * vx * vz / g
        # A simple balance: vz = 2.0 (approx 0.2m height), then vx = d * g / (2 * vz)
        target_vx = self.commands[:, 0] * 9.81 / 4.0
        vx_error = torch.square(self.liftoff_lin_vel[:, 0] - target_vx)
        vz_reward = torch.clamp(self.liftoff_lin_vel[:, 2], min=1.0) # More generous minimum
        return self.just_liftoff.float() * (torch.exp(-vx_error / 2.0) + vz_reward) # Widen vx error sigma to 2.0

    def _reward_jump_distance_tracking(self):
        # Reward matching the target distance with a wider sigma (0.4) for early exploration
        current_dist = (self.root_states[:, 0] - self.liftoff_pos_x).clip(min=0.0)
        dist_error = torch.square(current_dist - self.commands[:, 0])
        # Higher reward when airborne or just landed
        return (self.airborne | self.just_landed).float() * torch.exp(-dist_error / 0.4)

    def _reward_jump_stay_on_ground(self):
        # Penalize staying on ground when jump is requested
        jump_requested = self.commands[:, 3] > self.cfg.commands.jump_trigger_min
        staying = jump_requested & (~self.has_taken_off)
        return staying.float()

    def _reward_stand_still(self):
        # Penalize any horizontal movement when not in jump phase
        jump_requested = self.commands[:, 3] > self.cfg.commands.jump_trigger_min
        on_ground = (~self.airborne)
        not_jumping = on_ground & (~jump_requested)
        return not_jumping.float() * torch.exp(-torch.norm(self.base_lin_vel[:, :2], dim=1) * 10.0)

    def _reward_jump_trigger_tracking(self):
        # Penalize jumping when not requested, or staying on ground when requested
        contact = (self.contact_forces[:, self.feet_indices, 2] > 5.0).any(dim=1)
        jump_requested = self.commands[:, 3] > self.cfg.commands.jump_trigger_min
        mismatch = (contact == jump_requested) 
        return -mismatch.float()

    def _reward_jump(self):
        # Push-off reward when requested: more lenient contact requirement
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        jump_requested = self.commands[:, 3] > self.cfg.commands.jump_trigger_min
        pushing = jump_requested & contact.any(dim=1) & (~self.has_taken_off)
        return pushing.float() * torch.sum(self.torques, dim=1).clip(min=0.0) * 0.001

    def _reward_airborne_time(self):
        # Reward staying in the air when jump is requested
        jump_requested = self.commands[:, 3] > self.cfg.commands.jump_trigger_min
        return (jump_requested & self.airborne).float()

    def _reward_valid_jump(self):
        # Valid jump reward focused on distance and stability
        distance = (self.root_states[:, 0] - self.liftoff_pos_x).clip(min=0.0)
        stable = (
            (torch.norm(self.projected_gravity[:, :2], dim=1) < 0.4) # Slightly more lenient
            & (torch.norm(self.base_ang_vel[:, :2], dim=1) < 4.0)
        )
        valid = (
            self.just_landed
            & (self.airborne_time > 0.05) # Lower threshold for early learning
            & stable
        )
        return valid.float() * distance

    def _reward_landing_stability(self):
        landing_window = self.has_landed & (self.landing_time < self.cfg.rewards.landing_window_s)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 8.0)
        angular = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1))
        vertical = torch.exp(-torch.abs(self.base_lin_vel[:, 2]))
        return landing_window.float() * orientation * angular * vertical

    def _reward_default_hip_pos(self):
        joint_diff = (
            torch.abs(self.dof_pos[:, 0])
            + torch.abs(self.dof_pos[:, 3])
            + torch.abs(self.dof_pos[:, 6])
            + torch.abs(self.dof_pos[:, 9])
        )
        return torch.exp(-joint_diff * 4.0)

    def _reward_feet_clearance(self):
        feet_height = self.rigid_body_states[:, self.feet_indices, 2] - 0.02
        clearance = torch.clamp(feet_height / self.cfg.rewards.target_feet_height, min=0.0, max=1.0)
        return self.airborne.float() * torch.mean(clearance, dim=1)

    def _reward_lin_vel_z(self):
        return torch.exp(-torch.abs(self.base_lin_vel[:, 2]))

    def _reward_ang_vel_xy(self):
        return torch.exp(-torch.norm(torch.abs(self.base_ang_vel[:, :2]), dim=1))

    def _reward_orientation(self):
        return torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 10.0)

    def _reward_base_height(self):
        # Only reward target height when NOT jumping
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        jump_requested = self.commands[:, 3] > self.cfg.commands.jump_trigger_min
        return (~jump_requested).float() * torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 10.0)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(
                min=0.0,
                max=1.0,
            ),
            dim=1,
        )

    def _reward_torque_limits(self):
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0),
            dim=1,
        )

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_air_time = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        self.feet_air_time *= ~contact_filt
        return rew_air_time

    def _reward_feet_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_default_pos(self):
        joint_diff = torch.abs(self.dof_pos - self.default_dof_pos)
        calf_indices = [2, 5, 8, 11]
        joint_diff[:, calf_indices] *= 1.0
        return torch.sum(joint_diff, dim=1)

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(
                min=0.0
            ),
            dim=1,
        )
