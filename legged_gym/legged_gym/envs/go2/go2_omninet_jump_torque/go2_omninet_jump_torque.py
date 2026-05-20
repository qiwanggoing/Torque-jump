"""OmniNet-aligned jumping task.

Inherits SATA's torque + PD-fade + pd_alpha-obs infrastructure from
``GO2OmniJumpTorque`` and bolts on:
- OmniJump-style state machine (``has_jumped``, ``mid_air``, ``was_in_flight``,
  ``task_max_height``, ``max_height``, ``root_states_stored``) used by reward
  signals like *Successful jump* and *Height tracking*.
- The 12 reward terms from the OmniNet paper (Table I), with weights and
  formulas matching the OmniNet authors' open-source code under
  ``Omni-Jump/`` (with the three phase joint-angle rewards reconstructed from
  the paper formula since OmniJump's code only implements ``aerial``).
- Analytical IK (copied verbatim from ``Omni-Jump/legged_gym/utils/IK.py``)
  to derive the aerial / prelanding / ground joint-angle targets from the
  ``rel_foot_pos*`` foot positions specified in the config.
- An OmniJump-style ``_resample_commands`` that quantizes the sampled jump
  height to {0.32, 0.50, 0.68} m so the reward's ±0.04 m tolerance lines up
  with the training distribution.

Observation structure (69 dims) and torque-with-PD-prior control are unchanged
from the SATA baseline; only the reward/task layer is OmniNet-aligned.
"""

import numpy as np
import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.go2.go2_omnijump_torque.go2_omnijump_torque import GO2OmniJumpTorque
from legged_gym.envs.go2.go2_omninet_jump_torque.go2_omninet_jump_torque_config import (
    GO2OmniNetJumpTorqueCfg,
)
from legged_gym.utils.IK import Go2 as Go2IKLinkConfig, RobotIK


class GO2OmniNetJumpTorque(GO2OmniJumpTorque):
    cfg: GO2OmniNetJumpTorqueCfg

    # ------------------------------------------------------------------ #
    # 12 active rewards == OmniNet paper Table I.
    # All other rewards from the SATA infrastructure are filtered out by the
    # base class' ACTIVE_REWARD_WHITELIST mechanism in ``_prepare_reward_function``.
    # ------------------------------------------------------------------ #
    ACTIVE_REWARD_WHITELIST = {
        # Task
        "height_tracking",
        "successful_jump",
        "tracking_linear_velocity",
        "tracking_angular_velocity",
        # Pose
        "orientation",
        "joint_angle_aerial",
        "joint_angle_prelanding",
        "joint_angle_landing",
        # Safety
        "collision",
        # Smoothness
        "torques",
        "action_rate",
        "dof_acc",
    }

    # ====================================================================== #
    # Initialisation
    # ====================================================================== #
    def _init_buffers(self):
        super()._init_buffers()

        # ----- OmniJump state machine buffers -----
        self.omninet_has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.omninet_last_has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.omninet_mid_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.omninet_mid_air2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.omninet_was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.omninet_settled_after_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.omninet_settled_after_init_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.omninet_task_max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.omninet_max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.omninet_landing_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.omninet_command_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.omninet_last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device
        )
        self.omninet_contact_filt = torch.zeros_like(self.omninet_last_contacts)
        # Rolling buffer of root_states for check_landing (z[t-1] > z[t] when descending while mid_air)
        self.omninet_root_states_stored = torch.zeros(
            self.num_envs, 13, 10, dtype=torch.float, device=self.device
        )
        self.omninet_root_states_stored[:, :, 0] = self.root_states

        self.omninet_base_init_height = float(self.cfg.init_state.pos[2])

        # ----- IK-derived pose targets (q^air, q^pre, q^ground) -----
        self._init_omninet_pose_targets()

    def _init_omninet_pose_targets(self):
        """Run the analytical IK on the rel_foot_pos / rel_foot_pos_peak /
        rel_foot_pos_pre arrays from the config to derive joint-angle targets
        for the three jump phases used by the paper Table I joint-angle rewards.
        """
        ik = RobotIK(Go2IKLinkConfig)

        def _flatten(rel):
            # rel is [[x_FL, x_FR, x_RL, x_RR], [y_*], [z_*]] → flatten to
            # [FL_x, FL_y, FL_z, FR_x, FR_y, FR_z, RL_x, RL_y, RL_z, RR_x, RR_y, RR_z]
            feet = np.zeros(12, dtype=np.float64)
            for leg in range(4):
                feet[3 * leg + 0] = rel[0][leg]
                feet[3 * leg + 1] = rel[1][leg]
                feet[3 * leg + 2] = rel[2][leg]
            return feet

        feet_ground = _flatten(self.cfg.init_state.rel_foot_pos)
        feet_peak = _flatten(self.cfg.init_state.rel_foot_pos_peak)
        feet_pre = _flatten(self.cfg.init_state.rel_foot_pos_pre)
        zero_vel = np.zeros(12, dtype=np.float64)

        q_ground_np, _ = ik.computeIK(feet_ground, zero_vel)
        q_air_np, _ = ik.computeIK(feet_peak, zero_vel)
        q_pre_np, _ = ik.computeIK(feet_pre, zero_vel)

        # OmniJump IK output dof order matches our URDF (FL{hip,thigh,calf},
        # FR{...}, RL{...}, RR{...}).
        self.q_omninet_air = torch.tensor(q_air_np, dtype=torch.float, device=self.device)
        self.q_omninet_pre = torch.tensor(q_pre_np, dtype=torch.float, device=self.device)
        self.q_omninet_ground = torch.tensor(q_ground_np, dtype=torch.float, device=self.device)

    # ====================================================================== #
    # post_physics_step injection
    # ====================================================================== #
    def post_physics_step(self):
        # Mirror parent's post_physics_step but insert OmniJump's state machine
        # updates before reward computation, and check_has_jump_reset after.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        from isaacgym.torch_utils import quat_rotate_inverse  # local import (parent does same pattern)
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # === OmniJump state machine update (before reward computation) ===
        self.omninet_last_has_jumped[:] = self.omninet_has_jumped
        self._omninet_check_jump()
        self._omninet_store_root_states_roll()
        # task_max_height tracks the peak achieved during the current flight only:
        # mid_air & ~has_jumped & was_in_flight
        track_idx = self.omninet_mid_air & (~self.omninet_has_jumped) & self.omninet_was_in_flight
        if track_idx.any():
            self.omninet_task_max_height[track_idx] = torch.max(
                self.omninet_task_max_height[track_idx], self.root_states[track_idx, 2]
            )
        self.omninet_max_height[:] = self._omninet_compute_max_height()

        # SATA phase update (jumping_state, airborne, prelanding, landing) —
        # the parent's callback handles command resampling and our phase masks
        # used by the joint-angle rewards.
        self._post_physics_step_callback()

        # Termination check + reward computation (reward functions read the
        # OmniJump state buffers populated above).
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        # === OmniJump: AFTER rewards are computed, reset task_max_height /
        # was_in_flight / has_jumped for envs that have landed. This is what
        # makes ``successful_jump`` a sparse one-step reward instead of
        # repeating every frame after landing.
        self._omninet_check_has_jump_reset()

        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # ====================================================================== #
    # OmniJump state machine (verbatim from base/legged_robot.py:2402-2447,
    # 814-817, 823-830, 833-840, with tensor name remapping).
    # ====================================================================== #
    def _omninet_check_jump(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.omninet_last_contacts)
        self.omninet_contact_filt[:] = contact_filt

        # settle-after-init: agent is on the ground after spawning
        settled_after_init = torch.logical_and(
            torch.all(contact_filt, dim=1),
            self.root_states[:, 2] <= 0.34,
        )
        jump_filter = torch.all(~contact_filt, dim=1)   # all 4 feet off contact (combined w/ last frame)
        jump_filter2 = torch.all(~contact, dim=1)       # all 4 feet off contact (this frame only)

        self.omninet_mid_air[:] = jump_filter
        self.omninet_mid_air2[:] = jump_filter2

        idx_record_pose = torch.logical_and(settled_after_init, ~self.omninet_settled_after_init)
        self.omninet_settled_after_init_timer[idx_record_pose] = self.episode_length_buf[idx_record_pose].clone()
        self.omninet_settled_after_init[settled_after_init] = True

        # Only enter "in flight" once the robot has actually settled on the
        # ground after initialisation.
        self.omninet_was_in_flight[torch.logical_and(jump_filter, self.omninet_settled_after_init)] = True

        # has_jumped flips True the step the robot's first contact returns after flight.
        has_jumped = torch.logical_and(torch.any(contact_filt, dim=1), self.omninet_was_in_flight)
        self.omninet_has_jumped[has_jumped] = True

        self.omninet_landing_ids[:] = self._omninet_check_landing()
        self.omninet_last_contacts[:] = contact
        self.omninet_command_mask[:] = self.omninet_mid_air

    def _omninet_check_landing(self):
        # landing flag = mid_air and base z is descending (z at t-1 > z at t)
        descending = self.omninet_root_states_stored[:, 2, 1] > self.omninet_root_states_stored[:, 2, 0]
        return descending & self.omninet_mid_air

    def _omninet_store_root_states_roll(self):
        # Roll the 10-step history along the time axis and write current state at slot 0.
        self.omninet_root_states_stored[:] = torch.roll(self.omninet_root_states_stored, shifts=1, dims=-1)
        self.omninet_root_states_stored[:, :, 0] = self.root_states

    def _omninet_check_has_jump_reset(self):
        # After a successful jump-and-land cycle, reset has_jumped / was_in_flight
        # so the rewards stop firing repeatedly and the next jump can start fresh.
        contact_ids = torch.any(self.omninet_contact_filt, dim=1)
        contact_ids_all = torch.all(self.omninet_contact_filt, dim=1)
        finish_mask = contact_ids & self.omninet_has_jumped
        self.omninet_task_max_height[finish_mask] = self.omninet_base_init_height
        self.omninet_was_in_flight[finish_mask] = False
        self.omninet_landing_ids[contact_ids_all] = False
        self.omninet_has_jumped[finish_mask] = False

    def _omninet_compute_max_height(self):
        m = torch.max(self.omninet_max_height, self.root_states[:, 2])
        m[self.omninet_has_jumped] = 0.0
        return m

    # ====================================================================== #
    # reset_idx + command resampling
    # ====================================================================== #
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        # Reset OmniJump state machine buffers for the reset envs.
        self.omninet_has_jumped[env_ids] = False
        self.omninet_last_has_jumped[env_ids] = False
        self.omninet_mid_air[env_ids] = False
        self.omninet_mid_air2[env_ids] = False
        self.omninet_was_in_flight[env_ids] = False
        self.omninet_settled_after_init[env_ids] = False
        self.omninet_settled_after_init_timer[env_ids] = 0
        self.omninet_task_max_height[env_ids] = self.omninet_base_init_height
        self.omninet_max_height[env_ids] = 0.0
        self.omninet_landing_ids[env_ids] = False
        self.omninet_last_contacts[env_ids] = False
        self.omninet_contact_filt[env_ids] = False
        self.omninet_command_mask[env_ids] = False
        # Re-seed the rolling buffer with the current root_states so check_landing
        # doesn't trigger spurious descents from a stale t-1 entry.
        self.omninet_root_states_stored[env_ids] = (
            self.root_states[env_ids].unsqueeze(-1).expand(-1, -1, 10).clone()
        )

    def _resample_commands(self, env_ids):
        """OmniJump-style command sampling with height quantization (snap to
        0.32 / 0.50 / 0.68 m so the ±0.04 m successful_jump tolerance is
        meaningful at training time).
        """
        if len(env_ids) == 0:
            return

        # Velocity commands
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

        # Jump-height command (quantized) — mirrors Omni-Jump/legged_gym/envs/base/legged_robot.py:1314-1331
        height_z = torch_rand_float(
            self.command_ranges["height_z"][0],
            self.command_ranges["height_z"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        trot_ids = height_z < 0.46
        height_z[trot_ids] = 0.32
        mid_band = (height_z >= 0.46) & (height_z < 0.60)
        height_z[mid_band] = 0.50
        high_band = (height_z >= 0.60) & (height_z < 0.85)
        height_z[high_band] = 0.68
        self.commands[env_ids, 3] = height_z

        # Suppress tiny xy commands (OmniJump: <0.2 norm zeros out)
        small = torch.norm(self.commands[env_ids, :2], dim=1) < 0.2
        if small.any():
            small_ids = env_ids[small]
            self.commands[small_ids, 0] = 0.0
            self.commands[small_ids, 1] = 0.0

        # Single-jump bookkeeping for compatibility with parent's plumbing.
        if self.cfg.commands.num_commands > 4:
            single_jump_prob = float(getattr(self.cfg.commands, "single_jump_command_prob", 1.0))
            self.single_jump_command_mode[env_ids] = (
                torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1) < single_jump_prob
            )
            self.single_jump_command_done[env_ids] = False
            self.commands[env_ids, 4] = self.command_ranges["jump_command"][1]

    # ====================================================================== #
    # Reward functions — paper Table I (12 items).
    # The 5 standard regularisers (orientation, torques, dof_acc, action_rate,
    # collision) are inherited from ``LeggedRobot`` and already match OmniJump's
    # implementations verbatim; only the 7 jumping-specific terms need override.
    # ====================================================================== #
    def _reward_height_tracking(self):
        # Paper Table I: e^{-20·(z* - z)^2} with z* = commands[:, 3], z = task_max_height
        # OmniJump code (_reward_task_max_height): exp(-(task_max_height - cmd)^2 / 0.05)
        # Note: 1/0.05 = 20, so these are identical.
        err = self.omninet_task_max_height - self.commands[:, 3]
        sigma = float(getattr(self.cfg.rewards, "max_height_reward_sigma", 0.05))
        rew = torch.exp(-torch.square(err) / sigma)
        # Don't reward walking-height commands
        walking = self.commands[:, 3] < float(getattr(self.cfg.rewards, "omninet_walking_height_threshold", 0.38))
        rew[walking] = 0.0
        return rew

    def _reward_successful_jump(self):
        # Paper Table I: n_jump = 1 if peak ∈ [cmd_h - 0.04, cmd_h + 0.04]
        tol = float(getattr(self.cfg.rewards, "omninet_jump_height_tolerance", 0.04))
        up_bond = self.commands[:, 3] + tol
        low_bond = self.commands[:, 3] - tol
        rewed = (
            self.omninet_has_jumped
            & (self.omninet_task_max_height > low_bond)
            & (self.omninet_task_max_height < up_bond)
        )
        rew = torch.zeros(self.num_envs, device=self.device)
        rew[rewed] = 1.0
        walking = self.commands[:, 3] < float(getattr(self.cfg.rewards, "omninet_walking_height_threshold", 0.38))
        rew[walking] = 0.0
        return rew

    def _reward_tracking_linear_velocity(self):
        # Paper Table I: e^{-4·||v* - v_xy||^2} = exp(-err / 0.25)
        # Velocity in world frame (matches OmniJump's _reward_tracking_lin_vel).
        sigma = max(float(getattr(self.cfg.rewards, "tracking_sigma", 0.25)), 1e-3)
        err = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, 7:9]), dim=1)
        return torch.exp(-err / sigma)

    def _reward_tracking_angular_velocity(self):
        # Paper Table I: e^{-4·(ω*_yaw - ω_yaw)^2}
        sigma = max(float(getattr(self.cfg.rewards, "tracking_sigma", 0.25)), 1e-3)
        err = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-err / sigma)

    def _reward_joint_angle_aerial(self):
        # Paper Table I: Σ_j ||q_j - q^air_j||, active during aerial phase.
        active = (self.airborne & (~self.prelanding)).float()
        err = torch.sum(torch.abs(self.dof_pos - self.q_omninet_air.unsqueeze(0)), dim=1)
        return active * err

    def _reward_joint_angle_prelanding(self):
        # Paper Table I: Σ_j ||q_j - q^pre_j||, active during prelanding phase.
        active = self.prelanding.float()
        err = torch.sum(torch.abs(self.dof_pos - self.q_omninet_pre.unsqueeze(0)), dim=1)
        return active * err

    def _reward_joint_angle_landing(self):
        # Paper Table I: Σ_j ||q_j - q^ground_j||, active during landing phase.
        active = self.landing.float()
        err = torch.sum(torch.abs(self.dof_pos - self.q_omninet_ground.unsqueeze(0)), dim=1)
        return active * err
