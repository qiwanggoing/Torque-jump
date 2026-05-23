import numpy as np
import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.go2.go2_omnijump_torque.go2_omnijump_torque import GO2OmniJumpTorque
from legged_gym.envs.go2.go2_omnijump_curriculum_torque.go2_omnijump_curriculum_torque_config import (
    GO2OmniJumpCurriculumTorqueCfg,
)
from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfgPPO


class GO2OmniJumpCurriculumTorque(GO2OmniJumpTorque):
    cfg: GO2OmniJumpCurriculumTorqueCfg

    ACTIVE_REWARD_WHITELIST = GO2OmniJumpTorque.ACTIVE_REWARD_WHITELIST | {"aerial_dof_acc"}

    STAGE_NAMES = ("stand", "takeoff", "flight", "landing", "motion")

    def _update_growth_scale(self):
        """Warmup + Linear schedule (override base Gompertz):
        - step_count < warmup_steps  → general_scale = 0  (PD stays full strength)
        - warmup_steps ≤ step_count ≤ x0 → general_scale ramps linearly 0→1
        - step_count > x0  → general_scale = 1 (PD fully faded)
        Removes Gompertz S-curve's mid-burst; first 1000 iter has stable PD support.
        """
        self.step_count += 1
        if self.cfg.control.control_type == "T" or self.cfg.test.use_test:
            self.step_count = GO2TorqueCfgPPO().runner.num_steps_per_env * self.cfg.test.checkpoint

        warmup_steps = max(0.0, float(getattr(self.cfg.growth, "warmup_steps", 0)))
        fade_end_steps = max(float(self.cfg.growth.x0), warmup_steps + 1.0)
        if self.step_count < warmup_steps:
            self.general_scale = 0.0
        else:
            ramp_progress = (self.step_count - warmup_steps) / (fade_end_steps - warmup_steps)
            self.general_scale = min(1.0, max(0.0, ramp_progress))

        self.current_freq = self.general_scale * (self.max_freq - self.start_freq) + self.start_freq
        self.current_torque_limit_scale = (
            self.general_scale * (self.max_torque_scale - self.start_torque_scale) + self.start_torque_scale
        )
        self.r_leg_scaled = (
            self.general_scale * (self.max_rear_torque_scale - self.start_rear_torque_scale)
            + self.start_rear_torque_scale
        )

    REWARD_START_STAGES = {
        # Stage 0 — regularisation from step 1
        "termination": 0,
        "orientation": 0,
        "collision": 0,
        "torques": 0,
        "action_rate": 0,
        "dof_acc": 0,
        "maintain_contact": 0,
        # Stage 1 — velocity + height signal bootstrap
        "tracking_linear_velocity": 1,
        "tracking_angular_velocity": 1,
        "height_tracking": 1,
        "peak_height_progress": 1,
        "all_feet_airborne": 1,
        "takeoff_vertical_velocity": 1,
        "projected_peak": 1,
        # Stage 0 — pose guidance lives at stage 0 too (we run curriculum disabled / one-stage)
        "joint_angle_loaded": 0,
        "joint_angle_extended": 0,
        "horizontal_drift": 0,
        "takeoff_direction": 0,
        "default_pos": 0,
        "default_hip_pos": 0,
        "aerial_dof_acc": 0,

        # Stage 2 — airborne pose quality (legacy, weight 0)
        "joint_angle_aerial": 2,
        "joint_angle_prelanding": 2,
        # Stage 3 — full jump cycle
        "successful_jump": 3,
        "joint_angle_landing": 3,
    }

    CURRICULUM_METRICS = (
        "rew_height_tracking",
        "jump_flight_rate",
        "jump_landing_rate",
        "successful_jump_rate",
        "mean_peak_height",
    )

    def _init_buffers(self):
        force_stage = int(getattr(self.cfg.curriculum, "force_stage", -1))
        if force_stage >= 0:
            self.curriculum_stage_idx = max(0, min(force_stage, len(self.STAGE_NAMES) - 1))
        else:
            self.curriculum_stage_idx = 0
        self.curriculum_stage_updates = 0
        self.curriculum_metric_ema = {}
        super()._init_buffers()

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        self.curriculum_final_reward_scales = dict(self.reward_scales)
        missing_rewards = sorted(set(self.curriculum_final_reward_scales) - set(self.REWARD_START_STAGES))
        if missing_rewards:
            raise ValueError(f"Missing curriculum stages for rewards: {missing_rewards}")

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        if not getattr(self.cfg.curriculum, "enabled", False):
            return
        stage_before = self.curriculum_stage_idx
        self._update_curriculum_from_episode()
        if self.curriculum_stage_idx != stage_before:
            self._resample_commands(env_ids)

    def compute_reward(self):
        if getattr(self.cfg.curriculum, "enabled", False):
            self._apply_curriculum_reward_scales()
        super().compute_reward()

    def _log_jump_episode_stats(self, env_ids):
        super()._log_jump_episode_stats(env_ids)
        if "episode" not in self.extras:
            self.extras["episode"] = {}
        self.extras["episode"]["curriculum_stage"] = torch.tensor(
            float(self.curriculum_stage_idx), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_stage_updates"] = torch.tensor(
            float(self.curriculum_stage_updates), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_takeoff_gate"] = torch.tensor(
            float(self.curriculum_stage_idx >= 1), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_flight_gate"] = torch.tensor(
            float(self.curriculum_stage_idx >= 2), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_landing_gate"] = torch.tensor(
            float(self.curriculum_stage_idx >= 3), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_motion_gate"] = torch.tensor(
            float(self.curriculum_stage_idx >= 4), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_stand_still_ema"] = torch.tensor(
            self.curriculum_metric_ema.get("rew_stand_still", 0.0), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_stand_score_ema"] = torch.tensor(
            self._stand_stage_score(), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_zero_base_height_ema"] = torch.tensor(
            self.curriculum_metric_ema.get("rew_zero_command_base_height", 0.0),
            dtype=torch.float,
            device=self.device,
        )
        self.extras["episode"]["curriculum_default_hip_pos_ema"] = torch.tensor(
            self.curriculum_metric_ema.get("rew_default_hip_pos", 0.0), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_height_tracking_ema"] = torch.tensor(
            self.curriculum_metric_ema.get("rew_height_tracking", 0.0),
            dtype=torch.float,
            device=self.device,
        )
        self.extras["episode"]["curriculum_flight_rate_ema"] = torch.tensor(
            self.curriculum_metric_ema.get("jump_flight_rate", 0.0), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_success_rate_ema"] = torch.tensor(
            self.curriculum_metric_ema.get("successful_jump_rate", 0.0), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_landing_rate_ema"] = torch.tensor(
            self.curriculum_metric_ema.get("jump_landing_rate", 0.0), dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_mean_peak_height_ema"] = torch.tensor(
            self.curriculum_metric_ema.get("mean_peak_height", 0.0), dtype=torch.float, device=self.device
        )
        action_scale, _, _ = self._control_curriculum_values()
        self.extras["episode"]["curriculum_action_scale"] = torch.tensor(
            action_scale, dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_rl_prior"] = torch.tensor(
            float(self.rl_prior_alpha[0]) if hasattr(self, "rl_prior_alpha") else 0.5,
            dtype=torch.float, device=self.device
        )
        self.extras["episode"]["curriculum_pd_prior"] = torch.tensor(
            float(self.pd_prior_alpha[0]) if hasattr(self, "pd_prior_alpha") else 0.5,
            dtype=torch.float, device=self.device
        )

    def _apply_curriculum_reward_scales(self):
        for name, scale in self.curriculum_final_reward_scales.items():
            required_stage = self.REWARD_START_STAGES.get(name, 0)
            self.reward_scales[name] = scale if self.curriculum_stage_idx >= required_stage else 0.0

    def _update_curriculum_from_episode(self):
        episode = self.extras.get("episode", {})
        alpha = float(self.cfg.curriculum.ema_alpha)
        for name in self.CURRICULUM_METRICS:
            if name not in episode:
                continue
            value = self._to_float(episode[name])
            old_value = self.curriculum_metric_ema.get(name, value)
            self.curriculum_metric_ema[name] = (1.0 - alpha) * old_value + alpha * value

        self.curriculum_stage_updates += 1
        if self.curriculum_stage_updates < int(self.cfg.curriculum.min_updates_per_stage):
            return
        if self._current_stage_passed():
            self.curriculum_stage_idx = min(self.curriculum_stage_idx + 1, len(self.STAGE_NAMES) - 1)
            self.curriculum_stage_updates = 0

    def _current_stage_passed(self):
        metric = self.curriculum_metric_ema
        if self.curriculum_stage_idx == 0:
            return (
                self._stand_stage_score() >= self.cfg.curriculum.stand_stage_reward_threshold
                and metric.get("rew_zero_command_base_height", 0.0)
                >= self.cfg.curriculum.zero_command_base_height_threshold
                and metric.get("rew_default_hip_pos", 0.0) >= self.cfg.curriculum.default_hip_pos_threshold
            )
        if self.curriculum_stage_idx == 1:
            return (
                metric.get("rew_height_tracking", 0.0)
                >= self.cfg.curriculum.takeoff_vertical_velocity_threshold
                and metric.get("jump_flight_rate", 0.0) >= self.cfg.curriculum.takeoff_flight_rate_threshold
            )
        if self.curriculum_stage_idx == 2:
            return (
                metric.get("jump_flight_rate", 0.0) >= self.cfg.curriculum.flight_rate_threshold
                and metric.get("mean_peak_height", 0.0) >= self.cfg.curriculum.flight_mean_peak_height_threshold
            )
        if self.curriculum_stage_idx == 3:
            return (
                metric.get("jump_landing_rate", 0.0) >= self.cfg.curriculum.landing_rate_threshold
                and metric.get("successful_jump_rate", 0.0) >= self.cfg.curriculum.successful_jump_rate_threshold
            )
        return False

    def _to_float(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().item())
        return float(value)

    def _stand_stage_score(self):
        metric = self.curriculum_metric_ema
        return (
            metric.get("rew_stand_still", 0.0)
            + metric.get("rew_zero_command_linear_velocity", 0.0)
            + metric.get("rew_zero_command_angular_velocity", 0.0)
            + metric.get("rew_zero_command_base_height", 0.0)
            + metric.get("rew_default_hip_pos", 0.0)
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if not getattr(self.cfg.curriculum, "enabled", False):
            return super()._resample_commands(env_ids)

        if self.curriculum_stage_idx < 1:
            self._disable_jump_command(env_ids)
            if self.cfg.commands.num_commands > 4:
                self.single_jump_command_mode[env_ids] = False
                self.single_jump_command_done[env_ids] = False
            return

        if self.cfg.commands.num_commands > 4 and hasattr(self, "single_jump_command_done"):
            hold_mask = self.single_jump_command_done[env_ids]
            hold_ids = env_ids[hold_mask]
            self._disable_jump_command(hold_ids)
            env_ids = env_ids[~hold_mask]
            if len(env_ids) == 0:
                return

        stand_prob = float(self.cfg.curriculum.stand_command_prob_after_takeoff)
        stand_mask = torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1) < stand_prob
        stand_ids = env_ids[stand_mask]
        self._disable_jump_command(stand_ids)
        if self.cfg.commands.num_commands > 4:
            self.single_jump_command_mode[stand_ids] = False
            self.single_jump_command_done[stand_ids] = False
        env_ids = env_ids[~stand_mask]
        if len(env_ids) == 0:
            return

        motion_open = self.curriculum_stage_idx >= int(getattr(self.cfg.curriculum, "velocity_command_start_stage", 4))
        if motion_open:
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
        else:
            self.commands[env_ids, 0:3] = 0.0

        self.commands[env_ids, 3] = torch_rand_float(
            self.command_ranges["jump_height"][0],
            self.command_ranges["jump_height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.num_commands > 4:
            single_jump_prob = float(getattr(self.cfg.commands, "single_jump_command_prob", 0.5))
            self.single_jump_command_mode[env_ids] = (
                torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1) < single_jump_prob
            )
            self.single_jump_command_done[env_ids] = False
            self.commands[env_ids, 4] = self.command_ranges["jump_command"][1]

    def _compute_torques(self, actions):
        self._update_growth_scale()
        self._update_default_joint_pd_target()
        action_scale, _, _ = self._control_curriculum_values()

        # PD fades from pd_prior_weight → 0; RL grows from rl_prior_weight → 1.0
        pd_alpha = float(self.cfg.control.pd_prior_weight) * max(0.0, 1.0 - float(self.general_scale))
        rl_alpha = 1.0 - pd_alpha

        # RL residual bounded by physical torque limits (not action_scale)
        torques_limits_eff = torch.clamp(self.torque_limits, min=1e-6)
        residual_torques = actions[:, :12] * torques_limits_eff

        self.rl_prior_alpha[:] = rl_alpha
        self.pd_prior_alpha[:] = pd_alpha
        self.residual_torques_action = residual_torques * rl_alpha * self.current_torque_limit_scale
        self.pd_prior_torques = (
            self.p_gains * (self.default_joint_pd_target - self.dof_pos) - self.d_gains * self.dof_vel
        ) * pd_alpha

        self.torques_action = self.residual_torques_action + self.pd_prior_torques
        torques_limits = torch.clamp(self.torque_limits, min=1e-6).clone()

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
        self.torques = torch.clip(self.torques, -torques_limits, torques_limits)

        if self.cfg.control.motor_fatigue:
            self.motor_fatigue += torch.abs(self.torques) * self.dt
            self.motor_fatigue *= 0.9
        else:
            self.motor_fatigue = torch.zeros_like(self.motor_fatigue)

        if self.low_torque:
            self.torques[:, :3] = self.torques[:, :3] * 0.2

        return self.torques

    def _control_curriculum_values(self):
        if not getattr(self.cfg.curriculum, "enabled", False):
            return (
                float(self.cfg.control.action_scale),
                float(self.cfg.control.rl_prior_weight),
                float(self.cfg.control.pd_prior_weight),
            )
        denom = max(len(self.STAGE_NAMES) - 1, 1)
        progress = float(self.curriculum_stage_idx) / float(denom)
        action_scale = self._lerp(
            self.cfg.control.curriculum_action_scale_start,
            self.cfg.control.curriculum_action_scale_end,
            progress,
        )
        rl_alpha = self._lerp(
            self.cfg.control.curriculum_rl_prior_start,
            self.cfg.control.curriculum_rl_prior_end,
            progress,
        )
        pd_alpha = self._lerp(
            self.cfg.control.curriculum_pd_prior_start,
            self.cfg.control.curriculum_pd_prior_end,
            progress,
        )
        return action_scale, rl_alpha, pd_alpha

    def _lerp(self, start, end, alpha):
        return float(start) + (float(end) - float(start)) * float(alpha)

    def _zero_command_mask(self):
        if self.cfg.commands.num_commands > 4:
            jump_command_inactive = self.commands[:, 4] <= float(self.cfg.commands.jump_command_threshold)
        else:
            jump_command_inactive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return (
            (torch.norm(self.commands[:, :3], dim=1) < 0.05)
            & (torch.abs(self.commands[:, 3]) < 0.05)
            & jump_command_inactive
        )

    def _reward_zero_command_linear_velocity(self):
        speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        sigma = max(float(self.cfg.rewards.zero_command_velocity_sigma), 1e-3)
        return self._zero_command_mask().float() * torch.exp(-speed / sigma)

    def _reward_zero_command_angular_velocity(self):
        yaw_speed = torch.abs(self.base_ang_vel[:, 2])
        sigma = max(float(self.cfg.rewards.zero_command_yaw_sigma), 1e-3)
        return self._zero_command_mask().float() * torch.exp(-yaw_speed / sigma)

    def _reward_zero_command_base_height(self):
        height_error = torch.abs(self.root_states[:, 2] - self.cfg.rewards.base_height_target)
        return self._zero_command_mask().float() * torch.exp(-self.cfg.rewards.zero_command_height_gain * height_error)

    def _reward_default_hip_pos(self):
        hip_ids = [0, 3, 6, 9]
        hip_error = torch.sum(
            torch.abs(self.dof_pos[:, hip_ids] - self.default_dof_pos[:, hip_ids]),
            dim=1,
        )
        return torch.exp(-self.cfg.rewards.default_hip_pos_gain * hip_error)

    def _reward_default_pos(self):
        # mygo2jump-style L1 over ALL 12 joints (was hip-only — left thigh/calf
        # with no reward-side anchor, so RL never learned to hold standing pose
        # without PD).
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_aerial_dof_acc(self):
        # Airborne-only joint acceleration penalty. Global dof_acc covers full
        # episode; this term specifically targets in-air twitching/flailing
        # observed after PD fades out. Active only when all four feet off
        # contact and not yet in prelanding.
        active = (self.airborne & (~self.prelanding)).float()
        acc = (self.last_dof_vel - self.dof_vel) / self.dt
        return active * torch.sum(torch.square(acc), dim=1)

    def _reward_joint_angle_aerial(self):
        active = (self.airborne & (~self.prelanding)).float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_air_target.unsqueeze(0)), dim=1)
        return active * pose_error

    def _reward_joint_angle_prelanding(self):
        active = self.prelanding.float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_pre_target.unsqueeze(0)), dim=1)
        return active * pose_error

    def _reward_joint_angle_landing(self):
        active = self.landing.float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_ground_target.unsqueeze(0)), dim=1)
        return active * pose_error

    def _reward_left_right_contact_sync(self):
        contact = self._get_contact_state()
        front_sync = (contact[:, 0] == contact[:, 1]).float()
        rear_sync = (contact[:, 2] == contact[:, 3]).float()
        return self.jumping_state.float() * 0.5 * (front_sync + rear_sync)

    def _reward_straight_jump_joint_symmetry(self):
        cmd_y = self.commands[:, 1]
        cmd_yaw = self.commands[:, 2]
        straight_gate = torch.exp(
            -torch.square(cmd_y / max(float(self.cfg.rewards.symmetry_lateral_sigma), 1e-3))
            -torch.square(cmd_yaw / max(float(self.cfg.rewards.symmetry_yaw_sigma), 1e-3))
        )
        q = self.dof_pos - self.default_dof_pos
        front_err = torch.abs(q[:, 0] + q[:, 3]) + torch.abs(q[:, 1] - q[:, 4]) + torch.abs(q[:, 2] - q[:, 5])
        rear_err = torch.abs(q[:, 6] + q[:, 9]) + torch.abs(q[:, 7] - q[:, 10]) + torch.abs(q[:, 8] - q[:, 11])
        sigma = max(float(self.cfg.rewards.joint_symmetry_tracking_sigma), 1e-3)
        return self.jumping_state.float() * straight_gate * torch.exp(-(front_err + rear_err) / sigma)
