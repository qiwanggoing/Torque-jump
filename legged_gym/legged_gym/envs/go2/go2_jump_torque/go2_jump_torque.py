import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import *
from legged_gym.envs.go2.go2_torque.go2_torque import GO2Torque
from legged_gym.envs.go2.go2_jump_torque.go2_jump_torque_config import GO2JumpTorqueCfg
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import wrap_to_pi


class GO2JumpTorque(GO2Torque):
    cfg: GO2JumpTorqueCfg

    PHASE_PREPARE = 0
    PHASE_COMPRESSION = 1
    PHASE_EXTENSION = 2
    PHASE_RELEASE = 3
    PHASE_FLIGHT = 4
    PHASE_TOUCHDOWN = 5
    PHASE_STABILIZE = 6

    def __init__(self, cfg: GO2JumpTorqueCfg, sim_params, physics_engine, sim_device, headless):
        self.current_pd_prior_scale = 0.0
        self.curriculum_stage_index = -1
        self.curriculum_stage_name = "disabled"
        self.curriculum_enabled = False
        self.curriculum_auto_advance = False
        self.curriculum_cfg_dict = {}
        self.curriculum_stage_cfg = {}
        self.base_reward_scales_raw = {}
        self.active_rsi_cfg = {}
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_nonfoot_contact_indices()

    def _init_nonfoot_contact_indices(self):
        if not hasattr(self, "feet_indices") or self.feet_indices.numel() == 0:
            return
        all_body_indices = torch.arange(self.num_bodies, device=self.device, dtype=torch.long)
        foot_mask = torch.zeros(self.num_bodies, dtype=torch.bool, device=self.device)
        foot_mask[self.feet_indices.long()] = True
        self.nonfoot_contact_indices = all_body_indices[~foot_mask]

    def _init_buffers(self):
        super()._init_buffers()
        self.commands_scale = torch.ones(
            self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.aerial_joint_target = self._build_joint_target(self.cfg.rewards.aerial_pose_joint_angles)
        self.prelanding_joint_target = self._build_joint_target(self.cfg.rewards.prelanding_pose_joint_angles)
        self.landing_joint_target = self._build_joint_target(self.cfg.rewards.landing_pose_joint_angles)
        self.takeoff_squat_joint_target = self._build_joint_target(self.cfg.rewards.takeoff_squat_pose_joint_angles)

        current_height = self.root_states[:, 2].clone()
        current_yaw = self._get_base_yaw()
        self.jump_start_height = current_height.clone()
        self.jump_start_xy = self.root_states[:, :2].clone()
        self.jump_start_yaw = current_yaw.clone()
        self.target_landing_xy = self.root_states[:, :2].clone()
        self.target_landing_yaw = current_yaw.clone()
        self.takeoff_height = current_height.clone()
        self.ready_stance_height = current_height.clone()
        self.jump_reference_height = torch.full_like(current_height, float(self.cfg.jump_phase.stand_height_target))
        self.cycle_peak_height = current_height.clone()
        self.episode_peak_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_peak_base_height = current_height.clone()
        self.landing_dx_curriculum_progress = torch.zeros(
            1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.landing_dx_success_ema = torch.zeros(
            1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_landing_dx_success_rate = torch.zeros(
            1, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_landing_position_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_landing_forward_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_landing_lateral_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_landing_yaw_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_completed_peak_height = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_completed_base_peak_height = current_height.clone()
        self.last_completed_landing_position_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_completed_landing_forward_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_completed_landing_lateral_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_completed_landing_yaw_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_completed_landing_tilt = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_estimated_landing_position_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_estimated_landing_forward_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_estimated_landing_lateral_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_completed_clearance_gate = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_completed_valid_jump = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.last_completed_airborne_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )

        self.jump_phase = torch.full(
            (self.num_envs,), self.PHASE_PREPARE, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.cycle_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.prepare_stand_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.prepare_stand_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.airborne_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.cycle_max_airborne_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.landing_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.recovery_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.just_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.just_took_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.just_completed_cycle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.jump_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.landing_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.completed_cycles = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        self.jump_toggle_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.external_toggle_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.episode_rsi_used = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.term_dof_limit_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.term_nonfoot_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.term_rebound_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.term_takeoff_abort_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.term_upside_down_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.term_timeout_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self._init_curriculum()
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        self._lock_landing_target(torch.arange(self.num_envs, device=self.device))

    def _sync_commands_from_state(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = self.jump_toggle_state[env_ids].float()

    def _build_joint_target(self, joint_angles):
        target = self.default_dof_pos.clone()
        for dof_id, dof_name in enumerate(self.dof_names):
            if dof_name in joint_angles:
                target[:, dof_id] = float(joint_angles[dof_name])
        return target

    def _init_curriculum(self):
        if not hasattr(self.cfg, "curriculum"):
            if hasattr(self.cfg, "rsi"):
                self.active_rsi_cfg = class_to_dict(self.cfg.rsi)
            return

        self.curriculum_cfg_dict = class_to_dict(self.cfg.curriculum)
        self.curriculum_enabled = bool(self.curriculum_cfg_dict.get("enabled", False))
        self.curriculum_auto_advance = bool(self.curriculum_cfg_dict.get("auto_advance", False))
        self.base_reward_scales_raw = class_to_dict(self.cfg.rewards.scales)
        self.active_rsi_cfg = class_to_dict(self.cfg.rsi) if hasattr(self.cfg, "rsi") else {}

        if not self.curriculum_enabled:
            return

        stage_order = self.curriculum_cfg_dict.get("stage_order", [])
        if not stage_order:
            self.curriculum_enabled = False
            return

        initial_stage = self.curriculum_cfg_dict.get("initial_stage", stage_order[0])
        if initial_stage not in stage_order:
            initial_stage = stage_order[0]
        self._apply_curriculum_stage(initial_stage)

    def _apply_curriculum_stage(self, stage_name):
        stage_order = self.curriculum_cfg_dict.get("stage_order", [])
        if stage_name not in stage_order:
            return

        stage_cfg = self.curriculum_cfg_dict.get(stage_name, {})
        self.curriculum_stage_name = stage_name
        self.curriculum_stage_index = stage_order.index(stage_name)
        self.curriculum_stage_cfg = stage_cfg

        if hasattr(self.cfg, "rsi"):
            self.active_rsi_cfg = class_to_dict(self.cfg.rsi)
            for key, val in stage_cfg.items():
                if key in self.active_rsi_cfg:
                    self.active_rsi_cfg[key] = val

        self._apply_reward_scale_table(stage_cfg.get("reward_scales", {}))

        landing_dx_range = stage_cfg.get("landing_dx_range", None)
        if landing_dx_range is not None and "lin_vel_x" in self.command_ranges:
            self.command_ranges["lin_vel_x"][0] = float(landing_dx_range[0])
            self.command_ranges["lin_vel_x"][1] = float(landing_dx_range[1])

        print(
            f"[GO2JumpTorque] Curriculum stage {self.curriculum_stage_index}: {self.curriculum_stage_name}"
        )

    def _apply_reward_scale_table(self, stage_reward_scales):
        scale_factor = self.dt if getattr(self, "init_done", False) else 1.0
        reward_scales = {}
        for name, base_scale in self.base_reward_scales_raw.items():
            raw_scale = float(stage_reward_scales.get(name, base_scale))
            if raw_scale != 0.0:
                reward_scales[name] = raw_scale * scale_factor
        self.reward_scales = reward_scales

        if not getattr(self, "init_done", False) or not hasattr(self, "reward_functions"):
            return

        self.reward_functions = []
        self.reward_names = []
        for name in self.reward_scales.keys():
            if name == "termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))

        if not hasattr(self, "episode_sums"):
            self.episode_sums = {}
        for name in self.reward_scales.keys():
            if name not in self.episode_sums:
                self.episode_sums[name] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
                )

    def _maybe_advance_curriculum(self, reset_stats):
        if not self.curriculum_enabled or not self.curriculum_auto_advance:
            return

        stage_order = self.curriculum_cfg_dict.get("stage_order", [])
        if self.curriculum_stage_index < 0 or self.curriculum_stage_index >= len(stage_order) - 1:
            return

        thresholds = self.curriculum_cfg_dict.get("auto_thresholds", {})
        natural_flight_rate = float(reset_stats["natural_flight_rate"].item())
        clearance_gate = float(reset_stats["clearance_gate"].item())
        valid_jump_rate = float(reset_stats["valid_jump_rate"].item())
        stable_landing_forward_error = float(reset_stats["stable_landing_forward_error"].item())

        if self.curriculum_stage_name == "takeoff_foundation":
            if (
                natural_flight_rate >= float(thresholds.get("takeoff_foundation_natural_flight_rate", 1.0))
            ):
                self._apply_curriculum_stage("forward_launch")
        elif self.curriculum_stage_name == "forward_launch":
            if (
                natural_flight_rate >= float(thresholds.get("forward_launch_natural_flight_rate", 1.0))
                and valid_jump_rate >= float(thresholds.get("forward_launch_valid_jump_rate", 1.0))
                and stable_landing_forward_error
                >= float(thresholds.get("forward_launch_stable_landing_forward_error", 0.0))
            ):
                self._apply_curriculum_stage("target_landing")

    def _get_active_rsi_value(self, key, default):
        if key in self.active_rsi_cfg:
            return self.active_rsi_cfg[key]
        return default

    def _get_curriculum_flag(self, key, default=False):
        if not self.curriculum_enabled:
            return default
        return bool(self.curriculum_stage_cfg.get(key, default))

    def _reset_cycle_state(self, env_ids):
        if len(env_ids) == 0:
            return
        self.jump_start_height[env_ids] = self.root_states[env_ids, 2]
        self.jump_start_xy[env_ids] = self.root_states[env_ids, :2]
        self.jump_start_yaw[env_ids] = self._get_base_yaw()[env_ids]
        self.takeoff_height[env_ids] = self.root_states[env_ids, 2]
        self.ready_stance_height[env_ids] = self.root_states[env_ids, 2]
        self.jump_reference_height[env_ids] = torch.maximum(
            self.ready_stance_height[env_ids],
            torch.full_like(self.ready_stance_height[env_ids], float(self.cfg.jump_phase.stand_height_target)),
        )
        self.cycle_peak_height[env_ids] = self.root_states[env_ids, 2]
        self.cycle_steps[env_ids] = 0
        self.phase_steps[env_ids] = 0
        self.prepare_stand_steps[env_ids] = 0
        self.prepare_stand_ready[env_ids] = False
        self.airborne_steps[env_ids] = 0
        self.cycle_max_airborne_steps[env_ids] = 0
        self.landing_steps[env_ids] = 0
        self.recovery_steps[env_ids] = 0
        self.just_landed[env_ids] = False
        self.just_took_off[env_ids] = False
        self.just_completed_cycle[env_ids] = False
        self.jump_phase[env_ids] = self.PHASE_PREPARE
        self._lock_landing_target(env_ids)

    def _set_phase(self, mask, phase):
        env_ids = mask.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return env_ids
        self.jump_phase[env_ids] = phase
        self.phase_steps[env_ids] = 0
        return env_ids

    def _get_cycle_phase(self):
        cycle_time = max(float(self.cfg.jump_phase.cycle_time), 1e-3)
        return torch.remainder(self.cycle_steps.float() * self.dt / cycle_time, 1.0)

    def _get_timed_phase_masks(self):
        phase = self._get_cycle_phase()
        takeoff_start = float(self.cfg.jump_phase.takeoff_phase_start)
        flight_start = float(self.cfg.jump_phase.flight_phase_start)
        landing_start = float(self.cfg.jump_phase.landing_phase_start)
        stance_mask = phase < takeoff_start
        takeoff_mask = (phase >= takeoff_start) & (phase < flight_start)
        flight_mask = (phase >= flight_start) & (phase < landing_start)
        landing_mask = phase >= landing_start
        return stance_mask, takeoff_mask, flight_mask, landing_mask

    def _get_prelanding_mask(self):
        phase = self._get_cycle_phase()
        prelanding_start = float(self.cfg.rewards.prelanding_phase_start)
        _, _, flight_contact_mask, _ = self._get_jump_contact_states()
        return (phase >= prelanding_start) & flight_contact_mask

    def _get_target_frame_xy_error(self, env_ids):
        error_xy = self.root_states[env_ids, :2] - self.target_landing_xy[env_ids]
        yaw = self.jump_start_yaw[env_ids]
        return self._target_frame_xy_error_from_delta(error_xy, yaw)

    def _target_frame_xy_error_from_delta(self, error_xy, yaw):
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        forward_error = cos_yaw * error_xy[:, 0] + sin_yaw * error_xy[:, 1]
        lateral_error = -sin_yaw * error_xy[:, 0] + cos_yaw * error_xy[:, 1]
        return forward_error, lateral_error

    def _update_test_command_state(self):
        if not self.cfg.test.use_test:
            return

        test_command = self.cfg.test.vel.to(self.device)
        desired_toggle = torch.full_like(
            self.jump_toggle_state,
            test_command[0] > self.cfg.jump_phase.request_threshold,
        )
        if self.cfg.commands.num_commands > 1 and test_command.numel() >= self.cfg.commands.num_commands:
            self.commands[:, 1:self.cfg.commands.num_commands] = test_command[1:self.cfg.commands.num_commands]

        clear_mask = (~desired_toggle) & (self.jump_phase == self.PHASE_PREPARE)
        self.jump_toggle_state[clear_mask] = False

        rising_edge = desired_toggle & ~self.external_toggle_state
        self.jump_toggle_state |= rising_edge
        self.external_toggle_state[:] = desired_toggle

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if hasattr(self, "rigid_body_states"):
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_test_command_state()
        self._sync_commands_from_state()

        self.cycle_steps += 1
        self.phase_steps += 1
        self.just_landed[:] = False
        self.just_took_off[:] = False
        self.just_completed_cycle[:] = False
        self.term_takeoff_abort_buf[:] = False

        contact, num_contacts, flight_contact_mask, landing_contact_mask = self._get_jump_contact_states()
        (
            prepare_mask,
            compression_mask,
            extension_mask,
            release_mask,
            flight_mask,
            touchdown_mask,
            stabilize_mask,
        ) = self._get_jump_phase_masks()
        takeoff_mask = compression_mask | extension_mask | release_mask
        current_height = self.root_states[:, 2]
        stable_prepare_mask = prepare_mask & (num_contacts >= self.cfg.jump_phase.prepare_min_contacts)
        self.takeoff_height = torch.where(stable_prepare_mask, current_height, self.takeoff_height)

        self.cycle_peak_height = torch.where(
            prepare_mask,
            self.takeoff_height,
            torch.maximum(self.cycle_peak_height, current_height),
        )
        peak_rel_height = torch.clamp(self.cycle_peak_height - self.jump_reference_height, min=0.0)
        current_rel_height = torch.clamp(current_height - self.jump_reference_height, min=0.0)
        stand_clearance = torch.clamp(
            current_height - float(self.cfg.jump_phase.stand_height_target),
            min=0.0,
        )
        self.episode_peak_height = torch.where(
            prepare_mask,
            self.episode_peak_height,
            torch.maximum(self.episode_peak_height, peak_rel_height),
        )
        self.episode_peak_base_height = torch.where(
            prepare_mask,
            self.episode_peak_base_height,
            torch.maximum(self.episode_peak_base_height, current_height),
        )

        self.airborne_steps = torch.where(
            flight_contact_mask,
            self.airborne_steps + 1,
            torch.zeros_like(self.airborne_steps),
        )
        self.cycle_max_airborne_steps = torch.where(
            prepare_mask,
            self.cycle_max_airborne_steps,
            torch.maximum(self.cycle_max_airborne_steps, self.airborne_steps),
        )
        valid_takeoff_signal = (
            (current_rel_height > self.cfg.jump_phase.min_flight_height)
            | (self.root_states[:, 9] > self.cfg.jump_phase.min_takeoff_velocity)
        )
        min_stand_clearance = float(self.cfg.jump_phase.min_flight_stand_clearance)
        if min_stand_clearance > 0.0:
            valid_takeoff_signal &= stand_clearance > min_stand_clearance

        self.landing_steps = torch.where(
            (flight_mask | touchdown_mask) & landing_contact_mask,
            self.landing_steps + 1,
            torch.zeros_like(self.landing_steps),
        )

        tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
        xy_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        ang_speed_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        all_feet_down = num_contacts == len(self.feet_indices)
        ready_contacts = num_contacts >= getattr(self.cfg.jump_phase, "prepare_ready_min_contacts", len(self.feet_indices))
        stand_height_err = torch.abs(current_height - float(self.cfg.jump_phase.stand_height_target))
        stand_pose_err = torch.mean(torch.abs(self.dof_pos - self.default_joint_pd_target), dim=1)
        prepare_stand_condition = (
            prepare_mask
            & ready_contacts
            & (stand_height_err < self.cfg.jump_phase.prepare_stand_height_tol)
            & (stand_pose_err < self.cfg.jump_phase.prepare_stand_pose_tol)
            & (xy_speed < self.cfg.jump_phase.prepare_stand_max_lin_vel)
            & (torch.abs(self.base_lin_vel[:, 2]) < self.cfg.jump_phase.prepare_stand_max_vertical_vel)
            & (ang_speed_xy < self.cfg.jump_phase.prepare_stand_max_ang_vel)
            & (tilt < self.cfg.jump_phase.prepare_stand_max_tilt)
        )
        self.ready_stance_height = torch.where(
            prepare_stand_condition,
            current_height,
            self.ready_stance_height,
        )
        self.prepare_stand_steps = torch.where(
            prepare_stand_condition,
            self.prepare_stand_steps + 1,
            torch.zeros_like(self.prepare_stand_steps),
        )
        self.prepare_stand_ready |= self.prepare_stand_steps >= self.cfg.jump_phase.prepare_stand_hold_steps
        recovery_ready = (
            (touchdown_mask | stabilize_mask)
            & (num_contacts >= self.cfg.jump_phase.recovery_min_contacts)
            & (xy_speed < self.cfg.jump_phase.recovery_max_lin_vel)
            & (torch.abs(self.base_lin_vel[:, 2]) < self.cfg.jump_phase.recovery_max_vertical_vel)
            & (ang_speed_xy < self.cfg.jump_phase.recovery_max_ang_vel)
            & (tilt < self.cfg.jump_phase.recovery_max_tilt)
        )
        self.recovery_steps = torch.where(
            recovery_ready,
            self.recovery_steps + 1,
            torch.zeros_like(self.recovery_steps),
        )

        jump_requested = self.jump_toggle_state
        # Require a short, stable four-contact stand before the next takeoff.
        # This prevents continuous pogo-like re-triggering and gives the robot
        # time to recover between jumps.
        ready_for_takeoff = (
            prepare_mask
            & jump_requested
            & self.prepare_stand_ready
            & (num_contacts >= self.cfg.jump_phase.prepare_min_contacts)
        )
        enter_compression = ready_for_takeoff
        compression_ids = self._set_phase(enter_compression, self.PHASE_COMPRESSION)
        if len(compression_ids) > 0:
            self.jump_reference_height[compression_ids] = torch.maximum(
                self.ready_stance_height[compression_ids],
                torch.full_like(self.ready_stance_height[compression_ids], float(self.cfg.jump_phase.stand_height_target)),
            )
            self.jump_start_height[compression_ids] = current_height[compression_ids]
            self.jump_start_xy[compression_ids] = self.root_states[compression_ids, :2]
            self.jump_start_yaw[compression_ids] = self._get_base_yaw()[compression_ids]
            self._lock_landing_target(compression_ids)
            self.cycle_peak_height[compression_ids] = current_height[compression_ids]
            self.airborne_steps[compression_ids] = 0
            self.cycle_max_airborne_steps[compression_ids] = 0
            self.landing_steps[compression_ids] = 0
            self.recovery_steps[compression_ids] = 0

        squat_depth = torch.clamp(self.jump_reference_height - current_height, min=0.0)
        upward_trend = self.root_states[:, 9] - self.last_root_vel[:, 2]
        # Contact-driven: enter EXTENSION as soon as vz turns positive or body lifts.
        # No time gate (phase_steps removed).
        enter_extension = compression_mask & (
            (self.root_states[:, 9] > 0.0)
            | (squat_depth >= float(self.cfg.jump_phase.compression_target_depth))
            | (upward_trend > 0.02)
        )
        extension_ids = self._set_phase(enter_extension, self.PHASE_EXTENSION)
        if len(extension_ids) > 0:
            self.landing_steps[extension_ids] = 0
            self.recovery_steps[extension_ids] = 0

        # Contact-driven: enter RELEASE as soon as feet start leaving ground or vz is sufficient.
        # No time gate.
        enter_release = extension_mask & (
            (self.root_states[:, 9] >= float(self.cfg.jump_phase.takeoff_entry_velocity))
            | (current_rel_height > 0.0)
            | flight_contact_mask
        )
        release_ids = self._set_phase(enter_release, self.PHASE_RELEASE)
        if len(release_ids) > 0:
            self.landing_steps[release_ids] = 0
            self.recovery_steps[release_ids] = 0

        # Contact-driven: enter FLIGHT when fully airborne (no foot contacts).
        # No time gate, no phase_steps check.
        enter_flight = release_mask & flight_contact_mask & valid_takeoff_signal
        flight_ids = self._set_phase(enter_flight, self.PHASE_FLIGHT)
        if len(flight_ids) > 0:
            self.jump_count[flight_ids] += 1
            self.just_took_off[flight_ids] = True
            self.landing_steps[flight_ids] = 0
            self.recovery_steps[flight_ids] = 0

        abort_takeoff = (
            takeoff_mask
            & (~enter_flight)
            & (self.phase_steps >= self.cfg.jump_phase.takeoff_max_steps)
        )
        abort_ids = abort_takeoff.nonzero(as_tuple=False).flatten()
        if len(abort_ids) > 0:
            self.term_takeoff_abort_buf[abort_ids] = True
            if not getattr(self.cfg.jump_phase, "terminate_on_takeoff_abort", True):
                self._reset_cycle_state(abort_ids)

        # Contact-driven: enter TOUCHDOWN when feet make contact during flight.
        # No settle steps — contact itself is the signal.
        enter_touchdown = flight_mask & landing_contact_mask
        touchdown_ids = self._set_phase(enter_touchdown, self.PHASE_TOUCHDOWN)
        if len(touchdown_ids) > 0:
            self.just_landed[touchdown_ids] = True
            self.landing_count[touchdown_ids] += 1
            self.recovery_steps[touchdown_ids] = 0
            self.last_landing_position_error[touchdown_ids] = torch.norm(
                self.root_states[touchdown_ids, :2] - self.target_landing_xy[touchdown_ids],
                dim=1,
            )
            (
                self.last_landing_forward_error[touchdown_ids],
                self.last_landing_lateral_error[touchdown_ids],
            ) = self._get_target_frame_xy_error(touchdown_ids)
            self.last_landing_yaw_error[touchdown_ids] = torch.abs(
                wrap_to_pi(self._get_base_yaw()[touchdown_ids] - self.target_landing_yaw[touchdown_ids])
            )

        # Contact-driven: enter STABILIZE when 4 feet are down and vertical speed is low.
        enter_stabilize = touchdown_mask & (num_contacts >= self.cfg.jump_phase.recovery_min_contacts) & (torch.abs(self.root_states[:, 9]) < 0.3)
        stabilize_ids = self._set_phase(enter_stabilize, self.PHASE_STABILIZE)
        if len(stabilize_ids) > 0:
            self.recovery_steps[stabilize_ids] = 0

        # Complete cycle only after the robot has remained stably recovered for
        # several control steps. This creates a real stand-between-jumps phase.
        recovery_hold_steps = max(int(getattr(self.cfg.jump_phase, "recovery_hold_steps", 1)), 1)
        cycle_complete = (
            stabilize_mask
            & (num_contacts >= self.cfg.jump_phase.recovery_min_contacts)
            & (torch.abs(self.root_states[:, 9]) < 0.2)
            & (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.4)
            & (self.recovery_steps >= recovery_hold_steps)
        )
        completed_ids = cycle_complete.nonzero(as_tuple=False).flatten()
        if len(completed_ids) > 0:
            self.last_completed_peak_height[completed_ids] = torch.clamp(
                self.cycle_peak_height[completed_ids] - self.jump_reference_height[completed_ids],
                min=0.0,
            )
            self.last_completed_base_peak_height[completed_ids] = self.cycle_peak_height[completed_ids]
            self.last_completed_landing_position_error[completed_ids] = torch.norm(
                self.root_states[completed_ids, :2] - self.target_landing_xy[completed_ids],
                dim=1,
            )
            (
                self.last_completed_landing_forward_error[completed_ids],
                self.last_completed_landing_lateral_error[completed_ids],
            ) = self._get_target_frame_xy_error(completed_ids)
            self.last_completed_landing_yaw_error[completed_ids] = torch.abs(
                wrap_to_pi(self._get_base_yaw()[completed_ids] - self.target_landing_yaw[completed_ids])
            )
            self.last_completed_landing_tilt[completed_ids] = tilt[completed_ids]
            self.last_completed_clearance_gate[completed_ids] = self._get_clearance_gate()[completed_ids]
            self.last_completed_valid_jump[completed_ids] = self._get_valid_jump_gate()[completed_ids] >= 1.0
            self.last_completed_airborne_steps[completed_ids] = self.cycle_max_airborne_steps[completed_ids]
            self.completed_cycles[completed_ids] += 1
            if not getattr(self.cfg.commands, "repeat_jump_in_episode", False):
                self.jump_toggle_state[completed_ids] = False
                self._sync_commands_from_state(completed_ids)
            else:
                self._resample_commands(completed_ids)
            self._reset_cycle_state(completed_ids)
            self.just_completed_cycle[completed_ids] = True

        self._sync_commands_from_state()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        ranges = self.cfg.commands.ranges
        self.jump_toggle_state[env_ids] = (
            torch_rand_float(
                ranges.jump_toggle[0],
                ranges.jump_toggle[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            > self.cfg.jump_phase.request_threshold
        )
        self.external_toggle_state[env_ids] = False
        if self.cfg.commands.num_commands > 1:
            self.commands[env_ids, 1] = torch_rand_float(
                ranges.lin_vel_x[0], ranges.lin_vel_x[1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(
                ranges.lin_vel_y[0], ranges.lin_vel_y[1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 3] = torch_rand_float(
                ranges.ang_vel_yaw[0], ranges.ang_vel_yaw[1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self._sync_commands_from_state(env_ids)

    def _get_pd_residual_scale(self):
        if not getattr(self.cfg.control, "pd_residual_enabled", False):
            return 0.0
        if not getattr(self.cfg.control, "pd_fade_enabled", True):
            return 1.0

        warmup_steps = float(self.cfg.control.pd_warmup_steps)
        ramp_steps = max(float(self.cfg.control.pd_ramp_steps), 1.0)
        if self.step_count < warmup_steps:
            return 1.0

        progress = min(max((self.step_count - warmup_steps) / ramp_steps, 0.0), 1.0)
        return 1.0 - progress

    def _compute_torques(self, actions):
        self._update_growth_scale()
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        self.torques_action = actions_scaled
        torques_limits = self.current_torque_limit_scale * self.torque_limits
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
            residual_torques = self.activation_sign * torques_limits * (
                1 - torch.sign(self.activation_sign) * self.dof_vel / self.dof_vel_limits
            )
        else:
            residual_torques = self.activation_sign * torques_limits

        self.current_pd_prior_scale = self._get_pd_residual_scale()
        if getattr(self.cfg.control, "pd_residual_enabled", False) and self.current_pd_prior_scale > 0.0:
            gain_scale = float(self.cfg.control.pd_gain_scale) * self.current_pd_prior_scale
            pd_torques = (
                self.p_gains * gain_scale * (self.default_joint_pd_target - self.dof_pos)
                - self.d_gains * gain_scale * self.dof_vel
            )
            pd_limit = torch.full_like(pd_torques, float(self.cfg.control.pd_torque_limit))
            pd_limit_fraction = float(getattr(self.cfg.control, "pd_torque_limit_fraction", 1.0))
            if pd_limit_fraction > 0.0:
                pd_limit = torch.minimum(pd_limit, torques_limits * pd_limit_fraction)
            pd_torques = torch.maximum(torch.minimum(pd_torques, pd_limit), -pd_limit)
            self.torques = residual_torques + pd_torques
        else:
            self.torques = residual_torques

        self.torques = torch.clip(self.torques, -torques_limits, torques_limits)

        if self.cfg.control.motor_fatigue:
            self.motor_fatigue += torch.abs(self.torques) * self.dt
            self.motor_fatigue *= 0.9
        else:
            self.motor_fatigue = torch.zeros_like(self.motor_fatigue)

        if self.low_torque:
            self.torques[:, :3] = self.torques[:, :3] * 0.2

        return self.torques

    def compute_observations(self):
        base_lin_vel = self.base_lin_vel
        motor_fatigue = self.motor_fatigue.detach()
        foot_contact_obs = self._get_foot_contact_obs()
        cycle_phase = self._get_cycle_phase()
        phase_obs = torch.stack(
            (
                torch.sin(2.0 * torch.pi * cycle_phase),
                torch.cos(2.0 * torch.pi * cycle_phase),
            ),
            dim=1,
        )
        obs_buf = torch.cat(
            (
                base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.commands[:, :self.cfg.commands.num_commands] * self.commands_scale,
                phase_obs,
                foot_contact_obs,
                self.torques,
                motor_fatigue,
            ),
            dim=-1,
        )
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            obs_buf,
            self.obs_buf,
        )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:37] = 0.0
        noise_vec[37:39] = 0.0
        noise_vec[39:43] = noise_scales.contact * noise_level
        noise_vec[43:55] = 0.0
        noise_vec[55:67] = noise_scales.fatigue * noise_level / 10
        return noise_vec

    def check_termination(self):
        dof_pos_limits_up = self.termination_dof_pos_limits[:, 1]
        dof_pos_limits_low = self.termination_dof_pos_limits[:, 0]

        self.term_dof_limit_buf = torch.any(self.dof_pos > dof_pos_limits_up, dim=1) | torch.any(
            self.dof_pos < dof_pos_limits_low, dim=1
        )
        jump_active = (self.jump_phase != self.PHASE_PREPARE) | self.just_completed_cycle
        self.term_nonfoot_contact_buf = self._get_critical_nonfoot_contact_mask() & jump_active
        _, _, flight_contact_mask, _ = self._get_jump_contact_states()
        landing_or_recovery = (
            (self.jump_phase == self.PHASE_TOUCHDOWN)
            | (self.jump_phase == self.PHASE_STABILIZE)
            | self.just_landed
        )
        rebound_airborne_steps = int(getattr(self.cfg.jump_phase, "rebound_airborne_steps", 2))
        rebound_min_upward_velocity = float(getattr(self.cfg.jump_phase, "rebound_min_upward_velocity", 0.0))
        self.term_rebound_buf = (
            landing_or_recovery
            & flight_contact_mask
            & (self.airborne_steps >= rebound_airborne_steps)
            & (self.root_states[:, 9] > rebound_min_upward_velocity)
        )
        self.term_upside_down_buf = self.projected_gravity[:, 2] > 0
        self.term_timeout_buf = self.episode_length_buf > self.max_episode_length

        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_buf |= self.term_dof_limit_buf
        if not self._get_curriculum_flag("relax_nonfoot_termination", False):
            self.reset_buf |= self.term_nonfoot_contact_buf
        if getattr(self.cfg.jump_phase, "terminate_on_rebound", True):
            self.reset_buf |= self.term_rebound_buf
        if getattr(self.cfg.jump_phase, "terminate_on_takeoff_abort", True):
            self.reset_buf |= self.term_takeoff_abort_buf
        self.reset_buf |= self.term_upside_down_buf

        self.time_out_buf = self.term_timeout_buf
        self.reset_buf |= self.term_timeout_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_landing_dx_curriculum(env_ids)
        reset_stats = self._collect_episode_stats(env_ids)
        self._maybe_advance_curriculum(reset_stats)
        reset_stats["curriculum_stage_index"] = torch.full(
            (), float(self.curriculum_stage_index), dtype=torch.float, device=self.device
        )
        super().reset_idx(env_ids)
        self.extras["episode"].update(reset_stats)

        self.torques[env_ids] = 0.0
        if hasattr(self, "torques_action"):
            self.torques_action[env_ids] = 0.0

        self.episode_peak_height[env_ids] = 0.0
        self.episode_peak_base_height[env_ids] = self.root_states[env_ids, 2]
        self.jump_count[env_ids] = 0
        self.landing_count[env_ids] = 0
        self.completed_cycles[env_ids] = 0
        self.last_landing_position_error[env_ids] = 0.0
        self.last_landing_forward_error[env_ids] = 0.0
        self.last_landing_lateral_error[env_ids] = 0.0
        self.last_landing_yaw_error[env_ids] = 0.0
        self.last_completed_peak_height[env_ids] = 0.0
        self.last_completed_base_peak_height[env_ids] = self.root_states[env_ids, 2]
        self.last_completed_landing_position_error[env_ids] = 0.0
        self.last_completed_landing_forward_error[env_ids] = 0.0
        self.last_completed_landing_lateral_error[env_ids] = 0.0
        self.last_completed_landing_yaw_error[env_ids] = 0.0
        self.last_completed_landing_tilt[env_ids] = 0.0
        self.last_estimated_landing_position_error[env_ids] = 0.0
        self.last_estimated_landing_forward_error[env_ids] = 0.0
        self.last_estimated_landing_lateral_error[env_ids] = 0.0
        self.last_completed_clearance_gate[env_ids] = 0.0
        self.last_completed_valid_jump[env_ids] = False
        self.last_completed_airborne_steps[env_ids] = 0
        self.episode_rsi_used[env_ids] = False
        self.term_rebound_buf[env_ids] = False
        self.term_takeoff_abort_buf[env_ids] = False
        self._reset_cycle_state(env_ids)
        self._sync_commands_from_state(env_ids)
        self._apply_rsi_reset(env_ids)

    def _apply_rsi_reset(self, env_ids):
        if len(env_ids) == 0 or self.cfg.test.use_test:
            return
        if not hasattr(self.cfg, "rsi") and not self.active_rsi_cfg:
            return

        probability = float(self._get_active_rsi_value("probability", 0.0))
        if probability <= 0.0:
            return

        rsi_mask = torch.rand(len(env_ids), device=self.device) < probability
        rsi_ids = env_ids[rsi_mask]
        if len(rsi_ids) == 0:
            return

        takeoff_probability = min(max(float(self._get_active_rsi_value("takeoff_state_probability", 0.0)), 0.0), 1.0)
        takeoff_rsi_mask = torch.rand(len(rsi_ids), device=self.device) < takeoff_probability
        takeoff_rsi_ids = rsi_ids[takeoff_rsi_mask]
        flight_rsi_ids = rsi_ids[~takeoff_rsi_mask]

        height_range = self._get_active_rsi_value("height_offset_range", [0.01, 0.05])
        velocity_range = self._get_active_rsi_value("upward_velocity_range", [0.25, 0.9])
        height_offset = torch_rand_float(
            height_range[0], height_range[1], (len(rsi_ids), 1), device=self.device
        ).squeeze(1)
        upward_velocity = torch_rand_float(
            velocity_range[0], velocity_range[1], (len(rsi_ids), 1), device=self.device
        ).squeeze(1)
        forward_fraction_range = self._get_active_rsi_value("forward_velocity_fraction_range", [0.0, 0.0])
        forward_fraction = torch_rand_float(
            forward_fraction_range[0],
            forward_fraction_range[1],
            (len(rsi_ids), 1),
            device=self.device,
        ).squeeze(1)

        reference_height = self.root_states[rsi_ids, 2].clone()
        yaw = self._get_base_yaw()[rsi_ids]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        dx = self.commands[rsi_ids, 1]
        dy = self.commands[rsi_ids, 2]
        target_delta = torch.stack(
            (
                cos_yaw * dx - sin_yaw * dy,
                sin_yaw * dx + cos_yaw * dy,
            ),
            dim=1,
        )
        target_dist = torch.clamp(torch.norm(target_delta, dim=1), min=1e-3)
        target_dir = target_delta / target_dist.unsqueeze(1)
        forward_velocity = self._get_target_takeoff_forward_velocity()[rsi_ids] * forward_fraction

        self.root_states[rsi_ids, 2] = reference_height + height_offset
        self.root_states[rsi_ids, 7:13] = 0.0
        self.root_states[rsi_ids, 7:9] = target_dir * forward_velocity.unsqueeze(1)
        self.root_states[rsi_ids, 9] = upward_velocity

        self.dof_pos[rsi_ids] = self.aerial_joint_target.expand(len(rsi_ids), -1)
        self.dof_vel[rsi_ids] = 0.0

        if len(takeoff_rsi_ids) > 0:
            takeoff_height_range = self._get_active_rsi_value("takeoff_height_offset_range", [-0.04, 0.0])
            takeoff_velocity_range = self._get_active_rsi_value("takeoff_upward_velocity_range", [0.0, 0.15])
            takeoff_forward_fraction_range = self._get_active_rsi_value(
                "takeoff_forward_velocity_fraction_range", [0.0, 0.0]
            )
            takeoff_height_offset = torch_rand_float(
                takeoff_height_range[0],
                takeoff_height_range[1],
                (len(takeoff_rsi_ids), 1),
                device=self.device,
            ).squeeze(1)
            takeoff_upward_velocity = torch_rand_float(
                takeoff_velocity_range[0],
                takeoff_velocity_range[1],
                (len(takeoff_rsi_ids), 1),
                device=self.device,
            ).squeeze(1)
            takeoff_forward_fraction = torch_rand_float(
                takeoff_forward_fraction_range[0],
                takeoff_forward_fraction_range[1],
                (len(takeoff_rsi_ids), 1),
                device=self.device,
            ).squeeze(1)
            takeoff_reference_height = reference_height[takeoff_rsi_mask]
            takeoff_forward_velocity = (
                self._get_target_takeoff_forward_velocity()[takeoff_rsi_ids] * takeoff_forward_fraction
            )
            self.root_states[takeoff_rsi_ids, 2] = torch.clamp(
                takeoff_reference_height + takeoff_height_offset,
                min=float(self.cfg.jump_phase.stand_height_target),
            )
            self.root_states[takeoff_rsi_ids, 7:13] = 0.0
            self.root_states[takeoff_rsi_ids, 7:9] = (
                target_dir[takeoff_rsi_mask] * takeoff_forward_velocity.unsqueeze(1)
            )
            self.root_states[takeoff_rsi_ids, 9] = takeoff_upward_velocity
            self.dof_pos[takeoff_rsi_ids] = self.takeoff_squat_joint_target.expand(len(takeoff_rsi_ids), -1)

        rsi_ids_int32 = rsi_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(rsi_ids_int32),
            len(rsi_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(rsi_ids_int32),
            len(rsi_ids_int32),
        )

        self.base_quat[rsi_ids] = self.root_states[rsi_ids, 3:7]
        self.last_root_vel[rsi_ids] = self.root_states[rsi_ids, 7:13]
        self.base_lin_vel[rsi_ids] = quat_rotate_inverse(self.base_quat[rsi_ids], self.root_states[rsi_ids, 7:10])
        self.last_base_lin_vel[rsi_ids] = self.base_lin_vel[rsi_ids]
        self.base_ang_vel[rsi_ids] = quat_rotate_inverse(self.base_quat[rsi_ids], self.root_states[rsi_ids, 10:13])
        self.projected_gravity[rsi_ids] = quat_rotate_inverse(self.base_quat[rsi_ids], self.gravity_vec[rsi_ids])

        self.jump_toggle_state[rsi_ids] = True
        self.external_toggle_state[rsi_ids] = False
        self.jump_phase[rsi_ids] = self.PHASE_FLIGHT
        self.phase_steps[rsi_ids] = 0
        self.prepare_stand_steps[rsi_ids] = 0
        self.prepare_stand_ready[rsi_ids] = True
        self.airborne_steps[rsi_ids] = self.cfg.jump_phase.min_airborne_steps
        self.cycle_max_airborne_steps[rsi_ids] = self.cfg.jump_phase.min_airborne_steps
        self.landing_steps[rsi_ids] = 0
        self.recovery_steps[rsi_ids] = 0
        self.jump_count[rsi_ids] = 1
        self.episode_rsi_used[rsi_ids] = True

        flight_start = float(self.cfg.jump_phase.flight_phase_start)
        landing_start = float(self.cfg.jump_phase.landing_phase_start)
        phase_end = max(flight_start + 1e-3, min(landing_start - 0.05, 0.95))
        rsi_phase = torch_rand_float(flight_start, phase_end, (len(rsi_ids), 1), device=self.device).squeeze(1)
        cycle_time = max(float(self.cfg.jump_phase.cycle_time), 1e-3)
        self.cycle_steps[rsi_ids] = torch.clamp(
            (rsi_phase * cycle_time / self.dt).to(dtype=torch.long),
            min=0,
        )

        if len(takeoff_rsi_ids) > 0:
            self.jump_phase[takeoff_rsi_ids] = self.PHASE_EXTENSION
            self.airborne_steps[takeoff_rsi_ids] = 0
            self.cycle_max_airborne_steps[takeoff_rsi_ids] = 0
            self.jump_count[takeoff_rsi_ids] = 0
            self.landing_steps[takeoff_rsi_ids] = 0
            self.recovery_steps[takeoff_rsi_ids] = 0
            takeoff_start = float(self.cfg.jump_phase.takeoff_phase_start)
            flight_start = float(self.cfg.jump_phase.flight_phase_start)
            takeoff_phase_range = self._get_active_rsi_value(
                "takeoff_phase_range",
                [takeoff_start, max(takeoff_start + 1e-3, flight_start - 0.02)],
            )
            phase_min = max(takeoff_start, float(takeoff_phase_range[0]))
            phase_max = min(float(takeoff_phase_range[1]), flight_start - 1e-3)
            phase_max = max(phase_min + 1e-3, phase_max)
            takeoff_phase = torch_rand_float(
                phase_min,
                phase_max,
                (len(takeoff_rsi_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.cycle_steps[takeoff_rsi_ids] = torch.clamp(
                (takeoff_phase * cycle_time / self.dt).to(dtype=torch.long),
                min=0,
            )

        self.jump_start_height[rsi_ids] = reference_height
        self.jump_start_xy[rsi_ids] = self.root_states[rsi_ids, :2]
        self.jump_start_yaw[rsi_ids] = self._get_base_yaw()[rsi_ids]
        self.takeoff_height[rsi_ids] = reference_height
        self.ready_stance_height[rsi_ids] = reference_height
        self.jump_reference_height[rsi_ids] = torch.maximum(
            reference_height,
            torch.full_like(reference_height, float(self.cfg.jump_phase.stand_height_target)),
        )
        self.cycle_peak_height[rsi_ids] = self.root_states[rsi_ids, 2]
        self.episode_peak_height[rsi_ids] = height_offset
        self.episode_peak_base_height[rsi_ids] = self.root_states[rsi_ids, 2]
        if len(takeoff_rsi_ids) > 0:
            self.episode_peak_height[takeoff_rsi_ids] = 0.0
        self._lock_landing_target(rsi_ids)
        self._sync_commands_from_state(rsi_ids)

    def _mean_or_zero(self, values, mask=None):
        if mask is not None:
            if not torch.any(mask):
                return torch.zeros((), dtype=torch.float, device=self.device)
            values = values[mask]
        if values.numel() == 0:
            return torch.zeros((), dtype=torch.float, device=self.device)
        return torch.mean(values.float())

    def _collect_episode_stats(self, env_ids):
        landed_mask = self.landing_count[env_ids] > 0
        completed_mask = self.completed_cycles[env_ids] > 0
        natural_mask = ~self.episode_rsi_used[env_ids]
        return {
            "curriculum_stage_index": torch.full(
                (), float(self.curriculum_stage_index), dtype=torch.float, device=self.device
            ),
            "jump_peak_height": torch.mean(self.episode_peak_height[env_ids]),
            "jump_peak_base_height": torch.mean(self.episode_peak_base_height[env_ids]),
            "completed_jump_peak_height": self._mean_or_zero(
                self.last_completed_peak_height[env_ids],
                completed_mask,
            ),
            "completed_jump_peak_base_height": self._mean_or_zero(
                self.last_completed_base_peak_height[env_ids],
                completed_mask,
            ),
            "jump_flight_rate": torch.mean((self.jump_count[env_ids] > 0).float()),
            "jump_landing_rate": torch.mean((self.landing_count[env_ids] > 0).float()),
            "jump_count": torch.mean(self.jump_count[env_ids].float()),
            "jump_completed_cycles": torch.mean(self.completed_cycles[env_ids].float()),
            "rsi_used_rate": torch.mean(self.episode_rsi_used[env_ids].float()),
            "natural_flight_rate": self._mean_or_zero((self.jump_count[env_ids] > 0).float(), natural_mask),
            "natural_completed_cycles": self._mean_or_zero(self.completed_cycles[env_ids].float(), natural_mask),
            "takeoff_height": torch.mean(self.takeoff_height[env_ids]),
            "jump_reference_height": torch.mean(self.jump_reference_height[env_ids]),
            "target_jump_height": torch.full(
                (), float(self._get_target_jump_height()), dtype=torch.float, device=self.device
            ),
            "target_takeoff_velocity": torch.full(
                (), float(self._get_target_takeoff_velocity()), dtype=torch.float, device=self.device
            ),
            "target_takeoff_forward_velocity": torch.mean(self._get_target_takeoff_forward_velocity()),
            "prepare_stand_ready_rate": torch.mean(self.prepare_stand_ready[env_ids].float()),
            "target_landing_dx": torch.mean(self.commands[env_ids, 1]),
            "target_landing_dy": torch.mean(self.commands[env_ids, 2]),
            "target_landing_yaw": torch.mean(self.commands[env_ids, 3]),
            "target_landing_distance": torch.mean(torch.norm(self.commands[env_ids, 1:3], dim=1)),
            "target_launch_vel_x": torch.mean(self.commands[env_ids, 1]),
            "target_launch_vel_y": torch.mean(self.commands[env_ids, 2]),
            "target_launch_speed": torch.mean(torch.norm(self.commands[env_ids, 1:3], dim=1)),
            "target_lin_vel_x": torch.mean(self.commands[env_ids, 1]),
            "target_lin_vel_y": torch.mean(self.commands[env_ids, 2]),
            "target_yaw_rate": torch.mean(self.commands[env_ids, 3]),
            "landing_dx_curriculum_progress": self.landing_dx_curriculum_progress[0],
            "landing_dx_success_ema": self.landing_dx_success_ema[0],
            "landing_dx_success_rate": self.last_landing_dx_success_rate[0],
            "clearance_gate": torch.mean(self._get_clearance_gate()[env_ids]),
            "completed_clearance_gate": self._mean_or_zero(
                self.last_completed_clearance_gate[env_ids],
                completed_mask,
            ),
            "valid_jump_rate": torch.mean((self._get_valid_jump_gate()[env_ids] >= 1.0).float()),
            "completed_valid_jump_rate": self._mean_or_zero(
                self.last_completed_valid_jump[env_ids].float(),
                completed_mask,
            ),
            "completed_airborne_steps": self._mean_or_zero(
                self.last_completed_airborne_steps[env_ids].float(),
                completed_mask,
            ),
            "estimated_landing_position_error": torch.mean(self.last_estimated_landing_position_error[env_ids]),
            "estimated_landing_forward_error": torch.mean(self.last_estimated_landing_forward_error[env_ids]),
            "estimated_landing_lateral_error": torch.mean(self.last_estimated_landing_lateral_error[env_ids]),
            "landing_position_error": self._mean_or_zero(self.last_landing_position_error[env_ids], landed_mask),
            "landing_forward_error": self._mean_or_zero(self.last_landing_forward_error[env_ids], landed_mask),
            "landing_lateral_error": self._mean_or_zero(self.last_landing_lateral_error[env_ids], landed_mask),
            "landing_yaw_error": self._mean_or_zero(self.last_landing_yaw_error[env_ids], landed_mask),
            "stable_landing_position_error": self._mean_or_zero(
                self.last_completed_landing_position_error[env_ids],
                completed_mask,
            ),
            "stable_landing_forward_error": self._mean_or_zero(
                self.last_completed_landing_forward_error[env_ids],
                completed_mask,
            ),
            "stable_landing_lateral_error": self._mean_or_zero(
                self.last_completed_landing_lateral_error[env_ids],
                completed_mask,
            ),
            "stable_landing_yaw_error": self._mean_or_zero(
                self.last_completed_landing_yaw_error[env_ids],
                completed_mask,
            ),
            "term_dof_limit_rate": torch.mean(self.term_dof_limit_buf[env_ids].float()),
            "term_nonfoot_contact_rate": torch.mean(self.term_nonfoot_contact_buf[env_ids].float()),
            "term_rebound_rate": torch.mean(self.term_rebound_buf[env_ids].float()),
            "term_takeoff_abort_rate": torch.mean(self.term_takeoff_abort_buf[env_ids].float()),
            "term_upside_down_rate": torch.mean(self.term_upside_down_buf[env_ids].float()),
            "term_timeout_rate": torch.mean(self.term_timeout_buf[env_ids].float()),
            "pd_prior_scale": torch.full((), float(self.current_pd_prior_scale), dtype=torch.float, device=self.device),
        }

    def _get_jump_contact_states(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        num_contacts = torch.sum(contact.int(), dim=1)
        flight_contact_mask = num_contacts <= self.cfg.jump_phase.flight_max_contacts
        landing_contact_mask = num_contacts >= self.cfg.jump_phase.landing_min_contacts
        return contact, num_contacts, flight_contact_mask, landing_contact_mask

    def _get_foot_contact_obs(self):
        contact, _, _, _ = self._get_jump_contact_states()
        return contact.float()

    def _get_nonfoot_contact_mask(self):
        if not hasattr(self, "nonfoot_contact_indices") or len(self.nonfoot_contact_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.any(
            torch.norm(self.contact_forces[:, self.nonfoot_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )

    def _get_critical_nonfoot_contact_mask(self):
        if len(self.termination_contact_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )

    def _get_base_yaw(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        return torch.atan2(forward[:, 1], forward[:, 0])

    def _lock_landing_target(self, env_ids):
        if len(env_ids) == 0:
            return
        if not getattr(self.cfg.commands, "use_landing_target", False):
            commanded_xy_body = self.commands[env_ids, 1:3]
            yaw = self.jump_start_yaw[env_ids]
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            commanded_xy_world = torch.stack(
                (
                    cos_yaw * commanded_xy_body[:, 0] - sin_yaw * commanded_xy_body[:, 1],
                    sin_yaw * commanded_xy_body[:, 0] + cos_yaw * commanded_xy_body[:, 1],
                ),
                dim=1,
            )
            pseudo_displacement = commanded_xy_world * float(self.cfg.rewards.flight_time_ref)
            self.target_landing_xy[env_ids] = self.jump_start_xy[env_ids] + pseudo_displacement
            self.target_landing_yaw[env_ids] = self.jump_start_yaw[env_ids]
            return
        dx = self.commands[env_ids, 1]
        dy = self.commands[env_ids, 2]
        dyaw = self.commands[env_ids, 3]
        yaw = self.jump_start_yaw[env_ids]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        target_dx_world = cos_yaw * dx - sin_yaw * dy
        target_dy_world = sin_yaw * dx + cos_yaw * dy
        self.target_landing_xy[env_ids, 0] = self.jump_start_xy[env_ids, 0] + target_dx_world
        self.target_landing_xy[env_ids, 1] = self.jump_start_xy[env_ids, 1] + target_dy_world
        self.target_landing_yaw[env_ids] = wrap_to_pi(self.jump_start_yaw[env_ids] + dyaw)

    def _get_target_velocity_body(self):
        if getattr(self.cfg.commands, "use_landing_target", False):
            v_xy_des = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
            yaw_rate_des = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            return v_xy_des, yaw_rate_des
        v_xy_des = self.commands[:, 1:3]
        yaw_rate_des = self.commands[:, 3]
        return v_xy_des, yaw_rate_des

    def _get_landing_target_progress(self):
        target_delta = self.target_landing_xy - self.jump_start_xy
        current_delta = self.root_states[:, :2] - self.jump_start_xy
        target_dist = torch.clamp(torch.norm(target_delta, dim=1), min=1e-3)
        target_dir = target_delta / target_dist.unsqueeze(1)

        forward_distance = torch.sum(current_delta * target_dir, dim=1)
        forward_progress = torch.clamp(forward_distance / target_dist, min=0.0, max=1.0)
        lateral_vec = current_delta - forward_distance.unsqueeze(1) * target_dir
        lateral_error = torch.norm(lateral_vec, dim=1)
        position_error = torch.norm(self.root_states[:, :2] - self.target_landing_xy, dim=1)
        return forward_progress, lateral_error, position_error

    def _get_projected_landing_error(self):
        target_z = self.jump_reference_height
        z = self.root_states[:, 2]
        vz = self.root_states[:, 9]
        gravity = abs(float(self.cfg.sim.gravity[2]))
        gravity = max(gravity, 1e-3)

        dz = z - target_z
        discriminant = torch.clamp(torch.square(vz) + 2.0 * gravity * dz, min=0.0)
        time_to_landing = (vz + torch.sqrt(discriminant)) / gravity
        time_to_landing = torch.clamp(
            time_to_landing,
            min=float(self.cfg.rewards.estimated_landing_time_min),
            max=float(self.cfg.rewards.estimated_landing_time_max),
        )

        projected_xy = self.root_states[:, :2] + self.root_states[:, 7:9] * time_to_landing.unsqueeze(1)
        error_xy = projected_xy - self.target_landing_xy
        forward_error, lateral_error = self._target_frame_xy_error_from_delta(error_xy, self.jump_start_yaw)
        position_error = torch.norm(error_xy, dim=1)
        return forward_error, lateral_error, position_error

    def _get_landing_dx_range(self):
        stage_range = self.curriculum_stage_cfg.get("landing_dx_range", None) if self.curriculum_enabled else None
        if stage_range is not None:
            return float(stage_range[0]), float(stage_range[1])

        ranges = self.cfg.commands.ranges
        final_min = float(ranges.lin_vel_x[0])
        final_max = float(ranges.lin_vel_x[1])
        if not getattr(self.cfg.commands, "landing_dx_curriculum", False):
            return final_min, final_max

        start_range = getattr(self.cfg.commands, "landing_dx_start", [final_min, final_max])
        progress = self._get_landing_dx_curriculum_progress()
        dx_min = float(start_range[0]) + progress * (final_min - float(start_range[0]))
        dx_max = float(start_range[1]) + progress * (final_max - float(start_range[1]))
        return dx_min, dx_max

    def _update_landing_dx_curriculum(self, env_ids):
        if len(env_ids) == 0:
            return
        if not getattr(self.cfg.commands, "landing_dx_curriculum", False):
            return
        mode = getattr(self.cfg.commands, "landing_dx_curriculum_mode", "step")
        if mode != "success_rate":
            return

        natural_mask = ~self.episode_rsi_used[env_ids]
        if not torch.any(natural_mask):
            return

        natural_env_ids = env_ids[natural_mask]
        position_cutoff = float(
            getattr(
                self.cfg.commands,
                "landing_dx_success_position_cutoff",
                self.cfg.rewards.landing_position_cutoff,
            )
        )
        success = (
            (self.completed_cycles[natural_env_ids] > 0)
            & self.last_completed_valid_jump[natural_env_ids]
            & (self.last_completed_landing_position_error[natural_env_ids] <= position_cutoff)
        )
        success_rate = torch.mean(success.float())
        alpha = min(max(float(getattr(self.cfg.commands, "landing_dx_success_ema_alpha", 0.1)), 0.0), 1.0)
        self.landing_dx_success_ema[0] = (
            (1.0 - alpha) * self.landing_dx_success_ema[0]
            + alpha * success_rate
        )
        self.last_landing_dx_success_rate[0] = success_rate

        success_threshold = float(getattr(self.cfg.commands, "landing_dx_success_threshold", 0.8))
        regress_threshold = float(getattr(self.cfg.commands, "landing_dx_regress_threshold", 0.55))
        if self.landing_dx_success_ema[0] >= success_threshold:
            increment = float(getattr(self.cfg.commands, "landing_dx_progress_increment", 0.02))
            self.landing_dx_curriculum_progress[0] = torch.clamp(
                self.landing_dx_curriculum_progress[0] + increment,
                min=0.0,
                max=1.0,
            )
            self.landing_dx_success_ema[0] = 0.0
        elif self.landing_dx_success_ema[0] < regress_threshold:
            decrement = float(getattr(self.cfg.commands, "landing_dx_progress_decrement", 0.0))
            if decrement > 0.0:
                self.landing_dx_curriculum_progress[0] = torch.clamp(
                    self.landing_dx_curriculum_progress[0] - decrement,
                    min=0.0,
                    max=1.0,
                )

    def _get_landing_dx_curriculum_progress(self):
        if getattr(self.cfg.commands, "landing_dx_curriculum_mode", "step") == "success_rate":
            return float(self.landing_dx_curriculum_progress[0].item())
        warmup_steps = float(getattr(self.cfg.commands, "landing_dx_warmup_steps", 0.0))
        ramp_steps = max(float(getattr(self.cfg.commands, "landing_dx_ramp_steps", 1.0)), 1.0)
        return min(max((float(self.step_count) - warmup_steps) / ramp_steps, 0.0), 1.0)

    def _get_jump_phase_masks(self):
        prepare_mask = self.jump_phase == self.PHASE_PREPARE
        compression_mask = self.jump_phase == self.PHASE_COMPRESSION
        extension_mask = self.jump_phase == self.PHASE_EXTENSION
        release_mask = self.jump_phase == self.PHASE_RELEASE
        flight_mask = self.jump_phase == self.PHASE_FLIGHT
        touchdown_mask = self.jump_phase == self.PHASE_TOUCHDOWN
        stabilize_mask = self.jump_phase == self.PHASE_STABILIZE
        return (
            prepare_mask,
            compression_mask,
            extension_mask,
            release_mask,
            flight_mask,
            touchdown_mask,
            stabilize_mask,
        )

    def _get_target_jump_height(self):
        target = float(self.cfg.rewards.fixed_jump_height)
        if not getattr(self.cfg.rewards, "jump_height_curriculum", False):
            return target

        progress = self._get_jump_curriculum_progress()
        start = float(getattr(self.cfg.rewards, "jump_height_start", target))
        return start + progress * (target - start)

    def _get_target_takeoff_velocity(self):
        target = float(self.cfg.rewards.takeoff_velocity_target)
        if not getattr(self.cfg.rewards, "takeoff_velocity_curriculum", False):
            return target

        progress = self._get_takeoff_velocity_curriculum_progress()
        start = float(getattr(self.cfg.rewards, "takeoff_velocity_start", target))
        return start + progress * (target - start)

    def _get_target_takeoff_forward_velocity(self):
        if getattr(self.cfg.commands, "use_landing_target", False):
            target_delta = self.target_landing_xy - self.jump_start_xy
            target_dist = torch.norm(target_delta, dim=1)
            target_speed = target_dist / max(float(self.cfg.rewards.flight_time_ref), 1e-3)
        else:
            target_speed = torch.norm(self.commands[:, 1:3], dim=1)
        min_vel = float(getattr(self.cfg.rewards, "takeoff_forward_velocity_min", 0.0))
        max_vel = max(float(getattr(self.cfg.rewards, "takeoff_forward_velocity_max", 1.0)), min_vel + 1e-3)
        return torch.clamp(target_speed, min=min_vel, max=max_vel)

    def _get_takeoff_velocity_curriculum_progress(self):
        warmup_steps = float(getattr(self.cfg.rewards, "takeoff_velocity_warmup_steps", 0.0))
        ramp_steps = max(float(getattr(self.cfg.rewards, "takeoff_velocity_ramp_steps", 1.0)), 1.0)
        return min(max((float(self.step_count) - warmup_steps) / ramp_steps, 0.0), 1.0)

    def _get_jump_curriculum_progress(self):
        warmup_steps = float(getattr(self.cfg.rewards, "jump_height_warmup_steps", 0.0))
        ramp_steps = max(float(getattr(self.cfg.rewards, "jump_height_ramp_steps", 1.0)), 1.0)
        return min(max((float(self.step_count) - warmup_steps) / ramp_steps, 0.0), 1.0)

    def _get_peak_height_gate(self, floor, target):
        peak_rel_height = torch.clamp(self.cycle_peak_height - self.jump_reference_height, min=0.0)
        floor = float(floor)
        target = max(float(target), floor + 1e-3)
        return torch.clamp(
            (peak_rel_height - floor) / (target - floor),
            min=0.0,
            max=1.0,
        )

    def _get_completed_height_gate(self, floor, target):
        floor = float(floor)
        target = max(float(target), floor + 1e-3)
        return torch.clamp(
            (self.last_completed_peak_height - floor) / (target - floor),
            min=0.0,
            max=1.0,
        )

    def _get_peak_base_height_gate(self, floor, target):
        floor = float(floor)
        target = max(float(target), floor + 1e-3)
        return torch.clamp(
            (self.cycle_peak_height - floor) / (target - floor),
            min=0.0,
            max=1.0,
        )

    def _get_current_base_height_gate(self, floor, target):
        floor = float(floor)
        target = max(float(target), floor + 1e-3)
        return torch.clamp(
            (self.root_states[:, 2] - floor) / (target - floor),
            min=0.0,
            max=1.0,
        )

    def _get_completed_base_height_gate(self, floor, target):
        floor = float(floor)
        target = max(float(target), floor + 1e-3)
        return torch.clamp(
            (self.last_completed_base_peak_height - floor) / (target - floor),
            min=0.0,
            max=1.0,
        )

    def _get_clearance_gate(self):
        return self._get_peak_height_gate(
            self.cfg.rewards.clearance_gate_floor,
            self.cfg.rewards.clearance_gate_target,
        )

    def _get_completed_clearance_gate(self):
        floor = float(self.cfg.rewards.clearance_gate_floor)
        target = max(float(self.cfg.rewards.clearance_gate_target), floor + 1e-3)
        return torch.clamp(
            (self.last_completed_peak_height - floor) / (target - floor),
            min=0.0,
            max=1.0,
        )

    def _get_valid_jump_gate(self):
        min_airborne = int(getattr(self.cfg.rewards, "valid_jump_min_airborne_steps", 1))
        min_clearance = float(getattr(self.cfg.rewards, "success_clearance_min", 0.0))
        peak_clearance = torch.clamp(self.cycle_peak_height - self.jump_reference_height, min=0.0)
        clearance_gate = self._get_clearance_gate()
        enough_air = self.cycle_max_airborne_steps >= min_airborne
        enough_clearance = peak_clearance >= min_clearance
        return clearance_gate * enough_air.float() * enough_clearance.float()

    def _get_completed_valid_jump_gate(self):
        min_airborne = int(getattr(self.cfg.rewards, "valid_jump_min_airborne_steps", 1))
        min_clearance = float(getattr(self.cfg.rewards, "success_clearance_min", 0.0))
        clearance_gate = self._get_completed_clearance_gate()
        enough_air = self.last_completed_airborne_steps >= min_airborne
        enough_clearance = self.last_completed_peak_height >= min_clearance
        return clearance_gate * enough_air.float() * enough_clearance.float()

    def _get_landing_height_gate(self):
        if not getattr(self.cfg.rewards, "use_height_gates", True):
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        return self._get_clearance_gate()

    def _get_landing_aux_height_gate(self):
        if not getattr(self.cfg.rewards, "use_height_gates", True):
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        min_gate = min(max(float(self.cfg.rewards.landing_aux_height_gate_min), 0.0), 1.0)
        return min_gate + (1.0 - min_gate) * self._get_landing_height_gate()

    def _reward_height_tracking(self):
        peak_rel_height = torch.clamp(self.cycle_peak_height - self.jump_reference_height, min=0.0)
        target = max(self._get_target_jump_height(), 1e-3)
        _, _, _, _, _, _, stabilize_mask = self._get_jump_phase_masks()
        _, _, timed_flight_mask, timed_landing_mask = self._get_timed_phase_masks()
        reward = torch.clamp(peak_rel_height / target, min=0.0, max=1.0)
        reward *= self._get_peak_base_height_gate(
            self.cfg.rewards.base_height_reward_floor,
            self.cfg.rewards.base_height_reward_target,
        )
        phase_weight = (timed_flight_mask | timed_landing_mask | stabilize_mask).float()
        return reward * phase_weight * self.jump_toggle_state.float()

    def _reward_jumping_success(self):
        target = max(self._get_target_jump_height(), 1e-3)
        tolerance = max(float(self.cfg.rewards.success_height_tolerance), 1e-3)
        height_quality = torch.clamp(self.last_completed_peak_height / target, min=0.0, max=1.0)
        height_in_band = self.last_completed_peak_height >= (target - tolerance)
        base_height_quality = self._get_completed_base_height_gate(
            self.cfg.rewards.success_base_height_floor,
            self.cfg.rewards.success_base_height_target,
        )
        nonfoot_contact = self._get_nonfoot_contact_mask()
        return (
            self.just_completed_cycle.float()
            * height_in_band.float()
            * height_quality
            * base_height_quality
            * (~nonfoot_contact).float()
        )

    def _reward_valid_jump_success(self):
        # Sparse completion reward for a real jump cycle that achieved enough
        # clearance and airborne duration, without non-foot contact failures.
        if not torch.any(self.just_completed_cycle):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        nonfoot_contact = self._get_nonfoot_contact_mask()
        return (
            self.just_completed_cycle.float()
            * self._get_completed_valid_jump_gate()
            * (~nonfoot_contact).float()
        )

    def _reward_phase_contact_sync(self):
        contact, num_contacts, flight_contact_mask, _ = self._get_jump_contact_states()
        _, compression_mask, extension_mask, release_mask, flight_mask, touchdown_mask, stabilize_mask = self._get_jump_phase_masks()
        takeoff_support_mask = compression_mask | extension_mask

        contact_ratio = num_contacts.float() / float(len(self.feet_indices))
        front_ratio = torch.mean(contact[:, :2].float(), dim=1)
        rear_ratio = torch.mean(contact[:, 2:].float(), dim=1)
        balanced_contact = contact_ratio * torch.minimum(front_ratio, rear_ratio)
        takeoff_release_mask = (release_mask & flight_contact_mask) | self.just_took_off
        if getattr(self.cfg.rewards, "use_height_gates", True):
            height_gate = self._get_peak_height_gate(
                self.cfg.rewards.phase_sync_height_gate_floor,
                self.cfg.rewards.phase_sync_height_gate_target,
            )
            landing_height_gate = 0.2 + 0.8 * height_gate
        else:
            height_gate = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
            landing_height_gate = torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        reward = torch.where(takeoff_support_mask, 0.2 * balanced_contact, reward)
        reward = torch.where(takeoff_release_mask, flight_contact_mask.float(), reward)
        reward = torch.where(flight_mask, flight_contact_mask.float() * height_gate, reward)
        reward = torch.where(touchdown_mask | stabilize_mask, 0.2 * balanced_contact * landing_height_gate, reward)
        reward = torch.where(self.just_took_off, flight_contact_mask.float(), reward)
        return reward * self.jump_toggle_state.float()

    def _reward_landing_target_progress(self):
        _, _, _, _, flight_mask, touchdown_mask, stabilize_mask = self._get_jump_phase_masks()
        nonfoot_contact = self._get_nonfoot_contact_mask()
        forward_progress, lateral_error, _ = self._get_landing_target_progress()
        lateral_sigma = max(float(self.cfg.rewards.landing_progress_lateral_sigma), 1e-3)
        lateral_quality = torch.exp(-torch.square(lateral_error / lateral_sigma))
        phase_mask = flight_mask | touchdown_mask | stabilize_mask
        height_gate = self._get_landing_aux_height_gate()
        return (
            forward_progress
            * lateral_quality
            * phase_mask.float()
            * (~nonfoot_contact).float()
            * height_gate
            * self.jump_toggle_state.float()
        )

    def _reward_estimated_landing_target_tracking(self):
        _, _, _, _, flight_mask, _, _ = self._get_jump_phase_masks()
        nonfoot_contact = self._get_nonfoot_contact_mask()
        forward_error, lateral_error, position_error = self._get_projected_landing_error()
        self.last_estimated_landing_forward_error = forward_error.detach()
        self.last_estimated_landing_lateral_error = lateral_error.detach()
        self.last_estimated_landing_position_error = position_error.detach()

        forward_sigma = max(float(self.cfg.rewards.estimated_landing_sigma), 1e-3)
        lateral_sigma = max(float(self.cfg.rewards.estimated_landing_lateral_sigma), 1e-3)
        reward = (
            torch.exp(-torch.square(forward_error / forward_sigma))
            * torch.exp(-torch.square(lateral_error / lateral_sigma))
        )
        phase_mask = flight_mask
        return (
            reward
            * phase_mask.float()
            * (~nonfoot_contact).float()
            * self._get_landing_aux_height_gate()
            * self.jump_toggle_state.float()
        )

    def _reward_long_jump_success(self):
        nonfoot_contact = self._get_nonfoot_contact_mask()
        pos_error = self.last_completed_landing_position_error
        pos_sigma = max(float(self.cfg.rewards.landing_position_sigma), 1e-3)
        pos_cutoff = float(self.cfg.rewards.landing_position_cutoff)
        tilt_sigma = max(float(self.cfg.rewards.landing_success_tilt_sigma), 1e-3)
        yaw_sigma = max(float(self.cfg.rewards.landing_yaw_sigma), 1e-3)
        yaw_cutoff = float(self.cfg.rewards.landing_yaw_cutoff)
        tilt_cutoff = float(self.cfg.rewards.landing_tilt_cutoff)
        position_quality = torch.exp(-torch.square(pos_error / pos_sigma))
        tilt_quality = torch.exp(-torch.square(self.last_completed_landing_tilt / tilt_sigma))
        yaw_quality = torch.exp(-torch.square(self.last_completed_landing_yaw_error / yaw_sigma))
        valid_pose = (
            (pos_error <= pos_cutoff)
            & (self.last_completed_landing_tilt <= tilt_cutoff)
            & (self.last_completed_landing_yaw_error <= yaw_cutoff)
        )
        return (
            position_quality
            * tilt_quality
            * yaw_quality
            * valid_pose.float()
            * self._get_completed_valid_jump_gate()
            * self.just_completed_cycle.float()
            * (~nonfoot_contact).float()
        )

    def _get_takeoff_support_quality(self):
        contact, num_contacts, flight_contact_mask, _ = self._get_jump_contact_states()
        contact_ratio = num_contacts.float() / float(len(self.feet_indices))
        front_ratio = torch.mean(contact[:, :2].float(), dim=1)
        rear_ratio = torch.mean(contact[:, 2:].float(), dim=1)
        support_quality = contact_ratio * torch.minimum(front_ratio, rear_ratio)

        # During true flight or the exact takeoff transition, do not suppress
        # reward. Before that point, require balanced front/rear support so the
        # policy cannot farm reward by lifting only the front or rear pair.
        launch_free_mask = flight_contact_mask | self.just_took_off
        launch_gate = torch.where(launch_free_mask, torch.ones_like(support_quality), support_quality)
        return support_quality, launch_gate

    def _reward_landing_success(self):
        return self._reward_long_jump_success()

    def _reward_takeoff_height_progress(self):
        _, _, extension_mask, release_mask, _, _, _ = self._get_jump_phase_masks()
        current_rel_height = torch.clamp(self.root_states[:, 2] - self.jump_reference_height, min=0.0)
        target = max(min(float(self.cfg.rewards.takeoff_height_progress_target), self._get_target_jump_height()), 1e-3)
        rel_reward = torch.clamp(current_rel_height / target, min=0.0, max=1.0)
        base_gate = self._get_current_base_height_gate(
            self.cfg.rewards.base_height_reward_floor,
            self.cfg.rewards.base_height_reward_target,
        )
        _, launch_gate = self._get_takeoff_support_quality()
        reward = rel_reward * base_gate
        launch_mask = extension_mask | release_mask | self.just_took_off
        return reward * launch_gate * launch_mask.float() * self.jump_toggle_state.float()

    def _reward_takeoff_squat(self):
        contact, num_contacts, _, _ = self._get_jump_contact_states()
        _, compression_mask, _, _, _, _, _ = self._get_jump_phase_masks()
        support_contact_mask = num_contacts >= self.cfg.jump_phase.landing_min_contacts
        # Only activate during COMPRESSION — not PREPARE.
        # This prevents the robot from sitting in a crouch in PREPARE to farm reward.
        # The robot must first trigger prepare_stand_ready to enter COMPRESSION,
        # then it is rewarded for crouching deep before the push-off.
        active_mask = compression_mask & support_contact_mask

        reference_height = self.jump_reference_height
        squat_depth = torch.clamp(reference_height - self.root_states[:, 2], min=0.0)
        floor = float(self.cfg.rewards.takeoff_squat_floor)
        target = max(float(self.cfg.rewards.takeoff_squat_target), floor + 1e-3)
        height_reward = torch.clamp((squat_depth - floor) / (target - floor), min=0.0, max=1.0)

        contact_ratio = num_contacts.float() / float(len(self.feet_indices))
        front_ratio = torch.mean(contact[:, :2].float(), dim=1)
        rear_ratio = torch.mean(contact[:, 2:].float(), dim=1)
        balanced_contact = contact_ratio * torch.minimum(front_ratio, rear_ratio)

        xy_speed = torch.norm(self.root_states[:, 7:9], dim=1)
        max_xy_speed = max(float(self.cfg.rewards.takeoff_squat_max_xy_speed), 1e-3)
        quiet_xy = torch.clamp(1.0 - xy_speed / max_xy_speed, min=0.0, max=1.0)
        tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
        tilt_sigma = max(float(self.cfg.rewards.takeoff_squat_tilt_sigma), 1e-3)
        tilt_reward = torch.exp(-torch.square(tilt / tilt_sigma))

        return (
            height_reward
            * balanced_contact
            * quiet_xy
            * tilt_reward
            * active_mask.float()
            * self.jump_toggle_state.float()
        )

    def _reward_takeoff_impulse(self):
        """Reward upward impulse (positive vz delta) during takeoff phases.

        Unlike takeoff_squat which rewards crouching depth, this reward
        activates the moment the robot starts pushing upward, regardless
        of state machine phase depth. Sitting still gives zero reward
        because vz does not change. Only actively pushing off the ground
        and generating upward acceleration earns reward.
        """
        _, compression_mask, extension_mask, release_mask, _, _, _ = self._get_jump_phase_masks()
        active_mask = compression_mask | extension_mask | release_mask
        vz_now = self.root_states[:, 9]
        vz_prev = self.last_root_vel[:, 2]
        vz_delta = torch.clamp(vz_now - vz_prev, min=0.0)
        _, launch_gate = self._get_takeoff_support_quality()
        return vz_delta * launch_gate * active_mask.float() * self.jump_toggle_state.float()

    def _reward_takeoff_velocity(self):
        _, _, extension_mask, release_mask, _, _, _ = self._get_jump_phase_masks()
        contact, num_contacts, flight_contact_mask, _ = self._get_jump_contact_states()

        contact_ratio = num_contacts.float() / float(len(self.feet_indices))
        front_ratio = torch.mean(contact[:, :2].float(), dim=1)
        rear_ratio = torch.mean(contact[:, 2:].float(), dim=1)
        support_quality = contact_ratio * torch.minimum(front_ratio, rear_ratio)
        support_floor = min(max(float(self.cfg.rewards.takeoff_velocity_support_min), 0.0), 1.0)
        support_quality = support_floor + (1.0 - support_floor) * support_quality
        support_quality = torch.where(self.just_took_off, torch.ones_like(support_quality), support_quality)

        floor = float(self.cfg.rewards.takeoff_velocity_floor)
        target = max(self._get_target_takeoff_velocity(), floor + 1e-3)
        vz = torch.clamp(self.root_states[:, 9], min=0.0)
        reward = torch.clamp((vz - floor) / (target - floor), min=0.0, max=1.0)
        release_phase = flight_contact_mask | self.just_took_off
        release_quality = 1.0 - contact_ratio
        # Give 25% reward floor during EXTENSION so the policy gets gradient
        # even when feet are still on the ground. Full reward when airborne.
        min_gate = 0.25
        contact_gate = min_gate + (1.0 - min_gate) * release_quality
        contact_gate = torch.where(release_phase, release_quality, contact_gate)
        launch_mask = extension_mask | release_mask | self.just_took_off
        return reward * support_quality * contact_gate * launch_mask.float() * self.jump_toggle_state.float()

    def _reward_takeoff_release(self):
        _, _, _, release_mask, _, _, _ = self._get_jump_phase_masks()
        _, num_contacts, flight_contact_mask, _ = self._get_jump_contact_states()
        nonfoot_contact = self._get_nonfoot_contact_mask()

        contact_ratio = num_contacts.float() / float(len(self.feet_indices))
        release_quality = 1.0 - contact_ratio

        velocity_floor = float(self.cfg.rewards.takeoff_release_velocity_floor)
        velocity_target = max(float(self.cfg.rewards.takeoff_release_velocity_target), velocity_floor + 1e-3)
        vz = torch.clamp(self.root_states[:, 9], min=0.0)
        velocity_quality = torch.clamp((vz - velocity_floor) / (velocity_target - velocity_floor), min=0.0, max=1.0)

        clearance_floor = float(self.cfg.rewards.takeoff_release_clearance_floor)
        clearance_target = max(float(self.cfg.rewards.takeoff_release_clearance_target), clearance_floor + 1e-3)
        current_clearance = torch.clamp(self.root_states[:, 2] - self.jump_reference_height, min=0.0)
        clearance_quality = torch.clamp(
            (current_clearance - clearance_floor) / (clearance_target - clearance_floor),
            min=0.0,
            max=1.0,
        )
        clearance_min_gate = min(
            max(float(getattr(self.cfg.rewards, "takeoff_release_clearance_min_gate", 0.0)), 0.0),
            1.0,
        )
        clearance_quality = clearance_min_gate + (1.0 - clearance_min_gate) * clearance_quality

        release_event_mask = self.just_took_off | (release_mask & flight_contact_mask)
        return (
            velocity_quality
            * release_quality
            * clearance_quality
            * release_event_mask.float()
            * (~nonfoot_contact).float()
            * self.jump_toggle_state.float()
        )

    def _reward_ground_creeping(self):
        prepare_mask, compression_mask, extension_mask, release_mask, _, _, _ = self._get_jump_phase_masks()
        _, num_contacts, _, _ = self._get_jump_contact_states()
        active_mask = prepare_mask | compression_mask | extension_mask | release_mask
        contact_ratio = num_contacts.float() / float(len(self.feet_indices))
        displacement = torch.norm(self.root_states[:, :2] - self.jump_start_xy, dim=1)
        allowed_distance = float(self.cfg.rewards.ground_creeping_distance)
        creep = torch.clamp(displacement - allowed_distance, min=0.0)
        no_flight_yet = self.jump_count == 0
        return creep * contact_ratio * active_mask.float() * no_flight_yet.float() * self.jump_toggle_state.float()

    def _reward_takeoff_forward_push(self):
        contact, num_contacts, flight_contact_mask, _ = self._get_jump_contact_states()
        _, _, extension_mask, _, _, _, _ = self._get_jump_phase_masks()

        support_contact_mask = extension_mask & (~flight_contact_mask)
        if not torch.any(support_contact_mask):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        contact_ratio = num_contacts.float() / float(len(self.feet_indices))
        front_ratio = torch.mean(contact[:, :2].float(), dim=1)
        rear_ratio = torch.mean(contact[:, 2:].float(), dim=1)
        support_quality = contact_ratio * torch.minimum(front_ratio, rear_ratio)

        target_delta = self.target_landing_xy - self.jump_start_xy
        target_dist = torch.clamp(torch.norm(target_delta, dim=1), min=1e-3)
        target_dir = target_delta / target_dist.unsqueeze(1)

        world_xy_vel = self.root_states[:, 7:9]
        forward_speed = torch.sum(world_xy_vel * target_dir, dim=1)
        lateral_vel = world_xy_vel - forward_speed.unsqueeze(1) * target_dir
        lateral_speed = torch.norm(lateral_vel, dim=1)

        prev_world_xy_vel = self.last_root_vel[:, :2]
        prev_forward_speed = torch.sum(prev_world_xy_vel * target_dir, dim=1)
        forward_speed_gain = torch.clamp(forward_speed - prev_forward_speed, min=0.0)

        floor = float(getattr(self.cfg.rewards, "takeoff_forward_velocity_floor", 0.0))
        target_speed = torch.clamp(self._get_target_takeoff_forward_velocity(), min=floor + 1e-3)
        speed_reward = torch.clamp((forward_speed - floor) / (target_speed - floor), min=0.0, max=1.0)
        gain_reward = torch.clamp(forward_speed_gain / target_speed, min=0.0, max=1.0)

        lateral_sigma = max(float(self.cfg.rewards.takeoff_forward_velocity_lateral_sigma), 1e-3)
        lateral_quality = torch.exp(-torch.square(lateral_speed / lateral_sigma))

        vertical_floor = float(getattr(self.cfg.rewards, "takeoff_forward_velocity_vertical_gate_floor", 0.0))
        vertical_target = max(
            float(getattr(self.cfg.rewards, "takeoff_forward_velocity_vertical_gate_target", 0.2)),
            vertical_floor + 1e-3,
        )
        world_vertical_velocity = self.root_states[:, 9]
        vertical_gate = torch.clamp(
            (world_vertical_velocity - vertical_floor) / (vertical_target - vertical_floor),
            min=0.0,
            max=1.0,
        )
        min_gate = min(
            max(float(getattr(self.cfg.rewards, "takeoff_forward_velocity_vertical_gate_min", 0.0)), 0.0),
            1.0,
        )
        vertical_gate = min_gate + (1.0 - min_gate) * vertical_gate

        push_reward = 0.5 * speed_reward + 0.5 * gain_reward
        return (
            push_reward
            * support_quality
            * lateral_quality
            * vertical_gate
            * support_contact_mask.float()
            * self.jump_toggle_state.float()
        )

    def _reward_takeoff_forward_velocity(self):
        _, _, extension_mask, release_mask, _, _, _ = self._get_jump_phase_masks()
        target_delta = self.target_landing_xy - self.jump_start_xy
        target_dist = torch.clamp(torch.norm(target_delta, dim=1), min=1e-3)
        target_dir = target_delta / target_dist.unsqueeze(1)

        world_xy_vel = self.root_states[:, 7:9]
        forward_speed = torch.sum(world_xy_vel * target_dir, dim=1)
        lateral_vel = world_xy_vel - forward_speed.unsqueeze(1) * target_dir
        lateral_speed = torch.norm(lateral_vel, dim=1)

        floor = float(getattr(self.cfg.rewards, "takeoff_forward_velocity_floor", 0.0))
        target_speed = torch.clamp(self._get_target_takeoff_forward_velocity(), min=floor + 1e-3)
        forward_reward = torch.clamp((forward_speed - floor) / (target_speed - floor), min=0.0, max=1.0)
        lateral_sigma = max(float(self.cfg.rewards.takeoff_forward_velocity_lateral_sigma), 1e-3)
        lateral_quality = torch.exp(-torch.square(lateral_speed / lateral_sigma))
        vertical_floor = float(getattr(self.cfg.rewards, "takeoff_forward_velocity_vertical_gate_floor", 0.0))
        vertical_target = max(
            float(getattr(self.cfg.rewards, "takeoff_forward_velocity_vertical_gate_target", 0.2)),
            vertical_floor + 1e-3,
        )
        world_vertical_velocity = self.root_states[:, 9]
        vertical_gate = torch.clamp(
            (world_vertical_velocity - vertical_floor) / (vertical_target - vertical_floor),
            min=0.0,
            max=1.0,
        )
        min_gate = min(
            max(float(getattr(self.cfg.rewards, "takeoff_forward_velocity_vertical_gate_min", 0.0)), 0.0),
            1.0,
        )
        vertical_gate = min_gate + (1.0 - min_gate) * vertical_gate
        _, launch_gate = self._get_takeoff_support_quality()
        launch_mask = release_mask | extension_mask | self.just_took_off
        return (
            forward_reward
            * lateral_quality
            * vertical_gate
            * launch_gate
            * launch_mask.float()
            * self.jump_toggle_state.float()
        )

    def _reward_maximum_height(self):
        peak_rel_height = torch.clamp(self.cycle_peak_height - self.jump_reference_height, min=0.0)
        reference = max(float(self.cfg.rewards.reference_jump_height), 1e-3)
        height_floor = float(self.cfg.rewards.height_reward_floor)
        _, _, _, _, _, _, stabilize_mask = self._get_jump_phase_masks()
        _, _, timed_flight_mask, timed_landing_mask = self._get_timed_phase_masks()
        reward = torch.clamp(
            (peak_rel_height - height_floor) / max(reference - height_floor, 1e-3),
            min=0.0,
            max=1.0,
        )
        reward *= self._get_peak_base_height_gate(
            self.cfg.rewards.base_height_reward_floor,
            self.cfg.rewards.base_height_reward_target,
        )
        phase_weight = (timed_flight_mask | timed_landing_mask | stabilize_mask).float()
        return reward * phase_weight * self.jump_toggle_state.float()

    def _reward_landing_position(self):
        nonfoot_contact = self._get_nonfoot_contact_mask()
        pos_error = self.last_completed_landing_position_error
        sigma = max(float(self.cfg.rewards.landing_position_sigma), 1e-3)
        cutoff = float(self.cfg.rewards.landing_position_cutoff)
        reward = torch.exp(-torch.square(pos_error / sigma))
        completion_gate = self.just_completed_cycle.float() * (~nonfoot_contact).float()
        if getattr(self.cfg.commands, "use_landing_target", False):
            completion_gate = completion_gate * self._get_completed_valid_jump_gate()
        return (
            reward
            * (pos_error <= cutoff).float()
            * completion_gate
        )

    def _reward_landing_orientation(self):
        nonfoot_contact = self._get_nonfoot_contact_mask()
        yaw_error = self.last_completed_landing_yaw_error
        tilt = self.last_completed_landing_tilt
        yaw_sigma = max(float(self.cfg.rewards.landing_yaw_sigma), 1e-3)
        tilt_sigma = max(float(self.cfg.rewards.landing_tilt_sigma), 1e-3)
        yaw_cutoff = float(self.cfg.rewards.landing_yaw_cutoff)
        tilt_cutoff = float(self.cfg.rewards.landing_tilt_cutoff)
        reward = torch.exp(-torch.square(yaw_error / yaw_sigma)) * torch.exp(-torch.square(tilt / tilt_sigma))
        valid_landing_pose = (yaw_error <= yaw_cutoff) & (tilt <= tilt_cutoff)
        completion_gate = self.just_completed_cycle.float() * (~nonfoot_contact).float()
        if getattr(self.cfg.commands, "use_landing_target", False):
            completion_gate = completion_gate * self._get_completed_valid_jump_gate()
        return (
            reward
            * valid_landing_pose.float()
            * completion_gate
        )

    def _reward_tracking_linear_velocity(self):
        (
            _prepare_mask,
            _compression_mask,
            extension_mask,
            release_mask,
            flight_mask,
            _touchdown_mask,
            _stabilize_mask,
        ) = self._get_jump_phase_masks()
        v_xy_des, _ = self._get_target_velocity_body()
        sigma = max(float(self.cfg.rewards.tracking_lin_vel_sigma), 1e-3)
        err = torch.sum(torch.square(self.base_lin_vel[:, :2] - v_xy_des), dim=1)
        reward = torch.exp(-err / (sigma * sigma))
        support_quality, _ = self._get_takeoff_support_quality()
        # Velocity tracking is only a launch/flight objective. After touchdown,
        # landing should be governed by stability-oriented rewards instead.
        phase_weight = (
            0.25 * extension_mask.float() * support_quality
            + 0.6 * release_mask.float() * support_quality
            + flight_mask.float()
        )
        return reward * phase_weight * self.jump_toggle_state.float()

    def _reward_tracking_angular_velocity(self):
        _, compression_mask, extension_mask, release_mask, flight_mask, _touchdown_mask, _stabilize_mask = self._get_jump_phase_masks()
        takeoff_mask = compression_mask | extension_mask | release_mask
        _, yaw_rate_des = self._get_target_velocity_body()
        yaw_sigma = max(float(self.cfg.rewards.tracking_yaw_rate_sigma), 1e-3)
        xy_sigma = max(float(self.cfg.rewards.ang_vel_xy_sigma), 1e-3)
        yaw_reward = torch.exp(-torch.square((self.base_ang_vel[:, 2] - yaw_rate_des) / yaw_sigma))
        xy_reward = torch.exp(-torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) / (xy_sigma * xy_sigma))
        phase_weight = (
            0.5 * takeoff_mask.float()
            + flight_mask.float()
        )
        return yaw_reward * xy_reward * phase_weight * self.jump_toggle_state.float()

    def _reward_feet_clearance(self):
        _, _, _, _, flight_mask, _, _ = self._get_jump_phase_masks()
        _, _, timed_flight_mask, _ = self._get_timed_phase_masks()
        if not hasattr(self, "rigid_body_states"):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        feet_height = torch.clamp(self.rigid_body_states[:, self.feet_indices, 2], min=0.0)
        target = max(float(self.cfg.rewards.feet_clearance_target), 1e-3)
        clearance = torch.clamp(feet_height / target, max=1.0)
        peak_rel_height = torch.clamp(self.cycle_peak_height - self.jump_reference_height, min=0.0)
        floor = float(self.cfg.rewards.feet_clearance_height_gate_floor)
        gate_target = max(float(self.cfg.rewards.feet_clearance_height_gate_target), floor + 1e-3)
        height_gate = torch.clamp(
            (peak_rel_height - floor) / (gate_target - floor),
            min=0.0,
            max=1.0,
        )
        min_gate = min(max(float(self.cfg.rewards.feet_clearance_height_gate_min), 0.0), 1.0)
        height_gate = min_gate + (1.0 - min_gate) * height_gate
        phase_weight = (timed_flight_mask | flight_mask).float()
        return torch.mean(clearance, dim=1) * phase_weight * height_gate * self.jump_toggle_state.float()

    def _reward_stand_height(self):
        contact, num_contacts, _, _ = self._get_jump_contact_states()
        prepare_mask, _, _, _, _, _, _ = self._get_jump_phase_masks()
        all_feet_down = num_contacts == len(self.feet_indices)
        prepare_reward_window = self.phase_steps <= self.cfg.jump_phase.prepare_reward_max_steps
        active_mask = prepare_mask & (~self.prepare_stand_ready) & all_feet_down & prepare_reward_window
        target_height = float(self.cfg.rewards.stand_height_target)
        sigma = max(float(self.cfg.rewards.stand_height_sigma), 1e-3)
        reward = torch.exp(-torch.square((self.root_states[:, 2] - target_height) / sigma))
        return reward * active_mask.float()

    def _reward_default_pose_hold(self):
        _, num_contacts, _, _ = self._get_jump_contact_states()
        prepare_mask, _, _, _, _, _, _ = self._get_jump_phase_masks()
        all_feet_down = num_contacts == len(self.feet_indices)
        prepare_reward_window = self.phase_steps <= self.cfg.jump_phase.prepare_reward_max_steps
        active_mask = prepare_mask & (~self.prepare_stand_ready) & all_feet_down & prepare_reward_window
        sigma = max(float(self.cfg.rewards.default_pose_sigma), 1e-3)
        joint_err = torch.mean(torch.square(self.dof_pos - self.default_joint_pd_target), dim=1)
        reward = torch.exp(-joint_err / (sigma * sigma))
        return reward * active_mask.float()

    def _reward_body_attitude(self):
        prepare_mask, compression_mask, extension_mask, release_mask, flight_mask, touchdown_mask, stabilize_mask = self._get_jump_phase_masks()
        takeoff_mask = compression_mask | extension_mask | release_mask
        takeoff_weight = float(getattr(self.cfg.rewards, "body_attitude_takeoff_weight", 0.2))
        phase_weight = (
            takeoff_weight * takeoff_mask.float()
            + 0.8 * flight_mask.float()
            + 0.6 * touchdown_mask.float()
            + 0.5 * stabilize_mask.float()
        )
        tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
        ang_speed_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        tilt_sigma = max(float(self.cfg.rewards.body_tilt_sigma), 1e-3)
        ang_sigma = max(float(self.cfg.rewards.body_ang_vel_xy_sigma), 1e-3)
        reward = (
            torch.exp(-torch.square(tilt / tilt_sigma))
            * torch.exp(-torch.square(ang_speed_xy / ang_sigma))
        )
        return reward * phase_weight * self.jump_toggle_state.float()

    def _joint_pose_tracking_reward(self, target):
        sigma = max(float(self.cfg.rewards.joint_pose_sigma), 1e-3)
        joint_err = torch.mean(torch.square(self.dof_pos - target), dim=1)
        return torch.exp(-joint_err / (sigma * sigma))

    def _reward_joint_pose_aerial(self):
        _, _, _, _, flight_mask, _, _ = self._get_jump_phase_masks()
        _, _, flight_contact_mask, _ = self._get_jump_contact_states()
        prelanding_mask = self._get_prelanding_mask()
        active_mask = flight_mask & flight_contact_mask & (~prelanding_mask)
        if getattr(self.cfg.rewards, "use_height_gates", True):
            height_gate = 0.1 + 0.9 * self._get_peak_height_gate(
                self.cfg.rewards.phase_sync_height_gate_floor,
                self.cfg.rewards.phase_sync_height_gate_target,
            )
        else:
            height_gate = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        return (
            self._joint_pose_tracking_reward(self.aerial_joint_target)
            * active_mask.float()
            * height_gate
            * self.jump_toggle_state.float()
        )

    def _reward_joint_pose_prelanding(self):
        _, _, _, _, flight_mask, touchdown_mask, _ = self._get_jump_phase_masks()
        prelanding_mask = self._get_prelanding_mask()
        height_gate = self._get_landing_aux_height_gate()
        active_mask = prelanding_mask & (flight_mask | touchdown_mask)
        return (
            self._joint_pose_tracking_reward(self.prelanding_joint_target)
            * active_mask.float()
            * height_gate
            * self.jump_toggle_state.float()
        )

    def _reward_joint_pose_landing(self):
        _, num_contacts, _, _ = self._get_jump_contact_states()
        _, _, _, _, _, touchdown_mask, stabilize_mask = self._get_jump_phase_masks()
        nonfoot_contact = self._get_nonfoot_contact_mask()
        active_mask = (touchdown_mask | stabilize_mask) & (num_contacts >= self.cfg.jump_phase.landing_min_contacts)
        return (
            self._joint_pose_tracking_reward(self.landing_joint_target)
            * active_mask.float()
            * (~nonfoot_contact).float()
            * self._get_landing_aux_height_gate()
            * self.jump_toggle_state.float()
        )

    def _reward_landing_stability(self):
        _, _, _, _, _, touchdown_mask, stabilize_mask = self._get_jump_phase_masks()
        phase_mask = touchdown_mask | stabilize_mask
        nonfoot_contact = self._get_nonfoot_contact_mask()
        tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
        xy_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        ang_speed = torch.norm(self.base_ang_vel[:, :2], dim=1)
        vertical_speed = torch.abs(self.base_lin_vel[:, 2])

        reward = (
            torch.exp(-float(self.cfg.rewards.landing_tilt_scale) * tilt)
            * torch.exp(-float(self.cfg.rewards.landing_xy_vel_scale) * xy_speed)
            * torch.exp(-float(self.cfg.rewards.landing_ang_vel_scale) * ang_speed)
            * torch.exp(-float(self.cfg.rewards.landing_vertical_vel_scale) * vertical_speed)
        )
        return reward * phase_mask.float() * (~nonfoot_contact).float() * self._get_landing_aux_height_gate()

    def _reward_landing_impact(self):
        _, _, _, _, _, touchdown_mask, stabilize_mask = self._get_jump_phase_masks()
        phase_mask = self.just_landed | touchdown_mask | stabilize_mask
        nonfoot_contact = self._get_nonfoot_contact_mask()
        vertical_speed = torch.abs(self.base_lin_vel[:, 2])
        xy_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        tilt = torch.norm(self.projected_gravity[:, :2], dim=1)

        vertical_sigma = max(float(self.cfg.rewards.landing_impact_velocity_sigma), 1e-3)
        xy_sigma = max(float(self.cfg.rewards.landing_impact_xy_sigma), 1e-3)
        tilt_sigma = max(float(self.cfg.rewards.landing_impact_tilt_sigma), 1e-3)
        reward = (
            torch.exp(-torch.square(vertical_speed / vertical_sigma))
            * torch.exp(-torch.square(xy_speed / xy_sigma))
            * torch.exp(-torch.square(tilt / tilt_sigma))
        )
        return reward * phase_mask.float() * (~nonfoot_contact).float() * self._get_landing_aux_height_gate()

    def _reward_landing_contact(self):
        contact, num_contacts, _, _ = self._get_jump_contact_states()
        _, _, _, _, _, touchdown_mask, stabilize_mask = self._get_jump_phase_masks()
        phase_mask = touchdown_mask | stabilize_mask
        nonfoot_contact = self._get_nonfoot_contact_mask()

        contact_ratio = num_contacts.float() / float(len(self.feet_indices))
        front_ratio = torch.mean(contact[:, :2].float(), dim=1)
        rear_ratio = torch.mean(contact[:, 2:].float(), dim=1)
        reward = contact_ratio * torch.minimum(front_ratio, rear_ratio)
        return reward * phase_mask.float() * (~nonfoot_contact).float() * self._get_landing_aux_height_gate()

    def _reward_landing_body_clearance(self):
        _, _, _, _, _, touchdown_mask, stabilize_mask = self._get_jump_phase_masks()
        phase_mask = touchdown_mask | stabilize_mask
        nonfoot_contact = self._get_nonfoot_contact_mask()
        floor = float(self.cfg.rewards.landing_body_height_floor)
        target = max(float(self.cfg.rewards.landing_body_height_target), floor + 1e-3)
        body_height = torch.clamp((self.root_states[:, 2] - floor) / (target - floor), min=0.0, max=1.0)
        tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
        tilt_reward = torch.exp(-torch.square(tilt / max(float(self.cfg.rewards.body_tilt_sigma), 1e-3)))
        return (
            body_height
            * tilt_reward
            * phase_mask.float()
            * (~nonfoot_contact).float()
            * self._get_landing_aux_height_gate()
        )

    def _reward_jump_distance(self):
        # Weak completion bonus for forward displacement along the commanded
        # launch direction. This is not the primary task anymore when landing
        # targets are disabled; it only nudges the policy to use the commanded
        # direction after learning to complete a valid jump cycle.
        if not torch.any(self.just_completed_cycle):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        target_delta = self.target_landing_xy - self.jump_start_xy
        target_dist = torch.clamp(torch.norm(target_delta, dim=1), min=1e-3)
        target_dir = target_delta / target_dist.unsqueeze(1)

        displacement = self.root_states[:, :2] - self.jump_start_xy
        forward_dist = torch.sum(displacement * target_dir, dim=1)

        reference = torch.clamp(target_dist, min=float(getattr(self.cfg.rewards, "jump_distance_reference_min", 0.15)))
        reward = torch.clamp(forward_dist / reference, min=0.0, max=1.0)

        nonfoot_contact = self._get_nonfoot_contact_mask()
        valid_jump = self._get_completed_valid_jump_gate() >= 1.0
        return (
            reward
            * self.just_completed_cycle.float()
            * (~nonfoot_contact).float()
            * valid_jump.float()
        )

    def _reward_nonfoot_contact(self):
        _, compression_mask, extension_mask, release_mask, flight_mask, touchdown_mask, stabilize_mask = self._get_jump_phase_masks()
        jump_active = (
            compression_mask
            | extension_mask
            | release_mask
            | flight_mask
            | touchdown_mask
            | stabilize_mask
            | self.just_completed_cycle
        )
        return self._get_nonfoot_contact_mask().float() * jump_active.float()
