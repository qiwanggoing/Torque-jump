import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_rotate_inverse, torch_rand_float

from legged_gym.envs.go2.go2_omnijump_torque.go2_omnijump_torque_config import GO2OmniJumpTorqueCfg
from legged_gym.envs.go2.go2_torque.go2_torque import GO2Torque
from legged_gym.utils.math import wrap_to_pi


class GO2OmniJumpTorque(GO2Torque):
    cfg: GO2OmniJumpTorqueCfg

    ACTIVE_REWARD_WHITELIST = {
        "height_tracking",           # OmniNet Table I: e^{-20(z*-z)^2}, dense instantaneous
        "successful_jump",           # OmniNet Table I: n_jump sparse binary
        "tracking_linear_velocity",  # OmniNet Table I: e^{-4||Δv||^2}
        "tracking_angular_velocity", # OmniNet Table I: e^{-4(Δω)^2}
        "orientation",               # OmniNet Table I: g_xy
        "joint_angle_loaded",        # two-phase: fold legs during squat-down / pre-pushoff
        "joint_angle_extended",      # two-phase: straight legs through pushoff / flight / landing
        "joint_angle_aerial",        # legacy OmniNet (weight 0 by default)
        "joint_angle_prelanding",    # legacy OmniNet (weight 0 by default)
        "joint_angle_landing",       # legacy OmniNet (weight 0 by default)
        "collision",                 # OmniNet Table I: -n_collision
        "torques",                   # OmniNet Table I: ||τ||^2, w=-1e-5
        "action_rate",               # OmniNet Table I: ||Δa||^2, w=-0.01
        "dof_acc",                   # OmniNet Table I: ||q̈||^2, w=-2.5e-7
        "termination",
        "maintain_contact",
        "peak_height_progress",       # bootstrap: progress toward target peak height
        "all_feet_airborne",          # flight: reward being airborne
        "takeoff_vertical_velocity",  # stance pushoff: reward upward velocity after squat
        "projected_peak",             # Olsen: reward projected peak height during pushoff
        "horizontal_drift",           # enforce vertical-only jumps when xy cmd is zero
        "takeoff_direction",          # one-shot at just_took_off: vz / ||v|| → encourage vertical takeoff
        "default_pos",                # mygo2jump-style L1 toward q_squat (whole-body squat bias)
        "default_hip_pos",            # mygo2jump-style exp reward for hip joints near default
        "landing_stability",          # Atanassov-style: penalize lin/ang vel during landing buffer
    }

    def _prepare_reward_function(self):
        filtered_scales = {}
        for key, value in self.reward_scales.items():
            if key in self.ACTIVE_REWARD_WHITELIST:
                filtered_scales[key] = value
        self.reward_scales = filtered_scales
        super()._prepare_reward_function()

    def _init_buffers(self):
        super()._init_buffers()
        self.default_joint_pd_target = self.default_dof_pos.repeat(self.num_envs, 1)
        self.residual_torques_action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device
        )
        self.pd_prior_torques = torch.zeros_like(self.residual_torques_action)
        self.pd_prior_alpha = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.rl_prior_alpha = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)

        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            dtype=torch.float,
            device=self.device,
        )

        self.jumping_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rsi_episode_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_taken_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prelanding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.landing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Two-phase pose guidance: loaded (fold legs) ↔ extended (straighten legs).
        # Switch point is vz crossing 0 — the physical pushoff onset.
        self.phase_loaded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase_extended = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_took_off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_jump_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending_velocity_score = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_episode_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.single_jump_play_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.single_jump_command_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.single_jump_command_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.jump_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stand_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.landing_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.post_jump_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.airborne_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.peak_base_height = self.root_states[:, 2].clone()
        self.jump_min_base_z = self.root_states[:, 2].clone()   # lowest base z before takeoff (squat-depth gate)
        self.jump_min_pose_err = torch.full((self.num_envs,), 1e3, device=self.device)  # min |dof-q_squat| before takeoff (squat-POSE gate); 1e3 = not-yet-reached
        self.jump_min_contact = torch.full((self.num_envs,), 4, dtype=torch.long, device=self.device)  # min #feet in contact during the load (4 = no foot lifted yet); clean-takeoff re-plant detector
        self.jump_replant = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)           # a foot RE-CONTACTED during the load after lifting (stutter-step / run-up) -> clean-takeoff violation
        self.landing_plant_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)     # consecutive all-4-feet-contact steps in the landing buffer (clean-landing settle detector)
        self.landing_planted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)        # latched once the feet HELD (clean_landing_plant_hold steps) -> "settled, now watch for re-lift"
        self.landing_relift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)         # a foot LIFTED after the clean plant (hop / shuffle-step) -> clean-landing violation
        self.squat_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # consecutive steps held within squat_pose_threshold this jump (resets on pop-out)
        self.squat_qualified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)      # latched once squat HELD >= squat_hold_steps: the real squat-POSE gate (no flick-through)
        self.landing_min_height = self.root_states[:, 2].clone()
        self.takeoff_root_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.landing_root_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.last_success_velocity_score = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.jump_starts = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_flights = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.squat_qualified_takeoffs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # # takeoffs this episode preceded by a HELD squat (diagnostic numerator)
        self.jump_landings = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_completed_cycles = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.successful_jumps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.peak_height_error_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.peak_height_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.jump_evaluations = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.q_ground_target = self._solve_pose_from_foot_height(self.cfg.rewards.ground_foot_height)
        self.q_air_target = self._solve_pose_from_foot_height(self.cfg.rewards.air_foot_height)
        self.q_pre_target = self._solve_pose_from_foot_height(self.cfg.rewards.prelanding_foot_height)
        self.q_squat_target = self._solve_pose_from_foot_height(
            float(getattr(self.cfg.rewards, "squat_foot_height", 0.10))
        )
        self.default_joint_pd_target = self.default_dof_pos.repeat(self.num_envs, 1)
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self._log_jump_episode_stats(env_ids)
        self._reset_jump_buffers(env_ids)
        if self.cfg.commands.num_commands > 4:
            self._resample_commands(env_ids)
        # Activate jumping state for RSI envs so flight rewards fire from step 0
        rsi_envs = env_ids[self.rsi_episode_mask[env_ids]]
        if len(rsi_envs) > 0:
            self.jumping_state[rsi_envs] = True

    def _reset_jump_buffers(self, env_ids):
        self.jumping_state[env_ids] = False
        self.has_taken_off[env_ids] = False
        self.has_landed[env_ids] = False
        self.airborne[env_ids] = False
        self.prelanding[env_ids] = False
        self.landing[env_ids] = False
        self.phase_loaded[env_ids] = False
        self.phase_extended[env_ids] = False
        self.just_took_off[env_ids] = False
        self.just_landed[env_ids] = False
        self.last_jump_success[env_ids] = False
        self.pending_success[env_ids] = False
        self.jump_episode_failed[env_ids] = False
        self.single_jump_play_done[env_ids] = False
        self.single_jump_command_done[env_ids] = False
        if hasattr(self, "last_contacts"):
            self.last_contacts[env_ids] = False

        self.jump_step_counter[env_ids] = 0
        self.stand_step_counter[env_ids] = 0
        self.landing_step_counter[env_ids] = 0
        self.post_jump_step_counter[env_ids] = 0
        self.airborne_time[env_ids] = 0.0
        self.peak_base_height[env_ids] = self.root_states[env_ids, 2]
        self.jump_min_base_z[env_ids] = self.root_states[env_ids, 2]
        self.jump_min_pose_err[env_ids] = 1e3  # squat-POSE gate: reset to not-yet-reached sentinel
        self.jump_min_contact[env_ids] = 4     # clean-takeoff re-plant detector: reset to "no foot lifted yet"
        self.jump_replant[env_ids] = False
        self.landing_plant_hold[env_ids] = 0     # clean-landing detector: reset
        self.landing_planted[env_ids] = False
        self.landing_relift[env_ids] = False
        self.squat_hold_counter[env_ids] = 0
        self.squat_qualified[env_ids] = False
        self.landing_min_height[env_ids] = self.root_states[env_ids, 2]
        self.takeoff_root_xy[env_ids] = self.root_states[env_ids, :2]
        self.landing_root_xy[env_ids] = self.root_states[env_ids, :2]
        self.last_success_velocity_score[env_ids] = 0.0

        self.jump_starts[env_ids] = 0.0
        self.jump_flights[env_ids] = 0.0
        self.squat_qualified_takeoffs[env_ids] = 0.0
        self.jump_landings[env_ids] = 0.0
        self.jump_completed_cycles[env_ids] = 0.0
        self.successful_jumps[env_ids] = 0.0
        self.peak_height_error_sum[env_ids] = 0.0
        self.peak_height_sum[env_ids] = 0.0
        self.jump_evaluations[env_ids] = 0.0

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        self.rsi_episode_mask[env_ids] = False
        rsi_prob = float(getattr(self.cfg.rewards, "rsi_prob", 0.0))
        if rsi_prob > 0.0 and len(env_ids) > 0:
            rsi_mask = torch.rand(len(env_ids), device=self.device) < rsi_prob
            rsi_ids = env_ids[rsi_mask]
            if len(rsi_ids) > 0:
                # RSI from squat, two sub-modes (both with DOFs in the squat pose):
                #   - launch (vz 1-3): air-drop already rising -> bootstraps V(squat -> flight). [original]
                #   - static (vz ~ 0): air-drop AT REST in the deep squat + jumping_state, so the policy
                #     must PUSH OFF from a standstill squat. THIS is the exploration piece we were missing:
                #     it lets value learn "deep-squat-at-rest = high return", so the standing rollouts then
                #     discover "dip first, then push" (countermovement) on their own. Pairs with the
                #     squat-depth gate (RSI is gate-exempt, so the dropped env earns jump rewards straight
                #     from the dip state, planting the V(dip) the gate then makes the standing policy chase).
                squat_height = float(getattr(self.cfg.rewards, "stance_squat_height", 0.20))
                static_frac = float(getattr(self.cfg.rewards, "rsi_static_frac", 0.0))
                static_mask = torch.rand(len(rsi_ids), device=self.device) < static_frac

                height_offset_min = float(getattr(self.cfg.rewards, "rsi_height_offset_min", 0.0))
                height_offset_max = float(getattr(self.cfg.rewards, "rsi_height_offset_max", 0.1))
                rsi_height_offset = torch_rand_float(height_offset_min, height_offset_max, (len(rsi_ids), 1), device=self.device).squeeze(1)
                rsi_height_offset = torch.where(static_mask, torch.zeros_like(rsi_height_offset), rsi_height_offset)  # static sits AT the squat
                self.root_states[rsi_ids, 2] = squat_height + self.env_origins[rsi_ids, 2] + rsi_height_offset

                vel_z_min = float(getattr(self.cfg.rewards, "rsi_vel_z_min", 1.0))
                vel_z_max = float(getattr(self.cfg.rewards, "rsi_vel_z_max", 3.0))
                static_vz_min = float(getattr(self.cfg.rewards, "rsi_static_vel_z_min", -0.1))
                static_vz_max = float(getattr(self.cfg.rewards, "rsi_static_vel_z_max", 0.3))
                launch_vz = torch_rand_float(vel_z_min, vel_z_max, (len(rsi_ids), 1), device=self.device).squeeze(1)
                static_vz = torch_rand_float(static_vz_min, static_vz_max, (len(rsi_ids), 1), device=self.device).squeeze(1)
                rsi_vel_z = torch.where(static_mask, static_vz, launch_vz)
                self.root_states[rsi_ids, 9] = rsi_vel_z

                self.dof_pos[rsi_ids] = self.q_squat_target.unsqueeze(0)
                self.dof_vel[rsi_ids] = 0.0

                self.rsi_episode_mask[rsi_ids] = True
                env_ids_int32 = rsi_ids.to(dtype=torch.int32)
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32),
                    len(env_ids_int32),
                )
                self.gym.set_dof_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.dof_state),
                    gymtorch.unwrap_tensor(env_ids_int32),
                    len(env_ids_int32),
                )

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
        # Fraction of jump attempts whose takeoff was preceded by a HELD squat (squat_hold_steps).
        # Compare against jump_flight_rate: equal => every takeoff held the squat first (goal);
        # squat_qualified_rate << jump_flight_rate => takeoffs still firing without holding the squat.
        self.extras["episode"]["squat_qualified_rate"] = torch.mean(self.squat_qualified_takeoffs[env_ids] / jump_den)

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
            test_command = self.cfg.test.vel.to(self.device)
            if test_command.numel() != self.cfg.commands.num_commands:
                padded_command = torch.zeros(self.cfg.commands.num_commands, device=self.device)
                command_dim = min(test_command.numel(), self.cfg.commands.num_commands)
                padded_command[:command_dim] = test_command[:command_dim]
                test_command = padded_command
            self.commands[:] = test_command
            if getattr(self.cfg.test, "single_jump_play", False) and self.cfg.commands.num_commands > 4:
                done_ids = self.single_jump_play_done.nonzero(as_tuple=False).flatten()
                self._disable_jump_command(done_ids)
        self._update_jump_state()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.commands.num_commands > 4 and hasattr(self, "single_jump_command_done"):
            hold_mask = self.single_jump_command_done[env_ids]
            hold_ids = env_ids[hold_mask]
            self._disable_jump_command(hold_ids)
            env_ids = env_ids[~hold_mask]
            if len(env_ids) == 0:
                return
            stand_prob = float(getattr(self.cfg.commands, "stand_command_prob", 0.0))
            if stand_prob > 0.0:
                stand_mask = (
                    torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1) < stand_prob
                )
                stand_ids = env_ids[stand_mask]
                self._disable_jump_command(stand_ids)
                self.single_jump_command_mode[stand_ids] = False
                self.single_jump_command_done[stand_ids] = False
                env_ids = env_ids[~stand_mask]
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
        if self.cfg.commands.num_commands > 4:
            single_jump_prob = float(getattr(self.cfg.commands, "single_jump_command_prob", 0.5))
            self.single_jump_command_mode[env_ids] = (
                torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1) < single_jump_prob
            )
            self.single_jump_command_done[env_ids] = False
            self.commands[env_ids, 4] = torch_rand_float(
                self.command_ranges["jump_command"][0],
                self.command_ranges["jump_command"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            # Stand episodes: zero velocity so tracking_linear_velocity rewards staying still
            stand_mask = self.commands[env_ids, 4] <= float(self.cfg.commands.jump_command_threshold)
            if stand_mask.any():
                stand_env_ids = env_ids[stand_mask]
                self.commands[stand_env_ids, 0] = 0.0
                self.commands[stand_env_ids, 1] = 0.0

    def _disable_jump_command(self, env_ids):
        if len(env_ids) == 0 or self.cfg.commands.num_commands <= 4:
            return
        self.commands[env_ids, 4] = self.command_ranges["jump_command"][0]

    def _start_jump(self, env_ids):
        if len(env_ids) == 0:
            return
        self.jumping_state[env_ids] = True
        self.has_taken_off[env_ids] = False
        self.has_landed[env_ids] = False
        self.airborne[env_ids] = False
        self.prelanding[env_ids] = False
        self.landing[env_ids] = False
        self.phase_loaded[env_ids] = False
        self.phase_extended[env_ids] = False
        self.just_took_off[env_ids] = False
        self.just_landed[env_ids] = False
        self.last_jump_success[env_ids] = False
        self.pending_success[env_ids] = False
        self.jump_episode_failed[env_ids] = False
        self.jump_step_counter[env_ids] = 0
        self.landing_step_counter[env_ids] = 0
        self.post_jump_step_counter[env_ids] = 0
        self.airborne_time[env_ids] = 0.0
        self.peak_base_height[env_ids] = self.root_states[env_ids, 2]
        self.jump_min_base_z[env_ids] = self.root_states[env_ids, 2]
        self.jump_min_pose_err[env_ids] = 1e3  # squat-POSE gate: reset to not-yet-reached sentinel
        self.jump_min_contact[env_ids] = 4     # clean-takeoff re-plant detector: reset to "no foot lifted yet"
        self.jump_replant[env_ids] = False
        self.landing_plant_hold[env_ids] = 0     # clean-landing detector: reset
        self.landing_planted[env_ids] = False
        self.landing_relift[env_ids] = False
        self.squat_hold_counter[env_ids] = 0
        self.squat_qualified[env_ids] = False
        self.landing_min_height[env_ids] = self.root_states[env_ids, 2]
        self.takeoff_root_xy[env_ids] = self.root_states[env_ids, :2]
        self.landing_root_xy[env_ids] = self.root_states[env_ids, :2]
        self.last_success_velocity_score[env_ids] = 0.0
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
        self.phase_loaded[env_ids] = False
        self.phase_extended[env_ids] = False
        self.just_took_off[env_ids] = False
        self.just_landed[env_ids] = False
        self.jump_step_counter[env_ids] = 0
        self.landing_step_counter[env_ids] = 0
        self.airborne_time[env_ids] = 0.0
        self.stand_step_counter[env_ids] = 0
        self.peak_base_height[env_ids] = self.root_states[env_ids, 2]
        self.jump_min_base_z[env_ids] = self.root_states[env_ids, 2]
        self.jump_min_pose_err[env_ids] = 1e3  # squat-POSE gate: reset to not-yet-reached sentinel
        self.jump_min_contact[env_ids] = 4     # clean-takeoff re-plant detector: reset to "no foot lifted yet"
        self.jump_replant[env_ids] = False
        self.landing_plant_hold[env_ids] = 0     # clean-landing detector: reset
        self.landing_planted[env_ids] = False
        self.landing_relift[env_ids] = False
        self.squat_hold_counter[env_ids] = 0
        self.squat_qualified[env_ids] = False
        self.landing_min_height[env_ids] = self.root_states[env_ids, 2]
        if completed:
            self.jump_completed_cycles[env_ids] += 1.0

    def _update_jump_state(self):
        contact = self._get_contact_state()
        contact_filt = torch.logical_or(contact, self.last_contacts)
        any_foot_contact = torch.any(contact_filt, dim=1)
        all_feet_contact = torch.all(contact_filt, dim=1)

        # Stable stand for running jumps: all feet on ground, no body contact, upright orientation.
        # No velocity constraint — robot jumps from running state (OmniNet-style).
        body_contact = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_contact = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 0.1, dim=1)
        any_body_contact = body_contact | base_contact

        stable_stand = (
            all_feet_contact
            & (~any_body_contact)
            & (torch.abs(self.projected_gravity[:, 0]) < 0.1)
            & (torch.abs(self.projected_gravity[:, 1]) < 0.1)
        )
        self.stand_step_counter = torch.where(
            (~self.jumping_state) & stable_stand,
            self.stand_step_counter + 1,
            torch.zeros_like(self.stand_step_counter),
        )

        if self.cfg.commands.num_commands > 4:
            jump_command_active = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        else:
            jump_command_active = torch.ones_like(self.jumping_state)
        if self.cfg.test.use_test and getattr(self.cfg.test, "single_jump_play", False):
            can_start_single_jump = ~self.single_jump_play_done
        else:
            can_start_single_jump = torch.ones_like(self.jumping_state)
        if self.cfg.commands.num_commands > 4 and getattr(self.cfg.rewards, "one_jump_reward_per_episode", False):
            can_start_single_jump &= ~self.single_jump_command_done
        first_jump_ready = self.episode_length_buf >= self.cfg.rewards.first_jump_delay_steps
        first_jump_request = (self.jump_starts <= 0.0) & first_jump_ready
        rearm_ready = first_jump_ready
        ready_to_jump = (
            (~self.jumping_state)
            & jump_command_active
            & can_start_single_jump
            & rearm_ready
        )
        start_ids = ready_to_jump.nonzero(as_tuple=False).flatten()
        self._start_jump(start_ids)

        self.just_took_off = (
            self.jumping_state
            & (~self.has_taken_off)
            & torch.all(~contact_filt, dim=1)
        )
        self.has_taken_off |= self.just_took_off
        self.airborne = self.jumping_state & self.has_taken_off & (~self.has_landed) & (~any_foot_contact)
        self.airborne_time += self.airborne.float() * self.dt
        self.peak_base_height = torch.where(
            self.airborne,
            torch.maximum(self.peak_base_height, self.root_states[:, 2]),
            self.peak_base_height,
        )
        # Track lowest base height during the pre-takeoff load (the dip), for the squat-depth gate.
        loading = self.jumping_state & (~self.has_taken_off)
        # CLEAN-TAKEOFF re-plant detector (one clean jump): during the load (jumping, pre-takeoff), once a
        # foot has LIFTED (contact count dropped below 4), any foot RE-CONTACTING (count goes back UP) is a
        # stutter-step / run-up -- the policy shuffling its feet to build forward momentum before the real
        # all-feet-off takeoff. We FORBID it (check_termination ends the episode) so the jump is ONE clean
        # push. Gated post-discovery in check_termination so early failed-push attempts, which also look
        # like re-plants, do not block the from-scratch discovery of jumping.
        _n_contact = contact_filt.sum(dim=1)
        self.jump_replant |= loading & (self.jump_min_contact < 4) & (_n_contact > self.jump_min_contact)
        self.jump_min_contact = torch.where(loading, torch.minimum(self.jump_min_contact, _n_contact), self.jump_min_contact)
        self.jump_min_base_z = torch.where(
            loading,
            torch.minimum(self.jump_min_base_z, self.root_states[:, 2]),
            self.jump_min_base_z,
        )
        # Track closest approach to the loaded squat POSE before takeoff (kept for diagnostics).
        self.jump_min_pose_err = torch.where(
            loading,
            torch.minimum(self.jump_min_pose_err, self._squat_pose_err()),
            self.jump_min_pose_err,
        )
        # Squat-HOLD gate: the jump chain unlocks only once the squat POSE has been HELD for
        # >= squat_hold_steps CONSECUTIVE steps (counter resets the instant the pose pops back out
        # above threshold). Replaces the old "jump_min_pose_err <= thr touched once" criterion, which
        # let the policy flick through the pose for a single frame and harvest the whole flight while
        # earning almost no stance_squat. squat_qualified latches True (so a clean load->release: once
        # earned, the subsequent leg-extension that raises pose_err does NOT re-lock the gate).
        _hold_thr = float(getattr(self.cfg.rewards, "squat_pose_threshold", 0.0))
        if _hold_thr > 0.0:
            in_pose = loading & (self._squat_pose_err() <= _hold_thr)
            self.squat_hold_counter = torch.where(
                in_pose,
                self.squat_hold_counter + 1,
                torch.where(loading, torch.zeros_like(self.squat_hold_counter), self.squat_hold_counter),
            )
            _hold_steps = int(getattr(self.cfg.rewards, "squat_hold_steps", 40))
            self.squat_qualified = self.squat_qualified | (self.squat_hold_counter >= _hold_steps)

        descending = self.base_lin_vel[:, 2] < -0.05
        prelanding_height = torch.maximum(
            self.peak_base_height - self.cfg.rewards.prelanding_height_margin,
            torch.full_like(self.commands[:, 3], self.cfg.rewards.base_height_target + 0.04),
        )
        self.prelanding = self.airborne & descending & (self.root_states[:, 2] <= prelanding_height)

        self.just_landed = self.jumping_state & self.has_taken_off & (~self.has_landed) & any_foot_contact
        self.has_landed |= self.just_landed
        self.landing = self.jumping_state & self.has_landed
        # CLEAN-LANDING settle detector (mirror of the clean-takeoff re-plant, opposite direction): once all
        # four feet have HELD contact for clean_landing_plant_hold steps in the landing buffer, LATCH
        # landing_planted ("settled"); thereafter any foot LIFTING (hop / shuffle-step) trips landing_relift.
        # The hold-then-latch (like the squat-HOLD gate) skips the touchdown impact chatter -> only a
        # DELIBERATE re-lift after a real plant counts. Penalized via _reward_clean_landing (NOT terminated);
        # that reward is gated on the succ-latch so messy from-scratch landings are never punished.
        _n_contact_land = contact_filt.sum(dim=1)
        _planted_now = self.landing & (_n_contact_land >= 4)
        self.landing_plant_hold = torch.where(
            _planted_now,
            self.landing_plant_hold + 1,
            torch.where(self.landing, torch.zeros_like(self.landing_plant_hold), self.landing_plant_hold),
        )
        _plant_hold_steps = int(getattr(self.cfg.rewards, "clean_landing_plant_hold", 15))
        self.landing_planted = self.landing_planted | (self.landing_plant_hold >= _plant_hold_steps)
        self.landing_relift = self.landing_relift | (self.landing_planted & (_n_contact_land < 4))

        # Two-phase pose guidance — fold during squat-down, extend through pushoff/flight/landing.
        # Pose-latched boundary (was vz<=0): stay in the FOLD phase until this jump has actually
        # reached the loaded squat POSE, then commit to EXTEND. jump_min_pose_err is monotone, so it
        # never flips back on a vz bobble — a clean load-then-release instead of the old vz flip that
        # switched to "extend" the instant the body twitched up (so it never truly loaded). Only the
        # PD prior / default_pos pose target read phase_loaded (joint_angle_loaded/extended are off),
        # so this just makes the symmetric q_squat fold persist until the squat pose is reached.
        # vz<=0 fallback keeps behaviour unchanged for tasks with the gate disabled.
        vz = self.root_states[:, 9]
        _pose_thr = float(getattr(self.cfg.rewards, "squat_pose_threshold", 0.0))
        if _pose_thr > 0.0:
            self.phase_loaded = self.jumping_state & (~self.has_taken_off) & (~self.squat_qualified)
        else:
            self.phase_loaded = self.jumping_state & (~self.has_taken_off) & (vz <= 0.0)
        self.phase_extended = self.jumping_state & (~self.phase_loaded)
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

        # Simplified successful_jump cancel: only "actually fell" disqualifies during landing buffer.
        # Tolerate calf/thigh scuffs (natural during landing absorption); base contact terminates
        # the episode anyway via termination_contact_indices.
        tilt_threshold = float(getattr(self.cfg.rewards, "success_fallover_tilt", 0.7))
        excessive_tilt = (
            (torch.abs(self.projected_gravity[:, 0]) > tilt_threshold)
            | (torch.abs(self.projected_gravity[:, 1]) > tilt_threshold)
        )
        self.pending_success &= ~(self.landing & excessive_tilt)

        self.last_jump_success[:] = False
        self.jump_episode_failed[:] = False
        self.last_success_velocity_score[:] = 0.0
        if torch.any(self.just_took_off):
            self.jump_flights[self.just_took_off] += 1.0
            self.squat_qualified_takeoffs[self.just_took_off & self.squat_qualified] += 1.0
            self.takeoff_root_xy[self.just_took_off] = self.root_states[self.just_took_off, :2]
        if torch.any(self.just_landed):
            self.jump_landings[self.just_landed] += 1.0
            self.landing_root_xy[self.just_landed] = self.root_states[self.just_landed, :2]
            peak_err = torch.abs(self.peak_base_height - self.commands[:, 3])
            self.jump_evaluations += self.just_landed.float()
            self.peak_height_error_sum += peak_err * self.just_landed.float()
            self.peak_height_sum += self.peak_base_height * self.just_landed.float()
            min_peak = float(getattr(self.cfg.rewards, "successful_jump_min_peak_height", 0.30))
            real_jump = self.peak_base_height >= min_peak
            jump_height_commanded = self.commands[:, 3] >= 0.28
            # Squat-depth gate: a jump only counts if it dipped to <= squat_gate_height first
            # (forces a real countermovement; RSI air-drops exempt). squat_gate_height<=0 -> off.
            # success_requires_squat_qualified=False DECOUPLES success from the squat_qualified HOLD gate:
            # that latched gate flickers under pure-torque noise (25 consecutive in-pose steps), and since
            # success required it, succ oscillated wildly even though flight/peak were stable. It is
            # REDUNDANT for success -- real_jump (peak>=min_peak 0.40, well above idle 0.31) already
            # guarantees a real countermovement (you can't pure-torque jump that high without squatting).
            # The gate still gates the jump-REWARD chain (keeps forcing a real squat during learning).
            success_at_impact = self.just_landed & real_jump & jump_height_commanded
            if getattr(self.cfg.rewards, "success_requires_squat_qualified", True):
                success_at_impact = success_at_impact & self._squat_deep_enough()
            self.pending_success |= success_at_impact

            # cmd-aware Gaussian height score: penalize both overshoot and undershoot
            height_sigma = float(getattr(self.cfg.rewards, "success_height_sigma", 0.05))
            height_score = torch.exp(-torch.square(self.peak_base_height - self.commands[:, 3]) / max(height_sigma, 1e-4))
            success_velocity_score = self._get_successful_jump_velocity_score()
            if not getattr(self.cfg.rewards, "success_use_velocity_score", False):
                success_velocity_score = torch.ones_like(success_velocity_score)
            combined_score = height_score * success_velocity_score

            self.pending_velocity_score = torch.where(
                success_at_impact,
                combined_score,
                self.pending_velocity_score,
            )

        takeoff_timeout = self.jumping_state & (~self.has_taken_off) & (
            self.jump_step_counter > self.cfg.rewards.takeoff_timeout_steps
        )
        if torch.any(takeoff_timeout):
            timeout_ids = takeoff_timeout.nonzero(as_tuple=False).flatten()
            self.jump_episode_failed[timeout_ids] = True
            if self.cfg.commands.num_commands > 4 and getattr(self.cfg.rewards, "one_jump_reward_per_episode", False):
                self.single_jump_command_done[timeout_ids] = True
                self.post_jump_step_counter[timeout_ids] = 0
                self._disable_jump_command(timeout_ids)
            self._finish_jump(timeout_ids, completed=False)

        if self.cfg.test.use_test and getattr(self.cfg.test, "single_jump_play", False):
            self.single_jump_play_done |= self.last_jump_success
            done_ids = self.last_jump_success.nonzero(as_tuple=False).flatten()
            self._disable_jump_command(done_ids)
        if self.cfg.commands.num_commands > 4:
            one_jump_reward = getattr(self.cfg.rewards, "one_jump_reward_per_episode", False)
            single_command_finished = self.last_jump_success & (self.single_jump_command_mode | one_jump_reward)
            if torch.any(single_command_finished):
                self.single_jump_command_done |= single_command_finished
                done_ids = single_command_finished.nonzero(as_tuple=False).flatten()
                self.post_jump_step_counter[done_ids] = 0
                self._disable_jump_command(done_ids)
        if self.cfg.commands.num_commands > 4 and getattr(self.cfg.rewards, "one_jump_reward_per_episode", False):
            post_jump_done = self.single_jump_command_done & (~self.jumping_state)
            self.post_jump_step_counter = torch.where(
                post_jump_done,
                self.post_jump_step_counter + 1,
                torch.zeros_like(self.post_jump_step_counter),
            )

        # Finally, check if landing buffer is complete to finalize the jump and grant rewards.
        # success/finish is gated ONLY on the time buffer (NOT on pose return) so the jump-discovery
        # signal stays dense -- gating success on a pose the early policy can't hit starves the
        # successful_jump carrot and the policy never learns to jump. The "return to default pose
        # before the NEXT jump" rule is decoupled below: it delays the next jump WITHOUT withholding
        # success for this one.
        ready_to_finish = (
            self.jumping_state
            & self.has_landed
            & (self.landing_step_counter >= max(int(self.cfg.rewards.landing_buffer_steps), 1))
        )
        pose_gate = getattr(self.cfg.rewards, "next_jump_requires_default_pose", False)
        if torch.any(ready_to_finish):
            finish_ids = ready_to_finish.nonzero(as_tuple=False).flatten()
            self.last_jump_success[finish_ids] = self.pending_success[finish_ids]
            self.last_success_velocity_score[finish_ids] = self.pending_velocity_score[finish_ids]
            self.successful_jumps[finish_ids] += self.last_jump_success[finish_ids].float()
            self._finish_jump(finish_ids, completed=True)
            self._disable_jump_command(finish_ids)
            # CONTINUOUS jumping: single-jump-MODE envs always stop after one jump. Continuous-mode
            # envs (mode=False) normally re-enable immediately (done=mode=False). With the pose gate
            # we instead LOCK them (done=True) and only unlock once they have returned to the default
            # pose (below), so every jump in the sequence starts from the same canonical idle stand.
            if pose_gate:
                self.single_jump_command_done[finish_ids] = True
            else:
                self.single_jump_command_done[finish_ids] = self.single_jump_command_mode[finish_ids]
            # Clean up pending success
            self.pending_success[finish_ids] = False
            self.pending_velocity_score[finish_ids] = 0.0

        # Decoupled NEXT-JUMP pose gate (config-driven, default off): unlock a finished continuous-mode
        # env (re-enable jumping) once it has RETURNED to the default standing pose. Does NOT touch the
        # success/finish above -> preserves dense discovery signal while still forcing each jump to
        # start from the canonical idle pose. Metric = sum|dof - default_dof_pos| over the 12 joints.
        if pose_gate:
            pose_err = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
            unlock = (
                (~self.jumping_state)
                & self.single_jump_command_done
                & (~self.single_jump_command_mode)
                & (pose_err < float(getattr(self.cfg.rewards, "next_jump_default_pose_threshold", 1.5)))
            )
            self.single_jump_command_done[unlock] = False

        self.last_contacts[:] = contact

    def compute_observations(self):
        foot_contact_obs = self._get_contact_state().float()
        motor_fatigue = self.motor_fatigue.detach()
        height_obs = torch.cat(
            (
                self.root_states[:, 2:3] * 2.0,
                (self.commands[:, 3:4] - self.root_states[:, 2:3]) * 2.0,
            ),
            dim=-1,
        )
        obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.commands[:, 3:4] * 2.0,
                self.commands[:, 4:5],
                height_obs,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                foot_contact_obs,
                self.torques,
                motor_fatigue,
                self.pd_prior_alpha,
            ),
            dim=-1,
        )
        obs_buf = torch.nan_to_num(obs_buf, nan=0.0, posinf=100.0, neginf=-100.0)
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            obs_buf,
            self.obs_buf,
        )

        if self.num_privileged_obs is not None:
            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            feet_pos_local = feet_pos - self.root_states[:, :3].unsqueeze(1)
            feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
            feet_contact_forces = self.contact_forces[:, self.feet_indices, :]
            self.privileged_obs_buf = torch.cat(
                (
                    obs_buf,
                    self.root_states[:, 2:3],
                    self.base_lin_vel,
                    feet_pos_local.reshape(self.num_envs, -1),
                    feet_vel.reshape(self.num_envs, -1),
                    feet_contact_forces.reshape(self.num_envs, -1),
                ),
                dim=-1,
            )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_observations, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:16] = 0.0
        noise_vec[16:28] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[28:40] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[40:44] = 0.0
        noise_vec[44:56] = 0.0
        noise_vec[56:68] = noise_scales.fatigue * noise_level / 10.0
        noise_vec[68:69] = 0.0   # pd_alpha: deterministic curriculum scalar, no noise
        return noise_vec

    def _compute_torques(self, actions):
        self._update_growth_scale()
        self._update_default_joint_pd_target()
        residual_torques = actions[:, :12] * self.cfg.control.action_scale

        rl_alpha = self.cfg.control.rl_prior_weight
        pd_alpha = self.cfg.control.pd_prior_weight
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

    def _update_default_joint_pd_target(self):
        # Phase-aligned PD prior (matches reward joint_angle_* targets):
        #   loaded → q_squat
        #   pushoff / landing → q_ground (default)
        #   airborne (~prelanding) → q_air (tucked)
        #   prelanding → q_pre (semi-extended)
        #   non-jumping → default_dof_pos
        self.default_joint_pd_target[:] = self.q_ground_target.unsqueeze(0)
        self.default_joint_pd_target[self.phase_loaded] = self.q_squat_target.unsqueeze(0)
        airborne_only = self.airborne & (~self.prelanding)
        self.default_joint_pd_target[airborne_only] = self.q_air_target.unsqueeze(0)
        self.default_joint_pd_target[self.prelanding] = self.q_pre_target.unsqueeze(0)

        use_default_pose = (~self.jumping_state) & (self.jump_starts <= 0.0)
        if self.cfg.commands.num_commands > 4:
            jump_command_active = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
            use_default_pose |= (~self.jumping_state) & (~jump_command_active)
        self.default_joint_pd_target[use_default_pose] = self.default_dof_pos.expand(self.num_envs, -1)[use_default_pose]

    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        collision_cutoff = (
            torch.sum(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1), dim=-1) > 0.2
        )
        roll, _, _ = get_euler_xyz(self.base_quat)
        roll = wrap_to_pi(roll)
        roll_cutoff = torch.abs(roll) > 2.4
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= collision_cutoff
        self.reset_buf |= roll_cutoff
        # CLEAN-TAKEOFF: end the episode on a load-phase foot RE-PLANT (stutter-step / run-up). The policy
        # can then only reach far via ONE clean push -> the dx curriculum SELF-LIMITS at the clean-jump
        # range instead of cheating distance with a shuffle (no artificial dx_final cap needed). Gated to
        # fire only AFTER discovery (clean_takeoff_min_step): early failed pushes also re-plant, and the
        # stutter only emerges much later, so the gate never blocks discovery. Off by default (other tasks).
        if getattr(self.cfg.rewards, "clean_takeoff_terminate", False):
            if self.common_step_counter >= int(getattr(self.cfg.rewards, "clean_takeoff_min_step", 60000)):
                self.reset_buf |= self.jump_replant
        # PITCH/TILT TERMINATION (hard, user request): the soft pitch penalties got traded off and the body
        # still descends/lands NOSE-DOWN (front-feet-first, topple risk). End the episode if the base pitches
        # nose-down beyond a hard limit during the descent/landing -> "stay level into touchdown" becomes
        # NON-NEGOTIABLE (lose the whole jump reward if you don't). projected_gravity[:,0] > 0 is nose-down;
        # nose-UP (lean-back, protective) is left free. Gated on the SAME succ-rate latch as the takeoff-omega
        # damping (_takeoff_omega_on) -> never kills the messy from-scratch jumps. Off by default (thr<=0).
        tilt_thr = float(getattr(self.cfg.rewards, "landing_tilt_terminate", 0.0))
        if tilt_thr > 0.0 and getattr(self, "_takeoff_omega_on", False):
            self.reset_buf |= self.landing & (self.projected_gravity[:, 0] > tilt_thr)
        if getattr(self.cfg.rewards, "one_jump_episode", False):
            self.reset_buf |= self.last_jump_success | self.jump_episode_failed
        elif getattr(self.cfg.rewards, "one_jump_reward_per_episode", False) and self.cfg.commands.num_commands > 4:
            post_jump_stand_steps = int(getattr(self.cfg.rewards, "post_jump_stand_steps", 80))
            self.reset_buf |= self.single_jump_command_done & (self.post_jump_step_counter >= post_jump_stand_steps)

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

        target = self.default_dof_pos.squeeze(0).clone()
        target[[1, 4, 7, 10]] = thigh
        target[[2, 5, 8, 11]] = calf
        return target

    def _reward_height_tracking(self):
        # Two-target design: squat by default, jump command during flight
        #   on ground (pre-jump / stance / pushoff / landing): squat height (~0.20m)
        #   in air (has_taken_off, not landed): jump height command
        # No nominal-standing target — robot's resting state is squat.
        if self.cfg.commands.num_commands > 4:
            jump_cmd_active = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        else:
            jump_cmd_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        squat_height = float(getattr(self.cfg.rewards, "stance_squat_height", 0.20))

        target = torch.full_like(self.root_states[:, 2], squat_height)
        flight_mask = self.has_taken_off & (~self.has_landed)
        target = torch.where(flight_mask, self.commands[:, 3], target)

        active = jump_cmd_active.float()
        height_error = torch.square(self.root_states[:, 2] - target)
        sigma = max(float(getattr(self.cfg.rewards, "height_tracking_sigma", 0.05)), 1e-4)
        return active * torch.exp(-height_error / sigma)

    def _reward_peak_height_progress(self):
        active = (self.jumping_state & (~self.has_landed)).float()
        air_ratio = self._get_air_foot_ratio()
        return active * air_ratio * self._get_height_progress(self.peak_base_height)

    def _reward_task_max_height(self):
        target_height = self.commands[:, 3]
        height_error = self.peak_base_height - target_height
        sigma = max(float(getattr(self.cfg.rewards, "task_max_height_sigma", 0.05)), 1e-3)
        jump_height_commanded = target_height >= 0.38
        if self.cfg.commands.num_commands > 4:
            jump_command_active = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        else:
            jump_command_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        reward = torch.exp(-torch.square(height_error) / sigma)
        active_jump = self.jumping_state & self.has_taken_off
        return active_jump.float() * jump_command_active.float() * jump_height_commanded.float() * reward

    def _get_height_progress(self, height=None):
        if height is None:
            height = self.root_states[:, 2]
        floor = torch.full_like(self.commands[:, 3], self.cfg.rewards.height_progress_floor)
        target = torch.maximum(self.commands[:, 3], floor + 1e-3)
        return torch.clamp((height - floor) / (target - floor), min=0.0, max=1.0)

    def _get_contact_state(self):
        threshold = float(getattr(self.cfg.rewards, "contact_force_threshold", 1.0))
        return self.contact_forces[:, self.feet_indices, 2] > threshold

    def _get_air_foot_ratio(self):
        return torch.mean((~self._get_contact_state()).float(), dim=1)

    def _squat_pose_err(self):
        # L1 distance from the current whole-body joint pose to the loaded squat pose q_squat
        # (a clean symmetric vertical fold, neutral hips). The single scalar that defines "squatted".
        return torch.sum(torch.abs(self.dof_pos - self.q_squat_target.unsqueeze(0)), dim=1)

    def _squat_deep_enough(self):
        # Squat-POSE gate (countermovement): True once this jump has HELD the loaded squat POSE for
        # >= squat_hold_steps consecutive steps before takeoff (latched in squat_qualified). The whole
        # jump-reward chain is withheld until then, so the only way to earn it is to actually get INTO
        # the squat and STAY there -- a pose target (vs the old base_z height) cannot be farmed by
        # face-planting / leg-splaying to drop base_z cheaply, and the hold requirement closes the
        # flick-through-for-one-frame hole. RSI air-drops exempt (start already in q_squat).
        # threshold<=0 -> always True (gate off; default, other tasks unaffected).
        thr = float(getattr(self.cfg.rewards, "squat_pose_threshold", 0.0))
        if thr <= 0.0:
            return torch.ones_like(self.jumping_state)
        return self.rsi_episode_mask | self.squat_qualified

    def _reward_takeoff_vertical_velocity(self):
        base_height = self.root_states[:, 2]
        min_height = float(getattr(self.cfg.rewards, "ascending_min_base_height", 0.18))
        vz = self.root_states[:, 9]
        ascending = self.jumping_state & (vz > 0) & (~self.has_landed) & (base_height > min_height)
        ascending = ascending & self._squat_deep_enough()   # squat-depth gate: no dip -> no takeoff reward
        # cmd-aware target: vz needed to reach cmd[3] from standing height
        h_stand = float(getattr(self.cfg.rewards, "stance_standing_height", 0.30))
        target_vz = torch.sqrt((2.0 * 9.81 * (self.commands[:, 3] - h_stand)).clamp(min=0.01))
        upward_velocity = torch.clamp(vz / target_vz, min=0.0, max=1.0)
        return ascending.float() * upward_velocity

    def _reward_projected_peak(self):
        # Olsen 2025: project peak height from current state (h + vz^2/2g) and reward closeness to target.
        # Gated on has_taken_off: ballistic formula h+vz²/2g only holds in free flight; allowing it in
        # stance lets policy game by spiking vz while feet still push the ground (no real liftoff).
        base_height = self.root_states[:, 2]
        min_height = float(getattr(self.cfg.rewards, "ascending_min_base_height", 0.18))
        vz = self.root_states[:, 9]
        ascending = (
            self.jumping_state
            & self.has_taken_off
            & (vz > 0)
            & (~self.has_landed)
            & (base_height > min_height)
        )
        ascending = ascending & self._squat_deep_enough()   # squat-depth gate (countermovement)
        projected = base_height + torch.clamp(vz, min=0.0) ** 2 / (2.0 * 9.81)
        target = self.commands[:, 3]
        sigma = max(float(getattr(self.cfg.rewards, "projected_peak_sigma", 0.05)), 1e-4)
        reward = torch.exp(-torch.square(projected - target) / sigma)
        return ascending.float() * reward

    def _reward_takeoff_impulse(self):
        active = self.jumping_state & (~self.has_taken_off)
        contact = self._get_contact_state()
        support = torch.mean(contact.float(), dim=1)
        vertical_force = torch.sum(torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0), dim=1)
        force_floor = float(self.cfg.rewards.takeoff_force_floor)
        force_target = max(float(self.cfg.rewards.takeoff_force_target), force_floor + 1e-3)
        force_reward = torch.clamp((vertical_force - force_floor) / (force_target - force_floor), min=0.0, max=1.0)
        vertical_acc = torch.clamp(
            ((self.root_states[:, 9] - self.last_root_vel[:, 2]) / self.dt)
            / max(float(self.cfg.rewards.takeoff_acc_target), 1e-3),
            min=0.0,
            max=1.0,
        )
        return active.float() * support * (0.5 * force_reward + 0.5 * vertical_acc)

    def _reward_all_feet_airborne(self):
        height_progress = self._get_height_progress()
        active = self.airborne & self._squat_deep_enough()   # squat-depth gate: no dip -> no flight reward
        return active.float() * (0.25 + 0.75 * height_progress)

    def _get_successful_jump_velocity_score(self):
        min_time = max(float(getattr(self.cfg.rewards, "success_velocity_min_airborne_time", 0.08)), 1e-3)
        flight_time = torch.clamp(self.airborne_time, min=min_time)
        displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        displacement[:, :2] = self.landing_root_xy - self.takeoff_root_xy
        avg_vel = quat_rotate_inverse(self.base_quat, displacement)[:, :2] / flight_time.unsqueeze(1)

        cmd_vel = self.commands[:, :2]
        tracking_error = torch.sum(torch.square(avg_vel - cmd_vel), dim=1)
        gain = float(getattr(self.cfg.rewards, "success_velocity_tracking_gain", 4.0))
        tracking_score = torch.exp(-gain * tracking_error)

        zero_cmd_error = torch.sum(torch.square(avg_vel), dim=1)
        zero_cmd_score = torch.exp(-gain * zero_cmd_error)
        min_command = float(getattr(self.cfg.rewards, "success_velocity_min_command", 0.05))
        xy_commanded = torch.norm(cmd_vel, dim=1) > min_command
        velocity_score = torch.where(xy_commanded, tracking_score, zero_cmd_score)

        min_score = float(getattr(self.cfg.rewards, "success_velocity_min_score", 0.20))
        return min_score + (1.0 - min_score) * velocity_score

    def _reward_successful_jump(self):
        return self.last_jump_success.float() * self.last_success_velocity_score

    def _reward_grounded_jump(self):
        contact = self._get_contact_state()
        all_feet_contact = torch.all(contact, dim=1)
        grace_elapsed = self.jump_step_counter > self.cfg.rewards.grounded_grace_steps
        stuck_on_ground = self.jumping_state & (~self.has_taken_off) & grace_elapsed & all_feet_contact
        return stuck_on_ground.float() * (0.5 + self._get_height_progress())

    def _reward_clean_landing(self):
        # CLEAN-LANDING penalty (user request, strong): after landing_planted latches (all 4 feet HELD for
        # clean_landing_plant_hold steps -> the touchdown impact chatter is skipped), penalize per-step ANY
        # foot off the ground (hop / shuffle-step) so the landing is ONE clean settle -- the soft mirror of
        # the clean-takeoff one-push rule (here a PENALTY, not a termination, so the successful_jump bonus is
        # kept and a late foot-shift is not catastrophic). Penalty magnitude ~ how long/much it hops.
        # Gated on the succ-rate latch (_takeoff_omega_on) -> never penalizes the messy from-scratch landings
        # (discovery-safe), the SAME gate as the takeoff-omega damping / jump_pitch_extra.
        if not getattr(self, "_takeoff_omega_on", False):
            return torch.zeros(self.num_envs, device=self.device)
        contact = self._get_contact_state()
        lifted = (contact.sum(dim=1) < 4).float()
        return self.landing_planted.float() * lifted

    def _reward_maintain_contact(self):
        # Atanassov-style: reward all-feet contact whenever not airborne.
        # No pushoff exclusion — when robot leaves the ground, all_feet naturally goes to 0.
        contact = self._get_contact_state()
        all_feet = torch.all(contact, dim=1)
        return (~self.airborne).float() * all_feet.float()

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
        sigma = max(float(getattr(self.cfg.rewards, "joint_symmetry_tracking_sigma", 0.25)), 1e-3)
        return self.jumping_state.float() * straight_gate * torch.exp(-(front_err + rear_err) / sigma)

    def _reward_tracking_linear_velocity(self):
        if getattr(self.cfg.rewards, "tracking_linear_velocity_all_time", False):
            # Always active throughout episode (matches OmniNet): velocity tracking
            # is not gated by jump command so reward continues after jump completes.
            vel_error = torch.sum(torch.square(self.root_states[:, 7:9] - self.commands[:, :2]), dim=1)
            sigma = max(float(getattr(self.cfg.rewards, "tracking_sigma", 0.25)), 1e-3)
            return torch.exp(-vel_error / sigma)
        takeoff_phase = (
            self.jumping_state
            & (~self.has_taken_off)
            & (self.root_states[:, 9] > self.cfg.rewards.lin_vel_takeoff_min_z_vel)
        )
        active = (takeoff_phase | self.airborne).float()
        vel_error = torch.sum(torch.square(self.base_lin_vel[:, :2] - self.commands[:, :2]), dim=1)
        return active * torch.exp(-self.cfg.rewards.velocity_tracking_gain * vel_error)

    def _reward_tracking_angular_velocity(self):
        # Track the commanded yaw rate (commands[2]) over the flight. OmniNet-style: when no yaw is
        # commanded (ang_vel_damp_zero_command=True), stay active anyway so the error term becomes wz^2 =
        # a yaw-rate DAMP that holds heading (= OmniNet's tracking_ang_vel with omega_cmd=0). A nonzero
        # command would be turning (Stage 2). Default (flag off / other configs) keeps the old behaviour:
        # only reward when a yaw rate is actually commanded.
        if getattr(self.cfg.rewards, "ang_vel_damp_zero_command", False):
            gate = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            gate = torch.abs(self.commands[:, 2]) > getattr(self.cfg.rewards, "ang_vel_tracking_min_command", 0.05)
        if getattr(self.cfg.rewards, "tracking_linear_velocity_all_time", False):
            if self.cfg.commands.num_commands > 4:
                jump_command_active = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
            else:
                jump_command_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            active = (jump_command_active & gate).float()
        elif getattr(self.cfg.rewards, "ang_vel_damp_airborne_only", False):
            # AIRBORNE only (after takeoff, before landing). Otherwise the reward is active during the
            # whole jumping_state INCLUDING the squat-but-not-taken-off phase, so a strong weight pays
            # the policy to SIT in the squat holding yaw still and never jump (discovery collapse, run
            # Jun10_12-47-57: yaw_r spiked to 0.14 exactly as flight_rate -> 0). Restricting to airborne
            # removes the "don't jump" optimum (no reward without leaving the ground) while still damping
            # the yaw spin where it actually accumulates (flight).
            active = (self.has_taken_off & (~self.has_landed) & gate).float()
        else:
            active = (self.jumping_state & (~self.has_landed) & gate).float()
        yaw_error = torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2])
        sigma = max(float(getattr(self.cfg.rewards, "tracking_sigma", 0.25)), 1e-3)
        return active * torch.exp(-yaw_error / sigma)

    def _reward_stand_still(self):
        if self.cfg.commands.num_commands > 4:
            jump_command_inactive = self.commands[:, 4] <= float(self.cfg.commands.jump_command_threshold)
        else:
            jump_command_inactive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        no_command = (
            (torch.norm(self.commands[:, :3], dim=1) < 0.05)
            & (torch.abs(self.commands[:, 3]) < 0.05)
            & jump_command_inactive
        )
        contact = self._get_contact_state()
        all_feet_contact = torch.all(contact, dim=1)
        penalized_body_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(self.penalised_contact_indices) > 0:
            penalized_body_contact |= torch.any(
                torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
                > self.cfg.rewards.stand_still_max_body_contact_force,
                dim=1,
            )
        if len(self.termination_contact_indices) > 0:
            penalized_body_contact |= torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
                > self.cfg.rewards.stand_still_max_body_contact_force,
                dim=1,
            )
        standing_contact = (
            all_feet_contact
            & (~penalized_body_contact)
            & (self.root_states[:, 2] >= self.cfg.rewards.stand_still_min_base_height)
        )
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel), dim=1)
        ang_vel_error = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        joint_error = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        height_error = torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target)
        min_height = float(self.cfg.rewards.stand_still_min_base_height)
        height_floor = max(min_height - 0.10, 1e-3)
        height_gate = torch.clamp((self.root_states[:, 2] - height_floor) / (min_height - height_floor), 0.0, 1.0)
        foot_contact_quality = torch.mean(contact.float(), dim=1)
        body_clear_quality = (~penalized_body_contact).float()
        contact_quality = 0.15 + 0.35 * foot_contact_quality + 0.50 * standing_contact.float()
        stability = (
            torch.exp(-self.cfg.rewards.stand_still_velocity_gain * lin_vel_error)
            * torch.exp(-self.cfg.rewards.stand_still_velocity_gain * ang_vel_error)
            * torch.exp(-self.cfg.rewards.stand_still_orientation_gain * orientation_error)
            * torch.exp(-self.cfg.rewards.stand_still_joint_gain * joint_error)
            * torch.exp(-self.cfg.rewards.stand_still_height_gain * height_error)
        )
        return (
            no_command.float()
            * stability
            * contact_quality
            * (0.20 + 0.80 * body_clear_quality)
            * (0.20 + 0.80 * height_gate)
        )

    def _reward_horizontal_drift(self):
        # Penalize world-frame horizontal velocity through the entire jump cycle (before landing).
        # Goal: keep jumps vertical when xy velocity command is zero.
        active = self.jumping_state & (~self.has_landed)
        horizontal_vel_sq = (
            torch.square(self.root_states[:, 7]) + torch.square(self.root_states[:, 8])
        )
        return active.float() * horizontal_vel_sq

    def _reward_default_pos(self):
        # mygo2jump-style L1 penalty toward q_squat (we want robot to bias toward squat posture).
        # Active throughout episode — small weight because push/flight legitimately deviates.
        joint_diff = torch.sum(torch.abs(self.dof_pos - self.q_squat_target.unsqueeze(0)), dim=1)
        return joint_diff

    def _reward_default_hip_pos(self):
        # mygo2jump-style: exp reward keeping 4 hip joints near their default values (no outward/inward drift).
        hip_indices = [0, 3, 6, 9]
        hip_diff = torch.sum(
            torch.abs(self.dof_pos[:, hip_indices] - self.default_dof_pos[hip_indices].unsqueeze(0)),
            dim=1,
        )
        return torch.exp(-hip_diff * 4.0)

    def _reward_takeoff_direction(self):
        vel = self.root_states[:, 7:10]
        vz = vel[:, 2]
        vel_norm = torch.norm(vel, dim=1)
        safe_norm = vel_norm.clamp(min=0.1)
        vertical_frac = torch.where(
            vel_norm > 0.1,
            vz / safe_norm,
            torch.zeros_like(vel_norm),
        )
        base_height = self.root_states[:, 2]
        min_height = float(getattr(self.cfg.rewards, "ascending_min_base_height", 0.18))
        ascending = self.jumping_state & (~self.has_landed) & (vz > 0) & (base_height > min_height)
        ascending = ascending & self._squat_deep_enough()   # squat-depth gate: no dip -> no takeoff reward
        return ascending.float() * vertical_frac

    def _reward_joint_angle_loaded(self):
        # Phase 1: fold legs during squat-down + pre-pushoff (loaded/spring-loaded posture).
        # Positive bell-curve reward — bounded above, smoother critic targets than linear |error|.
        active = self.phase_loaded.float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_squat_target.unsqueeze(0)), dim=1)
        sigma = max(float(getattr(self.cfg.rewards, "pose_guidance_sigma", 1.5)), 1e-3)
        return active * torch.exp(-pose_error / sigma)

    def _reward_joint_angle_extended(self):
        # Phase 2: straight legs through pushoff + flight + landing (extended posture).
        active = self.phase_extended.float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_ground_target.unsqueeze(0)), dim=1)
        sigma = max(float(getattr(self.cfg.rewards, "pose_guidance_sigma", 1.5)), 1e-3)
        return active * torch.exp(-pose_error / sigma)

    def _reward_joint_angle_aerial(self):
        active = (self.airborne & (~self.prelanding)).float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_air_target.unsqueeze(0)), dim=1)
        sigma = max(float(getattr(self.cfg.rewards, "pose_guidance_sigma", 1.5)), 1e-3)
        return active * torch.exp(-pose_error / sigma)

    def _reward_joint_angle_prelanding(self):
        active = self.prelanding.float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_pre_target.unsqueeze(0)), dim=1)
        sigma = max(float(getattr(self.cfg.rewards, "pose_guidance_sigma", 1.5)), 1e-3)
        return active * torch.exp(-pose_error / sigma)

    def _reward_joint_angle_landing(self):
        active = self.landing.float()
        pose_error = torch.sum(torch.abs(self.dof_pos - self.q_ground_target.unsqueeze(0)), dim=1)
        sigma = max(float(getattr(self.cfg.rewards, "pose_guidance_sigma", 1.5)), 1e-3)
        return active * torch.exp(-pose_error / sigma)

    def _reward_landing_stability(self):
        # Penalty for velocity during the landing observation period
        active = self.landing.float()
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel), dim=1)
        ang_vel_error = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return active * (
            torch.exp(-lin_vel_error / 0.25) * torch.exp(-ang_vel_error / 0.5)
        )

