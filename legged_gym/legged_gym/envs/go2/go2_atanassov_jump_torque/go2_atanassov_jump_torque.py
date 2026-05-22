"""Atanassov 2025 Stage-1 (jumping in place) task.

Implements the Stage-1 portion of "Curriculum-Based Reinforcement Learning
for Quadrupedal Jumping: A Reference-Free Design" (IEEE RAM, June 2025) on
top of SATA's torque + PD-prior-fade infrastructure.

Key design elements (matching paper §STAGE 1, Figure 2, Table 1):
- Modified RSI: a fraction of episodes initialise with a random base height
  and upward velocity drawn from configurable ranges. This breaks the
  "standing in place" local optimum by dropping the agent into reward-rich
  airborne states from the very first step.
- Phase-aware base-position reward: stance target ``p_z = 0.20`` m (squat),
  flight target ``p_z = 0.7`` m (peak), landing target ``p_des`` (initial
  spawn xy projected onto the ground).
- Sparse task rewards (landing position / orientation, max height, jumping)
  fire only on the ``just_landed`` step.
- Multiplicative reward combination ``r_total = r^+ · exp(-||r^-||² / σ)``:
  regularization penalties scale down the positive task reward rather than
  subtracting from it, so the total is always non-negative and the policy
  can never minimise reward by simply not jumping.
- Strict termination: body collision, base height < 0.12, large orientation
  error, or large landing-position error all reset the episode.
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.envs.go2.go2_omnijump_torque.go2_omnijump_torque import GO2OmniJumpTorque
from legged_gym.envs.go2.go2_atanassov_jump_torque.go2_atanassov_jump_torque_config import (
    GO2AtanassovJumpTorqueCfg,
)


class GO2AtanassovJumpTorque(GO2OmniJumpTorque):
    cfg: GO2AtanassovJumpTorqueCfg

    # ------------------------------------------------------------------ #
    # Active reward set — paper Table 1 (Stage 1 in-place jump).
    # ------------------------------------------------------------------ #
    ACTIVE_REWARD_WHITELIST = {
        # Task (positive, summed into r^+)
        "atanassov_landing_position",
        "atanassov_landing_orientation",
        "atanassov_max_height",
        "atanassov_jumping_sparse",
        "atanassov_base_position",
        "atanassov_orientation_tracking",
        "atanassov_base_lin_vel",
        "atanassov_base_ang_vel",
        "atanassov_feet_clearance",
        "atanassov_symmetry",
        "atanassov_nominal_pose",
        "atanassov_maintain_contact",
        "atanassov_takeoff_vz",
        "orientation",  # parent's raw-form orientation penalty (curriculum-style); replaces atanassov_orientation_tracking
        "default_hip_pos",  # parent's exp(-gain × Σ|q_hip - q_hip_default|) — hip-specific anti-splay (mygo2jump-style, but target = configured default 0.1/-0.1)
        # Regularization (negative scale)
        "atanassov_energy",
        "atanassov_base_acceleration",
        "atanassov_contact_change",
        "atanassov_contact_forces",
        "atanassov_joint_limits",
        "action_rate",
        "dof_acc",
        "collision",   # paper Table 1 "Collisions": penalty for any non-foot body contact
    }

    # ====================================================================== #
    # Buffer / state initialisation
    # ====================================================================== #
    def _init_buffers(self):
        super()._init_buffers()

        # Desired landing pose (set per-episode in reset_idx)
        self.atan_p_des = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.atan_q_des = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.atan_q_des[:, 3] = 1.0   # identity quaternion (w=1)

        # Previous-step trackers used by regularization terms
        self.atan_last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device
        )
        self.atan_last_base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Reuse parent's last_dof_vel for joint accel computation; nothing else needed.

    # ====================================================================== #
    # Reset / RSI
    # ====================================================================== #
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        # After parent reset (which applied RSI and reset jump buffers),
        # capture the spawn xy as p_des and reset trackers.
        init_x = float(self.cfg.init_state.pos[0])
        init_y = float(self.cfg.init_state.pos[1])
        init_z = float(self.cfg.init_state.pos[2])
        self.atan_p_des[env_ids, 0] = self.env_origins[env_ids, 0] + init_x
        self.atan_p_des[env_ids, 1] = self.env_origins[env_ids, 1] + init_y
        self.atan_p_des[env_ids, 2] = self.env_origins[env_ids, 2] + init_z
        self.atan_q_des[env_ids] = 0.0
        self.atan_q_des[env_ids, 3] = 1.0

        self.atan_last_contacts[env_ids] = False
        self.atan_last_base_lin_vel[env_ids] = self.base_lin_vel[env_ids]

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)  # parent's SATA RSI is no-op (rsi_prob=0)
        if len(env_ids) == 0:
            return

        rsi_prob = float(getattr(self.cfg.rewards, "atanassov_rsi_prob", 0.0))
        if rsi_prob <= 0.0:
            return

        rsi_mask = torch.rand(len(env_ids), device=self.device) < rsi_prob
        rsi_ids = env_ids[rsi_mask]
        if len(rsi_ids) == 0:
            return

        h_min = float(self.cfg.rewards.atanassov_rsi_height_min)
        h_max = float(self.cfg.rewards.atanassov_rsi_height_max)
        self.root_states[rsi_ids, 2] = (
            self.env_origins[rsi_ids, 2]
            + torch_rand_float(h_min, h_max, (len(rsi_ids), 1), device=self.device).squeeze(1)
        )

        vz_min = float(self.cfg.rewards.atanassov_rsi_vel_z_min)
        vz_max = float(self.cfg.rewards.atanassov_rsi_vel_z_max)
        self.root_states[rsi_ids, 9] = torch_rand_float(
            vz_min, vz_max, (len(rsi_ids), 1), device=self.device
        ).squeeze(1)
        # Lateral velocity zero (vertical jump in place)
        self.root_states[rsi_ids, 7:9] = 0.0

        # Re-use parent's rsi_episode_mask so jumping_state activates for these envs
        self.rsi_episode_mask[rsi_ids] = True

        env_ids_int32 = rsi_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    # ====================================================================== #
    # Termination — contact-based only.
    # Parent's check_termination already terminates on contact with the body
    # parts listed in asset.terminate_after_contacts_on (base, trunk, hip,
    # thigh). We additionally enforce a minimum base height so the robot
    # can't flatten itself onto the floor and farm landing reward.
    # Removed (vs old version): tilt-based termination and landing-distance
    # termination — those were preventing the policy from exploring push-off
    # motions that briefly tilt the base or drift sideways.
    # ====================================================================== #
    def check_termination(self):
        super().check_termination()
        too_low = self.root_states[:, 2] < float(self.cfg.rewards.atanassov_terminate_base_height)
        self.reset_buf |= too_low

    # ====================================================================== #
    # Reward combination: standard legged_gym linear sum.
    # We deviate from Atanassov's multiplicative ``r_total = r^+ · exp(-||r^-||²/σ)``
    # because the paper's σ was not specified and our σ guess (5.0) caused the
    # exp kernel to saturate when regularization penalties grew, killing the
    # PPO gradient and letting noise_std diverge (saw σ_action ≈ 128 by iter
    # 1500). Standard linear sum is the legged_gym default and keeps PPO stable.
    # ====================================================================== #
    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            scale = self.reward_scales[name]
            if scale == 0.0:
                continue
            rew = self.reward_functions[i]() * scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if getattr(self.cfg.rewards, "only_positive_rewards", False):
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

        # Update last_base_lin_vel for base_acceleration reward (next-step diff)
        self.atan_last_base_lin_vel[:] = self.base_lin_vel
        # Update last contacts for contact_change reward (next-step diff)
        contact_now = self._get_contact_state()
        self.atan_last_contacts[:] = contact_now

    # ====================================================================== #
    # Linear PD-fade schedule (override base's Gompertz curve).
    # Base GO2Torque._update_growth_scale uses Gompertz which doesn't honour
    # warmup_steps and concentrates the fade in a narrow sigmoid window
    # around x0. We use the same warmup+linear-ramp layout as
    # GO2OmniJumpCurriculumTorque for predictable scheduling:
    #   step < warmup_steps:                general_scale = 0  (PD locked full)
    #   warmup_steps ≤ step ≤ x0:           ramps linearly 0 → 1
    #   step > x0:                          general_scale = 1  (PD fully faded)
    # ====================================================================== #
    def _update_growth_scale(self):
        self.step_count += 1
        if self.cfg.control.control_type == "T" or self.cfg.test.use_test:
            from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfgPPO
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

    # ====================================================================== #
    # PD-fade torque computation (override base which uses constant pd_alpha).
    # ``GO2OmniJumpTorque._compute_torques`` reads ``pd_prior_weight`` and
    # ``rl_prior_weight`` as constants — so the growth schedule (warmup_steps,
    # x0) doesn't actually fade PD. We override here to make pd_alpha scale
    # with ``general_scale`` (1 - g): full pd_prior_weight at warmup, 0 once
    # the fade window completes.
    # ====================================================================== #
    def _compute_torques(self, actions):
        self._update_growth_scale()
        self._update_default_joint_pd_target()
        residual_torques = actions[:, :12] * self.cfg.control.action_scale

        # PD fades from pd_prior_weight → 0 as general_scale 0 → 1
        pd_alpha = float(self.cfg.control.pd_prior_weight) * max(0.0, 1.0 - float(self.general_scale))
        rl_alpha = 1.0 - pd_alpha

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

    # ====================================================================== #
    # Phase-aware PD target (SATA-style assist during PD-fade phase).
    # Parent computes a single PD blend toward ``default_joint_pd_target`` each
    # step. We override the target each step so the PD prior physically pulls
    # the robot through the squat → tuck → stand sequence Atanassov's reward
    # design assumes. As PD fades, this assist disappears and the policy must
    # have internalised the sequence.
    # ====================================================================== #
    def _update_default_joint_pd_target(self):
        # Default: standing pose (robot idle / between jumps / fallback)
        self.default_joint_pd_target[:] = self.default_dof_pos.expand(self.num_envs, -1)
        # Phase masks come from parent's _update_jump_state (one-step lag but fine)
        stance = self._stance_mask()
        flight = self._flight_mask()
        landing = self._landing_mask()
        # Per-phase IK-derived targets (parent's _init_buffers computes these)
        if stance.any():
            self.default_joint_pd_target[stance] = self.q_squat_target.unsqueeze(0)
        if flight.any():
            self.default_joint_pd_target[flight] = self.q_air_target.unsqueeze(0)
        if landing.any():
            self.default_joint_pd_target[landing] = self.q_ground_target.unsqueeze(0)

    # ====================================================================== #
    # Phase mask helpers
    # ====================================================================== #
    def _stance_mask(self):
        # Jump command active, robot still on ground, has not yet landed
        return self.jumping_state & (~self.has_taken_off) & (~self.has_landed)

    def _flight_mask(self):
        return self.airborne

    def _landing_mask(self):
        return self.has_landed

    # ====================================================================== #
    # Reward functions — Atanassov Table 1
    # All return non-negative values in [0, 1] for task rewards (exp kernel)
    # or non-negative raw values for regularization.
    # ====================================================================== #

    # ---- Sparse task rewards (fire once per episode on the just_landed step) ----
    def _reward_atanassov_landing_position(self):
        # Gate by cmd[4]>0.5: RSI episodes with cmd[4]=0 land airborne robots and would
        # otherwise leak this reward into stand-episode training.
        jump_commanded = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        active = self.just_landed.float() * jump_commanded.float()
        err = torch.sum(torch.square(self.root_states[:, :2] - self.atan_p_des[:, :2]), dim=1)
        sigma = float(self.cfg.rewards.sigma_pos_landing)
        return active * torch.exp(-err / sigma)

    def _reward_atanassov_landing_orientation(self):
        # q_des is identity. Use projected_gravity tilt as orientation error.
        # Same cmd[4] gate as landing_position to keep stand episodes clean.
        jump_commanded = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        active = self.just_landed.float() * jump_commanded.float()
        err = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        sigma = float(self.cfg.rewards.sigma_ori_landing)
        return active * torch.exp(-err / sigma)

    def _reward_atanassov_max_height(self):
        # Paper Stage 1 target: fixed peak height (atanassov_target_peak, currently 0.6m).
        # cmd[4]>0.5 gate stops RSI episodes (cmd[4]=0, jumping_state bootstrapped airborne)
        # from leaking jump reward into stand-episode training.
        jump_commanded = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        active = self.just_landed.float() * jump_commanded.float()
        target = float(self.cfg.rewards.atanassov_target_peak)
        err = torch.square(self.peak_base_height - target)
        sigma = float(self.cfg.rewards.sigma_pos_max)
        return active * torch.exp(-err / sigma)

    def _reward_atanassov_jumping_sparse(self):
        # Paper "Jumping" sparse: 1 if the agent jumped at all this episode.
        # Gated by cmd[4]>0.5: RSI bootstraps jumping_state=True with has_taken_off in cmd[4]=0
        # episodes; without this gate the policy gets paid for "jumping" in stand episodes.
        jump_commanded = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        active = self.just_landed.float() * jump_commanded.float()
        return active * self.has_taken_off.float()

    # ---- Dense phase-aware task rewards ----
    def _reward_atanassov_base_position(self):
        stance = self._stance_mask().float()
        flight = self._flight_mask().float()
        landing = self._landing_mask().float()

        pz = self.root_states[:, 2]
        sigma_st = float(self.cfg.rewards.sigma_pz_stance)
        sigma_fl = float(self.cfg.rewards.sigma_pz_flight)
        sigma_la = float(self.cfg.rewards.sigma_pos_landing)

        h_st = float(self.cfg.rewards.atanassov_stance_height)
        h_fl = float(self.cfg.rewards.atanassov_flight_height)

        r_st = torch.exp(-torch.square(pz - h_st) / sigma_st)
        r_fl = torch.exp(-torch.square(pz - h_fl) / sigma_fl)
        r_la = torch.exp(
            -torch.sum(torch.square(self.root_states[:, :3] - self.atan_p_des), dim=1) / sigma_la
        )
        return stance * r_st + flight * r_fl + landing * r_la

    def _reward_atanassov_orientation_tracking(self):
        # All phases (stance + flight + landing). Paper Table 1 sets flight to
        # 0 to allow somersault-style rotations, but we want vertical jump in
        # place and our torque-direct policy lacks paper's PD@10kHz stabiliser,
        # so the in-air orientation needs an explicit anchor against roll/pitch.
        active = (
            self._stance_mask() | self._flight_mask() | self._landing_mask()
        ).float()
        err = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        sigma = float(self.cfg.rewards.sigma_ori_stance)
        return active * torch.exp(-err / sigma)

    def _reward_atanassov_base_lin_vel(self):
        # Flight only — penalise world-frame xy velocity (Stage 1: v_des = 0).
        # IMPORTANT: world frame via root_states[:, 7:9], NOT body-frame
        # base_lin_vel[:, :2]. Body-frame xy velocity stays ≈ 0 when the
        # robot tilts and pushes purely along body-z, so the previous
        # body-frame version completely missed tilt-induced lateral motion.
        # OmniJump open-source code uses world frame here for the same reason.
        active = self._flight_mask().float()
        err = torch.sum(torch.square(self.root_states[:, 7:9]), dim=1)
        sigma = float(self.cfg.rewards.sigma_v_flight)
        return active * torch.exp(-err / sigma)

    def _reward_atanassov_base_ang_vel(self):
        # Flight (full weight) + Landing (0.1× weight) — track ω_des = 0
        flight = self._flight_mask().float()
        landing = self._landing_mask().float()
        err = torch.sum(torch.square(self.base_ang_vel), dim=1)
        sigma = float(self.cfg.rewards.sigma_omega)
        rew = torch.exp(-err / sigma)
        return flight * rew + 0.1 * landing * rew

    def _reward_atanassov_feet_clearance(self):
        # Flight only. Paper: ||p_feet - p^nom_feet + [0, 0, -0.15]||²
        # encourages legs tucked in close to body in xy, and feet ~15 cm below CoM in z.
        active = self._flight_mask().float()
        # Feet positions relative to base, in base frame
        feet_world = self.rigid_body_states[:, self.feet_indices, :3]
        base_pos = self.root_states[:, :3].unsqueeze(1)
        rel = feet_world - base_pos
        # Rotate into body frame for each foot
        rel_body = torch.zeros_like(rel)
        for i in range(len(self.feet_indices)):
            rel_body[:, i, :] = quat_rotate_inverse(self.base_quat, rel[:, i, :])
        # Nominal body-frame foot positions (paper §FEET_CLEARANCE: nominal joint
        # positions q^nom map to a specific foot offset). For Stage 1 we use the
        # default-pose foot heights ~15 cm below CoM as the paper formula bakes in.
        target_offset = torch.tensor([0.0, 0.0, -0.15], device=self.device).view(1, 1, 3)
        # The reward is a penalty-style term (larger error = larger value). Caller
        # treats this as positive task reward — but since errors are bounded by
        # gravity, an exp kernel keeps it stable.
        err = torch.sum(torch.square(rel_body - target_offset), dim=-1).sum(dim=-1)  # sum over 4 feet
        return active * torch.exp(-err / 0.1)

    def _reward_atanassov_symmetry(self):
        # Paper: w_sym · Σ_joint |q_left - q_right|²
        # Encode mirror symmetry: hip joints add (sign-flipped); thigh/calf subtract.
        q = self.dof_pos
        # FL/FR pair (indices 0-2 / 3-5): FL_hip+FR_hip, FL_thigh-FR_thigh, FL_calf-FR_calf
        # RL/RR pair (indices 6-8 / 9-11): same pattern
        err = (
            torch.square(q[:, 0] + q[:, 3])
            + torch.square(q[:, 1] - q[:, 4])
            + torch.square(q[:, 2] - q[:, 5])
            + torch.square(q[:, 6] + q[:, 9])
            + torch.square(q[:, 7] - q[:, 10])
            + torch.square(q[:, 8] - q[:, 11])
        )
        # Pass through exp kernel so it lives in [0,1] as a task-style reward.
        return torch.exp(-err / 0.5)

    def _reward_atanassov_nominal_pose(self):
        # Phase-weighted pose anchor + idle term (covers pure-stand episodes where
        # all jump phase masks are False). exp(-||q - q_default||²/sigma) form.
        err = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        sigma = float(self.cfg.rewards.sigma_q_nominal)
        rew = torch.exp(-err / sigma)
        idle = (~self.jumping_state) & (~self.has_landed)
        weight = (
            1.0 * idle.float()
            + 0.5 * self._stance_mask().float()
            + 1.0 * self._flight_mask().float()
            + 1.0 * self._landing_mask().float()
        )
        return weight * rew

    def _reward_atanassov_maintain_contact(self):
        # Stance only — reward keeping ALL 4 feet in contact during pre-jump.
        # Strict (was averaged): only 4-feet-on-ground gets full reward; 3 feet
        # or fewer = 0. Prevents the "tilted push" exploit (lift one foot, push
        # with others to gain vz).
        active = self._stance_mask().float()
        contact = self._get_contact_state()
        all_four = torch.all(contact, dim=1).float()
        return active * all_four

    def _reward_atanassov_takeoff_vz(self):
        # Quadratic upward-velocity reward, gated on three conditions:
        #   1. Pre-takeoff (~has_taken_off) so it stops once airborne
        #   2. Significant vz (>0.8 m/s) — kills micro-jitter exploit
        #   3. cmd[4]>0.5 — RSI bootstrap gives vz up to 3 m/s for free in stand
        #      episodes (cmd[4]=0); without this gate the policy gets paid for
        #      that bootstrap velocity it didn't earn.
        # World-frame vz (root_states[:, 9]) — body-frame would let tilted-push exploit grow.
        vz_world = self.root_states[:, 9]
        jump_commanded = self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        active = (~self.has_taken_off) & (vz_world > 0.8) & jump_commanded
        clipped_vz = torch.clamp(vz_world, min=0.0, max=4.0)
        return active.float() * torch.square(clipped_vz)

    # ---- Regularization rewards (negative scale; squared into r^-) ----
    def _reward_atanassov_energy(self):
        # τ · q̇  — absolute power per joint, summed
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_atanassov_base_acceleration(self):
        # |v̇|² where v̇ ≈ (v_t - v_{t-1}) / dt
        acc = (self.base_lin_vel - self.atan_last_base_lin_vel) / self.dt
        return torch.sum(torch.square(acc), dim=1)

    def _reward_atanassov_contact_change(self):
        contact_now = self._get_contact_state()
        diff = (contact_now != self.atan_last_contacts).float()
        return torch.sum(diff, dim=1)

    def _reward_atanassov_contact_forces(self):
        # Flight only — penalize variance of feet contact forces (should all be 0 in flight)
        active = self._flight_mask().float()
        forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)  # [N, 4]
        mean_f = torch.mean(forces, dim=1, keepdim=True)
        return active * torch.sum(torch.abs(forces - mean_f), dim=1)

    def _reward_atanassov_joint_limits(self):
        # Paper: w_qlim · Σ_joint (q_j outside soft limit)²
        out = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # below lower
        out += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)   # above upper
        return torch.sum(torch.square(out), dim=1)
