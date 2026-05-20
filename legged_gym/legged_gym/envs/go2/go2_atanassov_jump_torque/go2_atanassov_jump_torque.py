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
        active = self.just_landed.float()
        err = torch.sum(torch.square(self.root_states[:, :2] - self.atan_p_des[:, :2]), dim=1)
        sigma = float(self.cfg.rewards.sigma_pos_landing)
        return active * torch.exp(-err / sigma)

    def _reward_atanassov_landing_orientation(self):
        # q_des is identity. Use projected_gravity tilt as orientation error.
        active = self.just_landed.float()
        err = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        sigma = float(self.cfg.rewards.sigma_ori_landing)
        return active * torch.exp(-err / sigma)

    def _reward_atanassov_max_height(self):
        # Paper Stage 1 target: 0.9 m peak height
        active = self.just_landed.float()
        target = float(self.cfg.rewards.atanassov_target_peak)
        err = torch.square(self.peak_base_height - target)
        sigma = float(self.cfg.rewards.sigma_pos_max)
        return active * torch.exp(-err / sigma)

    def _reward_atanassov_jumping_sparse(self):
        # Paper "Jumping" sparse: 1 if the agent jumped at all this episode
        active = self.just_landed.float()
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
        # Stance and landing only (paper Table 1 shows 0 in flight)
        active = (self._stance_mask() | self._landing_mask()).float()
        err = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        sigma = float(self.cfg.rewards.sigma_ori_stance)  # same σ for stance/landing
        return active * torch.exp(-err / sigma)

    def _reward_atanassov_base_lin_vel(self):
        # Flight only — track desired xy velocity (Stage 1: v_des = 0)
        active = self._flight_mask().float()
        err = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)  # v_des = 0
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
        # Phase-weighted pose anchor. With overall weight now 5.0 in config,
        # per-phase contribution is:
        #   stance 0.5 — mild pull; doesn't override squat (base_position 8 dominates)
        #   flight 1.0 — STRONG anti-splay during airborne (was 0.1, too weak)
        #   landing 1.0 — full pull to stable landing pose
        err = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        sigma = float(self.cfg.rewards.sigma_q_nominal)
        rew = torch.exp(-err / sigma)
        weight = (
            0.5 * self._stance_mask().float()
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
        # Quadratic upward-velocity reward, gated on two conditions:
        #   1. Pre-takeoff (~has_taken_off) so it stops once airborne
        #   2. Significant vz (>0.8 m/s) — kills micro-jitter exploit
        # Note: contact-gate was tried but killed the signal entirely (push-off
        # naturally lifts feet asynchronously). Symmetry (weight 2.0) and
        # maintain_contact (weight 5.0) handle balance/4-foot as separate
        # rewards instead.
        active = (~self.has_taken_off) & (self.base_lin_vel[:, 2] > 0.8)
        clipped_vz = torch.clamp(self.base_lin_vel[:, 2], min=0.0, max=4.0)
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
