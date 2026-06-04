"""Landing-point tracking on top of the proven vertical-jump curriculum env.

See go2_omnijump_landing_torque_config.py for the full design rationale. In
short: inherit the validated GO2OmniJumpCurriculumTorque jump machinery
unchanged and add a thin landing layer (landing target, yaw-frame error
observation, dense projected-landing + sparse landing-position rewards).
"""

import torch
from isaacgym.torch_utils import get_euler_xyz

from legged_gym.envs.go2.go2_omnijump_curriculum_torque.go2_omnijump_curriculum_torque import (
    GO2OmniJumpCurriculumTorque,
)
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque_config import (
    GO2OmniJumpLandingTorqueCfg,
)


class GO2OmniJumpLandingTorque(GO2OmniJumpCurriculumTorque):
    cfg: GO2OmniJumpLandingTorqueCfg

    # Inherit the proven jump-driving reward set and add the two landing terms.
    ACTIVE_REWARD_WHITELIST = GO2OmniJumpCurriculumTorque.ACTIVE_REWARD_WHITELIST | {
        "landing_position",
        "projected_landing",
        "foot_contact_sync",
        "stance_squat",
    }

    # Curriculum gate table requires an entry for every active reward. Curriculum
    # is disabled (one-stage), so the stage value only needs to exist; 0 = active
    # from step 1 alongside the rest of the regularisation/task stack.
    REWARD_START_STAGES = {
        **GO2OmniJumpCurriculumTorque.REWARD_START_STAGES,
        "landing_position": 1,
        "projected_landing": 1,
        "foot_contact_sync": 0,
        "stance_squat": 0,
    }

    # ------------------------------------------------------------------ #
    # Buffers
    # ------------------------------------------------------------------ #
    def _init_buffers(self):
        super()._init_buffers()
        # World-frame desired landing xy, set each reset from spawn + commanded
        # displacement. Initialised to spawn so step-0 obs is well-defined.
        self.landing_target = self.root_states[:, :2].clone()

        # Stage 2: widen the displacement ranges that the parent's
        # _resample_commands draws commands[0:2] from. Stage 1 leaves them [0,0]
        # (land in place -> behaviour identical to the proven vertical jumper).
        if int(getattr(self.cfg.commands, "landing_stage", 1)) >= 2:
            self.command_ranges["lin_vel_x"] = list(self.cfg.commands.landing_disp_x_stage2)
            self.command_ranges["lin_vel_y"] = list(self.cfg.commands.landing_disp_y_stage2)

    # ------------------------------------------------------------------ #
    # Reset — set landing target = spawn xy + commanded displacement.
    # The robot spawns facing +x (identity heading) so the displacement maps
    # directly to world xy. Stage 1: commands[0:2] == 0 -> target == spawn.
    # ------------------------------------------------------------------ #
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)  # parent resamples commands[0:2] (the displacement) first
        init_x = float(self.cfg.init_state.pos[0])
        init_y = float(self.cfg.init_state.pos[1])
        self.landing_target[env_ids, 0] = self.env_origins[env_ids, 0] + init_x + self.commands[env_ids, 0]
        self.landing_target[env_ids, 1] = self.env_origins[env_ids, 1] + init_y + self.commands[env_ids, 1]

    # ------------------------------------------------------------------ #
    # Observations — identical to the parent layout except the velocity-command
    # slot (commands[:, :3] * commands_scale) is replaced by the yaw-frame
    # landing-point error  Ryaw^T (p* - p_base) = [fwd_err, lat_err, 0].
    # Only the yaw component of orientation is used so pitch/roll in flight do
    # not scramble the target direction. 69-dim layout + mirror parity preserved.
    # ------------------------------------------------------------------ #
    def compute_observations(self):
        foot_contact_obs = self._get_contact_state().float()
        motor_fatigue = self.motor_fatigue.detach()

        err_world = self.landing_target[:, :2] - self.root_states[:, :2]
        _, _, yaw = get_euler_xyz(self.base_quat)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        err_fwd = cos_y * err_world[:, 0] + sin_y * err_world[:, 1]
        err_lat = -sin_y * err_world[:, 0] + cos_y * err_world[:, 1]
        landing_err_obs = torch.stack((err_fwd, err_lat, torch.zeros_like(err_fwd)), dim=-1)

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
                landing_err_obs,                    # <- replaces commands[:, :3] * commands_scale
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

    # ------------------------------------------------------------------ #
    # Landing rewards (new). The jump-driving stack (takeoff_vz, projected_peak,
    # successful_jump, ...) is inherited from the parent untouched.
    # ------------------------------------------------------------------ #
    def _jump_commanded(self):
        if self.cfg.commands.num_commands > 4:
            return self.commands[:, 4] > float(self.cfg.commands.jump_command_threshold)
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _reward_projected_landing(self):
        # Olsen 2025 densification: while airborne, project the final landing xy
        # from ballistic motion and reward closeness to the commanded landing
        # point. Converts the otherwise terminal-only landing signal into a dense
        # in-flight gradient.
        #   t_land solves  pz + vz·t - ½g·t² = h_land  (later/descending root).
        #
        # HEIGHT GATE (pz > min): the `airborne` flag alone (feet off ground) is
        # NOT enough — it is farmable. A legs-tucked sprawl (feet off the ground,
        # body collapsed to ~0.13m over the spawn point) keeps `airborne` True, so
        # the projected landing trivially ≈ target and this dense reward pays
        # ~1/step forever without any real jump (observed: rew exploded 0.2→5.5,
        # mean_peak collapsed to 0.13). Requiring the body above standing height
        # mirrors the real-jump peak gate on the sparse landing_position term:
        # a genuine jump cannot hover up there, so the farm is impossible while
        # the dense in-place landing control (Stage 1, target=spawn) is preserved.
        pz = self.root_states[:, 2]
        min_h = float(getattr(self.cfg.rewards, "projected_landing_min_height", 0.40))
        active = (self.airborne & (pz > min_h) & self._jump_commanded()).float()
        g = 9.81
        vz = self.root_states[:, 9]
        h_land = self.env_origins[:, 2] + float(self.cfg.rewards.base_height_target)
        disc = torch.clamp(vz * vz + 2.0 * g * (pz - h_land), min=0.0)
        t_land = (vz + torch.sqrt(disc)) / g
        proj_xy = self.root_states[:, :2] + self.root_states[:, 7:9] * t_land.unsqueeze(1)
        err = torch.sum(torch.square(proj_xy - self.landing_target[:, :2]), dim=1)
        sigma = max(float(getattr(self.cfg.rewards, "sigma_landing_proj", 0.10)), 1e-4)
        return active * torch.exp(-err / sigma)

    def _reward_landing_position(self):
        # Sparse terminal reward: fires on the touchdown step, exp kernel on the
        # actual landing xy vs the commanded landing point. Gated by a real-jump
        # peak so the body must genuinely rise (no farming via a tucked fake jump).
        min_peak = float(getattr(self.cfg.rewards, "landing_real_jump_min_peak", 0.40))
        real_jump = self.peak_base_height >= min_peak
        active = self.just_landed.float() * self._jump_commanded().float() * real_jump.float()
        err = torch.sum(torch.square(self.root_states[:, :2] - self.landing_target[:, :2]), dim=1)
        sigma = max(float(getattr(self.cfg.rewards, "sigma_pos_landing", 0.05)), 1e-4)
        return active * torch.exp(-err / sigma)

    def _reward_default_pos(self):
        # Strengthened pose anchor (weight raised in config) to hold posture after PD
        # has faded — the rear legs were drifting from the pose at pd_alpha=0 because
        # the RL never had to hold it itself (PD did). BUT zero it during the ground
        # push-off extension (jumping, not yet airborne, moving UP): there the legs must
        # extend BEYOND q_ground to launch, so penalizing that deviation would cap the
        # jump. Full strength in the held phases (stand / squat-load / flight-tuck /
        # landing) where holding the pose is exactly the goal; off only during the
        # explosive upward push (front-rear coordination there is handled by
        # pushoff_leg_sync, which stays active).
        l1 = torch.sum(torch.abs(self.dof_pos - self.default_joint_pd_target), dim=1)
        pushoff = self.jumping_state & (~self.has_taken_off) & (self.root_states[:, 9] > 0.0)
        return torch.where(pushoff, torch.zeros_like(l1), l1)

    def _reward_foot_contact_sync(self):
        # Four-foot CONTACT-timing sync: all four feet should LEAVE the ground together at
        # takeoff and TOUCH DOWN together at landing. Staggered contact (1-3 feet on the
        # ground = "mixed") at those transitions tilts the body (pitch/roll).
        #
        # Implemented as a PENALTY on the mixed state (returns 1 when 1-3 feet touch, 0 when
        # all-same = all-off or all-on); used with a NEGATIVE weight. A penalty shapes cleanly
        # (0 when synced) whereas a positive "all-together" reward is saturated at 1 most of
        # the time and barely shapes. Active in the ground-transition windows only:
        #   - pre-takeoff push (jumping & not yet taken off) -> catches staggered LIFT-OFF;
        #   - landing buffer (self.landing) -> catches staggered TOUCH-DOWN.
        # Direction-agnostic (no straight gate): a forward Stage-2 jump also wants a clean
        # simultaneous takeoff/landing.
        contact = self._get_contact_state()
        num = contact.sum(dim=1)
        mixed = ((num > 0) & (num < 4)).float()
        active = (self.jumping_state & (~self.has_taken_off)) | self.landing
        return active.float() * mixed

    def _reward_stance_squat(self):
        # Atanassov-style stance squat — the dense countermovement shaper we were missing.
        # While commanded to jump but still on the ground (jumping & not yet taken off),
        # reward the base descending toward the squat height (~0.20m). This pulls the policy
        # OUT of the "stand-and-pop" local optimum (Atanassov: "squatting down to a height of
        # 0.2 m while on the ground"): a deeper dip lengthens the push stroke (~0.09m -> ~0.20m),
        # so the body accelerates over a longer distance and takes off faster -> jumps higher.
        #
        # Gate = jumping & ~has_taken_off, with NO vz<=0 term. That vz<=0 (phase_loaded) was the
        # dead loop in joint_angle_loaded: it only fired once ALREADY dipping, so nothing ever
        # drove the dip. Rewarding the whole pre-takeoff window lets the dip emerge on its own.
        #
        # Height-based (not pose-based): during the explosive push the base rises and this reward
        # smoothly decays to ~0 instead of fighting leg extension — takeoff_vertical_velocity /
        # projected_peak take over there. Switches off the instant the feet leave the ground.
        squat_height = float(getattr(self.cfg.rewards, "stance_squat_height", 0.20))
        sigma = max(float(getattr(self.cfg.rewards, "stance_squat_sigma", 0.02)), 1e-4)
        base_z = self.root_states[:, 2] - self.env_origins[:, 2]
        reward = torch.exp(-torch.square(base_z - squat_height) / sigma)
        active = self.jumping_state & (~self.has_taken_off)
        return active.float() * reward
