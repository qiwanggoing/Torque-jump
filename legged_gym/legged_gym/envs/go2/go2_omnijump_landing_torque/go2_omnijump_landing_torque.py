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
        "base_ang_vel_xy",   # landing stability: flight+landing roll/pitch ω damping
        "landing_impact",    # landing stability: touchdown force-spike penalty
        "pitch_level",       # landing stability: pitch-specific tilt penalty
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
        "base_ang_vel_xy": 0,   # active from step 1 (curriculum disabled = one-stage)
        "landing_impact": 0,
        "pitch_level": 0,
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
        active = (self.airborne & (pz > min_h) & self._jump_commanded() & self._squat_deep_enough()).float()
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
        active = self.just_landed.float() * self._jump_commanded().float() * real_jump.float() * self._squat_deep_enough().float()
        err = torch.sum(torch.square(self.root_states[:, :2] - self.landing_target[:, :2]), dim=1)
        sigma = max(float(getattr(self.cfg.rewards, "sigma_pos_landing", 0.05)), 1e-4)
        return active * torch.exp(-err / sigma)

    def _reward_default_pos(self):
        # Strengthened pose anchor (weight raised in config) to hold posture after PD
        # has faded — the rear legs were drifting from the pose at pd_alpha=0 because
        # the RL never had to hold it itself (PD did). BUT zero it during the ground
        # push-off extension (jumping, not yet airborne, moving UP): there the legs must
        # extend BEYOND q_ground to launch, so penalizing that deviation would cap the
        # jump.
        #
        # ALSO zero it during the SQUAT-DOWN (jump commanded, not yet squatted to pose):
        # there default_joint_pd_target = q_squat, so this L1 penalty = -0.5*|stand - q_squat|
        # ~= -3.5/step the instant the jump is commanded -- bigger than every other term. That
        # turned the squat into "a penalty wall to flee", and the cheapest escape was to pop
        # (vz>0 -> pushoff exclusion zeroes it) instead of folding. So the policy popped to dodge
        # the penalty rather than squatting to earn the reward. Remove the wall: the dip is driven
        # purely by the POSITIVE stance_squat pull toward q_squat (guide to the target, do not
        # punish being mid-fold). Penalty resumes once squatted (-> q_ground target = encourages
        # the launch) and in the held flight/landing phases.
        l1 = torch.sum(torch.abs(self.dof_pos - self.default_joint_pd_target), dim=1)
        pushoff = self.jumping_state & (~self.has_taken_off) & (self.root_states[:, 9] > 0.0)
        squat_down = self.jumping_state & (~self.has_taken_off) & (~self._squat_deep_enough())
        return torch.where(pushoff | squat_down, torch.zeros_like(l1), l1)

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
        # Pose-guided countermovement (GUIDE to the target, don't block cheats). The earlier
        # height-based version rewarded "base_z low", which is degenerate -- a single scalar that
        # the policy farmed by face-planting forward (@3000) and splaying the legs sideways (@4900),
        # both of which drop base_z without a launchable squat. Whack-a-mole gating of each sprawl
        # never ends. Instead reward approaching the loaded squat POSE q_squat (a clean symmetric
        # vertical fold, neutral hips): a full-pose target has essentially ONE satisfying config,
        # so there is nothing degenerate to cheat -- only the clean squat scores, and base_z drops
        # as a consequence of the fold, not as the objective.
        #
        # Gate = jumping & ~has_taken_off & not-yet-in-pose (NO vz term -> no joint_angle_loaded
        # dead loop): drives stand -> squat from the moment the jump is commanded, then switches
        # off once the pose is reached so it never fights the leg extension during the push (the
        # squat-POSE gate _squat_deep_enough unlocks the jump chain from the same threshold).
        sigma = max(float(getattr(self.cfg.rewards, "squat_pose_sigma", 3.0)), 1e-4)
        reward = torch.exp(-self._squat_pose_err() / sigma)
        thr = float(getattr(self.cfg.rewards, "squat_pose_threshold", 0.0))
        # Keep paying the dip-shaping reward until the squat is QUALIFIED (held >= squat_hold_steps),
        # not merely touched: this is what gives the policy a reason to STAY folded through the hold
        # window instead of popping straight back up the instant pose_err first dips under thr.
        not_in_pose_yet = (~self.squat_qualified) if thr > 0.0 else torch.ones_like(self.jumping_state)
        active = self.jumping_state & (~self.has_taken_off) & not_in_pose_yet
        return active.float() * reward

    def _reward_base_ang_vel_xy(self):
        # Landing-stability lever borrowed from Olsen 2025 / Atanassov 2025 (papers): damp base
        # roll/pitch ANGULAR VELOCITY through flight AND landing so the body does not enter touchdown
        # already tumbling -- the root cause of the "lands then flips" failure. Returned as a positive
        # magnitude with a NEGATIVE scale (penalty), NOT a positive bell kernel: a kernel that only
        # fires during a short phase is cheap for the height/success drive to ignore (cf. joint_angle_*
        # sitting at ~0). It penalizes rotation RATE, not airborne time, so a CLEAN high jump (w~=0)
        # pays nothing -> it does not bias toward shorter/lower jumps. yaw (w_z) is excluded because it
        # may be commanded in Stage 2.
        active = (self.airborne | self.prelanding | self.landing).float()
        ang_vel_sq = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return active * ang_vel_sq

    def _reward_landing_impact(self):
        # Landing-stability lever borrowed from Olsen 2025 (papers: Ground force L2 / Soft impact):
        # penalize the hard vertical foot-force SPIKE during the landing window so touchdown is
        # cushioned instead of slammed. Bounded to [0,1] (saturates) and zero below ~standing weight
        # (landing_impact_force_floor) -> a soft floor, never an unbounded "wall" the policy flees.
        active = self.landing.float()
        fz = torch.sum(torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0), dim=1)
        floor = float(getattr(self.cfg.rewards, "landing_impact_force_floor", 150.0))
        norm = max(float(getattr(self.cfg.rewards, "landing_impact_force_norm", 1500.0)), 1e-3)
        excess = torch.clamp((fz - floor) / norm, min=0.0, max=1.0)
        return active * excess

    def _reward_pitch_level(self):
        # Strengthen PITCH attitude specifically. The base holds a persistent nose-down ("head-heavy")
        # pitch that the shared orientation term (-2.0, roll+pitch EQUALLY, hence weak on pitch alone)
        # does not flatten. projected_gravity[:,0] is the pitch tilt (0 when level, ~sin(theta) when
        # nose-down). Penalty (positive magnitude, negative scale) over the WHOLE jump cycle
        # (load -> push -> flight -> land) so the body is driven level throughout -- and since an
        # asymmetric front/rear push is what pitches the body, this also pressures a symmetric push.
        # Local to the landing task -> leaves the shared _reward_orientation untouched; stacks on top
        # of it so pitch ends up weighted more than roll. yaw/roll unaffected.
        active = self.jumping_state.float()
        pitch_tilt = torch.square(self.projected_gravity[:, 0])
        return active * pitch_tilt
