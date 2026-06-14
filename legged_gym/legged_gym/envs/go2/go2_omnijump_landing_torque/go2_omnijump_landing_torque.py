"""Landing-point tracking on top of the proven vertical-jump curriculum env.

See go2_omnijump_landing_torque_config.py for the full design rationale. In
short: inherit the validated GO2OmniJumpCurriculumTorque jump machinery
unchanged and add a thin landing layer (landing target, yaw-frame error
observation, dense projected-landing + sparse landing-position rewards).
"""

import math

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
        "dof_pos_limits",    # penalize folding joints to the limit (over-deep squat hits the "wall")
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
        "dof_pos_limits": 0,
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
        stage2 = int(getattr(self.cfg.commands, "landing_stage", 1)) >= 2
        # Curriculum only ever runs in Stage 2 (so the EMA buffers below are always initialised
        # before _update_dx_curriculum reads them).
        self.landing_dx_curriculum = stage2 and bool(getattr(self.cfg.commands, "landing_dx_curriculum", False))
        self.landing_dx_max = float(getattr(self.cfg.commands, "landing_dx_final", 0.40))
        if stage2:
            self.command_ranges["lin_vel_y"] = list(self.cfg.commands.landing_disp_y_stage2)
            # Reduce time spent IDLING. Each resample (every resampling_time s) draws commands[4] and
            # STANDS if it is <= jump_command_threshold (0.5). The default range [0,1] -> 50% stand
            # per resample. Narrow it so the per-resample stand prob ~= 0.1, so the robot jumps almost
            # every resample (the IMPORTANT standing — returning to a stable stand AFTER landing — is
            # still trained in every jump episode's post-landing buffer).
            jc = getattr(self.cfg.commands, "jump_command_range", None)
            if jc is not None:
                self.command_ranges["jump_command"] = list(jc)
            if self.landing_dx_curriculum:
                # DISTANCE CURRICULUM (Atanassov 2025 local-difficulty): start the forward dx
                # range tiny (default 0 = pure in-place = the proven vertical-jump discovery)
                # and grow the upper bound ONLY after the policy STABLY LANDS ON TARGET at the
                # current distance (combined hit-AND-success rate, see _update_dx_curriculum).
                # Sidesteps the from-scratch discovery cliff: at dx=0 the landing target IS the
                # spawn, so the (tight-sigma) landing reward is fully available during discovery.
                self.landing_dx_max = float(getattr(self.cfg.commands, "landing_dx_start", 0.0))
                self.command_ranges["lin_vel_x"] = [0.0, self.landing_dx_max]
                self._far_stable_sum = 0.0   # cumulative far-band stable-hits since last advance
                self._far_n_sum = 0.0        # cumulative far-band attempts since last advance
                self._dx_last_advance_step = 0
            else:
                self.command_ranges["lin_vel_x"] = list(self.cfg.commands.landing_disp_x_stage2)

        # Per-episode accumulator: jumps that landed within tol of the commanded landing point
        # (mirrors successful_jumps; consumed ONLY by the distance-curriculum advance gate).
        self.jump_target_hits = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Commanded displacement magnitude of the episode's jump (recorded at touchdown), used to
        # filter the FAR-BAND advance gate (only jumps near dx_max count).
        self._last_jump_cmd_dx = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

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
        self._update_dx_curriculum()

    # ------------------------------------------------------------------ #
    # Distance curriculum: jump-stat plumbing + advance logic.
    # ------------------------------------------------------------------ #
    def _reset_jump_buffers(self, env_ids):
        super()._reset_jump_buffers(env_ids)
        if hasattr(self, "jump_target_hits"):
            self.jump_target_hits[env_ids] = 0.0

    def _update_jump_state(self):
        super()._update_jump_state()
        # Accumulate "landed ON the commanded point" for the curriculum gate: a REAL jump
        # (peak gate, same as landing_position) whose recorded touchdown xy (self.landing_root_xy,
        # set by the parent at just_landed) is within tol of self.landing_target.
        if hasattr(self, "jump_target_hits") and torch.any(self.just_landed):
            tol = float(getattr(self.cfg.commands, "landing_dx_hit_tol", 0.12))
            min_peak = float(getattr(self.cfg.rewards, "landing_real_jump_min_peak", 0.40))
            err = torch.norm(self.landing_root_xy - self.landing_target[:, :2], dim=1)
            hit = self.just_landed & (self.peak_base_height >= min_peak) & (err <= tol)
            self.jump_target_hits += hit.float()
            # remember this jump's commanded distance (for the far-band advance gate)
            self._last_jump_cmd_dx[self.just_landed] = torch.norm(self.commands[self.just_landed, 0:2], dim=1)

    def _log_jump_episode_stats(self, env_ids):
        super()._log_jump_episode_stats(env_ids)
        jump_den = torch.clamp(self.jump_starts[env_ids], min=1.0)
        self.extras["episode"]["landing_hit_rate"] = torch.mean(self.jump_target_hits[env_ids] / jump_den)
        # COMBINED curriculum gate metric: a jump counts ONLY if the SAME jump BOTH landed on target
        # (jump_target_hits) AND ended in a stable/successful landing (successful_jumps). Single-jump
        # episodes -> each count is 0/1, so the AND of (>=1) is "this episode's jump did both". This
        # closes the loophole where "some jumps hit then topple + others land short but stable" cleared
        # the old separate hit/succ averages without any jump being both.
        stable_hit = (self.successful_jumps[env_ids] >= 1.0) & (self.jump_target_hits[env_ids] >= 1.0)
        self.extras["episode"]["landing_stable_hit_uniform"] = torch.mean(stable_hit.float() / jump_den)  # diagnostic (all dx)
        # FAR-BAND gate metric: only jumps whose commanded dx was in the top band [dx_max*(1-frac),
        # dx_max] count, so the curriculum advances only when the NEWEST/farthest distances are
        # stably hit -- not when the easy near commands carry a uniform average. (n pushed so the
        # gate can skip batches with no far-band jumps instead of biasing the EMA to 0.)
        far_frac = float(getattr(self.cfg.commands, "landing_dx_far_frac", 0.40))
        jumped = self.jump_starts[env_ids] >= 1.0
        far = jumped & (self._last_jump_cmd_dx[env_ids] >= self.landing_dx_max * (1.0 - far_frac))
        far_n = far.float().sum()
        far_hit = (stable_hit & far).float().sum()
        self.extras["episode"]["landing_farband_n"] = far_n
        self.extras["episode"]["landing_farband_hit"] = far_hit
        self.extras["episode"]["landing_stable_hit_rate"] = far_hit / torch.clamp(far_n, min=1.0)  # per-batch (noisy) — diagnostic only

    def _update_dx_curriculum(self):
        # Advance the forward dx range ONLY once the policy has TRULY mastered the far end of the
        # current range -- i.e. a CUMULATIVE far-band stable-hit rate (same jump lands on target AND
        # lands stably), over MANY samples, AFTER an adaptation window. The earlier per-batch EMA was
        # fooled by tiny far-band batches (rate 0/1) spiking to threshold -> it advanced on NOISE,
        # not mastery (pushed dx_max to 1.6 with only ~0.5 real far-band rate, then training on the
        # unreachable far commands collapsed the policy back to in-place). This robust gate:
        #   (a) skips the first min_hold steps after an advance (policy still adapting -> don't count),
        #   (b) then ACCUMULATES far-band hits/attempts,
        #   (c) advances only when >= min_far_samples have accumulated AND the cumulative rate >= thr,
        #   (d) resets the accumulators on advance.
        # -> the curriculum self-limits at the distance the policy can SUSTAINABLY hit; it cannot
        #    over-advance past the achievable ceiling. Inert outside training / when disabled.
        if not getattr(self, "landing_dx_curriculum", False):
            return
        if getattr(self.cfg.test, "use_test", False):
            return
        dx_final = float(getattr(self.cfg.commands, "landing_dx_final", 0.40))
        episode = self.extras.get("episode", {})
        fn = episode.get("landing_farband_n", None)
        fh = episode.get("landing_farband_hit", None)
        min_hold = int(getattr(self.cfg.commands, "landing_dx_min_hold_steps", 1500))
        adapting = (self.common_step_counter - self._dx_last_advance_step) < min_hold
        if (not adapting) and fn is not None and fh is not None:
            self._far_n_sum += self._to_float(fn)
            self._far_stable_sum += self._to_float(fh)
        cum_rate = self._far_stable_sum / max(self._far_n_sum, 1.0)
        # Surface the curriculum state to the training log.
        self.extras["episode"]["landing_dx_max"] = torch.tensor(self.landing_dx_max, dtype=torch.float, device=self.device)
        self.extras["episode"]["landing_dx_stable_cum"] = torch.tensor(cum_rate, dtype=torch.float, device=self.device)
        self.extras["episode"]["landing_dx_farsamples"] = torch.tensor(self._far_n_sum, dtype=torch.float, device=self.device)
        # Advance gate: enough SUSTAINED far-band samples AND cumulative rate cleared.
        if self.landing_dx_max >= dx_final - 1e-6:
            return
        min_samples = float(getattr(self.cfg.commands, "landing_dx_min_far_samples", 150))
        thr = float(getattr(self.cfg.commands, "landing_dx_stable_hit_threshold", 0.70))
        if self._far_n_sum >= min_samples:
            # A full WINDOW of far-band jumps accumulated -> evaluate. Advance if it cleared the bar;
            # EITHER WAY reset the window so the rate tracks the MOST RECENT window, not the whole
            # stage history. (Bug it fixes: a pure cumulative-since-advance never resets at the FIRST
            # stage -- no advance has happened -- so the early pre-/mid-discovery failures permanently
            # drag the average below thr and the gate stays stuck at dx_max=0 forever even after the
            # policy masters in-place: observed farsamples~950k, cum 0.26 while per-batch hit 0.93.)
            if cum_rate >= thr:
                step = float(getattr(self.cfg.commands, "landing_dx_step", 0.10))
                self.landing_dx_max = min(self.landing_dx_max + step, dx_final)
                self.command_ranges["lin_vel_x"] = [0.0, self.landing_dx_max]
                self._dx_last_advance_step = int(self.common_step_counter)  # re-adapt at the new distance
            self._far_stable_sum = 0.0
            self._far_n_sum = 0.0

    # ------------------------------------------------------------------ #
    # Pose-target override — during the post-touchdown landing buffer the parent leaves the
    # joint pd/reward target at q_ground (foot 0.30 = a TALL stance, base~0.34). With the PD
    # prior faded to 0 that tall pose never returns to the pre-jump idle pose (default_dof_pos,
    # base~0.31), so the cmd=0 stance is inconsistent pre/post jump -> topple + continuous-mode
    # drift. Pull the landing hold back to the canonical idle stand instead (this flows into both
    # the residual PD prior AND the default_pos reward, which read self.default_joint_pd_target).
    # ------------------------------------------------------------------ #
    def _update_default_joint_pd_target(self):
        super()._update_default_joint_pd_target()
        self.default_joint_pd_target[self.landing] = (
            self.default_dof_pos.expand(self.num_envs, -1)[self.landing]
        )

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

    def _landing_kernel(self, err_sq, base_sigma_key, norm_sigma_key):
        # exp kernel on a squared landing error, with optional DISTANCE-NORMALIZATION (Yang 2023):
        # divide err by the commanded displacement^2 so the reward is SCALE-INVARIANT -- a 15% miss
        # at 1.2m is judged like a 15% miss at 0.4m. A FIXED-sigma absolute kernel instead vanishes
        # at far targets (exp(-large) ~ 0), which is exactly why the policy plateaus at a ~constant
        # RELATIVE undershoot (lands at ~85% of the command). Floor avoids blow-up for in-place cmd.
        if bool(getattr(self.cfg.rewards, "landing_err_normalize", False)):
            floor = float(getattr(self.cfg.rewards, "landing_norm_dist_floor", 0.30))
            denom = torch.clamp(torch.norm(self.commands[:, 0:2], dim=1), min=floor) ** 2
            err_sq = err_sq / denom
            sigma = float(getattr(self.cfg.rewards, norm_sigma_key, 0.025))
        else:
            sigma = float(getattr(self.cfg.rewards, base_sigma_key, 0.10))
        return torch.exp(-err_sq / max(sigma, 1e-4))

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
        return active * self._landing_kernel(err, "sigma_landing_proj", "sigma_landing_proj_norm")

    def _reward_landing_position(self):
        # DENSE over the landing/settling phase (Atanassov 2025 'base position landing'), using the
        # FIXED touchdown xy (self.landing_root_xy, set at just_landed) so it cannot be farmed by
        # crawling toward the target after a short landing. Rewards "touched down on target" held
        # through the whole landing buffer -> revives this signal (the old SPARSE 1-step version
        # earned ~0 and was dead weight despite w30). Distance-normalized like projected_landing.
        # NOTE: dense -> ~150x the old per-jump magnitude, so its WEIGHT was cut hard in config.
        min_peak = float(getattr(self.cfg.rewards, "landing_real_jump_min_peak", 0.40))
        real_jump = self.peak_base_height >= min_peak
        active = self.landing.float() * real_jump.float()
        err = torch.sum(torch.square(self.landing_root_xy - self.landing_target[:, :2]), dim=1)
        return active * self._landing_kernel(err, "sigma_pos_landing", "sigma_pos_landing_norm")

    def _get_successful_jump_velocity_score(self):
        # LANDING-TASK OVERRIDE of the parent's success grading hook. The parent compares the average
        # FLIGHT VELOCITY (m/s) to commands[0:2] -- but here commands[0:2] are landing DISPLACEMENT (m),
        # so that score is unit-wrong (the bug that forced success_use_velocity_score=False). Instead
        # return a LANDING-ACCURACY score on the recorded touchdown xy, reusing the SAME distance-
        # normalized kernel as landing_position. The base then forms
        #     successful_jump = stay-upright(binary) x height_score x THIS
        # so the big completion bonus is paid ONLY for a jump that lands ON the commanded point AND
        # stays standing -- COUPLING precision with stability. A precise-but-topple jump scores 0 (the
        # binary), a stable-but-off-target jump scores low (this term, down to the floor). Kills the
        # farm where the policy banked the dense in-flight projected_landing for good AIM, then toppled
        # on touchdown (observed: hit 0.84 but succ 0.59). landing_root_xy is set at just_landed, one
        # line before the base reads this score, so it is the fresh touchdown position.
        err = torch.sum(torch.square(self.landing_root_xy - self.landing_target[:, :2]), dim=1)
        score = self._landing_kernel(err, "sigma_pos_landing", "sigma_pos_landing_norm")
        floor = float(getattr(self.cfg.rewards, "success_landing_min_score", 0.2))
        return floor + (1.0 - floor) * score

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
        # LANDING-FOCUSED leveling: the whole-cycle term above is diluted by the long level cruise, and a
        # nose-down touchdown (front-feet-first -> tumble) ends the episode fast -> that brief-but-FATAL
        # moment barely registers in the time-average, so there's almost no pressure exactly where it matters.
        # Pile an EXTRA pitch penalty on prelanding+landing so the body arrives & lands PARALLEL to the ground
        # (all four feet together). prelanding (the descent) accrues steps BEFORE a tumble can end the episode,
        # so this pressure actually lands. Symmetric (pushes toward LEVEL), not directional.
        land = (self.prelanding | self.landing).float()
        extra = float(getattr(self.cfg.rewards, "landing_pitch_extra", 5.0))
        return active * pitch_tilt + extra * land * pitch_tilt
