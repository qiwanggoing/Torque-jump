"""Landing-point tracking built on top of the PROVEN vertical-jump env.

This task extends ``go2_omnijump_curriculum_torque`` (the validated SATA
vertical jumper, run May28_22-34-51) into a landing-point-conditioned jump.
The design principle is *additive*: the entire jump-driving reward stack of the
parent is inherited UNCHANGED — we only graft a thin landing layer on top.

Landing layer
-------------
- ``commands[0:2]`` are REPURPOSED from target horizontal VELOCITY (m/s) to
  desired landing DISPLACEMENT (meters) in the spawn / heading frame:
    commands[0] = dx  (forward landing distance)
    commands[1] = dy  (lateral landing offset)
  The robot spawns facing +x (identity heading), so the displacement maps
  directly onto world xy. The env sets ``landing_target = spawn_xy + (dx, dy)``
  every reset.
- The observation slot that used to carry the velocity command is replaced by
  the yaw-frame landing-point error  Ryaw^T (p* - p_base) = [fwd_err, lat_err, 0]
  (Olsen 2025 navigation signal). The 69-dim layout and the obs mirror-symmetry
  parity (err_x even / err_y odd, same as the old vx/vy/yaw command) are kept,
  so PPO sym_loss needs no change.
- Two new rewards:
    * ``projected_landing``  — dense, in-flight ballistic projection of the
      landing xy (Olsen densification; horizontal analogue of ``projected_peak``).
    * ``landing_position``   — sparse, terminal exp(-||land_xy - p*||^2 / sigma),
      gated by a real-jump peak so a legs-tucked fake jump cannot farm it.
- ``tracking_linear_velocity`` disabled: its target ``commands[0:2]`` is now
  meters, not m/s, so the velocity-tracking semantics no longer apply.

Curriculum
----------
- Stage 1 (``landing_stage = 1``): displacement ranges = [0, 0] → target == spawn
  → behaviour is the proven vertical jump that *also* learns to land in place.
  Used to confirm the graft did not break jumping.
- Stage 2 (``landing_stage = 2``): open the forward/lateral displacement ranges
  → the same rewards now drive directed (forward / diagonal) jumps.
"""

from legged_gym.envs.go2.go2_omnijump_curriculum_torque.go2_omnijump_curriculum_torque_config import (
    GO2OmniJumpCurriculumTorqueCfg,
    GO2OmniJumpCurriculumTorqueCfgPPO,
)


class GO2OmniJumpLandingTorqueCfg(GO2OmniJumpCurriculumTorqueCfg):
    # NOTE: the deeper 0.25 ready-crouch experiment was REVERTED — it triggered a collapse
    # (~iter 700, still in warmup/full-PD): the near-folded crouch was hard to hold, the policy
    # splayed the hips to balance, default_hip_pos collapsed, and the jump fell apart. Back to the
    # proven ~0.30 default stance (inherited). Revisit launch depth later via a milder crouch +
    # stronger default_hip_pos if pursuing more height.
    class growth(GO2OmniJumpCurriculumTorqueCfg.growth):
        # PD fade EARLY (pure torque by ~iter500). The MIDDLE-window experiment (warmup 100000/x0 240000,
        # full PD until iter1450) FAILED: run Jun10_00-50-26 stayed squatQ=0 / peak 0.15 / NO jump for
        # 1087 iters even with full PD. Cause: with the squat-QUALIFIED gate, a 50% PD prior only gives a
        # shallow squat that never qualifies -> jump chain stays locked -> RL gets no jump-reward signal,
        # and the strong PD keeps it "comfortable" not squatting deep. Early fade WORKS precisely because
        # PD leaves fast and FORCES the RL to learn a qualifying squat-jump (Jun09_19-29-38: succ 0.90,
        # peak 0.57, continuous 0.65 jumps in play). The early-fade oscillation is cosmetic -- it still
        # converges to a great policy. ("PD-longer = smoother" held for Jun03 only because that older
        # config had no squat-qualified gate.) general_scale ramps 0->1 linearly warmup_steps->x0;
        # pd_alpha = 0.5*(1-general_scale). step_count ~= 69/iter.
        # SLOW GRADUAL FADE (user: "课程进化减慢从200到1200"): full PD only until ~iter200, then a long gradual
        # ramp to pure torque by ~iter1200, so the policy ADAPTS to pure torque smoothly instead of a sharp
        # transition (the iter~1100 dip was the PD/freq/torque transition). step rate ~96-97/iter (measured) ->
        # warmup 19200 ~= iter200, x0 115200 ~= iter1200 (verify the REAL general_scale trajectory from the run;
        # rate accelerates as freq ramps so it may complete a bit before 1200).
        # ⚠️ HISTORY: a MUCH slower PD-slow (warmup 100000 / x0 240000 = full PD until ~iter1450) FAILED HARD --
        # squatQ=0, NO jump for 1087 iters, because prolonged FULL PD gives a shallow squat that never qualifies
        # -> jump chain locked. THIS setting keeps the FULL-PD window SHORT (~iter200) to dodge that trap, then
        # fades slowly. WATCH the first ~iter300: if squatQ stays ~0 / no flight -> the short full-PD window is
        # still too long; shorten warmup. If it jumps fine and the iter~1100 dip is gentler -> the slow fade worked.
        # REVERTED to the ORIGINAL fast fade (user): PD should fade EARLY so the PURE-TORQUE policy is
        # established first, THEN the dx_max curriculum evolves ON pure torque (with the safety-revert in
        # _update_dx_curriculum so it plateaus instead of crashing, and re-advances as the pure-torque policy
        # gets stronger). The slow fade (warmup 19200/x0 115200) made dx_max hit the ceiling while PD was still
        # on (general_scale 0.45) -> the reach it found was PD-assisted, not the real pure-torque reach.
        # REVERTED to the Jun23_01-23-30 baseline (user): SLOW gradual fade -- full PD until ~iter200, then a
        # long ramp to pure torque by ~iter1200 (step ~96/iter -> warmup 19200 ~iter200, x0 115200 ~iter1200).
        # FAST FADE (user, 2026-06-24): full PD ~iter100, pure torque (general_scale=1) by ~iter500. The SLOW
        # fade was PROVEN to inflate dx: 3-checkpoint play (Jun24_15-00-38) showed dx advanced to 1.1 while PD
        # still assisted (model_1200 general_scale=0.73 hit cmd0.9 @0.87) but pure torque (model_3000
        # general_scale=1.0) undershoots (~0.77, hit 0.09). FAST fade makes the dx curriculum advance ON pure
        # torque -> dx_max settles at the HONEST reach (~0.8) so the commanded distances are actually hittable.
        # PIECEWISE fade (user, 2026-06-25): ramp 0->0.8 (full PD until warmup ~iter75, then ramp to 0.8 by
        # ~iter200), HOLD 0.8 (~10% PD) to ~iter500, then ramp 0.8->1.0 (pure torque) by ~iter1000. The plateau
        # lets the policy stabilise the ACTIVE squat under LIGHT PD before the final fade, so general_scale->1 is
        # a gentle ramp not the abrupt cliff where the PD-pre-loaded squat collapsed. NOTE: holding at 0.8 keeps
        # ~10% PD during part of the dx-curriculum advance, so dx may inflate SLIGHTLY (far less than the 27-33%
        # PD that inflated it to 1.1; the curriculum safety-revert pulls it back at pure torque). Iter targets are
        # APPROXIMATE -- step_count/iter varies ~48-95 with general_scale, so verify the real general_scale
        # trajectory in tensorboard (Episode/curriculum_general_scale) and nudge the step breakpoints if needed.
        warmup_steps = 7000            # full PD until ~iter75 (squat discovery via Option A pre-load)
        fade_hold_scale = 0.8          # plateau level (pd_alpha = 0.5*(1-0.8) = 0.1 = 10% PD assist)
        fade_ramp1_end = 15000         # reach 0.8 (~iter 200)
        fade_ramp2_start = 30000       # hold 0.8 until here (~iter 500), then ramp to 1.0
        x0 = 56000                     # reach 1.0 = pure torque (~iter 1000)

    class commands(GO2OmniJumpCurriculumTorqueCfg.commands):
        # Landing-point task: commands[0:2] repurposed velocity -> landing displacement (m).
        # Stage 1 keeps the displacement at [0,0] (land in place == proven vertical jump).
        # Set landing_stage = 2 to open the ranges below; the env widens
        # command_ranges["lin_vel_x"/"lin_vel_y"] accordingly at init.
        landing_stage = 2                      # STAGE 2 ON: env widens lin_vel_x/y ranges to the disp ranges below.
        landing_disp_x_stage2 = [0.0, 2.0]    # forward landing distance (m), Stage 2 (final range when curriculum off)
        landing_disp_y_stage2 = [0.0, 0.0]     # FORWARD-ONLY start (lateral off). Open to [-0.20,0.20] once forward jumps land.

        # ---- DISTANCE CURRICULUM (Atanassov 2025 local-difficulty) ----
        # Start the forward dx range at 0 (pure in-place = the proven vertical-jump discovery; the
        # landing reward is fully available because target==spawn) and grow the upper bound one
        # `step` at a time. Advance ONLY when, at the current distance, the policy both lands safely
        # (successful_jump_rate) AND lands near the commanded point (landing_hit_rate, |land-target|
        # <= hit_tol) — the hit gate stops the curriculum from outrunning the policy (success alone
        # is height-only and would let an in-place policy keep advancing). After each bump both rates
        # dip and must be re-earned at the new distance. Trains forward jumping in ONE from-scratch
        # run without the discovery cliff that a one-shot dx[0,0.40] open hits.
        landing_dx_curriculum = True
        # BIASED command sampling (Atanassov local-difficulty): concentrate most jump commands at the FAR
        # frontier (the goal = farthest landing point) instead of uniform over [0, dx_max]. The policy then
        # practices mostly where it counts; a spread fraction is kept for the easy->hard gradient + retention.
        # Only DISTANCE is biased (height untouched -- goal is a STABLE landing at the farthest point).
        landing_dx_biased = False              # DISABLED: 70/30 biased BACKFIRED (run iter4803) -- hit crashed 0.64->0.12,
                                               # dx_max regressed 1.1->0.9, stable_cum ~0. Concentrating on far STARVED the
                                               # near/mid commands that bootstrap distance-conditioning, and at far an UNDERSHOOT
                                               # makes the distance-normalized accuracy reward VANISH (exp(-big)~0) -> no gradient
                                               # -> the policy never learned to modulate distance (fixed high jump, lands off-target).
                                               # Uniform provides the easy->hard learning ladder -- keep it. The far-accuracy ceiling
                                               # (~1.1, hit 0.64) is a REWARD-gradient problem, not a sampling one.
        landing_dx_frontier_frac = 0.7         # (inert while landing_dx_biased=False)
        landing_dx_frontier_lo = 0.8           # (inert while landing_dx_biased=False)
        landing_dx_start = 0.0                 # initial dx upper bound (0 = in-place)
        landing_dx_final = 2.0                # final dx upper bound (the Stage-2 target)
        landing_dx_step = 0.10                 # increment per advance: 0 -> 0.1 -> 0.2 -> 0.3 -> 0.4
        # COMBINED advance gate: advance only when the SAME jump both lands on target AND lands
        # stably (landing_stable_hit_rate). Replaces the old two separate thresholds (succ + hit),
        # which let "hit-then-topple + short-but-stable" pass without any jump being both -> the
        # curriculum blew through to the cap. (succ/hit thresholds below are now unused.)
        landing_dx_stable_hit_threshold = 0.70 # advance needs CUMULATIVE far-band stable-hit rate >= this
        landing_dx_min_far_samples = 150       # ...over >= this many far-band jumps (post-adaptation), so the
                                               # gate reflects SUSTAINED mastery, not a noisy few-sample spike
                                               # (the old per-batch EMA spiked to thr on 1-2 jumps -> over-advanced
                                               # dx_max to 1.6 with only ~0.5 real far-band rate -> late collapse).
        # FAR-BAND: the stable-hit rate is measured ONLY over jumps whose commanded dx fell in the
        # top fraction [dx_max*(1-far_frac), dx_max] of the open range -> the gate requires the
        # NEWEST/farthest distances to be stably hit, not the easy near commands carrying a uniform
        # average. (uniform averaging let dx_max reach 1.2 while really mastering ~0.9.)
        landing_dx_far_frac = 0.30             # 0.20 -> 0.30: WIDEN the far band back a notch. 0.20 was TOO narrow:
                                               # ~1 far-band sample per log window (fb_n~1) -> the advance metric got
                                               # NOISY -> a lucky streak over-advanced dx_max from 0.9 to 1.0, past the
                                               # reliable reach (fb_hit cliff 0.79@0.9 -> 0.15@1.0 -> late decline).
                                               # 0.30 keeps "hit near the edge" but with enough samples for a STABLE
                                               # gate, so dx_max settles at the reliable reach instead of overshooting.
        landing_dx_succ_threshold = 0.80       # [unused — superseded by landing_dx_stable_hit_threshold]
        landing_dx_hit_threshold = 0.55        # [unused — superseded by landing_dx_stable_hit_threshold]
        landing_dx_hit_tol = 0.10              # a jump "hits" if |landing_xy - target| <= this (m).
                                               # KEEP < landing_dx_step (else in-place stays within tol of
                                               # the newly-opened distance and passes without forward motion).
        landing_dx_ema_alpha = 0.02            # EMA smoothing on the per-reset-batch stable-hit rate
        landing_dx_min_hold_steps = 1500       # min policy-steps held at a stage before it may advance (~30 iters)
        # Per-resample STAND probability: each resample (every resampling_time=1.8s) the robot STANDS
        # if commands[4] <= jump_command_threshold (0.5). Default range [0,1] -> 50% stand. Narrow to
        # [0.45,1.0] -> ~9% stand, so the robot idles far less and jumps almost every resample. (The
        # IMPORTANT standing — recovering to a stable stand after landing — is still trained in every
        # jump episode's post-landing buffer.)
        jump_command_range = [0.375, 1.0]  # SPLIT prob: P(draw<=0.5)=(0.5-lo)/(1-lo). lo=0.375 -> 20% STAND
                                            # (user: was 0.45 -> ~9%). More dedicated stand episodes so the policy
                                            # actually learns to STAND at cmd4=0 (+ stand_no_takeoff penalty bites
                                            # on them). cmd4 stays BINARY 0/1 via stand/jump_command_value.
        # BINARY jump command: 1.0 = jump, 0.0 = stand. The old scheme put STAND at the sampled [0.45,0.5]
        # band -- right under the 0.5 threshold -- and at a near-threshold stand command (e.g. 0.49) the
        # policy HESITATES and twitches a foot off (the stand-episode in-place hop). Pinning stand->0 and
        # jump->1 makes cmd4 an unambiguous binary far from the threshold, and drops the meaningless (0.5,1.0]
        # jitter from the cmd4 obs feature (its magnitude is never used as effort -- only cmd4>threshold).
        # Verified: at cmd4=0.45 the trained policy stands 800 steps dead still; 0.49 twitches. Needs RETRAIN
        # (the current model never saw a clean 0/1 -> OOD in play).
        stand_command_value = 0.0
        jump_command_value = 1.0
        # STAND-THEN-JUMP (user, 2026-06-24): a sampled JUMP first STANDS (cmd4=0) for a RANDOM delay, THEN
        # cmd4 flips to 1 and the jump fires -- so every jump starts from a real cmd4=0 STAND (now that the
        # stand is trained: stand_no_takeoff + 20% stand episodes), instead of jumping straight out of the
        # spawn free-fall. The random delay (vs a fixed one) stops the policy from timing a pre-squat.
        stand_before_jump = True
        jump_arm_delay_min = 55             # >= first_jump_delay_steps so first_jump_ready is met when cmd4 flips
        jump_arm_delay_max = 220            # up to ~1.1s pre-jump stand; random in [min,max] per episode
        # The moment the robot touches down, flip the jump command to STAND for the whole 0.75s landing
        # buffer (instead of holding cmd4=1 until the jump "finishes"). Without this the policy sees cmd4=1 +
        # the residual landing error during the buffer and HOPS to chase the undershot target. Verified in
        # play (force cmd4=0 at touchdown) that this kills the post-landing chase-hop. Landing-accuracy
        # rewards key off self.landing / touchdown-locked landing_root_xy, not cmd4, so scoring is unaffected.
        disable_jump_on_landing = True
        single_jump_command_prob = 1.0         # SINGLE JUMP per episode (reverted from 0.0=continuous). Continuous
                                               # (Jun09_13-24-40) gave a noisy, bistable training signal (succ/flght
                                               # oscillating 0.28-0.56) and lower peak; single-jump (Jun06_13-53-04) is
                                               # far cleaner (flght 0.97-1.00, succ 0.77-0.86 smooth, peak 0.58, no late
                                               # degradation). The post-landing topple is fixed WITHOUT continuous, via
                                               # landing_buffer_steps=150 below (success now requires surviving 0.75s
                                               # post-touchdown, so the ~0.75s topple is trained out) + the landing-pose
                                               # fixes (target->default_dof_pos, default_pos/orientation). Play stays
                                               # continuous regardless (play state machine is independent of train mode).

        class ranges(GO2OmniJumpCurriculumTorqueCfg.commands.ranges):
            jump_height = [0.40, 0.70]   # unchanged from the proven May28 baseline
            lin_vel_x = [0.0, 0.0]       # repurposed: landing dx (m). Stage 1 = land in place.
            lin_vel_y = [0.0, 0.0]       # repurposed: landing dy (m). Stage 1 = land in place.
            ang_vel_yaw = [0.0, 0.0]

    class rewards(GO2OmniJumpCurriculumTorqueCfg.rewards):
        # Landing-reward kernel widths + real-jump gate for the sparse terminal term.
        sigma_pos_landing = 0.06            # Stage-2: TIGHTENED from 0.12. At 0.12 an in-place jump at cmd dx=0.40
                                            # (err=0.16) still earned exp(-1.33)=0.26, and at the avg cmd dx=0.20
                                            # (err=0.04) earned 0.72 -> in-place farmed ~70% of the landing bonus.
                                            # 0.06 cuts those to 0.07 / 0.51 -> forces real forward motion to score.
        sigma_landing_proj = 0.05           # Stage-2: TIGHTENED from 0.10 (same reason). In-place at cmd dx=0.40
                                            # drops exp(-1.6)=0.20 -> exp(-3.2)=0.04; gradient at in-place still alive.
        # DISTANCE-NORMALIZE the landing kernels (Yang 2023): err is divided by the commanded
        # displacement^2 so the reward is SCALE-INVARIANT. Fixes the ~constant RELATIVE undershoot
        # (lands at ~85% of cmd) -- a fixed-sigma kernel's gradient vanishes at far targets, so the
        # policy plateaus short; normalized, a 15% miss at 1.2m is pushed as hard as at 0.4m.
        landing_err_normalize = True
        sigma_landing_proj_norm = 0.025     # kernel width on RELATIVE squared err (proj). ~5% miss->0.90, 15%->0.41
        sigma_pos_landing_norm = 0.04       # kernel width on RELATIVE squared err (terminal landing_position)
        landing_norm_dist_floor = 0.30      # min distance used in the normalizer (in-place/near cmds judged vs 0.30m)
        # landing_stability BRAKE kernel widths (read by _reward_landing_stability). MOVED here from class scales
        # where they were dead (cfg.rewards.<name> missing -> default 0.25 -> brake reward ~0 at landing speeds).
        # Loosened so the brake has gradient: |v|~2.5 -> exp(-2.5^2/2.0)~0.04 rising to 1.0 as it stops on the spot.
        landing_stability_lin_sigma = 2.0   # was effectively 0.25 (floored to ~0 for real landing |v|, no gradient)
        landing_stability_ang_sigma = 1.0   # was effectively 0.5
        four_leg_push_force_floor = 200.0   # N: total vertical GRF below which four_leg_push is NOT graded -> only the
                                            # REAL push (> body weight ~147N), not the static squat-load where legs
                                            # merely bear weight (which would be farmable by lingering in the squat).
        four_leg_push_force_target = 90.0   # N: per-leg vertical GRF for FULL credit (saturates -> surplus free, no
                                            # front==rear constraint). ~2.4x the static per-leg share; a leg below this
                                            # while still on the ground = "idling" -> graded down -> the policy wakes it.
        # NON-VANISHING far PULL on projected_landing (see _reward_projected_landing): the exp kernel ->0 for
        # a big undershoot at a far target -> no gradient -> curriculum plateaus (~1.3). A linear term gives
        # partial credit + a constant slope toward the target at any distance, so the policy keeps learning to
        # reach far. exp = precision near; linear = "reach to it" far. Discovery-safe (gated on a real jump).
        landing_lin_pull = True
        landing_lin_coef = 1.0              # 0.5 -> 1.0: STRENGTHEN the far pull. The far-band stalled at dx 1.0
                                            # (far accuracy ~0.47 < 0.70 gate); the policy undershoots far cmds to
                                            # ~0.85 because the reach-to-target pull was too soft vs the cost of a
                                            # bigger launch. Doubling coef doubles the far-undershoot slope
                                            # (coef/d_ref 0.33->0.67 /m) AND the far reward (cmd 1.0 land 0.6:
                                            # projected_landing 0.37->0.73). exp precision peak near target unchanged.
        landing_lin_ref = 1.5              # m: linear runs from 1 (on target) down to 0 at this miss distance
        # Takeoff-omega suppression (see _reward_base_ang_vel_xy): once succ_rate EMA >= gate, LATCH a stronger
        # ω penalty that also covers the PUSH -> kill the nose-down spin at the SOURCE (takeoff) so the body
        # flies level and lands flat. succ-rate gate (not a fixed step) = adapts to discovery speed, never
        # blocks the messy from-scratch pushes (an ungated strong ω penalty broke discovery, iter526 flight0).
        takeoff_omega_succ_gate = 0.80     # latch the stronger ω penalty once succ_rate EMA clears this
        takeoff_omega_gain = 4.0           # post-gate multiplier on base_ang_vel_xy (-0.15 -> ~-0.6 effective)
        # HARD pitch termination (see check_termination): end the episode if the base pitches NOSE-DOWN beyond
        # this (projected_gravity[:,0], ~sin(tilt)) AT TOUCHDOWN (the landing phase). Forces a level touchdown
        # (no front-feet-first), since soft penalties got traded off. Same succ-rate gate as takeoff_omega
        # (only after the robot can jump). 0 = off. MEASURED (play, model_10000): touchdown ~0.47 (28deg) nose-
        # down, consistently. 0.40 (~24deg) is a modest step below that -> forces ~4deg flatter (low collapse
        # risk); tighten over runs (0.40->0.35->...) to progressively flatten; loosen if dx/succ collapse.
        landing_tilt_terminate = 0.0       # OFF for now -- trying the SOFT (reward) route first (see below); flip to ~0.40 if soft fails
        # first_jump_delay_steps stays at the inherited 55 (0.275s). A 1s pre-jump idle (200) was
        # tried and BROKE from-scratch discovery (iter774 flight=0 vs the proven run's 0.914 by
        # iter500): 1s of standing rewards makes "don't jump" too comfortable -> the policy never
        # risks the squat-then-push (same failure mode as Jun09_11-29-05 strong default_pos/yaw).
        # The 1s settled-stance is a PLAY/visual nicety only -> set it in play_landing, not training.
        landing_real_jump_min_peak = 0.35   # was 0.40. lowered to 0.35 because stand_before_jump eliminates free-fall momentum.
                                            # (omnijump squat settles ~0.31, real jump peaks ~0.56)
        landing_buffer_steps = 150          # was 25 (=0.125s, inherited). A jump only "finishes" (success
                                            # credited + next jump re-enabled) after the robot stays stable
                                            # 150 steps (~0.75s) post-touchdown without exceeding fallover tilt.
                                            # = the "stand stable" requirement. The 25-step buffer let the policy
                                            # get credit after 0.125s then topple ~0.75s later (play roll_cutoff).
                                            # Toppling within these 150 steps -> roll_cutoff termination = penalty.
        # CONTINUOUS-jump NEXT-JUMP pose gate (DECOUPLED from success): success/finish is granted on
        # the time buffer alone (dense discovery signal preserved); this gate only delays the NEXT
        # jump until the robot has RETURNED to the default standing pose. Forces every jump in a
        # continuous sequence to start from the same canonical idle stand (kills chain drift) WITHOUT
        # withholding the successful_jump carrot from the still-learning policy. (The earlier version
        # gated success itself on pose -> starved discovery -> never learned to jump; Jun09_11-29-05.)
        # Gate metric = sum|dof - default_dof_pos| over the 12 joints. Threshold is non-critical now
        # (too tight = fewer jumps/episode, NOT stuck; too loose = no-op).
        next_jump_requires_default_pose = False   # continuous-only gate; no-op in single-jump mode (no "next jump").
        next_jump_default_pose_threshold = 1.5
        # Heading hold: keep tracking_angular_velocity active even at zero yaw command (commands[2]=0) so
        # its error term = wz^2 = a yaw-rate damp during flight -> stops the heading drift. (OmniNet does
        # this; our base reward otherwise only rewards tracking a NONZERO commanded yaw.)
        ang_vel_damp_zero_command = True
        # AIRBORNE-only: only damp yaw while actually in the air, NOT during the squat-but-not-taken-off
        # phase. Otherwise the strong (1.5) yaw reward pays the policy to sit in the squat holding still
        # and never jump -> discovery collapse (Jun10_12-47-57).
        ang_vel_damp_airborne_only = True
        # FIX: tracking_linear_velocity_all_time inherits True from omnijump_torque, but in
        # _reward_tracking_angular_velocity that branch is checked FIRST and OVERRIDES ang_vel_damp_airborne_only
        # above -> the 1.5-weight yaw damp wrongly stayed active during squat + landing (not just the air).
        # Force False here so the airborne-only gate actually takes effect (yaw damp in the air only, as
        # intended). The linear-velocity reward that also reads this flag is weight 0, so this is a no-op there.
        tracking_linear_velocity_all_time = False
        projected_landing_min_height = 0.40 # instantaneous height gate for the DENSE projected_landing:
                                            # blocks the legs-tucked sprawl farm (body ~0.13, feet off ground)
                                            # while keeping dense in-place landing control during real apex.
        # pose_guidance_sigma for joint_angle_aerial/prelanding/landing: kept inherited 5.0
        # (sharpening to 2.0 backfired — sharp exp saturates at the large air-pose error; the
        # fix for weak pose rewards is WEIGHT 1.5, not sigma). NOTE: the old joint-based
        # pushoff_leg_sync was replaced by contact-based foot_contact_sync, which uses no sigma.
        pose_guidance_sigma = 5.0

        # ---- Countermovement via a STANCE-SQUAT shaping reward (Atanassov 2025) ----
        # Root cause of "stand-and-pop" (no dip, capped height): nothing rewarded dipping
        # BEFORE the push, so the policy sat in the local optimum Atanassov explicitly warns
        # about ("standing in place"). Atanassov breaks it with RSI + a dense "squat to 0.2m
        # while on the ground" reward; we only had RSI. Fix = add that squat reward.
        # CRITICAL: its gate has NO vz<=0 condition. The old joint_angle_loaded used
        # phase_loaded (jumping & ~taken_off & vz<=0) — a DEAD LOOP: it only fired once the
        # robot was already dipping, so nothing ever drove the dip. See _reward_stance_squat.
        stance_squat_sigma = 0.02           # exp kernel width on (base_z - stance_squat_height);
                                            # 0.02 gives a strong gradient from the 0.31 stand
                                            # down to the 0.20 squat (stand value ~0.55 -> 1.0).
        # Squat-POSE gate (GUIDE to the target, replaces the gameable base_z height gate):
        # "squatted" = whole-body joint pose within squat_pose_threshold (L1 over 12 joints) of the
        # loaded pose q_squat. Standing is ~7.1 rad from q_squat (calf -1.5->-2.66, thigh 0.8/1.0
        # ->1.53, hips unchanged); q_squat itself = 0. Drives stand->squat, then unlocks the jump.
        squat_pose_sigma = 2.0              # TIGHTENED 5->2 (user): at 5 the kernel was so broad a SPLAT (body on the
                                            # ground, legs splayed/extended ~3-5 rad off q_squat) still scored ~0.45
                                            # -> the policy farmed stance_squat by collapsing FLAT ("dead fish") instead
                                            # of a clean controlled squat. At 2 a splat scores ~0.1 while a CLEAN q_squat
                                            # scores 1.0 -> only the clean squat pays. The old "strong pull from standing"
                                            # reason is moot: Option A pre-loads q_squat + stance_squat pays through the
                                            # armed stand, so no long-range approach pull is needed.
        squat_pose_threshold = 3.2          # was 2.8: EASED (stuck @ squatQ~0.48). "in the squat" = pose_err<=3.2, shallower from
                                            # standing (7.1). THE depth knob: stuck-not-jumping (can't fold
                                            # enough) -> RAISE; jumps too shallow / want a deeper load -> LOWER.
        squat_hold_steps = 25               # was 40 (0.2s) -> 25 (0.125s): EASED to unstick. jump chain unlocks only after the squat POSE is HELD within
                                            # squat_pose_threshold for this many CONSECUTIVE steps (= 0.2s at
                                            # sim dt 0.005s). Closes the "flick through the pose for one frame
                                            # and harvest the flight" hole. THE dwell knob: collapses to
                                            # not-jumping (can't hold long enough) -> LOWER (e.g. 20=0.1s);
                                            # want a more deliberate load before launch -> RAISE.
        # Give the dip+push room: a countermovement (~0.3-0.35s) does not fit the old 40-step
        # (0.2s) takeoff window — the dip would eat the budget and trip the timeout. 80 steps
        # = 0.4s. (step = sim dt 0.005s, counted on physics substeps, so freq-independent.)
        takeoff_timeout_steps = 200         # 1.0s: room for dip + push
        grounded_grace_steps = 80           # MUST-LAUNCH grace (was 12 in base): the squat dip+hold (~50 steps)
                                            # must finish BEFORE _reward_grounded_jump penalizes "still grounded",
                                            # else it would punish the normal deep squat -> premature pop. 80 = a
                                            # full squat-hold-launch fits inside; only DITHERING (no launch) past it
                                            # is penalized. (timeout is 200, so penalty fires over steps 80-200.)
        # Squat-depth gate (countermovement) — REPLACES the failed time-window. successful_jump and
        # projected_peak are WITHHELD until this jump has dipped to <= squat_gate_height before
        # takeoff. So "don't dip" = no main rewards AT ALL (not just a 0.5s blackout) -> a real
        # countermovement is the only way to score. The time-window failed because gating rewards
        # for N steps didn't stop the policy from physically insta-popping; a depth gate ties the
        # reward to the dip itself. RSI air-drops exempt. (stance_window_steps removed.)
        soft_dof_pos_limit = 0.9            # was 1.0 (no margin = penalty only AT the hard limit = useless).
                                            # 0.9 -> dof_pos_limits starts penalizing in the last 10% before the
                                            # hard URDF limit, so the over-deep squat stops before jamming the "wall".
        squat_gate_height = 0.24            # must dip base to <=0.24m (idle ~0.31) to unlock jump rewards
        squat_foot_height = 0.23            # 0.10 -> 0.23 (user). MEASURED: q_squat base == foot_height - 0.01, so
                                            # foot_height 0.23 -> base 0.219m (NOT the parent comment's bogus "base
                                            # ~0.20"; the OLD 0.10 was base ~0.09 = BELLY ON THE GROUND). At 0.10 the
                                            # legs fold to the EXTREME for Go2's long legs -> q_squat itself rests on
                                            # the ground, so the policy "dead-fish" splatted to BOTH farm the (now
                                            # always-on) stance_squat pose reward AND offload its weight onto the
                                            # ground (zeroing torques/dof_acc). At base ~0.22 the BELLY IS OFF THE
                                            # GROUND -> the squat must be ACTIVELY held against gravity (no free ride)
                                            # = a clean "combat crouch". Stays < squat_gate_height (0.24) -> qualifies.
        successful_jump_min_peak_height = 0.35  # was 0.40: lowered because dead-stop static jump is weaker initially
                                                # (= command floor 0.40; kills the low-jump shortcut)
        # REVERTED to True (the decouple test did NOT fix the collapse -- root was default_pos taxing the jump,
        # now fixed by converting default_pos to a reward). Back to the baseline: successful_jump =
        # stay-upright(binary) x height_score x landing-accuracy (couples precision with stability).
        success_use_velocity_score = True
        success_landing_min_score = 0.2     # REVERTED to safe (was 0.0 for the failed always-on test): the 0.2
                                            # floor keeps a stable-but-short jump earning the bonus (no give-up
                                            # death spiral). Isolate the soft clean_takeoff_bonus as the only change.
        # DECOUPLE success from the squat_qualified HOLD gate (which flickers under pure-torque noise ->
        # made succ oscillate 0.01-0.89 while flight/peak were stable). peak>=0.40 already guarantees a
        # real countermovement, so the gate is redundant FOR SUCCESS. It still gates the jump-REWARD chain.
        success_requires_squat_qualified = False  # REVERTED to safe (was True for the failed always-on test):
                                                  # isolate the soft clean_takeoff_bonus as the single new variable.
        # PRE-JUMP STANCE ANCHOR: scale the default_pos penalty (toward q_squat) UP only during the
        # ~jumping_state STAND (pre-jump + post-finish), so the policy holds a COLLECTED ready stance (feet
        # under body) instead of sprawling pre-jump and then needing a recovery STEP (which trips jump_replant).
        # The jump itself keeps the base default_pos weight (no extra tax). 1.0 = off. Read in _reward_default_pos,
        # gated post-discovery on _takeoff_omega_on (succ_ema>=0.80). WATCH noise_std (default_pos -0.7 ran away
        # once); if the stand still sprawls, raise; if discovery/jumping suffers, lower toward 1.0.
        default_pos_prejump_scale = 3.0
        # stand_no_takeoff penalty: ignore the first N env-steps of each episode so the SPAWN drop (init base
        # 0.42 -> natural rest ~0.30 free-fall, not a hop) is not penalized; the time-based jump reflex (the
        # real target) fires later (~ep90+), so 50 cleanly separates them. Read in _reward_stand_no_takeoff.
        stand_no_takeoff_grace = 50
        # RSI static deep-squat air-drop (the EXPLORATION piece): half the RSI envs start AT REST in the
        # deep squat + jumping, so value learns "deep-squat-at-rest = high return" (they're gate-exempt
        # and earn jump rewards from the dip). This plants V(dip) that the squat-depth gate then makes the
        # standing policy chase. (fe15103 tried this WITHOUT the gate and failed; now the gate backs it.)
        # RSI DISABLED (2026-06-05): rsi_prob 0.2 -> 0. RSI teleported 20% of envs INTO q_squat
        # (dof=q_squat) AND exempted them from the gate, so they grabbed the biggest rewards in the
        # system (takeoff_vz 15 / projected_peak 20 / successful 400) for pushing UP from a squat --
        # i.e. it demonstrated ONLY the JUMP half (push), never the FOLD half (stand->squat, it
        # teleports past it), training the shared policy toward popping. With RSI off, NO env earns
        # the jump chain until it actually folds-then-jumps, so stance_squat becomes the dominant
        # early reward (the squat is the only thing that scores) -- the intended guidance.
        rsi_prob = 0.0                    # was 0.2 (inherited). See note above. static_frac etc. now inert.
        rsi_static_frac = 0.5              # of the rsi_prob envs, half = static deep-squat, half = launch
        rsi_static_vel_z_min = -0.1       # near-rest vz at the squat bottom (slight down/up)
        rsi_static_vel_z_max = 0.3

        # ---- landing-stability borrows from the papers (Olsen 2025 / Atanassov 2025) ----
        # (4) tighten the success gate so a touch-down-then-topple is NOT counted as success.
        success_fallover_tilt = 0.4       # was 0.7 inherited (~46 deg). projected_gravity_xy above this
                                          # during the landing buffer cancels the pending success. 0.4 ~ 24 deg:
                                          # the 400-wt successful_jump now only pays for a landing that STAYS upright.
        # (2) landing_impact regularization knobs (see _reward_landing_impact):
        landing_impact_force_floor = 150.0  # N total vertical foot force below which no impact penalty (~standing weight)
        landing_impact_force_norm = 1500.0  # N normalizer; impact penalty saturates at (floor + norm)
        # (clean single jump) TERMINATE the episode on a load-phase foot RE-PLANT (stutter-step / run-up):
        # the jump must be ONE clean push (all feet leave together; no shuffling/stepping before takeoff).
        # WHY: as the dx curriculum pushed to 1.5-1.6 the policy invented a run-up (squat_qualified eroded
        # 0.90->0.73, play showed it stutter-stepping to build momentum) which also rotted the near commands
        # (cmd 1.0 undershot to 0.72 at model_9700 vs 0.98 at the clean model_5000). Forbidding the re-plant
        # makes the only way to reach far a clean push -> the dx curriculum SELF-LIMITS at the clean-jump
        # range (no artificial dx_final cap). Pure termination: a stutter loses the whole jump reward, and a
        # clean-but-short jump still earns more, so terminating is never an escape hatch.
        clean_takeoff_terminate = False     # HARD gate OFF (user): zeroing the jump-reward chain on a re-plant
                                            # killed discovery (robot couldn't learn to jump). Replaced by the
                                            # SOFT clean_takeoff_bonus reward (clean pays more, messy allowed).
                                            # =False also makes the _squat_deep_enough ~jump_replant gate inert.
        clean_takeoff_min_step = 0          # EXPERIMENT (user): 60000 -> 0 = LITERAL always-on, the gate bites
                                            # from step 0. Every re-planting jump forfeits the WHOLE jump-reward
                                            # chain from the very first iter -> the policy must find a CLEAN
                                            # single-push jump from scratch, never building a replant habit.
                                            # ⚠️ HIGH discovery risk (code's note: early failed pushes also
                                            # re-plant): with no early grace window, messy from-scratch attempts
                                            # all re-plant -> jump rewards ~0 -> the robot may sit in the squat
                                            # and never jump. WATCH squatQ/jump_flight in the first ~iter150; if
                                            # flight stays ~0 (no jumping), discovery died -> raise min_step.
        # CLEAN-LANDING (user request): no small hop / shuffle-step after touchdown -> ONE clean settle. Once
        # all 4 feet HOLD contact for clean_landing_plant_hold steps (skips the impact chatter), any foot
        # lifting is penalized per-step by _reward_clean_landing (weight `clean_landing` in scales). PENALTY,
        # not termination (keeps the successful_jump bonus). Gated post-discovery (succ-latch _takeoff_omega_on).
        clean_landing_plant_hold = 15       # consecutive all-4-contact steps (~0.075s) to latch "settled" before watching for re-lift
        landing_pitch_extra = 5.0           # EXTRA pitch-leveling multiplier on prelanding+landing (see _reward_pitch_level):
                                            # the whole-cycle pitch term is diluted by the long level cruise + the fast post-tumble
                                            # termination, so it barely presses the touchdown. At 5.0 the descent/touchdown pitch is
                                            # penalized (1+5)x -> land PARALLEL to the ground, all four feet together.
        jump_pitch_extra = 12.0            # EXTRA pitch penalty across ALL JUMP PHASES (load->push->flight->landing; gated on
                                            # the succ-latch). ROOT of the front-first landing: the body is already nose-down IN
                                            # THE AIR (launches tilted; takeoff_omega froze the rotation so it stays tilted). A
                                            # small per-step penalty did NOT change it (the lean buys reach), so penalize the tilt
                                            # HARD ((1+12)x EVERY jump step, +landing_pitch_extra on top at landing) -> the body
                                            # must be LEVEL the whole jump -> forces a level push/launch. Tune: still tilted ->
                                            # raise (15/20); if level flight shortens the jump (dx drops) the lean bought reach -> ease.

        class scales(GO2OmniJumpCurriculumTorqueCfg.rewards.scales):
            # ---- proven jump-driving stack inherited UNCHANGED ----
            #   takeoff_vertical_velocity=10, projected_peak=15, successful_jump=300,
            #   orientation=-1.6, collision=-3.0, default_pos=-0.3, default_hip_pos=0.3, ...
            # ---- landing layer (new) ----
            tracking_linear_velocity = 0.0   # was 0.5: commands[0:2] is now meters, not m/s
            tracking_angular_velocity = 1.5  # 0.5 -> 1.5: reverse-engineer of earned 0.027 showed wz~0.28 rad/s
                                             # (~12 deg drift/jump) = only half-damped. 3x to tighten the hold.
                                             # Still << main jump rewards (peak25/vz15); earned ~0.08. ACTIVATE (was 0):
                                             # OmniNet-style yaw-rate damping to hold heading.
                                             # ang_vel_damp_zero_command=True (below) keeps it active at zero yaw cmd
                                             # so the error term = wz^2 = damp spin during flight -> fixes the heading
                                             # drift. Kept WELL below the main jump rewards (peak25/vz15/landing20); it's
                                             # a stabilizer. Stage2: open commands[2] -> same term becomes turn-tracking.
            forward_reach = 30.0             # 20 -> 30: STRENGTHEN "farther = better" (user). Make distance-reach the
                                             # DOMINANT driver so the policy pushes HARD to reach far. WATCH: if it trades
                                             # away HEIGHT (peak drops) or PRECISION, ease back to 25; if unstable, 20.
                                             # distance-progressive EFFORT reward (see _reward_forward_reach). Rewards
                                             # the projected FORWARD reach (absolute m, capped at the command) -> jumping
                                             # FARTHER pays MORE + far command > near command, DECOUPLED from precise landing.
                                             # Fixes the chronic give-up: once a far command is hard to HIT precisely, all the
                                             # precision rewards go hit-or-miss and the policy abandons the big jump; this keeps a
                                             # STABLE positive signal for trying-its-best/reaching-far so it never gives up. Strong
                                             # (>= projected_peak 20) so DISTANCE outranks HEIGHT. TUNE vs precision rewards if it
                                             # overshoots near commands or starves precision.
            four_leg_push = 0.0             # DISABLED (user): it blew the critic (value_loss 0.18->4.0 @iter1031) and
                                             # never worked (rew flat ~0.017). _reward_four_leg_push still exists (push-up-to-
                                             # target per ON-GROUND leg, rear may dominate, front-first allowed) -> re-enable by
                                             # setting weight only AFTER re-verifying the "rear idle" premise with the FIXED
                                             # torque_diag (the old "rear thigh ~0.3" finding used the broken pd0/general_scale=1 eval).
            projected_landing = 15.0         # 10 -> 15: BOOST. USER PRINCIPLE: jump-DISTANCE/accuracy rewards must OUTRANK jump-HEIGHT
                                             # rewards. After projected_peak 25->20, height earned ~0.60 (pp 0.38 + takeoff_vz 0.22) still
                                             # exceeded distance ~0.43 (landing_position + this), so raise the distance side above height.
                                             # The earlier 20->10 HALVING was to curb a "precise-but-topple" farm -- but stability is now SOLVED
                                             # (succ 0.90 via landing-focused pitch), so the topple risk is gone and boosting AIM is safe; it now
                                             # drives the forward launch toward the target. Discovery-safe (gated on a real jump peak>=0.40).
            projected_peak = 20.0            # 25 -> 20: modest OVERALL height trim. Run trend showed the policy FARMS height (projected_peak
                                             # rose 0.30->0.50 while landing accuracy collapsed to ~0 once far accuracy got hard) -- height is
                                             # the biggest reward AND paid regardless of landing, so the policy dumps accuracy for it. A mild 20%
                                             # cut eases that without gating. NOTE: an earlier 25->15 broke discovery, but that was BUNDLED with
                                             # base_ang_vel_xy -0.4 (the real flight-penalty killer), now reverted to -0.15 -> a mild 20 should keep
                                             # discovery. If 20 is too mild (still farms height / 1.1 plateau holds) go lower or add a non-vanishing
                                             # far-accuracy pull. WATCH iter~500 flight recovers; if flight 0, 20 is still too low -> revert to 25.
            successful_jump = 1000.0          # Jun23_01-23-30 baseline (reverted). Sparse so weight is big but earned
                                             # modest (~0.25; also graded by height_score). Coupled to landing accuracy
                                             # via _get_successful_jump_velocity_score (success_landing_min_score floor).
            landing_position = 8.0           # 5 -> 8: BOOST (with projected_landing 15) so jump-DISTANCE/accuracy OUTRANKS height
                                             # (user principle). DENSE over the landing buffer (~150 steps, fixed touchdown xy). Target:
                                             # distance earned (landing_position + projected_landing ~0.67) > height (~0.60). Discovery-safe
                                             # (gated on a real jump peak>=0.40 -> can't be farmed by standing). VERIFY the earned balance in
                                             # the retrain (height < distance); if height still wins, boost these more / trim height further.
            # ---- Stage2-ready: DISABLE takeoff_direction (was inherited 3.0) ----
            # takeoff_direction = vz/‖v‖ rewards a PURELY VERTICAL takeoff — the only Stage1-specific
            # reward. It is redundant at command 0 (takeoff_vz + projected_landing already give the
            # ballistic vertical+horizontal target) and FIGHTS any directed jump once landing_stage=2
            # opens dx/dy (it would penalize the horizontal velocity you NEED to reach the target).
            # Removing it now makes the whole stack direction-general: switching to Stage2 = just open
            # the command ranges, zero reward surgery. Behaviour stays in-place while commands[0:2]=0.
            takeoff_direction = 0.0
            # ---- MUST-LAUNCH: penalize "commanded to jump but still grounded after the squat window" ----
            grounded_jump = 0.0              # DEACTIVATED: made it WORSE. The policy isn't "refusing" to jump --
                                             # it physically can't push as hard once PD fades (a CAPABILITY problem,
                                             # not motivation), so forcing a launch only produced worse weak jumps +
                                             # extra penalty. The real lever is the PD-fade slowdown (growth.x0), below.
            # ---- MERGED takeoff launch: velocity-VECTOR match (height + distance in one), replaces vertical-only ----
            takeoff_vertical_velocity = 0.0  # OFF: superseded by takeoff_velocity_match (which == it at dx=0)
            takeoff_velocity_match = 15.0    # reward takeoff velocity matching the ballistic launch to (landing
                                             # point + apex height). CAUSE-side "jump FAR and HIGH" driver — the
                                             # closeness rewards can't push reach (diminishing returns at undershoot).
                                             # = old takeoff_vz weight (15). At dx=0 it reduces to takeoff_vz (safe).
            # ---- structurally-inert rewards removed ----
            joint_angle_loaded = 0.0         # was 0.4: phase_loaded (jumping & ~taken_off & vz<=0) almost never
                                             # fires — the policy pre-squats during idle and pops straight up on
                                             # command, so there is no in-jump squat-down window. Contributed 0.0000.
                                             # SUPERSEDED by stance_squat below (same goal, NO vz<=0 dead-loop gate).
            # ---- countermovement: stance-squat shaping (the piece we were missing) ----
            stance_squat = 3.0               # was 1.5: now the PRIMARY early driver (RSI off -> jump chain locked
                                             # until folded, so this is the main thing firing early). pose-based
                                             # exp(-|dof-q_squat|/sigma). SHAPES the dip (how to get down); paid while loading and not yet at
                                             # squat_gate_height, then stops. The squat-depth gate on successful_jump/
                                             # projected_peak is the real forcing function (no dip -> no main rewards).
                                             # Farm-safe: stops at the gate, and squatting-without-jumping earns no
                                             # successful_jump anyway. (history: 0.5/2.0 weight-only + time-window all failed.)
            # ---- air-time + stay-planted boosts (user) ----
            clean_takeoff_bonus = 3.0        # SOFT clean-takeoff: extra flight reward for a no-re-plant (clean
                                             # single-push) takeoff. Clean jump earns all_feet_airborne + this;
                                             # a stutter-stepped jump earns all_feet_airborne only -> clean pays
                                             # MORE but messy is NOT forbidden (discovery-safe, replaces the hard
                                             # clean_takeoff_terminate gate). Tune: higher = stronger pull to clean.
            stand_no_takeoff = -5.0          # HARD penalty (user): cmd4=0 (STAND) but all feet leave the ground
                                             # (a hop) -> punish, so cmd4 is the real switch (cmd4=0 => stay grounded
                                             # in the resting squat, cmd4=1 => jump). Only fires at cmd4<=0.5 +
                                             # post-discovery + after grace -> never touches a commanded jump or
                                             # discovery. Tune harder (-8/-10) if the hop survives; the resting
                                             # squat (the desired stand) is unaffected (only LEAVING the ground costs).
            all_feet_airborne = 3.0          # 2.0 -> 3.0: more air-time pressure. Gated (squat_deep + height_progress)
                                             # so it can't be farmed by a tucked sprawl. The policy currently UNDERSHOOTS
                                             # the commanded apex (peak ~0.50 vs cmd ~0.55) -> this pushes it to the FULL
                                             # commanded height -> longer flight -> more reach when vx is calf-capped
                                             # (range = vx*T). MODERATE on purpose: over-boosting pushes jumping ABOVE
                                             # the command and fights projected_peak. Watch mean_peak_height + hit_rate.
            maintain_contact = 0.3           # 0.10 -> 0.3: POSITIVE "four feet planted when not airborne" — the incentive
                                             # complement to clean_landing for the post-touchdown shuffle (lifting a foot
                                             # in the settle now costs this). MODERATE: a big value also pays the pre-jump
                                             # STAND -> could make "don't jump" comfy (discovery risk). Watch jump_flight_rate.
            landing_stability = 1.0          # RE-ENABLED to BRAKE the landing momentum: per-step trace showed the
                                             # robot lands at vx~1.3 m/s, bounces (all feet off, +0.06m) and coasts
                                             # ~0.30m forward to a stop (the "post-landing slide"). This rewards LOW
                                             # base velocity during the landing buffer -> absorb/stop on the spot.
            # NOTE: landing_stability_lin_sigma / _ang_sigma live in `class rewards` (NOT here in scales) — the
            # reward fn reads cfg.rewards.<name>. They were MISPLACED here, so cfg.rewards.<name> fell back to the
            # default 0.25 -> exp(-2.5^2/0.25)~0 -> the brake had ZERO gradient (verified: stab~0.000, slide 0.7m).
            # clean_landing REMOVED (ineffective: detector never armed -> reward ~0). Post-landing slide handled by
            # landing_stability (brake momentum) + disable_jump_on_landing (no commanded re-jump); error obs real-time.
            # ---- four-foot contact-timing sync (penalty on staggered takeoff/landing) ----
            foot_contact_sync = -4.0         # was -3.0: STRENGTHEN four-foot takeoff/landing sync (less body tilt
                                             # at touchdown). config note: if peak drops too much, back off.
                                             # penalize 1-3 feet on the ground during the takeoff push / landing
                                             # window -> all four feet leave & touch down TOGETHER (less body tilt).
                                             # active = (jumping & ~taken_off) | landing, so this tightens BOTH
                                             # liftoff and touchdown timing. COST: stronger sync caps peak a bit
                                             # (-2.0 already 0.576->0.539); if still uneven go -4.0, if peak drops
                                             # too much back off. (replaced the old joint-based pushoff_leg_sync.)
            # ---- kill in-air flailing (PD faded -> RL flails legs in the ~unconstrained air phase) ----
            aerial_dof_acc = -3e-6           # was -1e-6: too weak. Air joint-accel actually ~180 rad/s^2 (flailing);
                                             # air pose is a near-zero-gradient dim (peak/airborne rewards ignore leg
                                             # pose) so RL leaves it noisy. x3 so "flail vs tuck" actually moves the
                                             # return. Watch it doesn't over-damp the necessary tuck/extend.
            # ---- general joint-accel smoothness: REDUCED (overrides inherited -2.5e-7). It competes with the
            #      explosive pure-torque pushoff; halving frees the jump. aerial_dof_acc kept at -3e-6 (unchanged, per call).
            dof_acc = -1.25e-7
            # ---- action_rate: RESTORED to -0.03 (the off=0.0 test confirmed action_rate was NOT the collapse cause
            #      -- it still collapsed at 0). Back to -0.03 for anti-fidget. (Root cause was default_pos taxing the
            #      jump -> now fixed by converting default_pos to a reward.) Phase-gate later if it damps the burst.
            action_rate = -0.03
            # ---- pose-shaping joint_angle_* REMOVED (cleanup, audit): each earned ~0 (robot never reached
            #      q_air/q_pre/q_ground) = dead weight. Landing ATTITUDE now held by orientation + foot_contact_sync
            #      (strengthened below); landing-POINT by projected_landing + landing_position (kept / revived).
            joint_angle_aerial = 0.0
            joint_angle_prelanding = 0.0
            joint_angle_landing = 0.0
            # ---- post-PD pose-holding (rear legs drifted once PD faded to 0) ----
            default_pos = -0.5              # HALVED penalty (was -1.0). At -1.0 it was the DOMINANT term (-0.81/s =
                                             # 57% of ALL penalties) and TAXED the jump (a jump deviates from the pose
                                             # target) -> jumping net-negative -> policy collapsed to NOT jumping (Jun21
                                             # runs: best ~iter500, collapse iter700-1100). Halving cuts the jump tax
                                             # ~in half (-0.81 -> ~-0.4) so jumping stays net-positive, kept as a PENALTY
                                             # (simpler than a reward; no standing-pose reward to make not-jumping comfy).
                                             # Zeroed during push-off / squat-down. NOTE: memory says -0.7 once caused
                                             # noise runaway -> watch noise_std; raise back if the anchor gets too loose.
            default_hip_pos = 1.0            # 0.3 -> 1.0: the policy slid the front feet INWARD (hip adduction) to shuffle
                                             # forward momentum (the stutter/run-up morphed into a SLIDE once the re-plant
                                             # termination forbade stepping). default_hip_pos keeps the 4 hip-abduction joints
                                             # near default; at 0.3 it earned only ~0.05 (hips drifting ~0.46 rad) -- too weak to
                                             # hold them. Raised so deviating forfeits a meaningful reward -> legs stay vertical
                                             # in the frontal plane (no inward collapse). Safe: a clean forward jump is sagittal
                                             # (thigh/calf) and never needs hip abduction. Tune up (1.5-2.0) if the slide persists;
                                             # if it persists even then it's pure ground-slip (not hip) -> add a foot_slip penalty.
            orientation = -3.5               # -3.0 -> -3.5 (DISCOVERY-SAFE: -4.5 + the strong default_pos made not-jumping
                                             # too comfortable from scratch, Jun09_11-29-05). Mild strengthen of the level-body
                                             # hold (late training showed g_xy^2 creeping 0.017->0.038 as the policy traded
                                             # attitude for jump magnitude). Vertical (Stage1) jump wants body level throughout.
                                             # the main landing-stability lever after joint_angle_landing removed. (pitch also via pitch_level.)
            # ---- (1)+(2) landing stability from the papers: stop "lands then flips" ----
            base_ang_vel_xy = -0.15          # REVERTED -0.4 -> -0.15: -0.4 penalized the MESSY exploratory flight so hard that "don't jump"
                                             #     won -> discovery died (flight 0 at iter526). -0.15 is the discovery-proven value. The nose-down
                                             #     ROTATION is a LATE concern -> if needed, re-add stronger damping GATED to post-discovery, not step 1.
                                             # (1) PENALTY on base roll/pitch angular velocity in flight+landing
                                             # (Olsen ϕσ(‖ω‖) / Atanassov "track zero ω after landing"). We had NO
                                             # ω damping -> body tumbled into touchdown. Penalty (not bell kernel) so
                                             # it bites; penalizes spin RATE not airborne time -> clean high jump unhurt.
                                             # THE knob: still flipping -> more negative; jumps get stiff/low -> back off.
            dof_pos_limits = -5.0            # ENABLE (was 0/off): penalize joints folding past the soft limit
                                             # (soft_dof_pos_limit=0.9 below = last 10% before the hard URDF limit).
                                             # Fix for the over-deep squat (base ~0.13) jamming the knees to the
                                             # "wall" + stalling, which also dropped peak. Stops the dip ~10% short
                                             # of the hard limit -> smoother push, should recover height. Tunable.
            landing_impact = 0.0             # DISABLED (user: "没什么用"). Was -2.0 (Olsen soft-impact). Landing
                                             # cushion now handled by landing_stability (brake, sigma fix) + the
                                             # touchdown dynamics; this term wasn't earning its keep. (Aside: it also
                                             # risks suppressing height -- softer impact = jump lower -- which fights
                                             # our distance goal.) Re-enable only if touchdowns start slamming.
            pitch_level = -6.0               # -4.5 -> -6.0: further STRENGTHEN (preemptive vs nose-dive when we push
                                             # height higher; back off if the jump gets stiff/peak drops). Nose-dive is the high-jump
                                             # failure mode -- Jun05_23-55-11 crashed at pitch 0.68). PITCH-specific
                                             # attitude penalty (projected_gravity_x^2) over the whole
                                             # jump. Fixes the persistent nose-down ("head-heavy") tilt that the
                                             # symmetric orientation (-2.0) is too weak on. Stacks on orientation ->
                                             # pitch weighted ~2.5x roll during the jump. THE knob: still nose-down ->
                                             # more negative; jump gets stiff/weak or peak drops -> back off.

    class logging(GO2OmniJumpCurriculumTorqueCfg.logging):
        # Decluttered TERMINAL print: drop the redundant metrics (jump_landing_rate / jump_completed_cycles
        # ≈ jump_flight_rate). EVERYTHING still goes to tensorboard/wandb -- this only filters the terminal.
        print_episode_keys = [
            k for k in GO2OmniJumpCurriculumTorqueCfg.logging.print_episode_keys
            if k not in ("jump_landing_rate", "jump_completed_cycles")
        ] + [
            "rew_takeoff_velocity_match",  # merged launch driver (jump far+high) — watch vs dx_max advancing
            "rew_projected_landing",
            "rew_landing_position",
            "rew_foot_contact_sync",
            "rew_stance_squat",     # countermovement shaping — watch vs successful_jump_rate
            # inherited-active but missing from the parent's print list — surface them
            "rew_base_ang_vel_xy",   # (1) flight+landing roll/pitch ω damping — the anti-tumble lever
            "rew_landing_impact",    # (2) touchdown force-spike penalty — cushion vs slam
            "rew_pitch_level",       # pitch-specific tilt penalty — fix persistent nose-down
            "rew_tracking_angular_velocity",  # OmniNet yaw-rate damping (hold heading) — watch it stays < jump rewards
            "rew_dof_pos_limits",    # joint-limit penalty — watch it shrinks as the over-deep squat stops jamming
            "squat_qualified_rate",  # frac of takeoffs preceded by a HELD squat; compare to jump_flight_rate
            # ---- distance curriculum (watch these to see the dx ramp progress) ----
            "landing_dx_max",            # current forward dx upper bound (grows as the curriculum advances)
            "landing_dx_stable_cum",     # CUMULATIVE far-band stable-hit the gate reads (>= thr AND enough samples -> advance)
            "landing_stable_hit_uniform",# 又准又稳 over all dx (uniform)
            "landing_hit_rate",          # accuracy (ignores stability)
        ]

    class test(GO2OmniJumpCurriculumTorqueCfg.test):
        vel = GO2OmniJumpCurriculumTorqueCfg.test.vel.clone()
        vel[0] = 0.0   # Stage 1: land in place (dx=0). Set vel[0]>0 to play directed jumps.
        vel[1] = 0.0
        single_jump_play = True


class GO2OmniJumpLandingTorqueCfgPPO(GO2OmniJumpCurriculumTorqueCfgPPO):
    class algorithm(GO2OmniJumpCurriculumTorqueCfgPPO.algorithm):
        sym_coef = 1.0   # was 0.5: match my_go2_jump — tighter LEFT-RIGHT mirror symmetry
                         # (front-rear is handled by the pushoff_leg_sync reward, not sym_loss)
        entropy_coef = 0.003   # 0.001 -> 0.003: MORE exploration. At 0.001 noise_std collapsed to ~0.04 -> the
                               # policy got too CONSERVATIVE (peak ~0.50, undershoots far) and plateaued; the old
                               # high+far run had noise ~0.39. 0.003 settles noise ~0.32 (memory) = that exploration
                               # level but STABLE (the runaway was 0.005 -> 0.86), and forward_reach now holds the
                               # floor so the bigger jumps don't degrade. WATCH noise_std: ~0.2-0.35 good; toward 0.5+
                               # = runaway -> drop to 0.002.
                               # Constant 0.005 (run Jun06_07-41-08) cracked the squat gate (squatQ->0.95) but then
                               # noise RAN AWAY to 1.33 -> degraded from iter3000, collapsed at 5400. Constant 0.003 was
                               # the opposite (noise 0.32 -> stuck). So: high early (discover/crack gate), low late
                               # (anneal = consolidate + kill the noise runaway). See entropy_coef_final.

    class runner(GO2OmniJumpCurriculumTorqueCfgPPO.runner):
        experiment_name = "go2_omnijump_landing_torque"
        run_name = "stage1_landing"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        max_iterations = 3000    # 10000 -> 3000 (user): PD fades early (~iter800), then plenty of room for the
                                 # pure-torque policy to consolidate + dx_max to evolve (with the safety-revert).
        # entropy_coef ANNEALS 0.005 -> 0.001 at entropy_anneal_iter (HARD STEP, on_policy_runner.py:129-133).
        # MOVED 2800 -> 500 for the real-Go2 actuator. The 0.005 START is the ONLY force pushing action_std UP;
        # with the weak real calf the precise squat-jump can't survive high noise, so noise_std running away
        # (0.36 -> 0.6-0.87) crashed every run at iter700-1100 (squatQ -> 0, value_loss -> 0). 2800 was tuned for
        # the OLD strong calf whose peak hit ~iter2500; the NEW calf cracks the squat gate by iter100 (squatQ 0.91)
        # and noise bottoms at ~iter300, so anneal at 500 LOCKS the early peak (~0.58) and kills the runaway BEFORE
        # it crosses ~0.55 and collapses the jump. (Raise to 600-800 if a fresh run discovers slower; data = iter100.)
        entropy_anneal_iter = 500
        entropy_coef_final = 0.001
