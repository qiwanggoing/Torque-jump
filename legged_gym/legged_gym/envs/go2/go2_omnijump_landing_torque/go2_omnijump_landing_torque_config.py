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
        warmup_steps = 7000        # full PD only for the first ~iter100, then start fading.
        x0 = 35000                 # linear-fade end -> pd_alpha=0 by ~iter500. (Proven config Jun09_19-29-38.)

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
        landing_dx_far_frac = 0.40
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
        jump_command_range = [0.45, 1.0]
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
        # first_jump_delay_steps stays at the inherited 55 (0.275s). A 1s pre-jump idle (200) was
        # tried and BROKE from-scratch discovery (iter774 flight=0 vs the proven run's 0.914 by
        # iter500): 1s of standing rewards makes "don't jump" too comfortable -> the policy never
        # risks the squat-then-push (same failure mode as Jun09_11-29-05 strong default_pos/yaw).
        # The 1s settled-stance is a PLAY/visual nicety only -> set it in play_landing, not training.
        landing_real_jump_min_peak = 0.40   # peak gate for the landing_position reward
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
        squat_pose_sigma = 5.0              # exp kernel on |dof - q_squat| for the dip reward. Raised 3->5: the
                                            # pull from standing (7.1 rad away) is (1/sigma)*e^(-7.1/sigma), which
                                            # is ~55% stronger at 5 than 3 (peaks near sigma~7). default_pos no
                                            # longer competes during the dip, so this positive pull now drives it.
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
        successful_jump_min_peak_height = 0.40  # was 0.30: a ~0.34 "low pop" no longer counts as success
                                                # (= command floor 0.40; kills the low-jump shortcut)
        # Grade successful_jump by a LANDING-ACCURACY score (the landing env OVERRIDES the parent's
        # velocity hook -- see _get_successful_jump_velocity_score there). Without this, successful_jump
        # is pure binary x height and is DECOUPLED from the landing point -> the policy farms the dense
        # in-flight projected_landing for AIM, then topples on touchdown (observed hit 0.84 but succ 0.59).
        success_use_velocity_score = True   # STAGE 2: ON, via the landing-accuracy OVERRIDE (NOT the parent's
                                            # velocity-tracking score, which is unit-wrong here: commands[0:2] are
                                            # landing DISPLACEMENT (m), not m/s). The override returns a distance-
                                            # normalized exp on touchdown xy vs target, so:
                                            #   successful_jump = stay-upright(binary) x height_score x landing-accuracy
                                            # -> the big completion bonus pays ONLY for landing ON the commanded point
                                            # AND staying standing. COUPLES precision with stability (precise-but-topple
                                            # -> 0 via the binary; stable-but-off-target -> low via the accuracy term).
        success_landing_min_score = 0.2     # floor on the landing-accuracy score: a stable but off-target jump still
                                            # earns 0.2*height (keeps stability rewarded while not yet on target); an
                                            # on-target stable jump earns the full height_score.
        # DECOUPLE success from the squat_qualified HOLD gate (which flickers under pure-torque noise ->
        # made succ oscillate 0.01-0.89 while flight/peak were stable). peak>=0.40 already guarantees a
        # real countermovement, so the gate is redundant FOR SUCCESS. It still gates the jump-REWARD chain.
        success_requires_squat_qualified = False
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
        clean_takeoff_terminate = True
        clean_takeoff_min_step = 60000      # ...but ONLY after this many env-steps (well past the ~iter500
                                            # discovery / PD-fade window; the stutter only emerges ~iter6000+,
                                            # so the gate is active long before it yet never terminates the
                                            # from-scratch jumping discovery, whose failed pushes also re-plant).
        landing_pitch_extra = 5.0           # EXTRA pitch-leveling multiplier on prelanding+landing (see _reward_pitch_level):
                                            # the whole-cycle pitch term is diluted by the long level cruise + the fast post-tumble
                                            # termination, so it barely presses the touchdown. At 5.0 the descent/touchdown pitch is
                                            # penalized (1+5)x -> land PARALLEL to the ground, all four feet together.

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
            successful_jump = 1000.0          # 400 -> 700: raise the completion reward to ~rank3 (just below
                                             # landing/peak). It's sparse so weight is big but earned modest
                                             # (~0.25; it's also ALREADY graded by height_score≈0.45 since peak
                                             # 0.5 < cmd 0.7). Paired with success_use_velocity_score=True so it's
                                             # graded by landing-drift too (less binary -> less oscillation amplify).
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
            landing_stability = 0.0          # was 1.0: exp(-landing_velocity/0.25) but touchdown velocities are
                                             # >> 0.25, so it floors at ~0 and never fires (contributed 0.0002).
                                             # If landing damping is wanted later, re-add with a much looser sigma.
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
            # ---- pose-shaping joint_angle_* REMOVED (cleanup, audit): each earned ~0 (robot never reached
            #      q_air/q_pre/q_ground) = dead weight. Landing ATTITUDE now held by orientation + foot_contact_sync
            #      (strengthened below); landing-POINT by projected_landing + landing_position (kept / revived).
            joint_angle_aerial = 0.0
            joint_angle_prelanding = 0.0
            joint_angle_landing = 0.0
            # ---- post-PD pose-holding (rear legs drifted once PD faded to 0) ----
            default_pos = -1.0               # back to -1.0 (the -0.7 RELAX destabilized: run Jun09_17-35-42 had noise runaway
                                             # to 0.66 + succ oscillating 0.46, same loosen-the-anchor failure as e6bd00f's
                                             # -0.5->-0.25). Anchor must stay tight. Height is pushed via projected_peak ONLY
                                             # now (isolated), not by loosening the policy. RESTORED from -0.25 originally; the
                                             # -0.25 (cleanup) removed the dominant pose
                                             # anchor -> looser, higher-variance policy (noise_std ~0.84 vs ~0.55) and
                                             # deterministic play idled/landed in a deep crouch (base_z~0.149). That
                                             # sustained high noise is what tipped the iter~2475 collapse. (zeroed
                                             # during push-off so it doesn't fight the jump.)
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
            landing_impact = -2.0            # (2) Olsen Ground-force/Soft-impact: penalize the vertical foot-force
                                             # SPIKE at touchdown (bounded, soft floor at ~standing weight) -> cushion,
                                             # don't slam. KEEP MODEST: too negative incentivizes jumping LOWER
                                             # (smaller fall = softer impact) and suppresses height. #1 (ω damping)
                                             # is the primary lever; this is secondary. peak drops -> back off toward 0.
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
        entropy_coef = 0.005   # START high for exploration; ANNEALS to entropy_coef_final=0.001 at iter2800.
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
        max_iterations = 10000   # distance curriculum needs room: ~in-place discovery + 4 dx stages
        # entropy_coef ANNEALS 0.005 -> 0.001 at iter2800 (re-enabled; "disable the anneal" was a misdiagnosis).
        # Data: constant 0.005 (Jun06_07-41-08) cracked the gate then noise RAN AWAY (0.45->1.33) -> degraded@3000,
        # collapsed@5400 as PD faded. Constant 0.003 -> noise 0.32 -> stuck. Neither constant works. The anneal is the
        # answer: 0.005 early to discover + crack the squat-HOLD gate, drop to 0.001 at 2800 to LOCK IN the peak
        # (squatQ 0.95 / peak 0.605 hit ~iter2500-2700) and prevent the late noise runaway / pure-torque collapse.
        entropy_anneal_iter = 2800
        entropy_coef_final = 0.001
