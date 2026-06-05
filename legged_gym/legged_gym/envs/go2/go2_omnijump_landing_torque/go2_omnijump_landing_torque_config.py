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
    class commands(GO2OmniJumpCurriculumTorqueCfg.commands):
        # Landing-point task: commands[0:2] repurposed velocity -> landing displacement (m).
        # Stage 1 keeps the displacement at [0,0] (land in place == proven vertical jump).
        # Set landing_stage = 2 to open the ranges below; the env widens
        # command_ranges["lin_vel_x"/"lin_vel_y"] accordingly at init.
        landing_stage = 1
        landing_disp_x_stage2 = [0.0, 0.40]    # forward landing distance (m), Stage 2
        landing_disp_y_stage2 = [-0.20, 0.20]  # lateral landing offset (m), Stage 2

        class ranges(GO2OmniJumpCurriculumTorqueCfg.commands.ranges):
            jump_height = [0.40, 0.70]   # unchanged from the proven May28 baseline
            lin_vel_x = [0.0, 0.0]       # repurposed: landing dx (m). Stage 1 = land in place.
            lin_vel_y = [0.0, 0.0]       # repurposed: landing dy (m). Stage 1 = land in place.
            ang_vel_yaw = [0.0, 0.0]

    class rewards(GO2OmniJumpCurriculumTorqueCfg.rewards):
        # Landing-reward kernel widths + real-jump gate for the sparse terminal term.
        sigma_pos_landing = 0.05            # terminal landing xy (sparse) — tight
        sigma_landing_proj = 0.10           # in-flight ballistic estimate — looser (noisy)
        landing_real_jump_min_peak = 0.40   # peak gate for the SPARSE landing_position reward
                                            # (omnijump squat settles ~0.31, real jump peaks ~0.56)
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
        squat_pose_threshold = 2.8          # "in the squat" = pose_err<=2.8, ~= 60% of the way down from
                                            # standing (7.1). THE depth knob: stuck-not-jumping (can't fold
                                            # enough) -> RAISE; jumps too shallow / want a deeper load -> LOWER.
        squat_hold_steps = 40               # jump chain unlocks only after the squat POSE is HELD within
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
        squat_gate_height = 0.24            # must dip base to <=0.24m (idle ~0.31) to unlock jump rewards
        successful_jump_min_peak_height = 0.40  # was 0.30: a ~0.34 "low pop" no longer counts as success
                                                # (= command floor 0.40; kills the low-jump shortcut)
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

        class scales(GO2OmniJumpCurriculumTorqueCfg.rewards.scales):
            # ---- proven jump-driving stack inherited UNCHANGED ----
            #   takeoff_vertical_velocity=10, projected_peak=15, successful_jump=300,
            #   orientation=-1.6, collision=-3.0, default_pos=-0.3, default_hip_pos=0.3, ...
            # ---- landing layer (new) ----
            tracking_linear_velocity = 0.0   # was 0.5: commands[0:2] is now meters, not m/s
            projected_landing = 10.0         # dense horizontal shaper (vs projected_peak=15 -> height stays prioritized)
            landing_position = 30.0          # sparse terminal landing-at-target bonus (real-jump gated)
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
            foot_contact_sync = -3.0         # was -2.0: still visibly uneven at takeoff/landing in play.
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
            # ---- air/landing pose quality: revived from 0.4 (near-dead) to actually pull pose ----
            joint_angle_aerial = 1.5         # was 0.4: tuck pose (q_air) in flight — main in-air stability lever
            joint_angle_prelanding = 1.5     # was 0.4: pre-landing pose (q_pre)
            joint_angle_landing = 1.5        # was 0.4: landing pose (q_ground)
            # ---- post-PD pose-holding (rear legs drifted once PD faded to 0) ----
            default_pos = -0.5               # was -0.3: stronger pose anchor so RL holds posture WITHOUT PD.
                                             # _reward_default_pos override zeros it during push-off so this does NOT cap the jump.
            orientation = -2.0               # was -1.6: mild bump — directly hold body attitude (less wobble post-PD)
            # ---- (1)+(2) landing stability from the papers: stop "lands then flips" ----
            base_ang_vel_xy = -0.05          # (1) PENALTY on base roll/pitch angular velocity in flight+landing
                                             # (Olsen ϕσ(‖ω‖) / Atanassov "track zero ω after landing"). We had NO
                                             # ω damping -> body tumbled into touchdown. Penalty (not bell kernel) so
                                             # it bites; penalizes spin RATE not airborne time -> clean high jump unhurt.
                                             # THE knob: still flipping -> more negative; jumps get stiff/low -> back off.
            landing_impact = -2.0            # (2) Olsen Ground-force/Soft-impact: penalize the vertical foot-force
                                             # SPIKE at touchdown (bounded, soft floor at ~standing weight) -> cushion,
                                             # don't slam. KEEP MODEST: too negative incentivizes jumping LOWER
                                             # (smaller fall = softer impact) and suppresses height. #1 (ω damping)
                                             # is the primary lever; this is secondary. peak drops -> back off toward 0.

    class logging(GO2OmniJumpCurriculumTorqueCfg.logging):
        print_episode_keys = GO2OmniJumpCurriculumTorqueCfg.logging.print_episode_keys + [
            "rew_projected_landing",
            "rew_landing_position",
            "rew_foot_contact_sync",
            "rew_stance_squat",     # countermovement shaping — watch vs successful_jump_rate
            # inherited-active but missing from the parent's print list — surface them
            "rew_joint_angle_aerial",
            "rew_joint_angle_prelanding",
            "rew_joint_angle_landing",
            "rew_base_ang_vel_xy",   # (1) flight+landing roll/pitch ω damping — the anti-tumble lever
            "rew_landing_impact",    # (2) touchdown force-spike penalty — cushion vs slam
            "squat_qualified_rate",  # frac of takeoffs preceded by a HELD squat; compare to jump_flight_rate
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

    class runner(GO2OmniJumpCurriculumTorqueCfgPPO.runner):
        experiment_name = "go2_omnijump_landing_torque"
        run_name = "stage1_landing"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        max_iterations = 6000
        # entropy_coef annealing (read by OnPolicyRunner.learn): keep 0.005 early to discover
        # the jump (~iter2000), then DROP to 0.001 at iter 2800 so the policy converges
        # (noise_std tightens) as PD fades over iter ~1000-5500. Confirmed winner: noise_std
        # converged 0.69->0.48, late decline gone, greedy play peak ~0.576.
        entropy_anneal_iter = 2800
        entropy_coef_final = 0.001
