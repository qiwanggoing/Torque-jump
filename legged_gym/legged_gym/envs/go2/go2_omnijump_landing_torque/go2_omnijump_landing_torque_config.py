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
        # Give the dip+push room: a countermovement (~0.3-0.35s) does not fit the old 40-step
        # (0.2s) takeoff window — the dip would eat the budget and trip the timeout. 80 steps
        # = 0.4s. (step = sim dt 0.005s, counted on physics substeps, so freq-independent.)
        takeoff_timeout_steps = 80          # was 40 (0.2s, only enough for stand-and-pop)

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
            stance_squat = 0.5               # dense: reward base dipping toward ~0.20m while commanded
                                             # to jump but still on the ground (jumping & ~taken_off).
                                             # Small weight on purpose: must NOT outweigh takeoff_vz(15)/
                                             # projected_peak(20)/successful_jump(400), so "squat then JUMP"
                                             # stays net-positive vs squatting idle. Est ~0.10/episode (≈ takeoff_vz);
                                             # squat-and-don't-jump farm caps ~0.2 but forfeits ~0.4 of jump rewards.
            landing_stability = 0.0          # was 1.0: exp(-landing_velocity/0.25) but touchdown velocities are
                                             # >> 0.25, so it floors at ~0 and never fires (contributed 0.0002).
                                             # If landing damping is wanted later, re-add with a much looser sigma.
            # ---- four-foot contact-timing sync (penalty on staggered takeoff/landing) ----
            foot_contact_sync = -2.0         # penalize 1-3 feet on the ground during the takeoff push / landing
                                             # window -> all four feet leave & touch down TOGETHER (less body tilt).
                                             # (replaced the old joint-based pushoff_leg_sync.)
            # ---- air/landing pose quality: revived from 0.4 (near-dead) to actually pull pose ----
            joint_angle_aerial = 1.5         # was 0.4: tuck pose (q_air) in flight — main in-air stability lever
            joint_angle_prelanding = 1.5     # was 0.4: pre-landing pose (q_pre)
            joint_angle_landing = 1.5        # was 0.4: landing pose (q_ground)
            # ---- post-PD pose-holding (rear legs drifted once PD faded to 0) ----
            default_pos = -0.5               # was -0.3: stronger pose anchor so RL holds posture WITHOUT PD.
                                             # _reward_default_pos override zeros it during push-off so this does NOT cap the jump.
            orientation = -2.0               # was -1.6: mild bump — directly hold body attitude (less wobble post-PD)

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
