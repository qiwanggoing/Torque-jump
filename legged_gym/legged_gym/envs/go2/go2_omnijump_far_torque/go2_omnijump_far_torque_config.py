"""Configuration for the Maximum Forward Reach / Far Jump task (go2_omnijump_far_torque).

Built directly on top of our latest proven landing-torque foundation (GO2OmniJumpLandingTorqueCfg),
this task strips away the "precise landing coordinate (dx, dy)" and "post-landing braking" constraints
so the policy is no longer penalized for forward momentum or sliding upon touchdown.

By combining:
1. Our pure-torque takeoff & countermovement protection net (squat_gate_height=0.24m, stance_squat,
   clean_takeoff_bonus, foot_contact_sync, jump_pitch_extra=12.0)
2. OmniNet's linear velocity tracking (tracking_linear_velocity=2.0)
3. Our physical forward reach distance reward (forward_reach=60.0)

We push the quadruped to release maximum physical impulse into forward jumping (Far Jump).
"""

from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque_config import (
    GO2OmniJumpLandingTorqueCfg,
    GO2OmniJumpLandingTorqueCfgPPO,
)


class GO2OmniJumpFarTorqueCfg(GO2OmniJumpLandingTorqueCfg):
    class commands(GO2OmniJumpLandingTorqueCfg.commands):
        # Disable the target-landing distance curriculum so we immediately demand high forward reach
        landing_dx_curriculum = False
        landing_stage = 2

        class ranges(GO2OmniJumpLandingTorqueCfg.commands.ranges):
            # Demand massive forward jumping reach / speed (1.0m to 2.0m target reach)
            lin_vel_x = [1.0, 2.0]
            lin_vel_y = [-0.3, 0.3]
            jump_height = [0.40, 0.60]
            ang_vel_yaw = [-0.5, 0.5]

    class rewards(GO2OmniJumpLandingTorqueCfg.rewards):
        # (quadratic forward_reach REVERTED -> linear, see far env: it squashed distance in the achievable
        # <1m range and made the policy a vertical hopper.)
        tracking_linear_velocity_all_time = True
        tracking_sigma = 0.5

        # ---- RSI-FORWARD (expand reach past the ~0.7m ceiling; Olsen 2025 RSI-along-projectile) ----
        # Air-drop a fraction of resets into a squat that is LAUNCHING FORWARD (matched ballistic vx,vz to a
        # far distance) so the value fn learns "far forward flight = high return" and the standing policy
        # chases a bigger forward push. This is the ONE paper-backed lever for breaking the local optimum
        # where the rear thighs sit idle (torque_diag: rear-thigh cmd/Y1<1 = unused gradient). The forward vx
        # is GATED on the succ-latch (_takeoff_omega_on) so it never fires before the jump is discovered.
        rsi_prob = 0.15                # 15% of resets = RSI air-drop
        rsi_static_frac = 0.4          # of those: 40% static-squat (push-from-standstill / countermovement
                                       # half, keeps the fold trained), 60% launch-forward (reach-expansion)
        rsi_forward_vx = True          # give the LAUNCH sub-mode a matched forward vx (teach DISTANCE)
        rsi_forward_dist_min = 0.8     # ballistic target distance for the forward-launch RSI: just BEYOND the
        rsi_forward_dist_max = 1.4     # current ~0.7 reach -> a REPRODUCIBLE "reach a bit farther" pull
        rsi_forward_vx_max = 4.0       # clamp the injected forward speed (avoid unphysical air-drops)
        rsi_vel_z_min = 1.5            # launch vz range (total apex ~0.31-0.66) that pairs with the forward vx
        rsi_vel_z_max = 3.0

        class scales(GO2OmniJumpLandingTorqueCfg.rewards.scales):
            # =========================================================================
            # 1. STRIP AWAY LANDING PRECISION & BRAKING CONSTRAINTS
            # =========================================================================
            # Remove penalties for landing away from a specific (dx, dy) coordinate
            projected_landing = 0.0
            landing_position = 0.0
            takeoff_velocity_match = 0.0
            # Re-enable post-landing velocity braking to absorb touchdown impact and stop post-landing bouncing
            landing_stability = 2.0

            # =========================================================================
            # 2. MAX-DISTANCE OBJECTIVE (2026-07-08 fix: the previous run became a VERTICAL HOPPER)
            # =========================================================================
            # forward_reach (LINEAR, inherited) = the SOLE dominant objective. It rewards the ballistic
            # projected forward reach (vx * flight_time) -> optimizes launch SPEED and ANGLE for max distance,
            # and IMPLICITLY values the height that lengthens flight time -> height needs no separate reward.
            forward_reach = 60.0
            # projected_peak (HEIGHT) -> 0: it was the vertical-hop attractor (earned rew 0.278, dominant). The
            # policy farmed pure height (peak 0.50) and barely moved forward (~0.13m). Kill it; let height EMERGE
            # from the ~45deg ballistic optimum via forward_reach's flight-time term.
            projected_peak = 0.0
            # tracking_linear_velocity -> 0: SEMANTICS BUG. commands[0:2] are METRES (landing displacement), not
            # m/s; tracking them as an all-time velocity target rewarded forward DRIFT/slide (earned rew 0.286,
            # dominant) and muddied the objective. forward_reach already carries the takeoff-velocity signal.
            tracking_linear_velocity = 0.0
            # USE ALL CAPABILITY (user): wake the IDLE REAR THIGHS. torque_diag proved the rear thighs sit at
            # cmd/Y1 < 1 (unused gradient) while the policy scales distance ONLY via the rear HIP. four_leg_push
            # grades each ON-GROUND leg up to a GRF target (surplus is free -> no front==rear constraint) so an
            # idling leg drags the mean down -> the policy recruits it. Gated (succ-latch + real-push force floor
            # > body weight) = discovery-safe. WATCH value_loss: it blew the critic once at a higher weight -> if
            # value_loss spikes or it won't jump forward, set this to 0 first when bisecting.
            four_leg_push = 5.0
            # Keep flight-time and jump success rewards (gated by our squat gate height)
            all_feet_airborne = 3.0
            successful_jump = 1000.0

            # =========================================================================
            # 3. ANTI-STUTTER & LANDING STABILITY ANCHORS
            # =========================================================================
            # Strongly incentivize a clean, single-push takeoff without any stutter-step (re-plant)
            clean_takeoff_bonus = 25.0  # raised from inherited 3.0 to overcome the +60.0 forward_reach pull
            # Reward four feet planted when not airborne to prevent post-landing foot shuffling
            maintain_contact = 0.5      # raised from inherited 0.3
            # stance_squat=3.0, foot_contact_sync=-4.0, jump_pitch_extra=12.0, etc.
            # are all inherited intact from GO2OmniJumpLandingTorqueCfg.rewards.scales!

    class logging(GO2OmniJumpLandingTorqueCfg.logging):
        print_episode_keys = [
            k for k in GO2OmniJumpLandingTorqueCfg.logging.print_episode_keys
            if k not in (
                "rew_projected_landing",
                "rew_landing_position",
                "rew_takeoff_velocity_match",
                "rew_landing_stability",
                "landing_dx_max",
                "landing_dx_mean",
                "landing_dx_min",
                "landing_dx_stable_cum",
                "landing_stable_hit_uniform",
                "landing_hit_rate",
                "landing_farband_hit_smooth",
            )
        ] + [
            "rew_forward_reach",       # ★ the distance objective — should now DOMINATE and rise
            "rew_four_leg_push",        # rear-thigh recruitment (use-all-capability lever) — watch it wake
            "rew_projected_peak",       # HEIGHT — now weight 0, should read ~0 (no vertical-hop farming)
        ]


class GO2OmniJumpFarTorqueCfgPPO(GO2OmniJumpLandingTorqueCfgPPO):
    class algorithm(GO2OmniJumpLandingTorqueCfgPPO.algorithm):
        entropy_coef = 0.005          # was 0.003: explore harder to ESCAPE the vertical-hop local optimum the
                                      # previous run collapsed into (noise_std fell to 0.044 by iter1000).
    class runner(GO2OmniJumpLandingTorqueCfgPPO.runner):
        experiment_name = "go2_omnijump_far_torque"
        run_name = "far_jump_v1"
        max_iterations = 3000
        entropy_anneal_iter = 1500    # was 500: keep exploring forward launches much longer before annealing.
        entropy_coef_final = 0.001    # WATCH noise_std: if it runs away past ~0.5, lower entropy_coef.
