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
        # Enable all-time linear velocity tracking matching OmniNet Table I semantics
        tracking_linear_velocity_all_time = True
        tracking_sigma = 0.5

        class scales(GO2OmniJumpLandingTorqueCfg.rewards.scales):
            # =========================================================================
            # 1. STRIP AWAY LANDING PRECISION & BRAKING CONSTRAINTS
            # =========================================================================
            # Remove penalties for landing away from a specific (dx, dy) coordinate
            projected_landing = 0.0
            landing_position = 0.0
            takeoff_velocity_match = 0.0
            # Remove post-landing velocity braking so high forward momentum is not taxed
            landing_stability = 0.0

            # =========================================================================
            # 2. ACTIVATE OMNINET & REACH-MAXIMIZATION DRIVERS
            # =========================================================================
            # Activate OmniNet Table I linear velocity tracking to drive forward momentum
            tracking_linear_velocity = 2.0
            # Keep our massive forward reach distance reward as the dominant objective
            forward_reach = 60.0
            # Keep flight-time and jump success rewards (gated by our squat gate height)
            all_feet_airborne = 3.0
            successful_jump = 1000.0

            # =========================================================================
            # 3. KEEP OUR PROVEN TORQUE TAKEOFF & ATTITUDE PROTECTION NET UNCHANGED
            # =========================================================================
            # stance_squat=3.0, clean_takeoff_bonus=3.0, foot_contact_sync=-4.0,
            # jump_pitch_extra=12.0, landing_pitch_extra=5.0, default_hip_pos=2.0, etc.
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
            "rew_forward_reach",
            "rew_tracking_linear_velocity",
        ]


class GO2OmniJumpFarTorqueCfgPPO(GO2OmniJumpLandingTorqueCfgPPO):
    class runner(GO2OmniJumpLandingTorqueCfgPPO.runner):
        experiment_name = "go2_omnijump_far_torque"
        run_name = "far_jump_v1"
        max_iterations = 3000
