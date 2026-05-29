from legged_gym.envs.go2.go2_omnijump_torque.go2_omnijump_torque_config import (
    GO2OmniJumpTorqueCfg,
    GO2OmniJumpTorqueCfgPPO,
)


class GO2OmniJumpCurriculumTorqueCfg(GO2OmniJumpTorqueCfg):
    class growth(GO2OmniJumpTorqueCfg.growth):
        start_torque_scale = 1.0   # disable cur_scale ramp; RL effective scale ramp reduced from 4× (5.875→23.5) to 2× (11.75→23.5) — matches mygo2jump 2.35× shape and gives RL ~12Nm authority from iter 0
        k = 0.0001                 # unused under linear schedule (kept for compatibility)
        warmup_steps = 96000       # PD stays at full 0.5 until step_count ≥ 96000 (~iter 1000 at freq=100). Lets RL bootstrap with PD support before fade starts.
        x0 = 384000                # linear-fade end: general_scale=1, pd_alpha=0 at step_count=384000 (~iter 4000 at ~96 step/iter). 3000-iter slow ramp; ~1.67pp PD drop per 100 iter (was 2.5pp).

    class curriculum:
        enabled = False
        stage = "metric"
        ema_alpha = 0.10
        min_updates_per_stage = 8
        stand_stage_reward_threshold = 1.80
        zero_command_base_height_threshold = 0.25
        default_hip_pos_threshold = 0.05
        takeoff_vertical_velocity_threshold = 0.45
        takeoff_flight_rate_threshold = 0.03
        flight_rate_threshold = 0.45
        flight_mean_peak_height_threshold = 0.20
        landing_rate_threshold = 0.55
        successful_jump_rate_threshold = 0.45
        stand_command_prob_before_takeoff = 1.0
        stand_command_prob_after_takeoff = 0.20
        velocity_command_start_stage = 1
        force_stage = -1
        play_stage = 4
        play_stand_command_prob = 0.0
        notes = "Disabled: train all jump rewards and commands in one stage, OmniNet-style."

    class control(GO2OmniJumpTorqueCfg.control):
        curriculum_action_scale_start = 10.0
        curriculum_action_scale_end = GO2OmniJumpTorqueCfg.control.action_scale
        curriculum_rl_prior_start = 0.50
        curriculum_rl_prior_end = 0.50
        curriculum_pd_prior_start = 0.50
        curriculum_pd_prior_end = 0.50

    class commands(GO2OmniJumpTorqueCfg.commands):
        single_jump_command_prob = 1.0
        # Curriculum: shift jump_height lower bound after step_count threshold
        jump_height_curriculum_switch_step = 240000   # ~iter 2500 (PD fade mid-period)
        jump_height_curriculum_lower_after = 0.55      # narrow range to [0.55, 0.70] to force higher jumps
        class ranges(GO2OmniJumpTorqueCfg.commands.ranges):
            jump_height = [0.40, 0.70]
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class rewards(GO2OmniJumpTorqueCfg.rewards):
        zero_command_velocity_sigma = 0.25
        zero_command_yaw_sigma = 0.25
        zero_command_height_gain = 10.0
        default_hip_pos_gain = 4.0
        pose_tracking_sigma = 0.20
        prelanding_tracking_sigma = 0.20
        joint_symmetry_tracking_sigma = 0.25
        success_height_tolerance = 0.10
        success_use_velocity_score = False
        task_max_height_sigma = 0.05
        height_tracking_sigma = 0.05
        tracking_linear_velocity_all_time = False
        landing_buffer_steps = 25    # was 50; shorter buffer = give policy faster credit for surviving landing
        stand_rearm_steps = 5
        one_jump_reward_per_episode = True
        rsi_prob = 0.2
        rsi_vel_z_min = 1.0
        rsi_vel_z_max = 3.0
        rsi_height_offset_min = 0.0
        rsi_height_offset_max = 0.1   # smaller spread; RSI now starts from squat (~0.20m), not standing
        post_jump_stand_steps = 300   # was 80 (~0.8s); extended to 3s so robot trains an explicit autonomous-stand phase under PD fade conditions

        class scales(GO2OmniJumpTorqueCfg.rewards.scales):
            maintain_contact = 0.10            # moderate: standing anchor without dominating takeoff signal
            peak_height_progress = 0.0         # disabled: projected_peak subsumes this
            all_feet_airborne = 2.0            # boosted (was 1.0): bigger airborne reward
            takeoff_vertical_velocity = 10.0   # boosted (was 4.0): strong stance push signal — primary lever to break "don't jump" mode
            projected_peak = 15.0              # sole height tracker: compensates for disabled height_tracking + peak_height_progress
            termination = -10.0                # not in OmniNet, kept for base-contact episodes
            orientation = -1.6                 # boosted (was -0.8): stronger upright pull during all phases
            collision = -3.0                   # boosted (was -1.0): kill leg-leg self-collision in air
            torques = -1e-5                    # OmniNet: -1e-5
            action_rate = -0.015               # halved from -0.03: allow more explosive pushoff
            dof_acc = -2.5e-7                  # restored to original: was over-penalizing fast (smooth) motion
            horizontal_drift = 0.0            # disabled: dense takeoff_direction subsumes this
            takeoff_direction = 3.0            # dense ascending vz/||v|| (was 80 one-shot; ~40 steps × 3.0 × 0.85 ≈ equivalent total)
            height_tracking = 0.0              # disabled: projected_peak subsumes this
            successful_jump = 300.0            # boosted (was 200): make completion reward dominate to overcome ep-short collapse
            tracking_linear_velocity = 0.5
            tracking_angular_velocity = 0.0
            joint_angle_loaded = 0.4
            joint_angle_extended = 0.0
            default_pos = -0.3                 # mygo2jump weight; now spans ALL 12 joints (fix: was hip-only and 1/3 weight). Strong pose anchor toward default standing pose — critical for surviving PD=0 stand.
            default_hip_pos = 0.3              # mygo2jump-style exp keep hip joints near default (no outward/inward drift)
            aerial_dof_acc = -1e-6             # airborne-only joint accel penalty (4× global dof_acc); targets in-air twitching/flailing observed after PD fades out
            joint_angle_aerial = 0.4
            joint_angle_prelanding = 0.4
            joint_angle_landing = 0.4
            landing_stability = 1.0

    class logging(GO2OmniJumpTorqueCfg.logging):
        print_episode_keys = [
            "rew_height_tracking",
            "rew_peak_height_progress",
            "rew_all_feet_airborne",
            "rew_maintain_contact",
            "rew_takeoff_vertical_velocity",
            "rew_projected_peak",
            "rew_successful_jump",
            "rew_orientation",
            "rew_joint_angle_loaded",
            "rew_joint_angle_extended",
            "rew_collision",
            "rew_termination",
            "rew_torques",
            "rew_action_rate",
            "rew_dof_acc",
            "rew_aerial_dof_acc",
            "rew_horizontal_drift",
            "rew_takeoff_direction",
            "rew_default_pos",
            "rew_default_hip_pos",
            "jump_flight_rate",
            "jump_landing_rate",
            "jump_completed_cycles",
            "successful_jump_rate",
            "mean_peak_height",
        ]

    class test(GO2OmniJumpTorqueCfg.test):
        vel = GO2OmniJumpTorqueCfg.test.vel.clone()
        single_jump_play = True    # mirror training: one jump per episode, then stand for post_jump_stand_steps


class GO2OmniJumpCurriculumTorqueCfgPPO(GO2OmniJumpTorqueCfgPPO):
    class policy(GO2OmniJumpTorqueCfgPPO.policy):
        init_noise_std = 0.50                # was 0.35; need more action-space coverage to discover coordinated push-off

    class algorithm(GO2OmniJumpTorqueCfgPPO.algorithm):
        entropy_coef = 0.005                 # fresh-start exploration; resist premature noise_std collapse

    class runner(GO2OmniJumpTorqueCfgPPO.runner):
        experiment_name = "go2_omnijump_curriculum_torque"
        run_name = "auto_curriculum"
        resume = False                       # from-scratch with positive pose rewards + new smoothness balance
        load_run = -1
        checkpoint = -1
        resume_path = None
        max_iterations = 6000
