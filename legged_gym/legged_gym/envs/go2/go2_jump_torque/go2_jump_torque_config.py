import torch

from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO


class GO2JumpTorqueCfg(GO2TorqueCfg):
    class env(GO2TorqueCfg.env):
        num_observations = 67
        num_actions = 12
        episode_length_s = 4

    class init_state(GO2TorqueCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class terrain(GO2TorqueCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False

    class asset(GO2TorqueCfg.asset):
        terminate_after_contacts_on = ["Head", "base"]
        penalize_contacts_on = ["thigh", "calf"]

    class commands:
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = False
        repeat_jump_in_episode = False
        use_landing_target = False
        landing_dx_curriculum = False
        landing_dx_curriculum_mode = "success_rate"
        landing_dx_start = [0.45, 0.55]
        landing_dx_warmup_steps = 50000
        landing_dx_ramp_steps = 200000
        landing_dx_success_threshold = 0.80
        landing_dx_success_ema_alpha = 0.10
        landing_dx_progress_increment = 0.02
        landing_dx_progress_decrement = 0.0
        landing_dx_regress_threshold = 0.55
        landing_dx_success_position_cutoff = 0.12

        class ranges:
            jump_toggle = [1.0, 1.0]
            lin_vel_x = [0.8, 1.4]   # target launch speed m/s
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class curriculum:
        enabled = True
        auto_advance = True
        initial_stage = "takeoff_foundation"
        stage_order = [
            "takeoff_foundation",
            "forward_launch",
            "target_landing",
        ]

        class takeoff_foundation:
            landing_dx_range = [0.15, 0.25]
            probability = 0.0
            takeoff_state_probability = 0.0
            takeoff_forward_velocity_fraction_range = [0.00, 0.30]
            forward_velocity_fraction_range = [0.10, 0.50]
            takeoff_upward_velocity_range = [0.20, 0.50]
            relax_nonfoot_termination = True

            class reward_scales:
                landing_target_progress = 0.0
                estimated_landing_target_tracking = 0.0
                landing_position = 0.0
                long_jump_success = 0.0
                jump_distance = 0.0
                valid_jump_success = 8.0
                takeoff_squat = 5.0
                takeoff_impulse = 60.0
                takeoff_release = 35.0
                takeoff_height_progress = 30.0
                takeoff_velocity = 24.0
                takeoff_forward_push = 2.0
                takeoff_forward_velocity = 1.0
                tracking_linear_velocity = 0.0
                feet_clearance = 0.5
                body_attitude = 3.0
                landing_stability = 0.5
                stand_height = 3.0
                default_pose_hold = 1.5
                joint_pose_aerial = 0.5
                joint_pose_prelanding = 1.0
                joint_pose_landing = 0.5
                nonfoot_contact = 0.0

        class forward_launch:
            landing_dx_range = [0.25, 0.40]
            probability = 0.18
            takeoff_state_probability = 1.00
            takeoff_forward_velocity_fraction_range = [0.20, 0.80]
            forward_velocity_fraction_range = [0.30, 0.90]
            takeoff_upward_velocity_range = [0.18, 0.45]

            class reward_scales:
                landing_target_progress = 0.0
                estimated_landing_target_tracking = 0.0
                landing_position = 0.0
                long_jump_success = 0.0
                jump_distance = 4.0
                valid_jump_success = 10.0
                takeoff_squat = 0.25
                takeoff_impulse = 20.0
                takeoff_release = 30.0
                takeoff_height_progress = 30.0
                takeoff_velocity = 20.0
                takeoff_forward_push = 10.0
                takeoff_forward_velocity = 8.0
                tracking_linear_velocity = 4.0
                feet_clearance = 2.0
                body_attitude = 4.0
                landing_stability = 1.5
                stand_height = 0.5
                default_pose_hold = 0.25
                joint_pose_aerial = 0.25
                joint_pose_landing = 1.0

        class target_landing:
            landing_dx_range = [0.45, 0.55]
            probability = 0.10
            takeoff_state_probability = 1.00
            takeoff_forward_velocity_fraction_range = [0.20, 0.80]
            forward_velocity_fraction_range = [0.40, 1.00]
            takeoff_upward_velocity_range = [0.15, 0.45]

            class reward_scales:
                landing_target_progress = 0.0
                estimated_landing_target_tracking = 0.0
                landing_position = 0.0
                long_jump_success = 0.0
                jump_distance = 4.0
                valid_jump_success = 12.0
                takeoff_squat = 0.0
                takeoff_impulse = 8.0
                takeoff_release = 0.0
                takeoff_height_progress = 10.0
                takeoff_velocity = 20.0
                takeoff_forward_push = 0.0
                takeoff_forward_velocity = 8.0
                tracking_linear_velocity = 6.0
                feet_clearance = 1.0
                body_attitude = 4.0
                landing_stability = 2.0
                stand_height = 0.0
                default_pose_hold = 0.0
                joint_pose_aerial = 0.0
                joint_pose_landing = 1.0

        class auto_thresholds:
            takeoff_foundation_natural_flight_rate = 0.30
            forward_launch_natural_flight_rate = 0.50
            forward_launch_valid_jump_rate = 0.30
            forward_launch_stable_landing_forward_error = -0.20

    class logging:
        # Keep TensorBoard complete, but make terminal logs readable during reward debugging.
        print_episode_keys = [
            "curriculum_stage_index",
            # --- 核心跳跃奖励 ---
            "rew_takeoff_squat",
            "rew_takeoff_impulse",
            "rew_takeoff_velocity",
            "rew_takeoff_forward_velocity",
            "rew_takeoff_release",
            "rew_takeoff_height_progress",
            "rew_takeoff_forward_push",
            "rew_feet_clearance",
            # --- 任务奖励 ---
            "rew_jump_distance",
            "rew_valid_jump_success",
            "rew_tracking_linear_velocity",
            # --- 辅助奖励 ---
            "rew_phase_contact_sync",
            "rew_body_attitude",
            "rew_landing_stability",
            "rew_ground_creeping",
            "rew_landing_impact",
            "rew_nonfoot_contact",
            # --- 跳跃统计 ---
            "jump_flight_rate",
            "jump_landing_rate",
            "jump_completed_cycles",
            "rsi_used_rate",
            "natural_flight_rate",
            "natural_completed_cycles",
            "prepare_stand_ready_rate",
            "target_landing_dx",
            "clearance_gate",
            "valid_jump_rate",
            "completed_airborne_steps",
            "stable_landing_forward_error",
            "term_nonfoot_contact_rate",
            "term_rebound_rate",
        ]

    class control(GO2TorqueCfg.control):
        pd_residual_enabled = True
        pd_fade_enabled = False
        pd_gain_scale = 1.0
        pd_torque_limit = 10.0
        pd_torque_limit_fraction = 0.4
        pd_warmup_steps = 200 * 24
        pd_ramp_steps = 800 * 24

    class noise(GO2TorqueCfg.noise):
        class noise_scales(GO2TorqueCfg.noise.noise_scales):
            contact = 0.05

    class jump_phase:
        request_threshold = 0.5
        cycle_time = 1.5
        takeoff_phase_start = 0.35
        flight_phase_start = 0.55
        landing_phase_start = 0.80
        stand_height_target = 0.35
        prepare_min_contacts = 4
        prepare_ready_min_contacts = 4
        flight_max_contacts = 0
        landing_min_contacts = 2
        recovery_min_contacts = 4
        prepare_min_steps = 1
        compression_min_steps = 1
        compression_target_depth = 0.001
        extension_min_steps = 1
        extension_entry_velocity = 0.01
        release_min_steps = 1
        prepare_reward_max_steps = 120
        prepare_stand_hold_steps = 6
        prepare_stand_height_tol = 0.15
        prepare_stand_pose_tol = 1.50
        prepare_stand_max_lin_vel = 0.50
        prepare_stand_max_vertical_vel = 0.50
        prepare_stand_max_ang_vel = 4.0
        prepare_stand_max_tilt = 0.50
        min_airborne_steps = 2
        landing_settle_steps = 2
        landing_complete_steps = 1
        recovery_settle_steps = 1
        recovery_hold_steps = 12
        takeoff_max_steps = 120
        rebound_airborne_steps = 2
        rebound_min_upward_velocity = 0.05
        terminate_on_rebound = True
        terminate_on_takeoff_abort = False
        min_flight_height = 0.02
        min_flight_stand_clearance = 0.0
        min_takeoff_velocity = 0.15
        takeoff_entry_velocity = 0.04
        takeoff_abort_velocity = 0.05
        recovery_max_lin_vel = 0.6
        recovery_max_vertical_vel = 0.5
        recovery_max_ang_vel = 2.5
        recovery_max_tilt = 0.45

    class rsi:
        probability = 0.0
        takeoff_state_probability = 1.00
        takeoff_height_offset_range = [-0.04, 0.00]
        takeoff_upward_velocity_range = [0.15, 0.45]
        takeoff_forward_velocity_fraction_range = [0.20, 0.80]
        takeoff_phase_range = [0.45, 0.53]
        height_offset_range = [0.03, 0.08]
        upward_velocity_range = [0.25, 0.65]
        forward_velocity_fraction_range = [0.40, 1.00]

    class rewards(GO2TorqueCfg.rewards):
        only_positive_rewards = False
        use_height_objective = False
        use_height_gates = True

        # Legacy height knobs are kept for ablations, but disabled for this task.
        fixed_jump_height = 0.12
        jump_height_curriculum = False
        jump_height_start = 0.04
        jump_height_warmup_steps = 60000
        jump_height_ramp_steps = 120000
        height_tracking_sigma = 0.06
        success_height_tolerance = 0.02
        reference_jump_height = 0.18
        height_reward_floor = 0.02
        base_height_reward_floor = 0.30
        base_height_reward_target = 0.40
        success_height_floor = 0.04
        success_height_target = 0.12
        success_base_height_floor = 0.34
        success_base_height_target = 0.42
        takeoff_height_progress_target = 0.08
        clearance_gate_floor = 0.005
        clearance_gate_target = 0.04
        success_clearance_min = 0.04
        valid_jump_min_airborne_steps = 8
        takeoff_velocity_floor = 0.0
        takeoff_velocity_target = 0.55
        takeoff_velocity_start = 0.20
        takeoff_velocity_curriculum = True
        takeoff_velocity_warmup_steps = 12000
        takeoff_velocity_ramp_steps = 60000
        takeoff_velocity_support_min = 0.25
        takeoff_velocity_contact_gate_min = 0.10
        takeoff_velocity_sigma = 0.5
        takeoff_release_velocity_floor = 0.12
        takeoff_release_velocity_target = 0.45
        takeoff_release_clearance_floor = 0.0
        takeoff_release_clearance_target = 0.02
        takeoff_release_clearance_min_gate = 0.25
        takeoff_squat_floor = 0.01
        takeoff_squat_target = 0.035
        takeoff_squat_max_xy_speed = 0.35
        takeoff_squat_tilt_sigma = 0.25
        takeoff_forward_velocity_floor = 0.0
        takeoff_forward_velocity_min = 0.15
        takeoff_forward_velocity_max = 0.85
        takeoff_forward_velocity_lateral_sigma = 0.45
        takeoff_forward_velocity_vertical_gate_floor = 0.0
        takeoff_forward_velocity_vertical_gate_target = 0.30
        takeoff_forward_velocity_vertical_gate_min = 0.0
        stand_height_target = 0.35
        stand_height_sigma = 0.05
        default_pose_sigma = 0.35
        landing_position_sigma = 0.08
        landing_position_cutoff = 0.15
        jump_distance_reference_min = 0.15
        landing_progress_lateral_sigma = 0.12
        ground_creeping_distance = 0.06
        estimated_landing_sigma = 0.24
        estimated_landing_lateral_sigma = 0.18
        estimated_landing_time_min = 0.03
        estimated_landing_time_max = 0.80
        landing_success_tilt_sigma = 0.25
        landing_yaw_sigma = 0.30
        landing_yaw_cutoff = 0.70
        landing_tilt_sigma = 0.25
        landing_tilt_cutoff = 0.45
        landing_reward_height_gate_floor = 0.02
        landing_reward_height_gate_target = 0.08
        landing_aux_height_gate_min = 0.00
        phase_sync_height_gate_floor = 0.02
        phase_sync_height_gate_target = 0.08
        prelanding_phase_start = 0.70
        flight_time_ref = 0.35
        tracking_lin_vel_sigma = 0.50
        tracking_yaw_rate_sigma = 1.00
        ang_vel_xy_sigma = 1.20
        feet_clearance_target = 0.06
        feet_clearance_height_gate_floor = 0.02
        feet_clearance_height_gate_target = 0.08
        feet_clearance_height_gate_min = 0.05
        feet_clearance_sigma = 0.04
        body_tilt_sigma = 0.25
        body_ang_vel_xy_sigma = 1.20
        body_attitude_takeoff_weight = 0.35
        body_attitude_height_gate_min = 0.40
        landing_body_height_floor = 0.26
        landing_body_height_target = 0.34
        landing_tilt_scale = 5.0
        landing_xy_vel_scale = 3.0
        landing_ang_vel_scale = 2.0
        landing_vertical_vel_scale = 4.0
        landing_impact_velocity_sigma = 0.35
        landing_impact_xy_sigma = 0.60
        landing_impact_tilt_sigma = 0.35
        landing_success_window_steps = 6
        joint_pose_sigma = 0.45

        aerial_pose_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 1.45,
            'RL_thigh_joint': 1.45,
            'FR_thigh_joint': 1.45,
            'RR_thigh_joint': 1.45,
            'FL_calf_joint': -2.5,
            'RL_calf_joint': -2.5,
            'FR_calf_joint': -2.5,
            'RR_calf_joint': -2.5,
        }
        prelanding_pose_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.65,
            'RL_thigh_joint': 0.85,
            'FR_thigh_joint': 0.65,
            'RR_thigh_joint': 0.85,
            'FL_calf_joint': -1.20,
            'RL_calf_joint': -1.20,
            'FR_calf_joint': -1.20,
            'RR_calf_joint': -1.20,
        }
        takeoff_squat_pose_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 1.10,
            'RL_thigh_joint': 1.20,
            'FR_thigh_joint': 1.10,
            'RR_thigh_joint': 1.20,
            'FL_calf_joint': -1.85,
            'RL_calf_joint': -1.95,
            'FR_calf_joint': -1.85,
            'RR_calf_joint': -1.95,
        }
        landing_pose_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

        class scales:
            # Task objective: execute a valid commanded jump and align launch direction.
            jump_distance = 4.0
            valid_jump_success = 12.0
            landing_target_progress = 0.0
            estimated_landing_target_tracking = 0.0
            long_jump_success = 0.0
            landing_success = 0.0
            landing_position = 0.0
            landing_orientation = 0.0

            # Jump mechanics.
            phase_contact_sync = 4.0
            takeoff_squat = 0.0
            takeoff_impulse = 8.0
            takeoff_release = 0.0
            takeoff_height_progress = 10.0
            takeoff_velocity = 20.0
            takeoff_forward_push = 0.0
            takeoff_forward_velocity = 8.0
            tracking_linear_velocity = 6.0
            feet_clearance = 1.0
            ground_creeping = -4.0

            # Stability and gesture.
            tracking_angular_velocity = 0.5
            body_attitude = 4.0
            joint_pose_aerial = 0.0
            joint_pose_prelanding = 2.0
            joint_pose_landing = 1.0

            # Touchdown and recovery.
            landing_contact = 0.0
            landing_stability = 2.0
            landing_impact = 2.5

            # Prepare.
            stand_height = 0.0
            default_pose_hold = 0.0

            # Safety and regularization.
            nonfoot_contact = -24.0
            soft_dof_pos_limits = -2.0
            torques = -1e-5
            dof_acc = -2e-7
            action_rate = -0.002
            collision = -0.5
            termination = -15.0

    class test:
        use_test = False
        checkpoint = 3000
        vel = torch.tensor([1.0, 0.50, 0.0, 0.0], dtype=torch.float32)


class GO2JumpTorqueCfgPPO(GO2TorqueCfgPPO):
    class algorithm(GO2TorqueCfgPPO.algorithm):
        entropy_coef = 0.002
        sym_loss = True
        obs_permutation = [
            0.0001, -1, 2,
            -3, 4, -5,
            6, -7, 8,
            -12, 13, 14, -9, 10, 11, -18, 19, 20, -15, 16, 17,
            -24, 25, 26, -21, 22, 23, -30, 31, 32, -27, 28, 29,
            33, 34, -35, -36,
            37, 38,
            40, 39, 42, 41,
            -46, 47, 48, -43, 44, 45, -52, 53, 54, -49, 50, 51,
            58, 59, 60, 55, 56, 57, 64, 65, 66, 61, 62, 63,
        ]
        act_permutation = [-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8]
        frame_stack = 1
        sym_coef = 1.0

    class runner(GO2TorqueCfgPPO.runner):
        run_name = ''
        experiment_name = 'SATA_jump'
        max_iterations = 3000
