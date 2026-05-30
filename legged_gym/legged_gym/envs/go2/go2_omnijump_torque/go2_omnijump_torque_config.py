import torch

from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO


class GO2OmniJumpTorqueCfg(GO2TorqueCfg):
    class env(GO2TorqueCfg.env):
        num_envs = 4096
        num_observations = 69   # +1 for pd_alpha (curriculum PD strength 0..0.5); lets policy condition on PD fade level
        num_privileged_obs = 109  # +1 propagated (privileged_obs concats obs_buf)
        num_actions = 12
        episode_length_s = 10.0
        env_spacing = 3.0
        send_timeouts = True

    class terrain(GO2TorqueCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class commands(GO2TorqueCfg.commands):
        curriculum = False
        heading_command = False
        num_commands = 5
        resampling_time = 1.8
        jump_command_threshold = 0.5
        stand_command_prob = 0.0
        single_jump_command_prob = 1.0

        class ranges:
            lin_vel_x = [0.8, 1.4]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [0.0, 0.0]
            jump_height = [0.48, 0.60]
            jump_command = [0.0, 1.0]

    class init_state(GO2TorqueCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(GO2TorqueCfg.control):
        control_type = "TG"
        activation_process = True
        hill_model = True
        motor_fatigue = True
        action_scale = 60.0
        decimation = 1
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.2}
        rl_prior_weight = 0.5
        pd_prior_weight = 0.5

    class asset(GO2TorqueCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "hip"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        default_dof_drive_mode = 3
        angular_damping = 0.0
        linear_damping = 0.0

    class domain_rand(GO2TorqueCfg.domain_rand):
        # Inherit SATA base randomization (friction, mass, CoM shift)
        push_robots = False      # no push during jump
        loss_action_obs = False  # keep all observations
        loss_rate = 0.0
        shifted_com_range_x = [-0.05, 0.05]  # reduced from [-0.2, 0.2]; large shift tilts robot past stable_stand threshold

    class rewards(GO2TorqueCfg.rewards):
        only_positive_rewards = False
        base_height_target = 0.42
        stance_squat_height = 0.20  # Atanassov stance phase squat target
        max_contact_force = 100.0
        contact_force_threshold = 1.0
        success_landing_min_base_height = 0.30
        success_landing_xy_vel_max = 0.60
        success_landing_z_vel_max = 0.25
        success_landing_ang_vel_max = 0.80
        success_height_tolerance = 0.08
        success_use_velocity_score = False
        task_max_height_sigma = 0.05
        tracking_linear_velocity_all_time = True
        pose_tracking_sigma = 0.20
        prelanding_tracking_sigma = 0.20
        joint_symmetry_tracking_sigma = 0.25
        height_progress_floor = 0.25
        height_tracking_gain = 20.0
        height_tracking_sigma = 0.05
        stand_still_velocity_gain = 6.0
        stand_still_orientation_gain = 10.0
        stand_still_joint_gain = 4.0
        stand_still_height_gain = 20.0
        stand_still_min_base_height = 0.27
        stand_still_max_body_contact_force = 0.05
        velocity_tracking_gain = 4.0
        angular_tracking_gain = 4.0
        ang_vel_tracking_min_command = 0.05
        takeoff_force_floor = 180.0
        takeoff_force_target = 360.0
        takeoff_acc_target = 8.0
        takeoff_velocity_target = 2.5
        squat_pushoff_height_threshold = 0.32
        projected_peak_sigma = 0.025       # was 0.05; sharper peak around cmd target — punish "just clear 0.30m" hops
        squat_foot_height = 0.10           # IK foot-to-hip distance for RSI squat pose (base ~0.20m)
        ascending_min_base_height = 0.18   # block ascending-reward farming when robot is flopped/sideways
        successful_jump_min_peak_height = 0.30  # successful_jump only counts when peak actually cleared squat by ≥10cm
        success_fallover_tilt = 0.7        # ~46°; only true fallovers cancel a pending success during landing buffer
        pose_guidance_sigma = 5.0          # bell-curve width for joint_angle_loaded/extended rewards
        lin_vel_takeoff_min_z_vel = 0.05
        first_jump_delay_steps = 55
        first_jump_min_base_height = 0.24
        first_jump_z_vel_max = 0.08
        first_jump_ang_vel_max = 0.60
        first_jump_dof_pos_error_max = 0.20
        takeoff_min_base_height = 0.24
        takeoff_min_z_vel = 0.05
        grounded_grace_steps = 12
        symmetry_lateral_sigma = 0.20
        symmetry_yaw_sigma = 0.45
        prelanding_height_margin = 0.06
        stand_rearm_steps = 20
        rsi_prob = 0.0
        rsi_vel_z_min = 1.0
        rsi_vel_z_max = 3.0
        rsi_height_offset_min = 0.0
        rsi_height_offset_max = 0.3
        landing_buffer_steps = 25
        takeoff_timeout_steps = 40
        state_switch_window_start = 28
        state_switch_window_end = 55
        ik_thigh_length = 0.213
        ik_calf_length = 0.213
        ik_nominal_foot_x = 0.02
        ground_foot_height = 0.30
        air_foot_height = 0.18
        prelanding_foot_height = 0.25

        class scales:
            termination = 0.0
            height_tracking = 1.0
            task_max_height = 2.0
            successful_jump = 40.0
            tracking_linear_velocity = 1.0
            tracking_angular_velocity = 0.5
            orientation = -0.4
            joint_angle_aerial = -0.5
            joint_angle_prelanding = -0.8
            joint_angle_landing = -0.15
            collision = -1.0
            torques = -1e-6
            action_rate = -0.001
            dof_acc = -2.5e-7
    class normalization(GO2TorqueCfg.normalization):
        class obs_scales(GO2TorqueCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(GO2TorqueCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(GO2TorqueCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.2
            gravity = 0.05

    class logging:
        print_episode_keys = [
            "rew_task_max_height",
            "rew_successful_jump",
            "rew_tracking_linear_velocity",
            "rew_tracking_angular_velocity",
            "rew_orientation",
            "rew_joint_angle_aerial",
            "rew_joint_angle_prelanding",
            "rew_joint_angle_landing",
            "rew_collision",
            "rew_torques",
            "rew_action_rate",
            "rew_dof_acc",
            "jump_flight_rate",
            "jump_landing_rate",
            "jump_completed_cycles",
            "successful_jump_rate",
            "peak_height_error",
            "mean_peak_height",
        ]

    class test(GO2TorqueCfg.test):
        use_test = False
        checkpoint = 3000
        vel = torch.tensor([1.0, 0.0, 0.0, 0.56, 1.0], dtype=torch.float32)
        single_jump_play = False
        post_landing_play_steps = 120


class GO2OmniJumpTorqueCfgPPO(GO2TorqueCfgPPO):
    class policy(GO2TorqueCfgPPO.policy):
        init_noise_std = 0.35
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        activation = "elu"

    class algorithm(GO2TorqueCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001
        num_learning_epochs = 5
        num_mini_batches = 8
        learning_rate = 1.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        sym_loss = True
        obs_permutation = [
            0.0001, -1, 2, -3, 4, -5, -6, 7, 8, 9, -10, -11,
            12, 13, 14, 15, -19, 20, 21, -16, 17, 18, -25, 26,
            27, -22, 23, 24, -31, 32, 33, -28, 29, 30, -37, 38,
            39, -34, 35, 36, 41, 40, 43, 42, -47, 48, 49, -44,
            45, 46, -53, 54, 55, -50, 51, 52, 59, 60, 61, 56,
            57, 58, 65, 66, 67, 62, 63, 64,
            68,   # pd_alpha (scalar curriculum value, mirror-invariant)
        ]
        act_permutation = [-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8]
        frame_stack = 1
        sym_coef = 0.5

    class runner(GO2TorqueCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 48
        max_iterations = 5000
        save_interval = 100
        experiment_name = "go2_omnijump_torque"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
