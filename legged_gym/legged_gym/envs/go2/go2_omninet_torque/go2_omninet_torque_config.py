import torch

from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO


class GO2OmniNetTorqueCfg(GO2TorqueCfg):
    """OmniNet-style jumping task with SATA torque output and PD+residual actuation."""

    class env(GO2TorqueCfg.env):
        history_length = 20
        num_single_obs = 46   # 3+3+4+12+12+12 = ang_vel, gravity, command, q, qdot, previous action
        num_estimator_targets = 10
        num_terrain_obs = 187
        num_observations = history_length * num_single_obs
        num_privileged_obs = num_estimator_targets + num_single_obs + num_terrain_obs
        num_actions = 12
        num_envs = 4096
        episode_length_s = 10.0
        env_spacing = 3.0
        send_timeouts = True

    class terrain(GO2TorqueCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = True

    class commands(GO2TorqueCfg.commands):
        curriculum = False
        heading_command = False
        num_commands = 4
        resampling_time = 1.8

        class ranges:
            lin_vel_x = [0.8, 1.4]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-1.2, 1.2]
            jump_height = [0.48, 0.60]

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
        action_scale = 5.0
        decimation = 1
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.2}
        pd_prior_start = 1.0
        pd_prior_end = 0.2

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
        randomize_friction = True
        friction_range = [0.4, 0.8]
        push_robots = True
        push_interval_s = 4.0
        max_push_vel_xy = 0.4
        max_push_vel_ang = 0.6
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        shifted_com_range_x = [-0.02, 0.02]
        shifted_com_range_y = [-0.02, 0.02]
        shifted_com_range_z = [-0.02, 0.02]
        loss_action_obs = False
        loss_rate = 0.0

    class rewards(GO2TorqueCfg.rewards):
        only_positive_rewards = False
        velocity_tracking_sigma = 0.35
        angular_tracking_sigma = 0.25
        height_tracking_sigma = 0.04
        pose_tracking_sigma = 0.20
        prelanding_tracking_sigma = 0.20
        success_height_tolerance = 0.04
        prelanding_height_margin = 0.06
        stand_rearm_steps = 8
        landing_buffer_steps = 8
        takeoff_timeout_steps = 40
        state_switch_window_start = 28
        state_switch_window_end = 55
        base_height_target = 0.42
        max_contact_force = 100.0
        ik_thigh_length = 0.213
        ik_calf_length = 0.213
        ik_nominal_foot_x = 0.02
        ground_foot_height = 0.30
        air_foot_height = 0.18
        prelanding_foot_height = 0.25

        class scales:
            termination = -10.0
            height_tracking = 8.0
            successful_jump = 40.0
            tracking_linear_velocity = 4.0
            tracking_angular_velocity = 3.0
            pose = -0.5
            prelanding = -0.8
            landing = -0.15
            action_rate = -0.01
            collision = -1.0

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

    class viewer(GO2TorqueCfg.viewer):
        ref_env = 0
        pos = [8.0, 0.0, 5.0]
        lookat = [9.0, 3.0, 2.0]

    class sim(GO2TorqueCfg.sim):
        dt = 0.005
        substeps = 1

    class logging:
        print_episode_keys = [
            "rew_height_tracking",
            "rew_successful_jump",
            "rew_tracking_linear_velocity",
            "rew_tracking_angular_velocity",
            "rew_pose",
            "rew_prelanding",
            "rew_landing",
            "rew_collision",
            "rew_action_rate",
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
        vel = torch.tensor([1.0, 0.0, 0.0, 0.60], dtype=torch.float32)


class GO2OmniNetTorqueCfgPPO(GO2TorqueCfgPPO):
    class policy(GO2TorqueCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        estimator_hidden_dims = [258, 128]
        estimator_activation = "relu"
        estimator_loss_coef = 0.5
        estimator_target_dim = 10
        single_obs_dim = 46
        history_length = 20

    class algorithm(GO2TorqueCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 8
        learning_rate = 1.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        sym_loss = False
        obs_permutation = []
        act_permutation = []
        frame_stack = 1
        sym_coef = 0.0

    class runner(GO2TorqueCfgPPO.runner):
        policy_class_name = "ActorCriticOmniNet"
        algorithm_class_name = "PPO"
        num_steps_per_env = 48
        max_iterations = 5000
        save_interval = 100
        experiment_name = "go2_omninet_torque"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
