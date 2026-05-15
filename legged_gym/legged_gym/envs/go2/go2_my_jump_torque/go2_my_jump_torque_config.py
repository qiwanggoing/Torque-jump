import torch

from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO


class GO2MyJumpTorqueCfg(GO2TorqueCfg):
    """OmniNet-style commanded single-jump task on top of SATA torque output."""

    class env(GO2TorqueCfg.env):
        frame_stack = 1
        c_frame_stack = 1
        num_single_obs = 46
        num_observations = frame_stack * num_single_obs
        single_num_privileged_obs = 67
        num_privileged_obs = c_frame_stack * single_num_privileged_obs
        num_actions = 12
        num_envs = 4096
        episode_length_s = 10.0
        env_spacing = 3.0
        joint_num = 12
        send_timeouts = True

    class terrain(GO2TorqueCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]

    class commands(GO2TorqueCfg.commands):
        curriculum = False
        max_curriculum = 3.5
        num_commands = 4
        resampling_time = 4.0
        heading_command = False
        # Jump trigger thresholds
        jump_trigger_min = 0.5 

        class ranges:
            jump_distance = [0.5, 1.5] # Target jump distance in meters
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            jump_trigger = [0.0, 1.0] # 0: Stand, 1: Jump


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
        pd_prior_start = 1.0
        pd_prior_end = 0.2

    class asset(GO2TorqueCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        default_dof_drive_mode = 3
        angular_damping = 0.0
        linear_damping = 0.0

    class domain_rand(GO2TorqueCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 0.8]
        push_robots = True
        push_interval_s = 4
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
        tracking_sigma = 0.25
        velocity_tracking_sigma = 0.35
        height_tracking_sigma = 0.04
        liftoff_velocity_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.42
        max_contact_force = 100.0
        min_air_time = 0.08
        min_jump_clearance = 0.04
        landing_window_s = 0.35
        target_feet_height = 0.08
        cycle_time = 1.5

        class scales:
            termination = -10.0
            jump_distance_tracking = 20.0 # Match the target distance
            stand_still = 5.0 # No movement when not jumping
            jump_trigger_tracking = 5.0 # Follow the trigger
            jump_stay_on_ground = -10.0 # Force liftoff
            liftoff_momentum = 8.0 # Initial momentum matching distance
            jump = 2.0 # Push-off power reward
            airborne_time = 5.0 # Stay in the air
            valid_jump = 10.0 # Reward stable landing
            landing_stability = 5.0
            orientation = 1.5
            ang_vel_xy = 0.4
            feet_clearance = 0.6
            torques = -0.0002
            dof_vel = -0.0001
            dof_acc = -5.5e-4
            collision = -1.0
            action_rate = -0.05
            default_pos = -0.05
            feet_contact_forces = -0.01

    class normalization(GO2TorqueCfg.normalization):
        class obs_scales(GO2TorqueCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            quat = 1.0

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
        pos = [10.0, 0.0, 6.0]
        lookat = [11.0, 5.0, 3.0]

    class sim(GO2TorqueCfg.sim):
        dt = 0.005
        substeps = 1

        class physx(GO2TorqueCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class test(GO2TorqueCfg.test):
        use_test = False
        checkpoint = 3000
        vel = torch.tensor([0.35, 0.0, 0.0, 0.51], dtype=torch.float32)


class GO2MyJumpTorqueCfgPPO(GO2TorqueCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy(GO2TorqueCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(GO2TorqueCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 16
        learning_rate = 1.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        sym_loss = False
        obs_permutation = []
        act_permutation = [-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8]
        frame_stack = 1
        sym_coef = 1.0

    class runner(GO2TorqueCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 64
        max_iterations = 15000
        save_interval = 100
        experiment_name = "go2_my_jump_torque"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
