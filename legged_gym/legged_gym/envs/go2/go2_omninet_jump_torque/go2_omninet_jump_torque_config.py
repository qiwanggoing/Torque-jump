"""Config for the OmniNet-aligned jumping task (paper Table I rewards + OmniJump
state machine, on top of SATA torque control + PD fade + pd_alpha observation).

Inherits the SATA torque infrastructure from GO2OmniJumpTorqueCfg and overrides:
- commands (OmniJump 5-dim layout with height quantization in env)
- init_state pose + rel_foot_pos for IK
- rewards (paper Table I 12 items, OmniJump weights and sigmas)
- asset termination contacts
- PPO hyperparameters (network size, lr, noise, rollout length)
"""

from legged_gym.envs.go2.go2_omnijump_torque.go2_omnijump_torque_config import (
    GO2OmniJumpTorqueCfg,
    GO2OmniJumpTorqueCfgPPO,
)


class GO2OmniNetJumpTorqueCfg(GO2OmniJumpTorqueCfg):
    class env(GO2OmniJumpTorqueCfg.env):
        num_envs = 4096
        num_observations = 69          # keep 68 + pd_alpha (no obs change vs torque baseline)
        num_privileged_obs = 109
        num_actions = 12
        episode_length_s = 20.0        # OmniJump
        env_spacing = 3.0
        send_timeouts = True

    class growth(GO2OmniJumpTorqueCfg.growth):
        # Keep our SATA PD fade schedule (paper uses POSE control so no fade)
        start_torque_scale = 1.0
        k = 0.0001
        warmup_steps = 96000
        x0 = 384000

    class control(GO2OmniJumpTorqueCfg.control):
        # Keep SATA torque + PD prior (unchanged from parent)
        control_type = "TG"
        action_scale = 60.0
        decimation = 1
        stiffness = {"joint": 40.0}    # matches OmniJump (K_p=40, K_d=1.2)
        damping = {"joint": 1.2}
        rl_prior_weight = 0.5
        pd_prior_weight = 0.5

    class commands(GO2OmniJumpTorqueCfg.commands):
        # OmniJump-style velocity tracking + height tracking command
        curriculum = False
        heading_command = False        # use ang_vel_yaw directly
        num_commands = 5               # [v_x, v_y, ω_yaw, h_target, jump_cmd]
        resampling_time = 10.0
        jump_command_threshold = 0.5
        stand_command_prob = 0.0
        single_jump_command_prob = 1.0
        height_command = True
        tracking_z = False
        bool_jump = False
        desired_jumping_height = 0.85
        jump_prob = 0.3

        class ranges(GO2OmniJumpTorqueCfg.commands.ranges):
            lin_vel_x = [-0.2, 1.0]            # OmniJump
            lin_vel_y = [-0.5, 0.5]            # OmniJump
            ang_vel_yaw = [-0.8, 0.8]          # OmniJump
            height_z = [0.32, 0.85]            # OmniJump (used by quantization)
            jump_height = [0.32, 0.85]         # alias for compatibility
            jump_command = [0.0, 1.0]
            heading = [-1.0, 1.0]
            vel_z_bool = [0, 1]

    class init_state(GO2OmniJumpTorqueCfg.init_state):
        pos = [0.1, 0.0, 0.34]                 # OmniJump init pose
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {               # OmniJump
            "FL_hip_joint": -0.05,
            "RL_hip_joint": -0.05,
            "FR_hip_joint": 0.05,
            "RR_hip_joint": 0.05,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
        # OmniJump rel foot positions for analytical IK (body-frame, FL/FR/RL/RR per axis)
        rel_foot_pos = [
            [0.194,  0.194, -0.193, -0.193],   # x
            [0.156, -0.156,  0.156, -0.156],   # y
            [-0.316, -0.316, -0.316, -0.316],  # z (ground stance)
        ]
        rel_foot_pos_peak = [
            [0.232,  0.232, -0.155, -0.155],   # x
            [0.148, -0.148,  0.148, -0.148],   # y
            [-0.112, -0.112, -0.115, -0.115],  # z (aerial peak — feet tucked up)
        ]
        # Prelanding pose: midway between aerial peak and ground (paper Table I needs q^pre)
        rel_foot_pos_pre = [
            [0.213,  0.213, -0.174, -0.174],
            [0.152, -0.152,  0.152, -0.152],
            [-0.214, -0.214, -0.215, -0.215],
        ]

    class asset(GO2OmniJumpTorqueCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base", "trunk", "hip", "thigh"]  # OmniJump
        self_collisions = 0
        default_dof_drive_mode = 3
        angular_damping = 0.0
        linear_damping = 0.0

    class domain_rand(GO2OmniJumpTorqueCfg.domain_rand):
        push_robots = False
        loss_action_obs = False
        loss_rate = 0.0
        shifted_com_range_x = [-0.05, 0.05]

    class rewards(GO2OmniJumpTorqueCfg.rewards):
        # OmniJump global reward settings (paper §III.A.4)
        only_positive_rewards = False
        base_height_target = 0.32
        max_contact_force = 1000.0
        max_height_reward_sigma = 0.2    # for height_tracking exp(-(z*-z)^2/0.05) = exp(-20(z*-z)^2)
        foot_height_target = 0.09
        tracking_sigma = 0.5             # tracking_lin_vel / ang_vel use this; 1/0.25 = 4 → matches paper exp(-4·err^2)
        # Successful jump tolerance: peak ∈ [h* - 0.04, h* + 0.04]
        omninet_jump_height_tolerance = 0.04
        omninet_walking_height_threshold = 0.38

        # SATA infrastructure params (used by _init_buffers and base methods)
        stance_squat_height = 0.20
        contact_force_threshold = 1.0
        success_landing_min_base_height = 0.30
        success_landing_xy_vel_max = 0.60
        success_landing_z_vel_max = 0.25
        success_landing_ang_vel_max = 0.80
        success_height_tolerance = 0.10
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
        projected_peak_sigma = 0.05
        squat_foot_height = 0.10
        ascending_min_base_height = 0.18
        successful_jump_min_peak_height = 0.30
        success_fallover_tilt = 0.7
        pose_guidance_sigma = 1.5
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
        rsi_prob = 0.0                    # disable our RSI; OmniJump doesn't use it
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

        class scales(GO2OmniJumpTorqueCfg.rewards.scales):
            # ============================================================
            # Paper Table I — 12 reward items (all other scales = 0)
            # ============================================================
            # Task
            height_tracking = 1.0                    # exp(-20·(z*-z)^2)
            successful_jump = 20.0                   # n_jump (peak ∈ ±0.04m of cmd)
            tracking_linear_velocity = 1.5           # exp(-4·||v*-v||^2)
            tracking_angular_velocity = 0.6          # exp(-4·(ω*-ω)^2)
            # Pose
            orientation = -0.8                       # ||g_xy||^2
            joint_angle_aerial = -0.4                # Σ||q-q^air||
            joint_angle_prelanding = -0.6            # Σ||q-q^pre||
            joint_angle_landing = -0.12              # Σ||q-q^ground||
            # Safety
            collision = -1.0                         # n_collision
            # Smoothness
            torques = -1e-5                          # ||τ||^2
            action_rate = -0.01                      # ||a_t - a_{t-1}||^2
            dof_acc = -2.5e-7                        # ||q̈||^2

            # ============================================================
            # Disable everything else SATA infra brings in
            # ============================================================
            termination = 0.0
            maintain_contact = 0.0
            peak_height_progress = 0.0
            all_feet_airborne = 0.0
            takeoff_vertical_velocity = 0.0
            projected_peak = 0.0
            horizontal_drift = 0.0
            takeoff_direction = 0.0
            default_pos = 0.0
            default_hip_pos = 0.0
            joint_angle_loaded = 0.0
            joint_angle_extended = 0.0
            aerial_dof_acc = 0.0
            tracking_pos = 0.0
            stand_still = 0.0
            task_max_height = 0.0   # this is OmniJump's name; we use height_tracking instead

    class logging(GO2OmniJumpTorqueCfg.logging):
        print_episode_keys = [
            "rew_height_tracking",
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
            "mean_peak_height",
        ]

    class test(GO2OmniJumpTorqueCfg.test):
        vel = GO2OmniJumpTorqueCfg.test.vel.clone()
        single_jump_play = True


class GO2OmniNetJumpTorqueCfgPPO(GO2OmniJumpTorqueCfgPPO):
    class policy(GO2OmniJumpTorqueCfgPPO.policy):
        # OmniJump network architecture and exploration
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(GO2OmniJumpTorqueCfgPPO.algorithm):
        # OmniJump PPO hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4              # OmniJump
        learning_rate = 1.0e-3            # OmniJump (vs 1e-4 SATA default)
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        sym_loss = True
        # obs_permutation / act_permutation inherited from parent (69-dim layout unchanged)

    class runner(GO2OmniJumpTorqueCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24            # OmniJump
        max_iterations = 20000            # OmniJump
        save_interval = 200
        experiment_name = "go2_omninet_jump_torque"
        run_name = "omninet_paper"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
