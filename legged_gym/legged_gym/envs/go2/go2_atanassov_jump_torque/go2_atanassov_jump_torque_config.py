"""Config for the Atanassov 2025 Stage 1 (jumping in place) task, aligned with
the paper "Curriculum-Based Reinforcement Learning for Quadrupedal Jumping:
A Reference-Free Design" (IEEE RAM, June 2025), Table 1.

Stage 1 specifics:
- No directional command. Only jump toggle is used (cmd[4]).
- Target peak height hardcoded to 0.9 m (paper §STAGE 1).
- Modified RSI bootstraps with random base height and upward velocity.
- Reward set follows Table 1 with phase-aware targets:
    Stance:   p_z = 0.20m (squat), maintain contact
    Flight:   p_z = 0.7m, track desired velocities, feet tucked
    Landing:  land near initial pose, match initial orientation
- Sparse task rewards (max_height, landing_position, landing_orientation,
  jumping) fire only in the landing phase.
- Reward combination is multiplicative: r_total = r^+ · exp(-||r^-||²/σ)
  so penalties scale down task reward instead of being subtracted.

Inherits SATA torque + PD-prior-fade + pd_alpha observation infrastructure.
"""

from legged_gym.envs.go2.go2_omnijump_torque.go2_omnijump_torque_config import (
    GO2OmniJumpTorqueCfg,
    GO2OmniJumpTorqueCfgPPO,
)


class GO2AtanassovJumpTorqueCfg(GO2OmniJumpTorqueCfg):
    class env(GO2OmniJumpTorqueCfg.env):
        num_envs = 4096
        num_observations = 69         # keep our 68 + pd_alpha layout
        num_privileged_obs = 109
        num_actions = 12
        episode_length_s = 10.0       # 10s episodes (long enough for a full jump cycle)
        env_spacing = 3.0
        send_timeouts = True

    class growth(GO2OmniJumpTorqueCfg.growth):
        # SATA curriculum-validated schedule: warmup iter ~1000, fade ends iter ~4000.
        # Linear ramp = (4000-1000) × 96 = 288000 step_count → 1.67pp PD drop per 100 iter
        # (vs the compressed 3.6pp/100iter that collapsed at iter 800).
        start_torque_scale = 1.0
        k = 0.0001
        warmup_steps = 96000    # PD locked at 50% until iter ~1000
        x0 = 384000             # PD → 0 by iter ~4000

    class control(GO2OmniJumpTorqueCfg.control):
        control_type = "TG"
        action_scale = 60.0
        decimation = 1
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.2}
        rl_prior_weight = 0.5
        pd_prior_weight = 0.5

    class commands(GO2OmniJumpTorqueCfg.commands):
        # Stage 1 only uses the jump toggle; everything else is fixed at 0.
        curriculum = False
        heading_command = False
        num_commands = 5
        resampling_time = 1e9        # never resample mid-episode (Stage 1 is single-jump)
        jump_command_threshold = 0.5
        stand_command_prob = 0.0
        single_jump_command_prob = 0.7    # 70% jump episodes + 30% pure-stand episodes (cmd[4]=0 throughout) — teaches policy "cmd[4]=0 → stand still"
        atanassov_target_height = 0.6   # default target peak (lowered from paper 0.9 — easier to reach for torque control)

        class ranges(GO2OmniJumpTorqueCfg.commands.ranges):
            jump_height = [0.6, 0.6]       # default 0.6m (Stage 1 fixed; Stage 2 will vary)
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            jump_command = [0.0, 1.0]

    class init_state(GO2OmniJumpTorqueCfg.init_state):
        # Standing init pose (same as SATA baseline)
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

    class asset(GO2OmniJumpTorqueCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base", "trunk", "hip", "thigh"]
        self_collisions = 0
        default_dof_drive_mode = 3
        angular_damping = 0.0
        linear_damping = 0.0

    class domain_rand(GO2OmniJumpTorqueCfg.domain_rand):
        push_robots = False
        loss_action_obs = False
        loss_rate = 0.0
        shifted_com_range_x = [-0.1, 0.1]    # paper Table 2: ±0.1m
        # Other Atanassov rand parameters can be inherited / matched at deploy

    class rewards(GO2OmniJumpTorqueCfg.rewards):
        default_hip_pos_gain = 4.0   # exp(-gain × hip_diff) sharpness for default_hip_pos reward
        # ---------- Phase targets (Table 1) ----------
        only_positive_rewards = False
        atanassov_target_peak = 0.6              # peak height target (lowered from paper 0.9)
        atanassov_stance_height = 0.20            # base z target during stance (squat)
        atanassov_flight_height = 0.6             # base z target during flight (matches peak target)
        atanassov_terminate_orientation = 3.0    # rad
        atanassov_terminate_base_height = 0.12   # m
        atanassov_terminate_landing_error = 0.15 # m (landing position)

        # ---------- Reward kernel sigmas (Table 1 σ_X) ----------
        sigma_pz_stance = 0.01     # 0.05 → 0.01 (5× sharper): wide bell at 0.05 made any base_z ∈ [0.10, 0.30] nearly max-reward, letting policy "crouch and stay" without pushing back up. Tight bell makes only exact 0.20m squat highly rewarded — base=0.13m drops from 0.91 → 0.61 reward.
        sigma_pz_flight = 0.05     # broader for peak target
        sigma_pos_landing = 0.05   # landing xy position
        sigma_pos_max = 0.05       # max height
        sigma_ori_stance = 0.05    # narrower (0.10 → 0.05): pitch sensitivity 2× sharper
        sigma_ori_landing = 0.10
        sigma_v_flight = 0.25
        sigma_omega = 0.25
        sigma_q_nominal = 0.3      # 0.5 → 0.3: sharper nominal_pose kernel — more sensitive to joint deviation

        # ---------- Multiplicative reward sigma ----------
        # r_total = r^+ * exp(-||r^-||^2 / sigma_reg)
        sigma_reg = 5.0

        # ---------- Modified RSI (Atanassov §STAGE 1) ----------
        atanassov_rsi_prob = 0.5                 # 50% episodes use RSI
        atanassov_rsi_height_min = 0.30          # m above ground
        atanassov_rsi_height_max = 0.90          # m above ground
        atanassov_rsi_vel_z_min = 0.0            # m/s
        atanassov_rsi_vel_z_max = 3.0            # m/s

        # ---------- Misc SATA infra (kept defaults) ----------
        base_height_target = 0.42
        max_contact_force = 100.0
        contact_force_threshold = 1.0
        stance_squat_height = 0.20
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
        projected_peak_sigma = 0.025
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
        rsi_prob = 0.0           # disable SATA's squat-only RSI; we use Atanassov's instead
        rsi_vel_z_min = 0.0
        rsi_vel_z_max = 0.0
        rsi_height_offset_min = 0.0
        rsi_height_offset_max = 0.0
        landing_buffer_steps = 100  # extend so "landing" phase covers from touchdown to episode end
        takeoff_timeout_steps = 40
        state_switch_window_start = 28
        state_switch_window_end = 55
        ik_thigh_length = 0.213
        ik_calf_length = 0.213
        ik_nominal_foot_x = 0.02
        ground_foot_height = 0.30
        air_foot_height = 0.18
        prelanding_foot_height = 0.25
        post_jump_stand_steps = 300   # 100 → 300 (1s → 3s): more cmd[4]=0 data after each jump so policy learns to stand instead of bouncing

        class scales(GO2OmniJumpTorqueCfg.rewards.scales):
            # =====================================================================
            # Atanassov Table 1 — Task rewards (positive weights, phase-gated)
            # =====================================================================
            # Sparse (landing phase only) — boosted because they fire ≤1× per ep
            atanassov_landing_position = 50.0      # 15 → 50 (3.3×) — strong landing-at-init incentive
            atanassov_landing_orientation = 10.0   # 3 → 10 (3.3×)
            atanassov_max_height = 100.0           # 50 → 100 (2×, dominant jump-completion signal)
            atanassov_jumping_sparse = 20.0        # 5 → 20 (4×)

            # Dense phase-aware
            atanassov_base_position = 15.0         # 8 → 15: stronger phase target signal; landing-phase 3D position pull doubled to push back-jumping correction
            atanassov_orientation_tracking = 0.0   # disabled — replaced by curriculum-style raw `orientation` below (more continuous gradient at large tilts)
            atanassov_base_lin_vel = 1.0           # flight
            atanassov_base_ang_vel = 0.5           # flight + 0.1·landing
            atanassov_feet_clearance = 1.0         # reduced 2 → 1: pose is secondary
            atanassov_symmetry = 2.0               # 0.2 → 2.0 (10×): force left-right symmetric joint angles to kill tilted-push exploit
            atanassov_nominal_pose = -3.0          # -1 → -3 (3×): previous magnitude not enough to hold default pose at play time when PD prior is fully faded — policy was drifting into deep squat under cmd[4]=0. Stronger constant L1 gradient makes default standing pose the local optimum even without PD support.
            atanassov_maintain_contact = 5.0       # 0.5 → 5.0 (10×): strict 4-foot-contact gate; main stance stability incentive
            atanassov_takeoff_vz = 15.0             # 20 → 15: paired with new linear-normalized impl (vz/2.5, clamp [0,1]). Max per step = 15 (was 320 with vz² × weight 20). Smooth gradient from vz=0 — no 0.8 cliff that made weight 5 give 0 reward.

            # =====================================================================
            # Regularization rewards (negative weights → multiplicative penalty)
            # =====================================================================
            atanassov_energy = -1e-5
            atanassov_base_acceleration = -1e-3
            atanassov_contact_change = -0.1
            atanassov_contact_forces = -1e-3       # flight only
            action_rate = -0.08            # -0.01 → -0.08 (8×, match curriculum): kill the high-freq leg-shake exploit policy used to fire takeoff_vz without real liftoff
            dof_acc = -1e-6                # -2.5e-7 → -1e-6 (4×, match curriculum): suppress joint-acc jitter that enables the same exploit
            atanassov_joint_limits = -1.0

            # =====================================================================
            # Disable everything else from the SATA / OmniJump infrastructure
            # =====================================================================
            termination = -20.0           # 0 → -20: penalty for non-timeout episode termination (collision/roll/too_low). Stops the "crash for reward" trade-off where policy was profitably maximizing takeoff_vz even if episodes ended in 1 step.
            maintain_contact = 0.0
            peak_height_progress = 0.0
            all_feet_airborne = 0.0
            takeoff_vertical_velocity = 0.0
            projected_peak = 0.0
            orientation = -1.6            # curriculum-style raw form (sum(square(projected_gravity_xy))); no exp saturation — pitch/roll penalty grows continuously with tilt
            collision = -3.0              # -1 → -3: stronger penalty per body-part contact-step; pairs with new termination=-20 to make crash-for-reward trade-off unprofitable
            torques = 0.0
            horizontal_drift = 0.0
            takeoff_direction = 0.0
            default_pos = 0.0
            default_hip_pos = 5.0          # 2 → 5: stronger hip-only anti-splay
            joint_angle_loaded = 0.0
            joint_angle_extended = 0.0
            joint_angle_aerial = 0.0
            joint_angle_prelanding = 0.0
            joint_angle_landing = 0.0
            aerial_dof_acc = 0.0
            tracking_linear_velocity = 0.0
            tracking_angular_velocity = 0.0
            height_tracking = 0.0
            successful_jump = 0.0

    class logging(GO2OmniJumpTorqueCfg.logging):
        print_episode_keys = [
            "rew_atanassov_base_position",
            "rew_atanassov_orientation_tracking",
            "rew_orientation",
            "rew_default_hip_pos",
            "rew_atanassov_base_lin_vel",
            "rew_atanassov_base_ang_vel",
            "rew_atanassov_feet_clearance",
            "rew_atanassov_symmetry",
            "rew_atanassov_nominal_pose",
            "rew_atanassov_maintain_contact",
            "rew_atanassov_takeoff_vz",
            "rew_collision",
            "rew_atanassov_landing_position",
            "rew_atanassov_landing_orientation",
            "rew_atanassov_max_height",
            "rew_atanassov_jumping_sparse",
            "rew_atanassov_energy",
            "rew_atanassov_base_acceleration",
            "rew_atanassov_contact_change",
            "rew_atanassov_contact_forces",
            "rew_atanassov_joint_limits",
            "rew_action_rate",
            "rew_dof_acc",
            "jump_flight_rate",
            "jump_landing_rate",
            "successful_jump_rate",
            "mean_peak_height",
            "atanassov_rsi_rate",
        ]

    class test(GO2OmniJumpTorqueCfg.test):
        vel = GO2OmniJumpTorqueCfg.test.vel.clone()
        vel[0] = 0.0   # atanassov 训练只用 lin_vel_x=[0,0]，父类默认 vx=1 会让 policy 困惑/前飘
        single_jump_play = True


class GO2AtanassovJumpTorqueCfgPPO(GO2OmniJumpTorqueCfgPPO):
    class policy(GO2OmniJumpTorqueCfgPPO.policy):
        # SATA OmniJump torque baseline parameters (these were validated on the
        # SATA torque task). Paper Atanassov / OmniNet use 1.0 noise + 0.01 entropy
        # but those are tuned for POSE+PD@10kHz control where actions get filtered
        # before reaching joints. Direct torque control needs much less noise.
        init_noise_std = 0.5         # 0.35 → 0.5: prior run noise collapsed to 0.01 by iter 2800, policy then froze and crashed when PD hit 0 (iter 4000+)
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(GO2OmniJumpTorqueCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005         # 0.001 → 0.005: prior 0.001 let noise_std collapse to 0.01 well before fade ended → no exploration to adapt to PD=0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-4       # SATA baseline (was 1e-3; paper uses 1e-3 but for POSE control). 10× smaller for stable torque PPO updates.
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        sym_loss = True

    class runner(GO2OmniJumpTorqueCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 48       # SATA baseline (was 24). Longer rollouts → more stable PPO updates.
        max_iterations = 5000          # warmup 0-1000, fade 1000-4000, pure-torque refinement 4000-5000
        save_interval = 200
        experiment_name = "go2_atanassov_jump_torque"
        run_name = "stage1_vertical"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
