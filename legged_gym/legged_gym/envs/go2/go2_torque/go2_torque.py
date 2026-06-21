from legged_gym.envs import LeggedRobot
from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO
from legged_gym.utils.math import wrap_to_pi
from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
import torch


def update_com(I_box, mass_box, com_box, mass_point, point_pos):
    new_com = (mass_box * com_box + mass_point * point_pos) / (mass_box + mass_point)
    return new_com


def parallel_axis_theorem(I_com, mass, d):
    d_x, d_y, d_z = d
    d_squared = np.array([
        [d_y ** 2 + d_z ** 2, -d_x * d_y, -d_x * d_z],
        [-d_x * d_y, d_x ** 2 + d_z ** 2, -d_y * d_z],
        [-d_x * d_z, -d_y * d_z, d_x ** 2 + d_y ** 2]
    ])

    return I_com + mass * d_squared


def update_inertia(I_box, mass_box, com_box, mass_point, point_pos):
    new_com = update_com(I_box, mass_box, com_box, mass_point, point_pos)

    displacement_box = com_box - new_com
    I_box_new = parallel_axis_theorem(I_box, mass_box, displacement_box)

    displacement_point = point_pos - new_com
    I_point = parallel_axis_theorem(np.zeros((3, 3)), mass_point, displacement_point)

    I_total = I_box_new + I_point

    return I_total, new_com


class GO2Torque(LeggedRobot):
    cfg: GO2TorqueCfg

    def __init__(self, cfg: GO2TorqueCfg, sim_params, physics_engine, sim_device, headless):
        self.max_torque_scale = cfg.growth.max_torque_scale
        self.start_torque_scale = cfg.growth.start_torque_scale
        self.max_rear_torque_scale = cfg.growth.max_rear_torque_scale
        self.start_rear_torque_scale = cfg.growth.start_rear_torque_scale
        self.max_freq = cfg.growth.max_freq
        self.start_freq = cfg.growth.start_freq
        self.step_count = 0
        self.current_dt = 0
        self.current_freq = self.start_freq
        self.low_torque = 0

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.motor_fatigue = torch.zeros(cfg.env.num_envs, self.num_dofs, device=sim_device)
        self.torques = torch.zeros(cfg.env.num_envs, self.num_dofs, device=sim_device)
        self.activation_sign = torch.zeros(cfg.env.num_envs, self.num_dofs, device=sim_device)
        if self.cfg.domain_rand.loss_action_obs == False:
            self.cfg.domain_rand.loss_rate = 0

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.termination_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                                          requires_grad=False)
            self.soft_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()

                self.termination_dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0] - 0.05
                self.termination_dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1] + 0.05

                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.soft_dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.soft_dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def check_termination(self):
        dof_pos_limits_up = self.termination_dof_pos_limits[:, 1]
        dof_pos_limits_low = self.termination_dof_pos_limits[:, 0]
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_buf |= torch.any(self.dof_pos > dof_pos_limits_up, dim=1)
        self.reset_buf |= torch.any(self.dof_pos < dof_pos_limits_low, dim=1)
        self.reset_buf |= self.projected_gravity[:, 2] > 0
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            com_rng_x = self.cfg.domain_rand.shifted_com_range_x
            com_rng_y = self.cfg.domain_rand.shifted_com_range_y
            com_rng_z = self.cfg.domain_rand.shifted_com_range_z
            rnd_mass = np.random.uniform(rng[0], rng[1])
            point_mass_pos = np.array([np.random.uniform(com_rng_x[0], com_rng_x[1]),
                                       np.random.uniform(com_rng_y[0], com_rng_y[1]),
                                       np.random.uniform(com_rng_z[0], com_rng_z[1])])

            # Add a point mass to the base
            props[0].mass += rnd_mass
            com_prev = np.array([props[0].com.x, props[0].com.y, props[0].com.z])
            inertia_prev = np.array([[props[0].inertia.x.x, props[0].inertia.x.y, props[0].inertia.x.z],
                                     [props[0].inertia.y.x, props[0].inertia.y.y, props[0].inertia.y.z],
                                     [props[0].inertia.z.x, props[0].inertia.z.y, props[0].inertia.z.z]])
            intertia, com = update_inertia(inertia_prev, props[0].mass, com_prev, rnd_mass, point_mass_pos)
            props[0].inertia.x += gymapi.Vec3(intertia[0, 0], intertia[0, 1], intertia[0, 2])
            props[0].inertia.y += gymapi.Vec3(intertia[1, 0], intertia[1, 1], intertia[1, 2])
            props[0].inertia.z += gymapi.Vec3(intertia[2, 0], intertia[2, 1], intertia[2, 2])
            props[0].com = gymapi.Vec3(com[0], com[1], com[2])
            for i in range(len(props)):
                props[i].mass += np.random.uniform(rng[0] / 16, rng[1] / 16)

        return props

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = (
                self.default_dof_pos * torch_rand_float(0.95, 1.05, (len(env_ids), self.num_dof), device=self.device))
        self.dof_vel[env_ids] = 0.
        self.activation_sign[env_ids] = 0.

        if self.add_noise:
            self.motor_fatigue[env_ids] = torch_rand_float(0, 0.2 * self.general_scale, (len(env_ids), 12),
                                                           device=self.device).squeeze(
                1)
        else:
            self.motor_fatigue[env_ids] = torch.zeros_like(self.motor_fatigue[env_ids])

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_growth_scale(self):
        self.step_count += 1
        if self.cfg.control.control_type == "T" or self.cfg.test.use_test:
            self.step_count = GO2TorqueCfgPPO().runner.num_steps_per_env * self.cfg.test.checkpoint

        # follow Gompertz curve
        self.general_scale = np.exp(-np.exp((-self.cfg.growth.k * (self.step_count - self.cfg.growth.x0))))

        self.current_freq = self.general_scale * (self.max_freq - self.start_freq) + self.start_freq
        self.current_torque_limit_scale = self.general_scale * (
                self.max_torque_scale - self.start_torque_scale) + self.start_torque_scale
        self.r_leg_scaled = self.general_scale * (
                self.max_rear_torque_scale - self.start_rear_torque_scale) + self.start_rear_torque_scale

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.actions = actions.to(self.device)
        # step physics and render each frame
        self.render()
        self.rew_buf[:] = 0.
        while self.current_dt * self.current_freq < 1:
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.current_dt += self.dt
            self.post_physics_step()
        self.current_dt %= (1 / self.current_freq)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        self._update_growth_scale()
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        self.torques_action = actions_scaled
        torques_limits = self.current_torque_limit_scale * self.torque_limits
        torques_limits[6:] = torques_limits[6:] * self.r_leg_scaled

        # muscle activation process
        if self.cfg.control.activation_process:
            current_activation_sign = torch.tanh(self.torques_action / torques_limits)
            activation_sign = (current_activation_sign - self.activation_sign) * 0.6 + self.activation_sign
        else:
            activation_sign = self.torques_action / torques_limits
        self.activation_sign = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            activation_sign, self.activation_sign)

        # hill model (T-N curve)
        if getattr(self.cfg.control, 'use_tn_curve', False) and self.cfg.control.hill_model:
            X1 = self.dof_vel_limits * self.cfg.control.tn_knee_speed_ratio
            X2 = self.dof_vel_limits * self.cfg.control.tn_max_speed_ratio
            Y1 = torques_limits
            Y2 = torques_limits * self.cfg.control.tn_peak_eccentric_ratio
            
            same_direction = (self.dof_vel * self.activation_sign) > 0
            max_effort = torch.where(same_direction, Y1, Y2)
            
            vel_abs = self.dof_vel.abs()
            k = -max_effort / (X2 - X1)
            decay_effort = k * (vel_abs - X1) + max_effort
            decay_effort = decay_effort.clip(min=0.0)
            
            max_effort = torch.where(vel_abs < X1, max_effort, decay_effort)
            self.torques = self.activation_sign * max_effort
        elif self.cfg.control.hill_model:
            self.torques = self.activation_sign * torques_limits * (
                1 - torch.sign(self.activation_sign) * self.dof_vel / self.dof_vel_limits)
        else:
            self.torques = self.activation_sign * torques_limits

        # fatigue
        if self.cfg.control.motor_fatigue:
            self.motor_fatigue += torch.abs(self.torques) * self.dt
            self.motor_fatigue *= 0.9
        else:
            self.motor_fatigue = torch.zeros_like(self.motor_fatigue)

        if self.low_torque:
            self.torques[:,:3] = self.torques[:,:3] * 0.2

        return self.torques

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:36] = 0.  # commands
        noise_vec[36:48] = 0.  # actions
        noise_vec[48:60] = noise_scales.fatigue * noise_level / 10
        if self.cfg.terrain.measure_heights:
            noise_vec[64:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy * self.general_scale
        self.root_states[:, 7:10] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 3),
                                                     device=self.device)  # lin vel x/y

        max_ang_vel = self.cfg.domain_rand.max_push_vel_ang * self.general_scale
        self.root_states[:, 10:13] = torch_rand_float(-max_ang_vel, max_ang_vel, (self.num_envs, 3),
                                                      device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def compute_observations(self):
        base_lin_vel = self.base_lin_vel
        motor_fatigue = self.motor_fatigue.detach()
        obs_buf = torch.cat((base_lin_vel * self.obs_scales.lin_vel,
                             self.base_ang_vel * self.obs_scales.ang_vel,
                             self.projected_gravity,
                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             self.commands[:, :3] * self.commands_scale,
                             self.torques,
                             motor_fatigue
                             ), dim=-1)
        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            obs_buf, self.obs_buf)

    def _init_buffers(self):
        super()._init_buffers()
        self.prev_dof_pos = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.hip_indices = []
        self.thigh_indices = []
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if 'hip_joint' in name:
                self.hip_indices.append(i)
            if 'thigh_joint' in name:
                self.thigh_indices.append(i)
        # ===== Actuator base values = real Go2 drivetrain (reconciled from two official Unitree sources) =====
        #   - unitree_rl_lab UnitreeActuatorCfg_Go2HV = the MOTOR torque-speed curve:
        #       Y1=20.2Nm (concentric) Y2=23.4Nm (eccentric)  X1=13.5 (knee point) X2=30 rad/s (no-load).
        #   - official unitree_ros URDF = the JOINT limits: hip/thigh 23.7Nm/30.1rad/s, calf 45.43Nm/15.70rad/s.
        # They AGREE once you account for a knee reduction: 30.1/15.70 = 45.43/23.7 = 1.917 -> the Go2 KNEE has a
        # ~1.917:1 reduction (calf joint = motor scaled: torque x1.917, speed /1.917). hip/thigh: no reduction.
        # rl_lab applies the bare MOTOR curve to ALL joints (".*"), omitting the knee reduction -- fine for walking
        # (knee never saturates) but WRONG for jumping (torque_diag shows the knee saturates at push-off). So we
        # model it PER-JOINT. With the cfg.control ratios global (knee X1/X2=0.45, eccentric Y2/Y1=1.1584), set the
        # base torque_limits (=Y1) and dof_vel_limits (=X2):
        #   hip/thigh : Y1=20.2  X2=30.0   -> Y2=23.4  X1=13.5
        #   calf      : Y1=38.7  X2=15.65  -> Y2=44.8  X1=7.04   (motor x/ 1.917; matches URDF calf 45.43/15.70)
        # Changes the physics -> all policies must be retrained.
        KNEE_REDUCTION = 1.917   # Go2 knee linkage (official unitree_ros URDF: 30.1/15.70 = 45.43/23.7)
        calf_mask = torch.tensor(['calf' in n for n in self.dof_names], device=self.device)
        self.torque_limits[:] = 20.2
        self.dof_vel_limits[:] = 30.0
        self.torque_limits[calf_mask] = 20.2 * KNEE_REDUCTION    # 38.7 Nm   (Y2 = 44.8 ~ URDF 45.43)
        self.dof_vel_limits[calf_mask] = 30.0 / KNEE_REDUCTION   # 15.65 rad/s (~ URDF 15.70)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.general_scale = 0

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.cfg.test.use_test and self.cfg.control.control_type == "T":
            self.commands[:] = self.cfg.test.vel
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                                             self.command_ranges["ang_vel_yaw"][0] * self.general_scale,
                                             self.command_ranges["ang_vel_yaw"][1] * self.general_scale)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        x_cmd_sum = self.command_ranges["lin_vel_x"][1] + self.command_ranges["lin_vel_x"][0]
        x_cmd_diff = self.command_ranges["lin_vel_x"][1] - self.command_ranges["lin_vel_x"][0]
        self.commands[env_ids, 0] = torch_rand_float(
            max(x_cmd_sum * 0.5 - x_cmd_diff * self.general_scale, self.command_ranges["lin_vel_x"][0]),
            min(x_cmd_sum * 0.5 + x_cmd_diff * self.general_scale, self.command_ranges["lin_vel_x"][1]),
            (len(env_ids), 1),
            device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0] * self.general_scale,
                                                     self.command_ranges["lin_vel_y"][1] * self.general_scale,
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0] * self.general_scale,
                                                         self.command_ranges["ang_vel_yaw"][1] * self.general_scale,
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

    ##############################################################################################################

    def _reward_soft_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.soft_dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.soft_dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_forward(self):
        # Reward for moving forward
        return torch.exp(
            -torch.abs(self.base_lin_vel[:, 0] - (
                    self.cfg.commands.ranges.lin_vel_x[1] + self.cfg.commands.ranges.lin_vel_x[0]) / 2 * max((
                    1 - self.general_scale * 2), 0) - self.commands[:, 0] * min(self.general_scale * 2, 1)) /
            self.cfg.rewards.tracking_sigma)
            # max((0.5 - self.general_scale), self.cfg.rewards.tracking_sigma))

    def _reward_head_height(self):
        # Reward for keeping the head high
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1).clip(
            max=self.cfg.rewards.base_height_target)
        head_up = -(self.projected_gravity[:, 0].clip(min=min(0, -0.2 * (1.5 - self.general_scale * 2))))
        return base_height * (1 + self.general_scale) + head_up

    def _reward_moving_y(self):
        return torch.exp(-torch.abs(
            self.base_lin_vel[:, 1] - self.commands[:, 1]) / self.cfg.rewards.tracking_sigma) * self.general_scale

    def _reward_moving_yaw(self):
        return torch.exp(-torch.abs(
            self.base_ang_vel[:, 2] - self.commands[:, 2]) / self.cfg.rewards.tracking_sigma) * self.general_scale

    def _reward_motor_fatigue(self):
        return torch.sum(self.motor_fatigue * torch.abs(self.torques_action), dim=1)

    def _reward_roll(self):
        return torch.abs(self.projected_gravity[:, 1])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])
