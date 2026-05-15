# GO2 OmniJump Torque Training Settings Snapshot

Date: 2026-04-30

Task: `go2_omnijump_torque`

Purpose: record the current training setup before changing the height reward design. The current observed behavior is continuous jumping with limited height variation. The main open question is whether the current per-step `height_tracking` reward is making the policy stabilize around one height instead of learning peak-height jump variation.

## Current Result Snapshot

Latest local log inspected:

- Log directory: `SATA/legged_gym/logs/go2_omnijump_torque/Apr30_00-49-15_`
- Last printed iteration: `1091 / 5000`
- Stop reason in log: `KeyboardInterrupt`
- Total timesteps at stop: `214,695,936`

Iteration 1091 metrics:

- Mean reward: `82.91`
- Mean episode length: `1622.62`
- `rew_all_feet_airborne`: `0.4600`
- `rew_height_tracking`: `1.3054`
- `rew_peak_height_progress`: `1.6693`
- `rew_successful_jump`: `0.2220`
- `rew_takeoff_impulse`: `1.3475`
- `rew_takeoff_vertical_velocity`: `1.5443`
- `rew_tracking_linear_velocity`: `2.1409`
- `jump_flight_rate`: `0.8685`
- `jump_landing_rate`: `0.8643`
- `jump_completed_cycles`: `11.0979`
- `successful_jump_rate`: `0.7629`
- `peak_height_error`: `0.1516`
- `mean_peak_height`: `0.3497`

Interpretation at this snapshot:

- The policy is reliably entering flight and landing.
- Successful stable jump cycles are common.
- The learned behavior appears to repeat jumps continuously.
- Base peak height remains much lower than the command range and does not vary much.

## Env And Command Settings

Source: `go2_omnijump_torque_config.py`

- `num_envs = 4096`
- `num_observations = 79`
- `num_privileged_obs = None`
- `num_actions = 12`
- `episode_length_s = 10.0`
- `env_spacing = 3.0`
- Terrain: plane, no terrain curriculum, height measurement enabled.
- Command dimensions: `[lin_vel_x, lin_vel_y, yaw_rate, jump_height]`
- Command resampling time: `1.8 s`
- Command ranges:
  - `lin_vel_x = [0.8, 1.4]`
  - `lin_vel_y = [-0.4, 0.4]`
  - `ang_vel_yaw = [-1.2, 1.2]`
  - `jump_height = [0.48, 0.60]`
- Play/test command:
  - `[1.0, 0.0, 0.0, 0.56]`

## Initial State

- Base position: `[0.0, 0.0, 0.42]`
- Base rotation: `[0.0, 0.0, 0.0, 1.0]`
- Initial linear velocity: `[0.0, 0.0, 0.0]`
- Initial angular velocity: `[0.0, 0.0, 0.0]`
- Default joint angles:
  - FL/RL hip: `0.1`
  - FR/RR hip: `-0.1`
  - FL/FR thigh: `0.8`
  - RL/RR thigh: `1.0`
  - all calves: `-1.5`

## Control Settings

- Control type: `TG`
- Action dimension: `12`
- Action scale: `60.0`
- Decimation: `1`
- Stiffness: `40.0`
- Damping: `1.2`
- RL residual prior weight: `0.5`
- PD prior weight: `0.5`
- Activation process: enabled
- Hill model: enabled
- Motor fatigue: enabled

Torque output in code:

- `residual_torques = actions[:, :12] * action_scale`
- `residual_torques_action = residual_torques * 0.5`
- `pd_prior_torques = PD(default_joint_pd_target - dof_pos, dof_vel) * 0.5`
- Final requested torque before actuator limits:
  - `torques_action = residual_torques_action + pd_prior_torques`

## Domain Randomization And Noise

- Friction randomization: disabled
- Pushes: disabled
- Base mass randomization: disabled
- Observation/action loss: disabled
- `loss_rate = 0.0`
- Observation noise: enabled
- Noise level: `1.0`
- Noise scales:
  - `dof_pos = 0.01`
  - `dof_vel = 1.5`
  - `ang_vel = 0.2`
  - `gravity = 0.05`

## PPO Settings

- Policy: `ActorCritic`
- Algorithm: `PPO`
- Actor hidden dims: `[256, 256, 256]`
- Critic hidden dims: `[256, 256, 256]`
- Activation: `elu`
- Initial action noise std: `0.35`
- Num steps per env: `48`
- Max iterations: `5000`
- Save interval: `100`
- Value loss coef: `1.0`
- Clip param: `0.2`
- Entropy coef: `0.001`
- Learning epochs: `5`
- Mini batches: `8`
- Learning rate: `1e-4`
- Schedule: `adaptive`
- Gamma: `0.99`
- Lambda: `0.95`
- Desired KL: `0.01`
- Max grad norm: `1.0`
- Symmetry loss: enabled
- Symmetry coef: `0.5`
- Frame stack: `1`
- Observation permutation length: `79`
- Action permutation length: `12`
- Resume: `False`

## Observation Layout

Current policy observation is 79D:

- `0:3`: base linear velocity
- `3:6`: base angular velocity
- `6:9`: projected gravity
- `9:12`: commanded xy/yaw velocity
- `12:13`: commanded jump height
- `13:15`: height observation
  - current base height scaled by `2.0`
  - command height minus current base height scaled by `2.0`
- `15:27`: joint position error from default pose
- `27:39`: joint velocity
- `39:51`: previous actions
- `51:55`: foot contact flags
- `55:67`: torques
- `67:79`: motor fatigue

Noise is zeroed for command, height observation, actions, contact, and torque channels.

## Jump State Machine

Jump start:

- `stable_stand` requires:
  - all feet contact
  - absolute z velocity `< 0.15`
  - roll/pitch angular velocity norm `< 0.6`
- Start a jump when not already jumping and `stand_step_counter >= 8`.

Takeoff:

- `just_took_off = jumping_state & ~has_taken_off & ~any_foot_contact`
- There is no extra base-height or z-velocity gate for takeoff.

Airborne:

- `airborne = jumping_state & has_taken_off & ~has_landed & ~any_foot_contact`
- `peak_base_height` is updated while `jumping_state` is true.

Landing:

- `just_landed = jumping_state & has_taken_off & ~has_landed & any_foot_contact`
- Peak height statistics are logged on `just_landed`.

Successful finish:

- Requires landing, all feet contact, switch-zone timing, small rebound from landing minimum, stable landing, and `landing_step_counter >= 8`.
- Stable landing thresholds:
  - base height `>= 0.30`
  - xy velocity `< 0.60`
  - z velocity `< 0.25`
  - roll/pitch angular velocity `< 0.80`
- `successful_jump` no longer requires peak height close to command height.

## Reward Constants

- `base_height_target = 0.42`
- `success_landing_min_base_height = 0.30`
- `success_landing_xy_vel_max = 0.60`
- `success_landing_z_vel_max = 0.25`
- `success_landing_ang_vel_max = 0.80`
- `height_progress_floor = 0.25`
- `height_tracking_gain = 20.0`
- `velocity_tracking_gain = 4.0`
- `angular_tracking_gain = 4.0`
- `takeoff_force_floor = 180.0`
- `takeoff_force_target = 360.0`
- `takeoff_acc_target = 8.0`
- `takeoff_velocity_target = 1.2`
- `grounded_grace_steps = 12`
- `symmetry_lateral_sigma = 0.20`
- `symmetry_yaw_sigma = 0.45`
- `prelanding_height_margin = 0.06`
- `stand_rearm_steps = 8`
- `landing_buffer_steps = 8`
- `takeoff_timeout_steps = 40`
- `state_switch_window_start = 28`
- `state_switch_window_end = 55`

## Reward Scales

- `termination = 0.0`
- `height_tracking = 30.0`
- `peak_height_progress = 30.0`
- `takeoff_impulse = 35.0`
- `takeoff_vertical_velocity = 40.0`
- `all_feet_airborne = 8.0`
- `successful_jump = 40.0`
- `grounded_jump = -2.0`
- `left_right_contact_sync = -1.0`
- `straight_jump_joint_symmetry = -1.0`
- `tracking_linear_velocity = 50.0`
- `tracking_angular_velocity = 1.0`
- `orientation = -0.8`
- `joint_angle_aerial = -0.2`
- `joint_angle_prelanding = -0.25`
- `joint_angle_landing = -0.04`
- `collision = -1.0`
- `torques = -1e-5`
- `action_rate = -0.01`
- `dof_acc = -1.5e-7`

## Current Reward Logic Summary

Height tracking:

- Active while `jumping_state & ~has_landed`.
- Multiplied by current air-foot ratio.
- Multiplied by current height progress.
- Also includes an exponential current-height tracking term against `commands[:, 3]`.
- This is the suspected source of fixed-height behavior.

Peak height progress:

- Active while `jumping_state & ~has_landed`.
- Multiplied by air-foot ratio.
- Uses `peak_base_height` instead of current base height.

Height progress:

- `floor = height_progress_floor`
- `target = max(command_height, floor + 1e-3)`
- `progress = clamp((height - floor) / (target - floor), 0, 1)`

Takeoff vertical velocity:

- Active before takeoff.
- Rewards positive vertical velocity toward `takeoff_velocity_target`.
- Also includes `0.3 * height_progress`.

Takeoff impulse:

- Active before takeoff.
- Rewards support-contact vertical force and vertical acceleration.

All feet airborne:

- Active only when all feet are off ground after takeoff and before landing.
- Reward is `0.25 + 0.75 * current_height_progress`.

Grounded jump penalty:

- Penalizes staying with all feet on ground after grace steps while in jump state and before takeoff.

Velocity tracking:

- Linear/yaw velocity tracking is active while `jumping_state & ~has_landed`.
- It is not currently restricted to takeoff-only.

Successful jump:

- Reward is one frame of `last_jump_success`.
- Success is stable completion of a jump cycle, not peak-height closeness.

## Omni-Jump Height Reward Comparison

Original Omni-Jump Go2 baseline config:

- `height_track = 0.0`
- `max_track = 0.0`
- `task_max_height = 2.0`
- `jumping = 40.0`

Important difference:

- Omni-Jump Go2 baseline does not mainly use per-step current-height tracking.
- It uses `task_max_height`, a maximum/peak height variable, to reward whether the jump reached the commanded height.
- Its `jumping` reward checks whether `task_max_height` is within a small band around the commanded height.

Implication for the next experiment:

- The current setup should be treated as the baseline before changing height rewards.
- A likely next change is to reduce or disable `height_tracking` and rely more on peak-height based reward, without adding extra reward names.

## Current Open Issue

The current trained behavior can repeatedly jump and land, but height variation is limited. The likely reason is that `height_tracking` rewards current base height during the jump, which can encourage a stable repeated height rather than a commanded peak-height jump. The next experiment should compare against an Omni-Jump-style peak-height objective.
