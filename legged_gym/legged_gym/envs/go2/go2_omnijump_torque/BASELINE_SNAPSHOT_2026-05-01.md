# GO2 Omni-Jump Torque Baseline Snapshot

Date: 2026-05-01

Task: `go2_omnijump_torque`

Purpose: record the current working jump baseline before starting a separate curriculum-training branch.

## Current Interface

- Action: 12-dimensional policy output, interpreted as residual joint torque and mixed with PD prior torque.
- Observation: 80 dimensions.
- Command: 5 dimensions: `[lin_vel_x, lin_vel_y, ang_vel_yaw, jump_height, jump_command]`.
- Zero command semantics: `[0, 0, 0, 0, 0]` means stand still.
- Jump command active when `jump_command > 0.5`.
- Training command mix:
  - `stand_command_prob = 0.20`
  - `single_jump_command_prob = 0.50`
  - jump samples use `lin_vel_x in [0.8, 1.4]`, `lin_vel_y in [-0.4, 0.4]`, `ang_vel_yaw = 0`, `jump_height in [0.48, 0.60]`.

## Control

- `control_type = "TG"`
- `action_scale = 60.0`
- `rl_prior_weight = 0.5`
- `pd_prior_weight = 0.5`
- Default no-command PD target is the default standing pose.
- Jump PD target switches through ground/air/prelanding IK targets.

## Important State Gates

- First jump requires a short standing delay:
  - `first_jump_delay_steps = 55`
  - base height at least `0.24`
  - small vertical velocity and roll/pitch angular velocity
  - mean joint position error from default pose below `0.20`
- Takeoff requires no foot contact plus:
  - `base_z >= 0.24`
  - `world_z_vel > 0.05`
- A completed jump requires landing, stable all-feet contact, rebound window, and `landing_buffer_steps = 8`.
- Single-jump commands are disabled after a completed stable jump, not immediately at touchdown.

## Reward Scales

- `peak_height_progress = 60.0`
- `takeoff_impulse = 35.0`
- `takeoff_vertical_velocity = 40.0`
- `all_feet_airborne = 8.0`
- `successful_jump = 100.0`
- `grounded_jump = -2.0`
- `tracking_linear_velocity = 100.0`
- `tracking_angular_velocity = 0.0`
- `stand_still = 3.0`
- `left_right_contact_sync = -1.0`
- `straight_jump_joint_symmetry = -1.0`
- `orientation = -0.8`
- `collision = -1.0`
- `torques = -1e-5`
- `action_rate = -0.01`
- `dof_acc = -1.5e-7`

## Latest Useful Training Readout

From the run before the stricter no-command stand contact gate was added:

- iteration: 1644 / 5000
- mean reward: 51.68
- mean episode length: 1643.01
- `jump_flight_rate = 0.8744`
- `jump_landing_rate = 0.8709`
- `jump_completed_cycles = 3.2236`
- `successful_jump_rate = 0.7592`
- `mean_peak_height = 0.3461`
- `rew_tracking_linear_velocity = 3.5436`
- `rew_peak_height_progress = 1.0958`
- `rew_stand_still = 0.0289`

Interpretation: the policy can learn repeated commanded jumps, but the no-command behavior was not a reliable natural stand yet.

## Latest Stand-Gate Change

After the readout above, `stand_still` was tightened so that no-command reward requires:

- all four feet in contact
- no thigh/calf/hip/base contact
- base height at least `0.27`
- low base velocity, low roll/pitch angular velocity, upright body, near-default joints, and near-target base height

This stricter stand gate has been locally sanity checked but not yet trained long enough to judge final behavior.

## Known Issues

- The policy can prefer repeated jumps when jump commands stay active.
- No-command standing still needs isolated training; otherwise the policy may find crouched or body-contact poses.
- Linear velocity reward can dominate if it is opened before stand/takeoff/landing are reliable.
- Height shaping helps discover jumping, but it can stabilize around one jump height instead of teaching a clean staged skill by itself.
