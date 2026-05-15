# GO2 Jump Torque Reward Functions

This document describes the current `go2_jump_torque` reward design after
removing height as a task objective.

## OmniNet Reference

Han et al. 2025, OmniNet, uses rewards to track:

- commanded jump height
- horizontal linear velocity
- yaw/angular velocity
- robot gesture in the air
- action smoothness
- collision avoidance
- pre-landing posture

The paper also uses a height/state estimator and an analytical-IK pose reward.

For our current task, the important difference is:

- OmniNet is a height-aware command-tracking jump controller.
- Our current task is a landing-target jump controller.
- We do not care about commanded height as a primary objective.

Therefore, we borrow only these OmniNet ideas:

- directional/targeted jumping needs a command-conditioned objective;
- yaw/body stability should be rewarded;
- aerial and pre-landing posture rewards are useful;
- action smoothness and collision penalties remain useful.

We intentionally remove OmniNet-style height dense/sparse rewards from the
active reward set.

## Current Task Definition

- The final target stage uses lightweight RSI: `rsi.probability = 0.20`.
- Training starts from the long-jump curriculum stage
  `takeoff_foundation`, then auto-advances to `forward_launch` and
  `target_landing`.
- `commands[0]`: jump toggle.
- `commands[1]`: landing target forward offset `dx`.
- `commands[2]`: landing target lateral offset `dy`.
- `commands[3]`: landing target yaw offset `dyaw`.
- `commands.use_landing_target = True`.
- `commands[1]` is not a velocity target.
- Landing `dx` is fixed to the `0.45-0.55m` range for direct 50 cm training.
- Takeoff velocity has its own curriculum; it no longer uses the disabled
  jump-height curriculum. It also starts after a `12000` step warmup.
- RSI currently samples only takeoff-like states:
  `takeoff_state_probability = 1.00`.
- Takeoff RSI uses random launch hints:
  `vz = 0.15-0.45m/s`, forward velocity fraction `0.20-0.80`, and
  cycle phase `0.45-0.53`.

## Reward Pipeline

The legged-gym reward framework uses every non-zero entry in
`cfg.rewards.scales`.

At initialization:

```text
effective_scale_i = cfg.rewards.scales.i * dt
```

At every reward step:

```text
reward_total = sum(raw_reward_i * effective_scale_i)
```

`termination` is added after the normal reward sum.

For this task:

```text
dt = sim.dt * control.decimation = 0.005 * 1 = 0.005
only_positive_rewards = False
```

## Notation

```text
clip01(x) = clamp(x, 0, 1)
I(c)      = 1 if condition c is true else 0
```

Jump phases are now contact-driven:

```text
prepare, takeoff, flight, landing, recovery
```

The timed cycle is kept only as a soft schedule/debug signal. The state machine
uses contact events:

```text
prepare -> takeoff:
  command requested, stable ready stance, enough support contacts

takeoff -> flight:
  num_contacts <= flight_max_contacts for min_airborne_steps
  and either upward velocity or clearance is present

flight -> landing:
  num_contacts >= landing_min_contacts for landing_settle_steps

landing -> recovery:
  stable support contacts for recovery_settle_steps
```

Landing target geometry:

```text
target_delta = target_landing_xy - jump_start_xy
current_delta = base_xy - jump_start_xy
target_dist = norm(target_delta)
target_dir = target_delta / target_dist

forward_distance = dot(current_delta, target_dir)
forward_progress = clip01(forward_distance / target_dist)
lateral_error = norm(current_delta - forward_distance * target_dir)
landing_position_error = norm(base_xy - target_landing_xy)
```

## Active Main Rewards

### `landing_target_progress`

Scale: `4.0`

Role: dense target-progress reward for forward/lateral jump commands.

Formula:

```text
lateral_quality = exp(-(lateral_error / landing_progress_lateral_sigma)^2)
phase_active = I(flight or landing or recovery)

raw = forward_progress
    * lateral_quality
    * phase_active
    * I(no nonfoot contact)
    * I(jump_toggle)
```

Why it exists:

- It replaces height tracking as the dense task reward.
- It gives the policy learning signal before final landing success happens.
- It is only active after leaving PREPARE, so it should not reward idle standing.

### `long_jump_success`

Scale: `30.0`

Role: sparse success reward for completing a jump and landing near the target.

Formula:

```text
position_quality = exp(-(last_completed_landing_position_error / landing_position_sigma)^2)
tilt_quality = exp(-(last_completed_landing_tilt / landing_success_tilt_sigma)^2)

raw = position_quality
    * tilt_quality
    * yaw_quality
    * I(last_completed_landing_position_error <= landing_position_cutoff)
    * valid_jump_gate
    * I(just_completed_cycle)
    * I(no nonfoot contact)
```

Why it exists:

- This is the main success reward now.
- It is gated by minimum clearance and airtime, but does not reward extra height.

### `landing_position`

Scale: `6.0`

Role: completed-cycle landing position accuracy reward.

Formula:

```text
position_reward = exp(-(last_completed_landing_position_error / landing_position_sigma)^2)

raw = position_reward
    * I(last_completed_landing_position_error <= landing_position_cutoff)
    * I(just_completed_cycle)
    * I(no nonfoot contact)
```

Why it exists:

- It gives a smoother final target-position reward than the binary success term.
- It is no longer height-gated.

## Active Jump-Mechanics Rewards

### `phase_contact_sync`

Scale: `4.0`

Role: enforce the basic jump contact pattern.

Definitions:

```text
contact_ratio = num_contacts / num_feet
front_ratio = mean(front_foot_contacts)
rear_ratio = mean(rear_foot_contacts)
balanced_contact = contact_ratio * min(front_ratio, rear_ratio)
```

Formula:

```text
takeoff_release = just_took_off or (state == takeoff and num_contacts <= flight_max_contacts)

if state == takeoff and not takeoff_release:
    raw = 0.2 * balanced_contact
elif takeoff_release:
    raw = I(num_contacts <= flight_max_contacts)
elif state == flight:
    raw = I(num_contacts <= flight_max_contacts)
elif state == landing:
    raw = 0.2 * balanced_contact
else:
    raw = 0

raw *= I(jump_toggle)
```

Why it exists:

- Takeoff and landing want balanced support.
- Flight wants no foot contact.
- Height gates are validity gates, not task objectives.
- It is gated by the real contact state, so PREPARE cannot collect
  flight/contact rewards without a state transition.
- TAKEOFF support is rewarded while feet are in contact; release is rewarded
  when contacts disappear or on the `just_took_off` transition.

### `takeoff_velocity`

Scale: `20.0`

Role: encourage vertical impulse during takeoff.

This is not a height objective. It is a mechanism reward that helps create
airborne time.

Formula:

```text
support_quality = (num_contacts / num_feet) * min(front_ratio, rear_ratio)
support_quality = takeoff_velocity_support_min
                + (1 - takeoff_velocity_support_min) * support_quality

vz = max(base_lin_vel_z, 0)
velocity_reward = clip01((vz - takeoff_velocity_floor)
                         / (target_takeoff_velocity - takeoff_velocity_floor))
release_quality = 1 - contact_ratio if no foot contact or just_took_off
                else takeoff_velocity_contact_gate_min
                     + (1 - takeoff_velocity_contact_gate_min) * (1 - contact_ratio)

raw = velocity_reward
    * support_quality
    * release_quality
    * I(state == takeoff)
    * I(jump_toggle)
```

Why it exists:

- Without some upward impulse, landing-target rewards can be hacked by crawling
  or crouch-sliding.
- It should stay auxiliary, not become a height-tracking objective.
- Its target ramps from `takeoff_velocity_start` to `takeoff_velocity_target`
  through `takeoff_velocity_curriculum`.
- It is only active after the state machine enters TAKEOFF.
- During TAKEOFF, full support still gives a small gate so pushing upward can be
  learned. Once contact is released, the reward is credited on the
  `just_took_off` event.

### `takeoff_forward_velocity`

Scale: `8.0`

Role: create horizontal launch velocity toward the landing target.

Formula:

```text
target_delta = target_landing_xy - jump_start_xy
target_dir = target_delta / norm(target_delta)
target_speed = clip(norm(target_delta) / flight_time_ref,
                    takeoff_forward_velocity_min,
                    takeoff_forward_velocity_max)

forward_speed = dot(world_xy_velocity, target_dir)
lateral_speed = norm(world_xy_velocity - forward_speed * target_dir)

forward_reward = clip01((forward_speed - takeoff_forward_velocity_floor)
                        / (target_speed - takeoff_forward_velocity_floor))
lateral_quality = exp(-(lateral_speed / takeoff_forward_velocity_lateral_sigma)^2)

raw = forward_reward
    * lateral_quality
    * I(state == takeoff)
    * I(jump_toggle)
```

Why it exists:

- Vertical takeoff alone produces natural flight but lands behind forward
  targets.
- This term gives a dense takeoff-stage signal for the commanded landing
  direction without treating `commands[1]` as a steady walking velocity.

## Active Stability And Gesture Rewards

### `tracking_angular_velocity`

Scale: `0.5`

Role: suppress yaw/roll/pitch angular velocity error.

In landing-target mode, yaw-rate target is zero.

Formula:

```text
yaw_reward = exp(-((base_ang_vel_z - yaw_rate_des) / tracking_yaw_rate_sigma)^2)
xy_reward = exp(-sum(base_ang_vel_xy^2) / ang_vel_xy_sigma^2)

phase_weight =
    0.5  * I(state == takeoff)
  + 1.0  * I(state == flight)
  + 0.5  * I(state == landing)
  + 0.25 * I(state == recovery)

raw = yaw_reward * xy_reward * phase_weight * I(jump_toggle)
```

### `body_attitude`

Scale: `2.0`

Role: keep the trunk level and reduce roll/pitch angular velocity.

Formula:

```text
tilt = norm(projected_gravity_xy)
ang_speed_xy = norm(base_ang_vel_xy)

phase_weight =
    body_attitude_takeoff_weight * I(state == takeoff)
  + 0.8  * I(state == flight)
  + 0.6  * I(state == landing)
  + 0.5  * I(state == recovery)

raw = exp(-(tilt / body_tilt_sigma)^2)
    * exp(-(ang_speed_xy / body_ang_vel_xy_sigma)^2)
    * phase_weight
    * I(jump_toggle)
```

### `joint_pose_aerial`

Scale: `1.0`

Role: regulate aerial posture.

Formula:

```text
joint_pose_reward(target) = exp(-mean((dof_pos - target)^2) / joint_pose_sigma^2)

active = I(state == flight
           and num_contacts <= flight_max_contacts
           and not prelanding)

raw = joint_pose_reward(aerial_joint_target)
    * active
    * I(jump_toggle)
```

Height gates are disabled.

### `joint_pose_prelanding`

Scale: `2.0`

Role: move legs toward a safe touchdown shape before landing.

Formula:

```text
prelanding = I(cycle_phase >= prelanding_phase_start
               and num_contacts <= flight_max_contacts
               and (state == flight or state == landing))

raw = joint_pose_reward(prelanding_joint_target)
    * prelanding
    * I(jump_toggle)
```

### `joint_pose_landing`

Scale: `1.0`

Role: encourage the landing stance after touchdown.

Formula:

```text
active = I(landing and num_contacts >= landing_min_contacts)

raw = joint_pose_reward(landing_joint_target)
    * active
    * I(no nonfoot contact)
    * I(jump_toggle)
```

## Active Landing Reward

### `landing_impact`

Scale: `2.5`

Role: reward quiet touchdown and recovery without adding a separate dense
landing controller.

Formula:

```text
stability =
    exp(-landing_tilt_scale * norm(projected_gravity_xy))
  * exp(-landing_xy_vel_scale * norm(base_lin_vel_xy))
  * exp(-landing_ang_vel_scale * norm(base_ang_vel_xy))
  * exp(-landing_vertical_vel_scale * abs(base_lin_vel_z))

raw = stability
    * I(landing or recovery)
    * I(no nonfoot contact)
```

## Active Prepare Rewards

### `stand_height`

Scale: `3.0`

Role: bring the robot to the ready stance before takeoff.

Formula:

```text
active = I(prepare
           and not prepare_stand_ready
           and all_feet_down
           and phase_steps <= prepare_reward_max_steps)

raw = exp(-((base_z - stand_height_target) / stand_height_sigma)^2)
    * active
```

This is a ready-stance reward, not a jump-height objective.

### `default_pose_hold`

Scale: `1.5`

Role: keep joints near the ready pose during PREPARE.

Formula:

```text
joint_err = mean((dof_pos - default_joint_pd_target)^2)

raw = exp(-joint_err / default_pose_sigma^2)
    * active
```

## Active Penalties

### `nonfoot_contact`

Scale: `-24.0`

Formula:

```text
raw = I(nonfoot_contact)
    * I(takeoff or flight or landing or recovery or just_completed_cycle)
```

### `soft_dof_pos_limits`

Scale: `-2.0`

Formula:

```text
raw = sum(max(soft_lower - dof_pos, 0) + max(dof_pos - soft_upper, 0))
```

### `dof_acc`

Scale: `-2e-7`

Formula:

```text
raw = sum(((last_dof_vel - dof_vel) / dt)^2)
```

### `action_rate`

Scale: `-0.002`

Formula:

```text
raw = sum((last_actions - actions)^2)
```

### `collision`

Scale: `-0.5`

Formula:

```text
raw = count(norm(contact_force_on_penalized_body) > 0.1)
```

### `termination`

Scale: `-15.0`

Formula:

```text
raw = I(reset_buf and not time_out_buf)
```

## Disabled As Redundant For The Current Task

These terms are intentionally not listed in the active `cfg.rewards.scales`
table, so legged-gym does not register or call them during training. The helper
functions can remain in the code for later ablations.

When curriculum is enabled, early stages may temporarily activate some
takeoff-bootstrap helpers, then phase them out as the task advances. The final
target stage remains the concise landing-target stack summarized below.

### Commanded-height objective terms

These are removed because we do not care about commanded height:

- `height_tracking`
- `jumping_success`
- `maximum_height`

Height gates remain enabled only for jump validity and downstream reward
activation; they do not pay extra reward for jumping higher after the gate
saturates.

### Velocity target term

`tracking_linear_velocity`

Reason: `commands[1]` is landing target `dx`, not a desired velocity.

### Extra posture/height shaping

`feet_clearance`

Reason: aerial posture is already handled by `joint_pose_aerial` and
`joint_pose_prelanding`. Keeping feet clearance active made the reward set more
height-like and redundant.

### Disabled optional terms

- `takeoff_squat`
- `takeoff_release`
- `takeoff_height_progress`
- `takeoff_forward_push`
- `landing_contact`
- `landing_stability`
- `landing_orientation`
- `landing_body_clearance`
- `motor_fatigue`
- `roll`

## Current Reward Stack Summary

The normalized reward stack is now:

1. **Task objective**:
   `landing_target_progress`, `estimated_landing_target_tracking`,
   `landing_position`, `long_jump_success`.
2. **Jump event mechanics**:
   `phase_contact_sync`, `takeoff_velocity`,
   `takeoff_forward_velocity`, `ground_creeping`.
3. **Gesture and stability**:
   `body_attitude`, `tracking_angular_velocity`,
   `joint_pose_aerial`, `joint_pose_prelanding`, `joint_pose_landing`.
4. **Landing recovery**:
   `landing_impact`.
5. **Prepare**:
   `stand_height`, `default_pose_hold`.
6. **Regularization and safety**:
   `nonfoot_contact`, `soft_dof_pos_limits`, `dof_acc`,
   `action_rate`, `collision`, `termination`.

The key training signal during the takeoff-RSI bootstrap should be:

```text
rew_takeoff_release > 0
rew_takeoff_velocity increasing
natural_flight_rate eventually > 0
```

If `rew_takeoff_release` stays at zero, the policy is still not discovering
contact release. If release becomes nonzero but `natural_flight_rate` stays
near zero, the remaining bottleneck is likely takeoff mechanics, action scale,
torque availability, or prepare-state quality.
