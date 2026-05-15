# GO2 Omni-Jump Curriculum Plan

Date: 2026-05-01

Goal: learn the full jump in one training run with metric-based reward and command gates, instead of asking PPO to discover standing, crouching, takeoff, flight, landing, and commanded motion all from iteration 0.

## Principle

Keep the current `go2_omnijump_torque` baseline intact. Use a separate task, `go2_omnijump_curriculum_torque`, to run the same reward family with automatic gates. Do not manually add one reward at a time between runs; the task opens reward groups only when the previous group's EMA metrics pass thresholds.

## Stage 0: Stand

Objective: with zero command, stand on four feet in the default pose.

Commands:

- all command dimensions zero
- no jump command

Open rewards:

- `stand_still`
- `orientation`
- `collision`
- `torques`
- `action_rate`
- `dof_acc`

Success metrics:

- long episode length
- nonzero `rew_stand_still`
- four feet contact
- no body contact
- base height near `base_height_target`

## Stage 1: Preload

Objective: from stable standing, learn a controlled crouch/preload without falling or body contact.

Open rewards:

- keep all Stage 0 rewards
- add a preload pose or preload-height reward
- add contact symmetry if needed

Do not open flight rewards yet.

## Stage 2: Takeoff

Objective: learn to push upward from preload.

Open rewards:

- keep Stage 0 and Stage 1 rewards
- add `takeoff_impulse`
- add `takeoff_vertical_velocity`
- add `grounded_jump` after a grace period

Success metrics:

- `takeoff_vertical_velocity` rises
- `jump_flight_rate` starts increasing
- grounded penalty stays small after learning

## Stage 3: Flight And Height

Objective: make four-foot airborne jumps and shape the peak height.

Open rewards:

- keep previous stages
- add `all_feet_airborne`
- add `peak_height_progress`

Success metrics:

- `jump_flight_rate` high
- `mean_peak_height` rises
- `peak_height_error` decreases

## Stage 4: Landing

Objective: finish one jump by landing stably and returning to stand.

Open rewards:

- keep previous stages
- add landing joint target penalties
- add stable completion / `successful_jump`

Success metrics:

- `jump_landing_rate` high
- `successful_jump_rate` rises
- no body contact
- stable all-feet standing after landing

## Stage 5: Commanded Jump

Objective: use command to select stand, single jump, continuous jump, direction, and height.

Open rewards:

- keep previous stages
- add `tracking_linear_velocity` only during takeoff/flight
- add yaw tracking only when yaw command is nonzero
- add command sampling for single-jump and continuous-jump modes

Success metrics:

- zero command stands still
- single-jump command completes one jump and returns to stand
- continuous command repeats jumps
- forward/side velocity commands affect jump direction without breaking landing

## First Experiment

`go2_omnijump_curriculum_torque` uses the following first metric schedule:

- stage 0, stand: zero-command stand only. Rewards follow the `my_go2_jump` style: zero horizontal velocity, zero yaw velocity, base height, default pose, default hip pose, and stand stability. Open next stage when the combined stand score, base-height EMA, and hip-pose EMA pass thresholds.
- stage 1, takeoff: open vertical jump commands plus `takeoff_impulse`, `takeoff_vertical_velocity`, and `grounded_jump`. Open next stage when takeoff velocity reward and `jump_flight_rate` pass thresholds.
- stage 2, flight: open `all_feet_airborne` and `peak_height_progress`. Open next stage when `jump_flight_rate` and `mean_peak_height` pass thresholds.
- stage 3, landing: open landing pose, contact symmetry, and `successful_jump`. Open next stage when landing rate and success rate pass thresholds.
- stage 4, motion: open horizontal/side velocity command sampling and velocity tracking rewards.

These thresholds are config values, so the schedule can be tuned without changing reward functions.
