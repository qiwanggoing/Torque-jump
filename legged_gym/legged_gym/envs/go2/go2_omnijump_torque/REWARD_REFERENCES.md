# Reward References

Before adding or changing reward terms for `go2_omnijump_torque`, first check
the jumping paper notes in `/home/qiwang/torque_jump2/papers`.

## Current Task

This task is closest to an OmniNet-style commanded jump primitive:

- commanded jump height
- commanded planar velocity
- commanded yaw rate
- no explicit landing-target command yet

So reward changes should mainly borrow from height/velocity/phase-aware jump
papers, not from full landing-target papers unless the task definition changes.

## Useful Sources

- `papers/JUMPING_PAPERS_FULL_EXTRACTION.md`
  - Atanassov: phase-aware rewards, symmetry, feet clearance, contact changes,
    landing stability.
  - OmniNet: commanded height, linear velocity, yaw velocity, IK-based aerial
    pose, pre-landing pose, action smoothness, collision avoidance.
  - Yang: compact landing-target reward and contact consistency.
  - Bellegarda: sparse final landing position/orientation reward.
  - Olsen/Guan notes: liftoff velocity and jump-height shaping.

- `papers/GO2_LONG_JUMP_REWARD_DESIGN.md`
  - If natural flight rate is zero, first strengthen takeoff/foundation terms.
  - Do not add more final landing rewards before the policy can naturally
    leave the ground.
  - Gate aerial pose rewards by flight/clearance so the robot cannot collect
    them while standing on the ground.

## Mapping For Current Rewards

- `height_tracking`, `successful_jump`: OmniNet commanded-height tracking.
- `tracking_linear_velocity`, `tracking_angular_velocity`: OmniNet commanded
  horizontal/yaw tracking, but should be gated so standing/running on the ground
  cannot dominate.
- `joint_angle_aerial`, `joint_angle_prelanding`, `joint_angle_landing`:
  OmniNet analytical-IK gesture and pre-landing rewards.
- `takeoff_impulse`, `takeoff_vertical_velocity`, `all_feet_airborne`,
  `grounded_jump`: early takeoff-foundation shaping, consistent with the
  staged-learning diagnosis in `GO2_LONG_JUMP_REWARD_DESIGN.md`.
- `left_right_contact_sync`, `straight_jump_joint_symmetry`: contact consistency
  and symmetry ideas from Atanassov/Yang-style phase/contact rewards.
- `orientation`, `collision`, `torques`, `action_rate`, `dof_acc`: common
  stability, safety, energy, and smoothness terms across the jumping papers.

## Rule

Every new reward term should state which paper/task idea it borrows from and
which failure mode it is intended to fix. If there is no paper support, keep it
as a temporary diagnostic term and remove or rename it after the experiment.
