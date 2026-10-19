# Pure-Torque Commanded Jumping for Unitree Go2

I developed and evaluated a reinforcement-learning controller that lets a Unitree Go2 perform a commanded single jump, land near a target, and recover to a stable stand. The final policy uses **direct joint-torque control with zero PD torque assistance** (`α_PD = 0`). It uses a phase-dependent posture reference to guide the different parts of the jump.

## My contribution

- Implemented and trained the Go2 jumping task with PPO in Isaac Gym.
- Developed the pure-torque controller and phase-dependent posture reference for take-off, flight, landing, and recovery.
- Evaluated landing accuracy and post-landing stability under randomized robot mass and ground friction.
- Wrote more than ten Python scripts for repeatable evaluation and diagnosis, including batch command sweeps, jump-phase analysis, joint torque–speed checks, and configurable randomization scans.

## Results

The final policy was evaluated under the full domain-randomization protocol: robot mass varied by ±1 kg and ground friction ranged from 0.5 to 1.25. A landing counted as a hit when the robot touched down within 0.10 m of its target. Post-landing recovery used a strict settling criterion.

| Measure | Result |
|---|---:|
| Landing hit rate (0.10 m tolerance) | 65% |
| Mean landing error | 0.091 m |
| Strict post-landing settling | 98% |
| Mean settling time | 0.39 s |

These results show that the robot can land and settle reliably with no direct PD torque assistance, while target accuracy remains an important metric for further improvement.

## Demo

<p align="center">
  <video src="media/go2_pure_torque_jump.mp4" controls width="78%"></video>
</p>

## Code

The task implementation is in `legged_gym/legged_gym/envs/go2/go2_omnijump_landing_torque/`. Evaluation and analysis utilities are in `legged_gym/legged_gym/scripts/` and `paper_tools/`.

The torque-control base is adapted from SATA; the original research and method are by Li et al., [SATA: Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning](https://arxiv.org/abs/2502.12674), RSS 2025.
