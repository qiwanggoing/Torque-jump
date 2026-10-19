# Pure-Torque Commanded Jumping for Unitree Go2

I developed and evaluated a reinforcement-learning controller that lets a Unitree Go2 perform a commanded single jump, land near a target, and recover to a stable stand. The final policy uses **direct joint-torque control with zero PD torque assistance** (`α_PD = 0`). It uses a phase-dependent posture reference to guide the different parts of the jump.

## My contribution

- Implemented and trained the Go2 jumping task with PPO in Isaac Gym.
- Developed the pure-torque controller and phase-dependent posture reference for take-off, flight, landing, and recovery.
- Evaluated landing accuracy and post-landing stability under randomized robot mass and ground friction.
- Wrote more than ten Python scripts for repeatable evaluation and diagnosis, including batch command sweeps, jump-phase analysis, joint torque–speed checks, and configurable randomization scans.

## Results

The strongest command range was 0.80–0.90 m in the nominal simulation evaluation (fixed friction and +0.1 kg base mass). A landing counted as a hit when touchdown was within 0.10 m of the target; post-landing recovery used a strict settling criterion.

| Forward jump command | Landing hit rate | Mean landing error | Strict settling |
|---:|---:|---:|---:|
| 0.80 m | 98% | 0.053 m | 100% |
| 0.90 m | 98% | 0.031 m | 100% |

The policy was also stress-tested with domain randomization of robot mass (±1 kg) and ground friction (0.5–1.25). The table above reports the best-performing command bins under nominal simulation conditions, not the domain-randomized aggregate.

## Demo

<p align="center">
  <video src="media/go2_pure_torque_jump.mp4" controls width="78%"></video>
</p>

## Code

The task implementation is in `legged_gym/legged_gym/envs/go2/go2_omnijump_landing_torque/`. Evaluation and analysis utilities are in `legged_gym/legged_gym/scripts/` and `paper_tools/`.

The torque-control base is adapted from SATA; the original research and method are by Li et al., [SATA: Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning](https://arxiv.org/abs/2502.12674), RSS 2025.
