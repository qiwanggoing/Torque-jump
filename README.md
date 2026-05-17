# SATA

Official Implementation for **[SATA: Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning](https://arxiv.org/abs/2502.12674)**

Accepted at **Robotics: Science and Systems (RSS) 2025**

LI Peizhuo*. LI Hongyi*, Ge SUN, [Jin CHENG](https://jin-cheng.me), Xinrong YANG, Guillaume BELLEGARDA, Milad SHAFIEE, [Yuhong CAO](https://www.yuhongcao.online), Auke IJSPEERT, [Guillaume SARTORETTI](https://cde.nus.edu.sg/me/staff/sartoretti-guillaume-a/)

<p align="center">
  <img src="sata_overview.png" width="100%" />
</p>

---

## 🧭 Overview

**SATA** is a torque-based reinforcement learning framework inspired by how animals progressively acquire locomotion capabilities.  
It introduces a biologically motivated **growth curriculum** that schedules torque limits and control frequency to evolve safely during training.

---

## 🦘 OmniJump Torque Task (this fork)

This fork adds `go2_omnijump_curriculum_torque` — an omnidirectional jumping task built on top of the SATA torque framework. The goal is a policy that **survives the full PD prior fade-out** (PD weight `0.5 → 0`) and executes pure-torque jumps.

### Train and play

```bash
python legged_gym/scripts/train.py --task=go2_omnijump_curriculum_torque --headless
python legged_gym/scripts/play.py  --task=go2_omnijump_curriculum_torque --checkpoint=4500
```

### Why pure-torque jumping is hard

A torque policy trained with PD support tends to learn a *PD-assisted* strategy that collapses once PD fades. Without explicit reward signals or observation hooks for PD strength, the policy can't decouple from the PD prior — when PD reaches 0%, the robot can't even stand, let alone jump. Our earlier baselines crashed to episode length ~24 steps once PD reached 0.

### What made pure-torque jumping work

| Fix | Why |
|---|---|
| **Slow linear PD fade** (`growth.x0: 288k → 384k`) | Fade ends near iter ~4000 instead of ~3000; gentler dynamics drift gives PPO time to adapt |
| **Disable torque-cap ramp** (`growth.start_torque_scale: 0.5 → 1.0`) | RL gets ~12 Nm authority from iter 0 instead of ~5.8 Nm; effective RL scale ramp drops from 4× to 2× over training |
| **`pd_alpha` added to observation (+1 dim, 68 → 69)** | Policy directly observes the current PD blend weight, learning a progress-conditional action `π(state, pd_alpha)` instead of an unconditional compromise |
| **`default_pos` reward fixed to span all 12 joints** | Was hip-only — thighs/calves had no reward-side anchor toward standing pose. Whole-body L1 to `default_dof_pos` with weight `-0.3` (mygo2jump value) gives RL a continuous signal to maintain standing throughout training |
| **`post_jump_stand_steps: 80 → 300`** (~0.8 s → 3 s) | Each episode now contains an explicit 3-second autonomous-stand phase after landing, exposing the policy to `cmd[4]=0` dynamics with reduced PD support |
| **`aerial_dof_acc` reward (weight `-1e-6`)** | Airborne-only joint-acceleration penalty (4× stricter than global `dof_acc`); suppresses in-air twitching/flailing observed after PD fully fades |

### Training schedule

| Phase | Iterations | PD weight |
|---|---|---|
| Warmup | 0 → 1000 | 50% (constant) |
| Linear fade | 1000 → 4000 | 50% → 0% |
| Pure-torque refinement | 4000 → 5000 | 0% |

`max_iterations = 5000`.

### Observation layout (69 dims)

```
base_lin_vel (3) + base_ang_vel (3) + projected_gravity (3) +
commands (5) + height_obs (2) +
dof_pos (12) + dof_vel (12) + foot_contact (4) +
torques (12) + motor_fatigue (12) + pd_alpha (1)
```

Single-frame, no stacking. `pd_alpha` is the only curriculum-aware channel.

---

## Training in Simulation

### Pipeline to Install and Train ABS

**Note**: Before running our code, it's highly recommended to first play with [RSL's Legged Gym version](https://github.com/leggedrobotics/legged_gym) to get a basic understanding of the Isaac-LeggedGym-RslRL framework.
   <!-- <br/><br/> -->

1. Create environment and install torch

   ```text
   conda create -n xxx python=3.8  # or use virtual environment/docker
   
   pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116  
   # used version during this work: torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
   # for older cuda ver:
   pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```

   

2. Install Isaac Gym preview 4 release https://developer.nvidia.com/isaac-gym

   unzip files to a folder, then install with pip:

   `cd isaacgym/python && pip install -e .`

   check it is correctly installed by playing: 

   ```cmd
   cd examples && python 1080_balls_of_solitude.py
   ```

   

3. Clone this codebase and install our `rsl_rl`

   ```cmd
   pip install -e rsl_rl
   ```



4. Install our `legged_gym`

   ```cmd
   pip install -e legged_gym
   ```

   Ensure you have installed the following packages:
    + pip install numpy==1.21 (must < 1.24, >1.20)
    + pip install tensorboard
    + pip install setuptools==59.5.0
    + pip install wandb
    
5. Try training.

   can use "--headless" to disable gui, press "v" to pause/resume gui play.

   for go2, in `SATA/legged_gym/legged_gym/envs/go2/go2_torque`,
    ```text
   python scripts/train.py --task=go2_torque
   ```
6. Play the trained policy

   ```cmd
   python scripts/play.py --task=go2_torque
   ```
## Troubleshooting:
### Contact
+ Corresponding author: CAO Yuhong: caoyuhong@nus.edu.sg
+ Deployment: LI Peizhuo: lipeizhuo@u.nus.edu 
+ Policy Learning in Sim: LI Hongyi: hongyi.li@u.nus.edu


### Issues
You can create an issue if you meet any bugs, except:
+ If you cannot run the [vanilla RSL's Legged Gym](https://github.com/leggedrobotics/legged_gym), it is expected that you first go to the vanilla Legged Gym repo for help.
+ There can be CUDA-related errors when there are too many parallel environments on certain PC+GPU+driver combination: we cannot solve thiss, you can try to reduce num_envs.
+ Our codebase is only for our hardware system showcased above. We are happy to make it serve as a reference for the community, but we won't tune it for your own robots.

## Credit
If our work does help you, please consider citing us and the following works:
```bibtex
@article{li2025sata,
  title={SATA: Safe and Adaptive Torque-Based Locomotion Policies Inspired by Animal Learning},
  author={Li, Peizhuo and Li, Hongyi and Sun, Ge and Cheng, Jin and Yang, Xinrong and Bellegarda, Guillaume and Shafiee, Milad and Cao, Yuhong and Ijspeert, Auke and Sartoretti, Guillaume},
  journal={arXiv preprint arXiv:2502.12674},
  year={2025}
}
```

