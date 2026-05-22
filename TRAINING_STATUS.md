# 训练状态记录

最后更新: 2026-05-22

两个任务并行训练中，记录当前 setup + 已知问题。

---

## 1. Atanassov 任务 (`go2_atanassov_jump_torque`)

### 当前 checkpoint
- 训完到 iter 5000（一次完整 5000-iter from-scratch 训练）
- 表现：success 68%, peak 0.42m, noise_std 0.16, episode 1830 step
- 能跳高、能连续跳

### 已知问题
- **起跳时后腿 hip 内扣过多**（约 ±15-25°）。物理上 rear hip 内扣有利于 vertical push，policy 学到这个 trick 换更高跳，但视觉不好看
- `rew_default_hip_pos = 2.67`（weight 5），反推 Σ\|hip 偏差\| ≈ 1.06 rad

### 关键 config
```python
# Schedule
warmup_steps = 96000      # iter 1000
x0 = 384000               # iter 4000
max_iterations = 5000

# PPO
init_noise_std = 0.5
entropy_coef = 0.005
learning_rate = 1e-4

# Key rewards
atanassov_nominal_pose = 8.0
default_hip_pos = 5.0
atanassov_takeoff_vz = 5.0
atanassov_landing_position = 50.0
atanassov_max_height = 100.0
atanassov_jumping_sparse = 20.0
sigma_q_nominal = 0.3
```

### 待办（如果要修 hip 内扣）
- A：`default_hip_pos: 5 → 10`，resume + skip fade (warmup=0, x0=1), train 2000 iter
- B：写 L1 form hip reward 替代 exp form（exp 在大偏差时梯度消失）

---

## 2. Curriculum 任务 (`go2_omnijump_curriculum_torque`)

### 当前 checkpoint
- 上次 from-scratch 训练到 iter ~2000 后 abort
- 原因：iter 2000 model 因 `just_took_off` bug 学到「蹭一下拿 success」捷径——peak 0.30m，base 不动，takeoff_timeout fail
- **该 checkpoint 已废，不能用**

### 关键 env bug 修复（已 commit 但未训）

#### Bug 1：`just_took_off` 蹭一下就触发
旧逻辑：`jumping_state & ~has_taken_off & all_feet_contact_off`
- 蹲下时脚 contact 短暂消失 → 误判 takeoff
- has_taken_off=True → just_landed 自然触发 → success 拿到
- robot 学到「不真跳也能拿钱」

修复：加 vz > 0.3 m/s 门
```python
just_took_off = jumping_state & ~has_taken_off & all_feet_contact_off & (vz > 0.3)
```

#### Bug 2：`_reward_takeoff_direction` 负值
症状：旧 buggy just_took_off 在 vz<0 的瞬间触发 → vz/‖v‖ 是负的
修复：bug 1 修复后自动消失

### Pose 两态化（新设计）
旧：3 个 PD/reward target（q_squat / q_ground / standing）—— q_ground 是个奇怪的中间 pose
新：只 2 个
- `phase_loaded`（蓄力）→ q_squat
- 其它所有阶段 → default_dof_pos (standing)

改动地方：
- `_reward_default_pos`：phase-aware target
- `_update_default_joint_pd_target`：同样 phase-aware

### 引导跳跃的 reward 加权
为了避开「policy 学不动反而更划算」局部最优：
```python
all_feet_airborne:         2.0 → 6.0    (3×)
takeoff_vertical_velocity: 10.0 → 25.0  (2.5×)
```

### 关键 config
```python
# Schedule
warmup_steps = 96000       # iter 1000
x0 = 384000                # iter 4000
max_iterations = 5000

# PPO
init_noise_std = 0.5
entropy_coef = 0.005

# Cmd
jump_height = [0.4, 0.7]
lin_vel_x/y = [0.0, 0.0]   # 纯垂直
single_jump_command_prob = 1.0

# Active rewards (with weights)
maintain_contact = 0.10
peak_height_progress = 5.0
all_feet_airborne = 6.0          ← boosted
takeoff_vertical_velocity = 25.0 ← boosted
projected_peak = 7.0
termination = -10.0
orientation = -1.6
collision = -3.0
torques = -1e-5
action_rate = -0.08
dof_acc = -1e-6
horizontal_drift = -1.5
takeoff_direction = 80.0
height_tracking = 1.0
successful_jump = 300.0
default_pos = -0.3
default_hip_pos = 0.3
aerial_dof_acc = -3e-6
task_max_height = 12.0
landing_stability = 15.0

# Landing stability sigma (configurable in env)
landing_stability_lin_vel_sigma = 1.0  # was 0.25 hardcoded
landing_stability_ang_vel_sigma = 1.5  # was 0.5 hardcoded

# RSI (mid-jump bootstrap)
rsi_prob = 0.2
rsi_vel_z_min = 1.0
rsi_vel_z_max = 3.0
rsi_height_offset_max = 0.1
```

### Play 设置（镜像训练）
```python
PRE_JUMP_IDLE_SECONDS = 2.0
POST_JUMP_STAND_SECONDS = 6.0    # 匹配训练 post_jump_stand_steps=300
CONTINUOUS_JUMP = False          # 镜像 single-jump training
test.vel = [0.0, 0.0, 0.0, 0.7, 1.0]  # vx=0（之前 bug 是 1.0）
```

### 下一步
1. 从头训 5000 iter（新 env code 全应用 + 新 weights）
2. 监测点：
   - iter 200: `rew_takeoff_vertical_velocity` > 0.05
   - iter 500: success > 25%, peak > 0.15m
   - iter 1000: warmup 结束, success > 40%
   - iter 3000: success > 70%, peak > 0.40m
   - iter 5000: success > 80%, peak > 0.50m

---

## 共同的 env code 改动（影响两个任务）

`legged_gym/envs/go2/go2_omnijump_torque/go2_omnijump_torque.py`:
- `just_took_off` 加 vz > 0.3 门
- `_reward_default_pos` 两态 target
- `_update_default_joint_pd_target` 两态 target
- `_reward_landing_stability` sigma 可配置

注：Atanassov 任务**不依赖**这些（用 `atanassov_*` reward），所以 env 改动对它无影响。Curriculum 任务依赖 parent 的 `default_pos / default_hip_pos / takeoff_direction / landing_stability`。

## 决策日志
- 选 single-jump training，post_jump_stand=300 给 6s 站立
- single_jump_command_prob 在两任务一直保持 1.0（纯跳跃训练，不混 stand-still episode）
- PD fade 用 linear schedule，由 curriculum/atanassov 各自 override 父类 Gompertz
- noise_std 0.5 + entropy 0.005 是稳定收敛的组合（0.35 + 0.001 会让 noise 早 collapse）
