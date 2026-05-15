# GO2 SATA 跳跃任务完整规格

Date: 2026-05-11

本文档记录 `go2_omnijump_curriculum_torque` 任务的当前完整设计。

## 1. 任务目标

```
给一次 jump command → 跳一次 → 落地站稳 → episode 结束
jump command = 0 时 → 安静站立
```

不是 height tracking 任务。高度只作为跳跃质量的辅助信号。

## 2. 继承链

```
GO2OmniJumpCurriculumTorque
  → GO2OmniJumpTorque
    → GO2Torque (SATA 基类)
      → LeggedRobot (Isaac Gym)
```

## 3. 控制架构

### 3.1 力矩输出

```
torque = PD_prior_torque + residual_torque

PD_prior_torque = (Kp * (q_target - q) - Kd * qdot) * pd_alpha
residual_torque = policy_action * action_scale * rl_alpha * torque_limit_growth
```

- `Kp = 40.0, Kd = 1.2`（所有关节统一）
- `pd_alpha = 0.50`（固定）
- `rl_alpha = 0.50`（固定）
- `action_scale = 60.0`
- `torque_limit_growth`：SATA Gompertz 曲线，从 0.3 增长到 1.0

### 3.2 PD 先验目标随 phase 切换

PD prior 的 `q_target` 不是固定的，而是根据跳跃阶段动态切换：

| Phase | q_target | 来源 |
|-------|----------|------|
| 非跳跃 / idle | `default_dof_pos` | 初始站姿 |
| 蓄力/起跳 | `q_ground_target` | IK: foot_height=0.30m（腿伸直） |
| 腾空（上升+顶点） | `q_air_target` | IK: foot_height=0.18m（收腿） |
| 预着陆（下降末段） | `q_pre_target` | IK: foot_height=0.25m（半伸腿） |
| 着地恢复 | `q_ground_target` | IK: foot_height=0.30m |

IK 目标在 `__init__` 时一次性计算，训练中不变。
四条腿用相同的 thigh/calf 目标，hip ab/ad 保持 default。

### 3.3 SATA 特性

经过 `activation_process`（肌肉激活低通滤波）→ `hill_model`（力-速度关系）→ `motor_fatigue`（疲劳累积）后输出最终力矩。

### 3.4 SATA Gompertz 频率和力矩增长

```
general_scale = exp(-exp(-k * (step - x0)))
    k = 0.00003, x0 = 24000

频率:     100Hz → 200Hz (general_scale * 100 + 100)
残差力矩上限: 30% → 100% (general_scale * 0.7 + 0.3)
```

增长曲线（近似）：

| 训练步数 | scale | 频率 | 残差力矩上限 |
|---------|-------|------|------------|
| 0 | 0.13 | 113 Hz | 39% |
| 20000 | 0.32 | 132 Hz | 53% |
| 50000 | 0.63 | 163 Hz | 74% |
| 100000 | 0.90 | 190 Hz | 93% |
| 200000 | 0.99 | 200 Hz | 100% |

注意：`torque_limit_growth` 只作用在 `residual_torque` 上，PD prior 不受限制。

## 4. Command 设计

5 维 command：

```
commands[0] = lin_vel_x     范围 [0.8, 1.4]
commands[1] = lin_vel_y     范围 [-0.4, 0.4]
commands[2] = ang_vel_yaw   范围 [0.0, 0.0]（当前关闭）
commands[3] = jump_height   范围 [0.35, 0.55]
commands[4] = jump_command  范围 [0.0, 1.0] 均匀采样
```

- `jump_command > 0.5` → 触发跳跃
- `jump_command ≤ 0.5` → 站立，不跳
- `jump_command` 从 [0, 1] 均匀采样，约 50% 跳跃 / 50% 站立
- 跳完后 `jump_command` 被设为 0.0（不再跳）
- 所有 jump episode 都是单次跳模式（`single_jump_command_prob = 1.0`）
- 一个 episode 最多跳一次（`one_jump_reward_per_episode = True`）
- `resampling_time = 1.8s`

## 5. Observation 设计

### 5.1 Actor Observation（68 维）

跟 SATA 原版一致：用力矩和疲劳代替 previous action。

| 维度 | 内容 | obs_scale |
|------|------|-----------|
| 0-2 | base 线速度 | 2.0 |
| 3-5 | base 角速度 | 0.25 |
| 6-8 | 重力投影 | 1.0 |
| 9-11 | 速度命令 (vx, vy, yaw) | lin_vel/ang_vel scale |
| 12 | jump_height 命令 | ×2.0 |
| 13 | jump_command | ×1.0 |
| 14 | 当前 base 高度 | ×2.0 |
| 15 | 高度误差 (cmd_h - base_h) | ×2.0 |
| 16-27 | 关节角偏移 (dof_pos - default) | 1.0 |
| 28-39 | 关节角速度 | 0.05 |
| 40-43 | 足端接触状态 (4维 binary) | 1.0 |
| 44-55 | 上一步实际力矩 | 1.0 |
| 56-67 | 电机疲劳 | 1.0 |

obs_scales 的作用：不同物理量数值范围差异大（角速度 0~10 rad/s，关节角速度 0~30 rad/s），
乘以 scale 后归一化到 O(1) 量级，让网络更容易学习。

力矩和疲劳时序说明（SATA 原版设计）：

```
step() 执行顺序:
  ① actions = policy_output           （当前动作）
  ② torques = _compute_torques(actions) （经 activation + hill model 的实际力矩）
  ③ apply torques → simulate           （施加力矩 → 物理仿真）
  ④ refresh state                       （读取新的 dof_pos, dof_vel 等）
  ⑤ compute_observations()             （构建观测）

在第⑤步构建观测时:
  - dof_pos, dof_vel = 施加力矩后的新状态（来自④）
  - self.torques = 导致这个新状态的力矩（来自②）
  - motor_fatigue = 累积到当前的疲劳水平

所以 self.torques 的语义是"导致当前状态的上一步力矩"，
从策略决策的视角看是 last_torque，因果关系正确。
```

不包含 raw previous action 的原因：
SATA 是力矩控制，`self.torques` 是 raw action 经过 activation process + hill model +
fatigue 处理后的实际力矩输出，信息比 raw action 更丰富——策略能从中了解电机的真实状态。
SATA 原版 go2_torque 也采用相同设计。

当前 ground truth 使用说明：
base 线速度和高度目前使用 ground truth（仿真器直接提供）。
真机部署时 IMU 无法直接测量这些量，需要替换为 estimator。
后续可参考 OmniNet 的做法训练一个 height estimator。

### 5.2 Privileged Critic Observation（108 维）

Critic 看到 actor obs + 额外特权信息（仅训练时可用）：

| 维度 | 内容 |
|------|------|
| 0-67 | 与 actor 相同的 68 维 obs |
| 68 | ground truth base 高度 |
| 69-71 | ground truth base 线速度 |
| 72-83 | 足端在 body frame 下的位置 (4×3) |
| 84-95 | 足端速度 (4×3) |
| 96-107 | 足端接触力 (4×3) |

## 6. Action 设计

12 维：每个关节一个 residual torque 系数。

```
action ∈ R^12
residual_torque = action * action_scale * rl_alpha * torque_limit_growth
```

## 7. Episode 流程

### 7.1 jump_command ≥ 0.5 的 episode（跳跃）

```
1. 站立等待: first_jump_delay_steps = 55 步
   - 策略收到 jump_command > 0.5
   - 必须先站稳 (stand_rearm_steps = 50 步)
   - PD prior → default standing pose

2. 起跳触发: stable_stand 检测通过后
   - jumping_state = True
   - PD prior → q_ground_target
   - 策略的 residual torque 产生向上冲量
   - takeoff_timeout = 40 步内必须起飞

3. 腾空: 四足全部离地后
   - airborne = True
   - PD prior → q_air_target（收腿）
   - 记录 peak_base_height

4. 预着陆: 下降 + 高度低于阈值
   - prelanding = True
   - PD prior → q_pre_target（半伸腿）

5. 着陆: 任一足端触地
   - has_landed = True, landing = True
   - PD prior → q_ground_target
   - landing_buffer_steps = 50 步缓冲期
   - 期间检查: 姿态倾斜/身体碰撞 → 取消成功判定

6. 成功判定:
   - 缓冲期满 → 检查 pending_success
   - 条件: 峰值高度误差 < 0.08m && 稳定着地

7. 跳后站立:
   - jump_command → 0.0
   - post_jump_stand_steps = 80 步站立恢复
   - 然后 episode reset
```

### 7.2 jump_command < 0.5 的 episode（站立）

```
- 策略收到 jump_command < 0.5
- 不触发跳跃，整个 episode 练习站立
- stand_still reward 持续给
- episode 到 max_length (10s) 后 timeout reset
```

## 8. Reward 设计

### 8.1 当前活跃的 reward（curriculum disabled 时全部活跃）

#### 任务 reward

| 名称 | Scale | 形式 | 活跃条件 |
|------|-------|------|---------|
| `task_max_height` | +60.0 | exp(-height_error²/σ) | jumping + has_taken_off |
| `successful_jump` | +80.0 | binary × velocity_score | 跳跃完成且满足判定条件 |
| `tracking_linear_velocity` | +10.0 | exp(-vel_error/σ) | jump_command active |
| `tracking_angular_velocity` | +5.0 | exp(-yaw_error/σ) | jump_command active & yaw > 0.05 |

#### 姿态 reward（OmniNet L1 惩罚形式）

| 名称 | Scale | 形式 | 活跃条件 |
|------|-------|------|---------|
| `joint_angle_aerial` | **-0.5** | Σ\|q - q_air\| | airborne & ~prelanding |
| `joint_angle_prelanding` | **-0.8** | Σ\|q - q_pre\| | prelanding |
| `joint_angle_landing` | **-0.15** | Σ\|q - q_ground\| | landing |
| `landing_stability` | +10.0 | exp(-lin_vel²) × exp(-ang_vel²) | landing |

#### 正则化 reward

| 名称 | Scale | 形式 |
|------|-------|------|
| `orientation` | -2.0 | gravity_xy² |
| `collision` | -20.0 | 非足端接触次数 |
| `torques` | -1e-6 | Σ torque² |
| `action_rate` | -0.001 | Σ(a_t - a_{t-1})² |
| `dof_acc` | -2.5e-7 | Σ joint_acc² |
| `termination` | -10.0 | 非超时终止 |

#### 未启用但代码中存在的 reward（scale = 0）

- `takeoff_vertical_velocity`
- `takeoff_impulse`
- `all_feet_airborne`

### 8.2 姿态 reward 改动说明

原来：`exp(-mean(error²) / sigma)` 正奖励（exp kernel）
现在：`Σ|error|` × 负权重（L1 惩罚，OmniNet 形式）

改动原因：exp kernel 在偏差大时梯度消失，L1 惩罚梯度恒定。
但由于 PD prior 始终在工作，姿态引导的主要力来自 PD prior 而非 reward。

### 8.3 IK 姿态目标

从 foot height 反解关节角（余弦定律 + atan2）：

| 目标 | foot_height | thigh (rad) | calf (rad) | 用途 |
|------|------------|-------------|------------|------|
| `q_ground` | 0.30m | ~0.78 | ~-1.50 | 站立/起跳/着陆 |
| `q_air` | 0.18m | ~1.23 | ~-2.20 | 空中收腿 |
| `q_pre` | 0.25m | ~0.96 | ~-1.78 | 预着陆伸腿 |

IK 参数：`thigh_length = 0.213m, calf_length = 0.213m, nominal_foot_x = 0.02m`

## 9. Phase 判定逻辑

```
jump_command > 0.5
  && stable_stand 持续 stand_rearm_steps 步
  → jumping_state = True

四足全部离地 (contact_filt 全 False)
  → has_taken_off = True, airborne = True

base_lin_vel_z < -0.05 && root_z ≤ prelanding_height
  → prelanding = True
  prelanding_height = max(peak_height - 0.06, base_height_target + 0.04)

任一足端触地 (contact_filt 任一 True)
  → has_landed = True, landing = True

landing_step_counter ≥ landing_buffer_steps
  → 判定成功/失败 → finish_jump
  → jump_command 被关闭
```

## 10. Termination 条件

- base 接触地面（terminate_after_contacts_on = ["base"]）
- roll 角 > 2.4 rad
- episode 超时（10s）
- 单次跳模式：跳完 + 站立恢复后 reset

## 11. 网络架构

```
Actor:  MLP [256, 256, 256] → 12, ELU activation, input = 68 (actor obs)
Critic: MLP [256, 256, 256] → 1,  ELU activation, input = 108 (privileged obs)
```

Asymmetric Actor-Critic：actor 和 critic 看不同的观测。
critic 在训练时使用特权信息（足端位置/速度/接触力等），部署时只用 actor。

- PPO, adaptive learning rate, lr = 1e-4
- γ = 0.99, λ = 0.95
- symmetry loss: obs/act permutation, sym_coef = 0.5
- num_steps_per_env = 48, num_mini_batches = 8
- max_iterations = 5000

## 12. Domain Randomization

当前全部关闭（`randomize_friction = False`, `push_robots = False`, `randomize_base_mass = False`）。
后续跳跃稳定后再逐步打开。

## 13. 初始状态

```
base position: [0, 0, 0.42]
base orientation: [0, 0, 0, 1] (identity)
base velocity: [0, 0, 0]
关节角: 前腿 thigh=0.8, 后腿 thigh=1.0, calf=-1.5, hip=±0.1
```

无 RSI（Reference State Initialization），所有 episode 从站立开始。

## 14. 今日改动汇总（2026-05-11）

1. 姿态 reward 从 exp kernel 改为 OmniNet 的 L1 惩罚形式
2. SATA torque_limit_growth 应用到残差力矩上（之前漏了）
3. 去掉 stand_command_prob，改为 jump_command 从 [0,1] 均匀采样
4. 打开 one_jump_reward_per_episode = True（单次跳模式）
5. single_jump_command_prob = 1.0
6. _disable_jump_command 不再清零速度命令，只关 jump_command
7. Observation 去掉 self.actions（12维），保留 SATA 原版的 torques + fatigue 设计
8. num_observations: 80 → 68
9. 新增 privileged critic obs（108维）：足端位置/速度/接触力
10. obs_permutation 更新为 68 维版本

## 15. 论文参考依据

| 设计决策 | 参考论文 |
|---------|---------|
| jump toggle 在 observation 中 | Atanassov 2025, Guan 2024 |
| L1 姿态惩罚 | OmniNet (Han 2025) |
| prelanding > aerial 权重 | OmniNet (Han 2025) |
| PD prior + residual torque | SATA (Li 2025) |
| 频率 + 力矩渐进增长 | SATA (Li 2025) |
| phase-aware PD target 切换 | 自有设计（OmniNet 只用在 reward，我们同时用在 PD prior） |
| 解析 IK 姿态目标 | OmniNet (Han 2025) |
| 单次跳训练模式 | Olsen 2025, Bellegarda 2024 |
| ~50/50 跳立比例 | Guan 2024 |
