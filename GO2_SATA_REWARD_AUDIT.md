# GO2 SATA 跳跃 Reward 设计审计与论文对比

Date: 2026-05-11

## 1. 当前 Reward 状态总览

### 1.1 实际活跃的 reward（在 whitelist 中且 scale ≠ 0）

| 名称 | Scale | 返回值范围 | 每步贡献量级 | 活跃阶段 |
|------|-------|----------|------------|---------|
| `successful_jump` | +80.0 | 0 或 1 | 0 或 80（极稀疏） | 跳跃完成判定瞬间 |
| `task_max_height` | +60.0 | 0~1 | 0~60 | 起飞后整个跳跃过程 |
| `tracking_linear_velocity` | +10.0 | 0~1 | 0~10 | jump_command > 0.5 时 |
| `landing_stability` | +10.0 | 0~1 | 0~10 | 着陆缓冲期 |
| `tracking_angular_velocity` | +5.0 | 0~1 | 0~5 | jump_command > 0.5 且 yaw > 0.05 |
| `orientation` | -2.0 | 0~1 | 0~-2 | 始终 |
| `joint_angle_prelanding` | -0.8 | 0~12+ | 0~-10 | prelanding |
| `joint_angle_aerial` | -0.5 | 0~12+ | 0~-6 | 腾空（非 prelanding） |
| `joint_angle_landing` | -0.15 | 0~12+ | 0~-2 | 着陆缓冲期 |
| `collision` | -20.0 | 0~N | 0~-60 | 始终 |
| `action_rate` | -0.001 | 0~100+ | 0~-0.1 | 始终 |
| `torques` | -1e-6 | 0~6000+ | 0~-0.006 | 始终 |
| `dof_acc` | -2.5e-7 | 0~1e6+ | 0~-0.25 | 始终 |

### 1.2 有代码但未激活的 reward

| 名称 | 原因 | 影响 |
|------|------|------|
| `stand_still` | **不在 whitelist** | **严重：jump_command < 0.5 时没有站立奖励** |
| `zero_command_linear_velocity` | 不在 whitelist | 同上 |
| `zero_command_angular_velocity` | 不在 whitelist | 同上 |
| `zero_command_base_height` | 不在 whitelist | 同上 |
| `default_hip_pos` | 不在 whitelist | 无 hip 姿态引导 |
| `left_right_contact_sync` | 不在 whitelist | 无对称接触引导 |
| `straight_jump_joint_symmetry` | 不在 whitelist | 无关节对称引导 |
| `takeoff_vertical_velocity` | scale = 0 | 无起跳引导 |
| `takeoff_impulse` | scale = 0 | 无蹬地引导 |
| `all_feet_airborne` | scale = 0 | 无腾空引导 |
| `termination` | **不在 whitelist** | **终止惩罚无效** |

## 2. 发现的问题

### 问题 1：没有站立 reward（严重）

`stand_still` 不在 `ACTIVE_REWARD_WHITELIST` 中。

当 `jump_command < 0.5`（约 50% 的 episode），策略：
- 没有任何正向奖励鼓励站立
- 只有负向惩罚（orientation, collision, torques）
- 结果：策略不知道"不跳的时候应该安静站着"

所有论文都有站立引导：
- **Atanassov**：stance 阶段有 contact maintenance + nominal pose
- **OmniNet**：通过 estimator 高度切换状态机
- **Guan**：walking reward 和 jump reward 50/50 混合

### 问题 2：没有起跳 dense reward（中等）

`takeoff_vertical_velocity`、`takeoff_impulse`、`all_feet_airborne` 全是 scale=0。

策略从"站立"到"腾空"之间没有任何 dense 引导。
唯一的正向信号是 `task_max_height`（需要先跳起来才有值）。

这产生了一个 bootstrap 困难：
- 策略需要先学会产生向上冲量 → 才能腾空 → 才能获得 height reward
- 但产生向上冲量这件事本身没有 reward

论文做法：
- **Atanassov**：stance 阶段有 base position / velocity dense reward
- **Olsen**：有 projectile-based estimated jump height（不需要真的腾空就有信号）
- **Guan**：有 dense jump reward（-std(F_foot)）引导起跳

### 问题 3：termination 不在 whitelist（中等）

`termination` reward（scale=-10.0）在 config 里定义了，但不在 whitelist。
策略摔倒/碰撞后被 reset，但不会收到额外惩罚。
collision 惩罚部分弥补了这一点（scale=-20.0），但 termination 应该也是有意义的。

### 问题 4：reward 量级严重不平衡

与 OmniNet 对比倍率：

| Reward | OmniNet | 我们 | 倍率 |
|--------|---------|------|------|
| successful_jump | 20 | 80 | **4x** |
| lin_vel_tracking | 1.5 | 10 | **6.7x** |
| ang_vel_tracking | 0.6 | 5 | **8.3x** |
| collision | -1 | -20 | **20x** |
| action_rate | -0.01 | -0.001 | 0.1x |

`successful_jump=80` 和 `task_max_height=60` 的量级远大于其他项。
这会导致：
- 策略只追求"跳成功"和"跳到目标高度"
- 忽略姿态质量、动作平滑度、着陆稳定性
- collision=-20 非常重，可能导致策略过于保守

### 问题 5：tracking_linear_velocity 在全时段活跃

当前配置 `tracking_linear_velocity_all_time = True`，
速度跟踪 reward 在 jump_command active 时始终给。

但我们的目标是"跳得远"，不是"全程跟踪一个水平速度"。
在起跳前、着陆后给速度跟踪 reward 是不合理的——
站在原地蹲下蓄力时策略不应该被要求有前向速度。

论文做法：
- **OmniNet**：速度跟踪在 jumping 阶段给（不是全时段）
- **Atanassov**：base velocity reward 在 flight 阶段权重更大

## 3. 论文 Reward 体系对比

### 3.1 OmniNet（Han 2025）

```
任务: height tracking + omnidirectional
reward 项数: 12
特点: 简洁、均衡

正向:
  height_tracking     = 1.0  （dense, exp kernel）
  successful_jump     = 20   （sparse, ±0.04m）
  lin_vel_tracking    = 1.5
  ang_vel_tracking    = 0.6

姿态（L1 惩罚）:
  joint_aerial        = -0.4
  joint_prelanding    = -0.6
  joint_landing       = -0.12

正则化:
  orientation         = -0.8
  collision           = -1.0
  torque              = -1e-5
  action_rate         = -0.01
  joint_acc           = -2.5e-7

最大正向/步: ~23     最大负向/步: ~-3
比例: 正:负 ≈ 8:1
```

### 3.2 Atanassov（2025）

```
任务: goal-conditioned landing-target jump
reward 项数: 19+
特点: 最丰富、phase-aware、乘法聚合

稀疏任务:
  landing_position        （着陆时）
  landing_orientation     （着陆时）
  max_height              （着陆时）
  jumping_detection       （着陆时）

稠密任务:
  base_position           （stance/flight/landing 不同目标）
  orientation_tracking    （stance + landing）
  base_linear_velocity    （主要 flight）
  base_angular_velocity   （flight + landing）
  feet_clearance          （flight）
  symmetry                （flight）
  nominal_pose            （三相都有）

正则化:
  energy
  base_acceleration
  contact_change
  maintain_contact        （stance）
  contact_forces
  action_rate
  joint_acceleration
  joint_limits

注意：负奖励通过指数形式缩放总奖励，不是简单求和。
```

### 3.3 Olsen（2025）

```
任务: vertical jump + horizontal jump（分开训练）
reward 项数: ~9 per task
特点: projectile densification

竖直跳:
  jump_height             （主任务）
  estimated_jump_height   （抛体估计，densification 关键）
  symmetry
  angular_velocity
  orientation_error
  desired_joint_positions  （目标姿态）
  ground_force_L2
  catch_landing
  damp_landing
```

### 3.4 Guan（2024）

```
任务: walking → jumping 扩展
reward 项数: walking rewards + 2 jump rewards
特点: 最精简

新增 jump rewards:
  dense_jump    = -std(F_foot)       （足端力分布均匀 → 有组织的起跳）
  sparse_jump   = exp(-(V - V*)²)    （liftoff velocity 匹配）
```

## 4. 建议的修改

### 4.1 必须修复

**加入 stand_still 到 whitelist：**

没有站立 reward，策略在 jump_command=0 时没有学习信号。

**加入 termination 到 whitelist：**

终止惩罚应该生效。

### 4.2 建议启用的 reward

**takeoff_vertical_velocity（建议 scale: 5.0~10.0）：**

给起跳阶段 dense 引导。没有这个，策略很难 bootstrap 出第一次起飞。

**left_right_contact_sync 或 straight_jump_joint_symmetry（建议 scale: 1.0~3.0）：**

前跳时左右对称很重要，Atanassov 和 Olsen 都有。

### 4.3 建议调整的权重

**降低 successful_jump：80 → 30~40**

当前太大，会主导整个 reward landscape。
OmniNet 只用 20。

**降低 task_max_height：60 → 15~25**

同上。

**降低 collision：-20 → -5~-8**

当前太重，策略可能过于保守不敢蹬地。
OmniNet 只用 -1。

**提高 action_rate：-0.001 → -0.005~-0.01**

当前太小，几乎没有平滑作用。
OmniNet 用 -0.01。

**tracking_linear_velocity 改为仅 flight 阶段：**

不应该在起跳前/着陆后要求速度跟踪。
设置 `tracking_linear_velocity_all_time = False`。

### 4.4 建议参考的权重比例（基于 OmniNet 的均衡比例）

```
# 任务
task_max_height            = 20.0    （OmniNet height_tracking=1 但 dense; 我们是 semi-sparse 需更大）
successful_jump            = 40.0    （OmniNet 20，我们的判定更严格）
tracking_linear_velocity   = 3.0     （OmniNet 1.5，仅 flight 阶段）
tracking_angular_velocity  = 1.5     （OmniNet 0.6）

# 起跳引导（新启用）
takeoff_vertical_velocity  = 8.0

# 站立
stand_still                = 5.0     （新启用）

# 姿态（已改为 L1，保持不变）
joint_angle_aerial         = -0.5
joint_angle_prelanding     = -0.8
joint_angle_landing        = -0.15
landing_stability          = 5.0

# 对称（新启用）
left_right_contact_sync    = 2.0

# 正则化
orientation                = -1.5
collision                  = -5.0
termination                = -5.0    （新启用）
torques                    = -1e-5
action_rate                = -0.005
dof_acc                    = -2.5e-7
```

## 5. 跳跃各阶段的 reward 覆盖度对比

| 阶段 | 我们当前 | OmniNet | Atanassov | 建议补充 |
|------|---------|---------|-----------|---------|
| **站立（jump_cmd=0）** | 无 ❌ | 通过状态机 | contact + nominal pose | stand_still |
| **蓄力/起跳** | 无 ❌ | height_tracking | base_pos + maintain_contact | takeoff_vertical_velocity |
| **腾空** | task_max_height ✅ | height_tracking + vel | base_vel + clearance + symmetry | OK |
| **空中姿态** | joint_aerial ✅ | joint_aerial ✅ | nominal_pose | OK |
| **预着陆** | joint_prelanding ✅ | joint_prelanding ✅ | -- | OK |
| **着陆缓冲** | landing_stability ✅ | joint_landing ✅ | landing_pos + landing_orient | OK |
| **成功奖励** | successful_jump ✅ | successful_jump ✅ | landing + height (sparse) | OK |
| **全程正则化** | orientation + collision ✅ | 同 ✅ | energy + contact_force + 更多 | 可选 |
| **对称** | 无 ❌ | 无 | symmetry ✅ | left_right_contact_sync |

站立和起跳阶段完全没有正向 reward 引导——这是当前最大的 gap。
