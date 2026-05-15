# SATA 跳跃训练奖励参考手册

最后更新: 2026-05-15

任务: `go2_omnijump_curriculum_torque`

---

## 重要前提 — RL 怎么"奖励"

**RL 不能直接奖励"力"或"动作"**，只能奖励**状态变量的结果**（位置、速度、接触、姿态等）。

- 想让 robot "用力推地" → 实际是 reward **base 向上速度 vz**（推地的物理结果）
- 想让 robot "稳定落地" → 实际是 reward **base 高度 + 速度 + 姿态在某区间**

Policy 自己学：什么 action 能产生 reward 高的状态。

---

## 1. 正向奖励

| Reward | Weight | 公式 / 触发条件 | 物理意义 |
|---|---|---|---|
| successful_jump | +300 | `last_jump_success × velocity_score`。one-shot, `_finish_jump`（landing buffer 25 步结束）那一帧 fire。Success 条件：peak ≥ 0.30m + 落地 25 步内 base 没翻倒（proj_gravity_xy < 0.7） | **测量 jump cycle 是否成功（binary）**。完成完整 jump cycle（蹲→推→飞→稳定落地）的终极大奖。一个 episode 最多 fire 一次 |
| takeoff_direction | +80 | `vz / ‖v‖`（如果 ‖v‖ < 0.1 fallback 0）。one-shot, `just_took_off`（四脚同时离地的那一帧）fire | **测量起跳瞬间速度向量的"垂直分量比例"**。读取 `root_states[:, 7:10]` 即 base 速度 (vx,vy,vz)。输出范围 [-1,1]：1=纯垂直, 0.71=45°斜跳, 0=纯水平。鼓励起跳方向垂直向上 |
| takeoff_vertical_velocity | +10 | `clamp(vz / 2.5, 0, 1)`。触发：`jumping_state & vz>0 & ~has_landed & base>0.18`。dense | **测量 base 向上速度 vz** (m/s)。读取 `root_states[:, 9]`。线性 cap 在 vz=2.5 m/s 时拿满分。**不直接奖励"力/torque"**，但 policy 学到"输出让 vz 大的 torque" = 主动推地。SATA torque control 的"训练扶手" |
| projected_peak | +7 | `exp(-(base_height + vz²/19.62 - cmd_height)² / 0.025²)`。触发：`jumping_state & has_taken_off & vz>0 & ~has_landed & base>0.18`。dense | **测量"弹道学预测 peak"跟 cmd 高度的偏差**。物理：自由飞行下 h + vz²/2g 守恒 = 未来 peak。钟形奖励——飞行中"预测 peak"越接近 cmd 越高分。`has_taken_off` gate 确保只在真飞行时算（防 stance vz 飙高 gaming）|
| peak_height_progress | +5 | `air_ratio × clamp((peak_base_height - 0.25)/(cmd - 0.25), 0, 1)`。触发：`jumping_state & ~has_landed`。dense | **测量 episode 内 base 创新高的"进度"**。读取 `peak_base_height`（episode 内 airborne 时 base 最高值）。线性从 0.25m → cmd 归一化。`air_ratio`（脚不接触地的比例）让奖励在真飞行时更高 |
| all_feet_airborne | +2 | `airborne × (0.25 + 0.75 × height_progress(current_base))`。触发：`airborne = jumping_state & has_taken_off & ~has_landed & 四脚都不接触地`。dense | **测量"是否真在飞行"**。读取 4 脚接触状态。四脚全离地 + base 高 = 高分。base 高度越接近 cmd 加成越多（0.25-1.0 范围）|
| height_tracking | +1 | `exp(-(base_height - target)² / 0.05²)`。target 按 phase 切换：站立时 0.42m，跳时 cmd_height。dense, 全程激活 | **测量 base 高度跟目标的偏差**。读取 `root_states[:, 2]`。钟形奖励。站立期间 robot 保持 0.42m，跳时 base 接近 cmd 高度 |
| default_hip_pos | +0.3 | `exp(-Σ_{hip∈{0,3,6,9}} \|q_hip - q_hip_default\| × 4.0)`。dense, 全程激活 | **测量 4 个 hip 关节角度跟 default 的偏差**。读取 `dof_pos[:, [0,3,6,9]]`。default 值: hip_FL/RL=0.1, hip_FR/RR=-0.1。鼓励 hip 不外撇 / 不内扣 |
| maintain_contact | +0.10 | `1.0 if 四脚都接触地 else 0.0`。触发：`~airborne`。dense | **测量"四脚是否都着地"**。读取 `contact_forces[:, feet, 2] > 1.0`。非飞行时锚住站立。weight 故意小防 "站着不动" 局部最优 |

---

## 2. 负向惩罚

| Reward | Weight | 公式 / 触发条件 | 物理意义 |
|---|---|---|---|
| termination | -10.0 | `1 if base_contact_force > 0.2 else 0`。摔倒触发 + episode 立即 reset。最多 fire 一次 | **测量 base 是否接触地面（摔倒）**。读取 base 部位 `contact_forces`。摔倒重罚 + 强制 reset。"防摔倒" 主信号 |
| collision | -3.0 | `Σ_{i∈{thigh,calf,hip}} (force_norm_i > 0.1)`。12 个部位累加。dense, 全程激活 | **测量 thigh/calf/hip 部位接触力 > 0.1N 的数量**。读取 `contact_forces[:, [thigh+calf+hip]_indices, :]`。腿撞腿 / 腿撞 base 都罚。每个部位接触算 1 分 |
| orientation | -1.6 | `‖projected_gravity_xy‖²`。dense, 全程激活 | **测量 base 偏离垂直的程度**。读取 `projected_gravity[:, :2]`（重力 [0,0,-1] 在 base 坐标系下 xy 分量）。base 垂直时 xy=0 → reward 0；倾斜越大 xy 越大 → 罚越多 |
| horizontal_drift | -1.5 | `vx² + vy²`。触发：`jumping_state & ~has_landed`。dense | **测量跳跃中 base 水平速度的平方**。读取 `root_states[:, 7:9]` 即 (vx, vy)。cmd 是原地跳 (lin_vel_x/y=0)，所以 vx vy 应该 ≈ 0。罚水平飘移（防后跳/侧跳）|
| default_pos | -0.1 | `Σ_{i=1}^{12} \|q_i - q_squat_i\|`。dense, 全程激活 | **测量 12 关节当前角度跟 q_squat 的 L1 距离**。读取 `dof_pos` + IK 算的 `q_squat_target`。整体偏蹲姿引导。weight 故意小，让 push/flight 短暂偏离 q_squat OK |
| action_rate | -0.03 | `Σ_{j=1}^{12} (a_j(t) - a_j(t-1))²`。dense, 全程激活 | **测量连续两帧 policy 输出 action 差的平方**。**不惩罚"大幅度动作"，只惩罚"快速变化"**。抗高频抖动 |
| dof_acc | -2.5e-7 | `Σ_{j=1}^{12} q̈_j²`。dense, 全程激活 | **测量关节加速度的平方和**。读取 `(dof_vel[t] - dof_vel[t-1]) / dt`。weight 故意小不阻止爆发推地，主要 regularization |
| torques | -1e-5 | `Σ_{j=1}^{12} τ_j²`。dense, 全程激活 | **测量 12 个关节力矩平方和**。读取 `torques`。能量损耗惩罚。weight 极小几乎不影响 |

---

## 3. Disabled rewards（weight=0，仅参考）

| Reward | 原因 |
|---|---|
| tracking_linear_velocity | cmd=0 时奖励"不动" → 创造让 robot 学到"不跳"的局部最优 |
| tracking_angular_velocity | 同上 |
| joint_angle_loaded | sigma=1.5 bug：12 关节 L1 误差 5-7，exp 几乎为 0 |
| joint_angle_extended | 同上 |
| joint_angle_aerial / prelanding / landing | 旧 4-phase 设计，已被 default_pos 取代 |

---

## 4. 关键状态变量

| Flag | 类型 | 设置时机 | 重置时机 |
|---|---|---|---|
| `jumping_state` | bool | cmd[4]>0.5 触发 `_start_jump` | `_finish_jump`（landing buffer 末 or takeoff_timeout）|
| `has_taken_off` | bool (latch) | 四脚同时离地那一帧 | env.reset_idx |
| `has_landed` | bool (latch) | has_taken_off 后任一脚触地 | env.reset_idx |
| `airborne` | bool | `jumping_state & has_taken_off & ~has_landed & 四脚都不接触` | 反向条件 |
| `just_took_off` | bool | has_taken_off 从 False → True 那**一帧** | 下一帧 |
| `just_landed` | bool | has_landed 从 False → True 那**一帧** | 下一帧 |
| `single_jump_command_done` | bool | `_finish_jump` 或 `takeoff_timeout` | env.reset_idx |
| `peak_base_height` | float | airborne 时累积 max(base_height) | env.reset_idx |
| `pending_success` | bool | just_landed 时 set；landing buffer 25 步内 excessive_tilt cancel | env.reset_idx |
| `last_jump_success` | bool | `_finish_jump` 那一帧 = pending_success | 每步开头 reset 为 False |

---

## 5. 物理量速查表

| 索引 | 含义 | 单位 |
|---|---|---|
| `root_states[:, 0:3]` | base 位置 (x, y, z) | m |
| `root_states[:, 3:7]` | base 旋转四元数 | — |
| `root_states[:, 7:10]` | base 线速度 (vx, vy, vz) | m/s |
| `root_states[:, 10:13]` | base 角速度 (ωx, ωy, ωz) | rad/s |
| `dof_pos[:, :12]` | 12 关节角度 | rad |
| `dof_vel[:, :12]` | 12 关节速度 | rad/s |
| `contact_forces[:, foot, 2]` | 脚的接触力 z 分量 | N |
| `projected_gravity` | 重力 [0,0,-1] 在 base 坐标系下投影 | — (单位向量) |
| `q_squat_target` | IK 算的蹲姿 (base ~0.20m) | rad/joint |
| `q_ground_target` | IK 算的站立姿 (base ~0.42m) | rad/joint |

---

## 6. Phase 定义

| Phase | 公式 | PD prior target | 物理意义 |
|---|---|---|---|
| 非 jumping | `~jumping_state` | `default_dof_pos`（站立, base ≈ 0.42m）| 等命令 / post-jump 站立 |
| phase_loaded | `jumping_state & ~has_taken_off & (vz ≤ 0)` | `q_squat`（蹲, base ≈ 0.20m）| stance 蹲下蓄力 |
| phase_extended | `jumping_state & ~phase_loaded` | `q_ground`（站立 IK, base ≈ 0.42m）| 推地 + 飞行 + 落地 |

---

## 7. Jump cycle 各 phase 激活的 reward

| Phase | 激活的正向 reward | 激活的惩罚 |
|---|---|---|
| Idle 等命令 | height_tracking, maintain_contact, default_hip_pos | default_pos (距 q_squat 远), orientation, action_rate, dof_acc, torques |
| Stance 蹲下 (vz≤0) | maintain_contact, default_hip_pos | (default_pos 罚减小) |
| Stance 推地 (vz>0) | **takeoff_vertical_velocity (10)**, peak_height_progress | collision (脚力大), action_rate |
| just_took_off | **takeoff_direction (80, one-shot)** | — |
| Flight 上升 (vz>0) | takeoff_vertical_velocity, **projected_peak (7)**, peak_height_progress, all_feet_airborne | horizontal_drift, collision |
| Flight 下降 (vz<0) | peak_height_progress, all_feet_airborne | horizontal_drift |
| just_landed | — | — (但 pending_success 触发判定) |
| Landing buffer (25 步) | — | excessive_tilt cancel pending |
| _finish_jump | **successful_jump (300, one-shot if success)** | — |
| Post-jump stand | height_tracking, maintain_contact, default_hip_pos | default_pos, orientation, action_rate |

---

## 8. 权重总览（按值排序）

| Reward | Weight | 类型 |
|---|---|---|
| successful_jump | +300 | sparse one-shot |
| takeoff_direction | +80 | one-shot |
| takeoff_vertical_velocity | +10 | dense |
| projected_peak | +7 | dense |
| peak_height_progress | +5 | dense |
| all_feet_airborne | +2 | dense |
| height_tracking | +1 | dense |
| default_hip_pos | +0.3 | dense |
| maintain_contact | +0.10 | dense |
| termination | -10.0 | sparse one-shot |
| collision | -3.0 | dense |
| orientation | -1.6 | dense |
| horizontal_drift | -1.5 | dense |
| default_pos | -0.1 | dense |
| action_rate | -0.03 | dense |
| dof_acc | -2.5e-7 | dense |
| torques | -1e-5 | dense |

---

## 9. PD prior fade schedule

| 阶段 | iter 范围 | PD weight | RL action_scale | 控制频率 |
|---|---|---|---|---|
| Warmup | 0 - 1000 | 0.50 | 10 | 100 Hz |
| Linear ramp | 1000 - 3000+ | 0.50 → 0 (linear) | 10 → 60 (linear) | 100 → 200 Hz (linear) |
| Pure RL | 3000+ | 0 | 60 | 200 Hz |

公式：`pd_alpha = pd_prior_weight × (1 - general_scale)`
其中 `general_scale = clamp((step_count - warmup_steps) / (x0 - warmup_steps), 0, 1)`
配置：`warmup_steps = 96000`, `x0 = 288000`

---

## 10. 命令空间

| Index | 名称 | 范围 | 含义 |
|---|---|---|---|
| cmd[0] | lin_vel_x | [0.0, 0.0] | 横向速度 x（curriculum 锁 0）|
| cmd[1] | lin_vel_y | [0.0, 0.0] | 横向速度 y |
| cmd[2] | ang_vel_yaw | [0.0, 0.0] | 偏航角速度 |
| cmd[3] | jump_height | [0.40, 0.70] | 该跳多高（m）|
| cmd[4] | jump_command | [0.0, 1.0] | >0.5 触发 jumping_state |

`single_jump_command_prob = 1.0` — 每次 resample 100% single jump 模式
`one_jump_reward_per_episode = True` — 一个 episode 只跳一次

---

## 11. 参考文献

| 论文 / 项目 | 贡献 |
|---|---|
| OmniNet (Lee et al. 2024) | Table I 基础 reward (height_tracking, successful_jump, orientation, collision, torques, action_rate, dof_acc, all_feet_airborne) |
| Olsen 2025 | projected_peak 弹道学预测思想 |
| Atanassov 2025 | phase-aware rewards, RSI, maintain_contact |
| Soni 2023 | takeoff_vertical_velocity 思想 |
| SATA (Li 2025) | 力矩控制框架 + PD 先验 + Gompertz/linear growth schedule |
| mygo2jump (内部) | default_pos / default_hip_pos 姿态引导设计 |
