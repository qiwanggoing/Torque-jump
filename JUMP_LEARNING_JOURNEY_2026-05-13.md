# 跳跃训练心路历程：从 OmniNet 路线到 Atanassov 路线的演进

**记录时间**: 2026-05-12 至 2026-05-13
**主环境**: `go2_omnijump_curriculum_torque`
**目标**: Go2 用 SATA 力矩控制学全方向跳跃

---

## 1. 起点：失败的 OmniNet 风格 run

**run**: May12 `auto_curriculum`，跑到 3221 迭代彻底卡住

**症状**: 机器人摆出"瑜伽姿势"——站得很高 + 抬起一条腿 + 不动。flight_rate = 0。

**诊断**:
- `stable_stand` 状态机要求"四脚着地 + 身体水平 + 倾角 < 5.7°"持续 5 步才能触发 `jumping_state`
- 域随机化 `shifted_com_range_x = [-0.2, 0.2]m` 导致机器人天生倾斜超过 5.7°
- → stable_stand 永远不满足 → jumping_state 永远不触发 → 跳跃奖励永远不发
- 机器人只能找局部最优：站高拿 height_tracking + 抬腿调 CoM + 不动避免 action_rate 惩罚

---

## 2. 关键架构改动顺序

### Step 1: 缩小 CoM 偏移

`shifted_com_range_x`: `[-0.2, 0.2]` → `[-0.05, 0.05]`

让 stable_stand 物理上可达。但仍然没有奖励引导机器人去满足它。

### Step 2: 去掉 stable_stand 前置门槛

```python
# go2_omnijump_torque.py:378
rearm_ready = first_jump_ready  # 原: stand_step_counter >= stand_rearm_steps
```

让 `jumping_state` 在 episode 跑满 55 步后自动激活，不再需要先稳定站立 5 步。

**理由**: stable_stand 包含 `all_feet_contact`，但没有任何奖励引导机器人去满足"四脚同时着地"。靠不可达的状态机门槛卡死整个流程不合理。

### Step 3: 加 RSI（Reference State Initialization）

Atanassov 2025 论文核心 bootstrap 技术：

```python
def _reset_root_states(self, env_ids):
    # 40% episode 初始化时给 1.0-3.0 m/s 向上速度 + 0-0.3m 随机初始高度
    ...
```

**目标**: 让机器人在还不会跳的时候，先尝过"飞起来 = 高奖励"的甜头。

### Step 4: RSI 必须同时激活 jumping_state

**重要 bug 发现**: 最初的 RSI 实现只给了向上速度，但 `jumping_state` 仍然在 step 55 才触发。RSI 把机器人甩到空中时 `jumping_state = False`，所有飞行奖励（`peak_height_progress`、`all_feet_airborne`、`joint_angle_aerial`）都不发——**RSI 完全空炮**。

修法: 在 `_reset_root_states` 用 `rsi_episode_mask` 记录哪些 envs 拿到 RSI，然后在 `_reset_jump_buffers` 之后强制把这些 envs 的 `jumping_state` 设为 True。

```python
def reset_idx(self, env_ids):
    ...
    self._reset_jump_buffers(env_ids)
    rsi_envs = env_ids[self.rsi_episode_mask[env_ids]]
    if len(rsi_envs) > 0:
        self.jumping_state[rsi_envs] = True
```

### Step 5: Phase-aware `height_tracking`（核心）

原始 OmniNet 风格: height_tracking 全程目标 = `commands[:, 3]`（跳跃高度）。

Atanassov 风格: 每个 phase 用不同的 base position 目标：
- **Stance** (蓄力期): 目标 0.20m（蹲下）
- **Flight** (腾空): 目标 = 跳跃命令 0.40-0.70m
- **Landing/post-landing**: 回到默认站立

最终演进到**只有两个目标**——蹲下 (0.20m) 或飞行目标。**取消"站立 0.42m"目标**，让蹲下成为机器人的默认静止状态。

```python
def _reward_height_tracking(self):
    squat_height = 0.20
    target = torch.full_like(self.root_states[:, 2], squat_height)
    flight_mask = self.has_taken_off & (~self.has_landed)
    target = torch.where(flight_mask, self.commands[:, 3], target)
    ...
```

### Step 6: 去掉 velocity tracking

`vel_cmd = 0` + `tracking_linear_velocity` 创造"原地不动也满分"的局部最优，机器人不愿动。

scales 中设 `tracking_linear_velocity = 0.0`、`tracking_angular_velocity = 0.0`，让基类自动过滤掉。

### Step 7: 加分阶段的辅助奖励

引入 phase-aware 的飞行奖励（whitelist 加入）：

- `peak_height_progress`: jumping_state & ~has_landed 时，奖励向跳跃峰值高度推进
- `all_feet_airborne`: airborne 阶段奖励四脚离地
- `joint_angle_aerial/prelanding/landing`: 各阶段惩罚关节偏离对应目标姿势

### Step 8: 加 pushoff 子阶段 + Olsen 投影奖励

**问题**: 即使有 squat 目标和 RSI，机器人还是只在 RSI episode 中飞。自主起跳率几乎为 0。原因——从"蹲下"到"腾空"的中间过渡**没有奖励信号**。

**修法**: 引入 `pushoff_phase` 子阶段（base ≤ pushoff threshold）：

```python
pushoff_phase = jumping_state & (~has_taken_off) & (base_height <= 0.32)
```

在这个子阶段开启两个奖励：

1. **`takeoff_vertical_velocity`**: 奖励向上速度
   ```python
   reward = pushoff_phase * clamp(vz / takeoff_velocity_target, 0, 1)
   ```

2. **`projected_peak`** (Olsen 2025 启发): 用抛体公式预测机器人能跳多高，奖励接近目标
   ```python
   projected = base_height + max(0, vz)² / (2g)
   reward = pushoff_phase * exp(-(projected - target)² / sigma)
   ```

### Step 9: 修复 maintain_contact 奖励悬崖

**Bug**: 之前在 `pushoff_phase` 关掉了 `maintain_contact`（"防止鼓励机器人不起跳"）。结果：

```
浅蹲 0.35m (非 pushoff): maintain_contact 0.3 + height_track 0.64 = 0.94
碰到门槛 0.32m (进入 pushoff): maintain_contact 0   + height_track 0.75 = 0.75
                                                                      ↑
                                                              悬崖 -0.19/步
```

机器人学会**主动避免进入 pushoff phase**，停在 0.33-0.35m。

修法: 去掉 pushoff 排除条件，maintain_contact 只要 `~airborne` 就给奖励（脚离地时它自然为 0）。

```python
def _reward_maintain_contact(self):
    return (~self.airborne).float() * all_feet.float()
```

### Step 10: 加速 PD 先验衰减

SATA 框架特有: `pd_alpha = 0.5 * (1 - general_scale)`，Gompertz 曲线 `general_scale = exp(-exp(-k * (step - x0)))`。

PD 先验把关节拉向默认站立姿势，限制机器人深蹲和大力蹬地。

- 原配置 `x0 = 120000` → inflection 在 iter 2500，整个 5000 迭代训练里 PD 主导
- 改成 `x0 = 60000` → inflection 在 iter 1250，让 RL 早一些拿到完整控制权

### Step 11: 放宽 successful_jump 容忍度

`success_height_tolerance`: 0.04m → **0.10m**

原来要求峰值高度误差 ≤ 4cm，几乎没法满足。放宽到 10cm 让成功率从 < 1% 上升到 4-6%。

### Step 12: 起跳权重大幅提升（最新改动 2026-05-13）

**分析**: 起跳是短促事件（5-10 步），蹲着可以持续 300+ 步。要让起跳真正划算，单步起跳奖励必须远高于单步蹲着奖励。

最终权重:

| 奖励 | 权重 |
|------|------|
| `height_tracking` | 1.0 |
| `maintain_contact` | **0.15**（降低 baseline 诱惑） |
| `takeoff_vertical_velocity` | **4.0**（boost） |
| `projected_peak` | **5.0**（boost，dominant 信号） |
| `peak_height_progress` | 1.0 |
| `all_feet_airborne` | 1.0 |
| `successful_jump` | 20.0 |
| `joint_angle_aerial` | -0.4 |
| `joint_angle_prelanding` | -0.6 |
| `joint_angle_landing` | -0.12 |
| `orientation` | -0.8 |
| `collision` | -1.0 |
| `torques` | -1e-5 |
| `action_rate` | -0.01 |
| `dof_acc` | -2.5e-7 |
| `termination` | -10.0 |

---

## 3. 重要踩坑记录

### 坑 1: 跟随 Atanassov 路线"纯粹化"

曾经为了"纯按论文"删掉 `takeoff_impulse` 和 `takeoff_vertical_velocity`，结果完全学不出来。

**教训**: Atanassov 用 **PD 位置控制**，蹬地动作是一个协调的关节角伸展，相对容易学。我们用 **力矩控制**，每个关节力矩独立，协调蹬地动作探索难度指数级提高，需要显式的"蹬地"奖励信号作为训练扶手。

### 坑 2: 单独"先学站立"阶段

曾经设 `single_jump_command_prob = 0.0` 想先训练站立稳定，再加跳跃。结果机器人收敛到 noise_std = 0.08 的"站着不动"策略，恢复跳跃命令后也学不动。

**教训**: 论文路线（Atanassov）不分"先站立后跳跃"，而是用 RSI 直接 bootstrap 跳跃状态。我们也应该一开始就开跳跃命令 + RSI。

### 坑 3: maintain_contact 局部最优

`maintain_contact = 0.3`（高权重）+ `tracking_linear_velocity` 还在的时候，机器人发现"站着不动 = 持续拿满分"，noise_std 急速塌缩。

**教训**: 任何"持续给分"的稳定状态奖励，都可能成为局部最优。要么权重很低，要么相对于跳跃奖励显著小。

### 坑 4: RSI 概率过高

曾经 `rsi_prob = 0.4`，机器人发现"等 RSI 就有奖励"，自主跳跃率反而下降。

**教训**: RSI 是 bootstrap，目的是给机器人**初次**体验飞行奖励，让它知道"飞起来 = 好"。一旦机器人学到这个关联，就应该自己去复现。RSI 概率太高反而打乱学习。

最终设 `rsi_prob = 0.2`。

### 坑 5: 奖励悬崖

任何 phase 切换处的奖励不连续都会被策略主动避开。例如 maintain_contact 在 pushoff_phase 关掉，机器人学会停在 pushoff 门槛之外。

**教训**: 设计 phase-aware 奖励时，要检查 phase 边界的奖励是否连续。如果某个 phase 失去某些奖励，必须有等量或更多的新奖励补上。

### 坑 6: Gompertz 时间表

`x0` 决定 PD 先验衰减节奏。原 120000 太慢（整个训练 PD 都在），原 12000 太快（机器人还没学会稳定就脱手）。最终 60000 是折中。

**教训**: 跳跃训练需要 RL 在中期就拿到大部分控制权（iter 1000-2500），所以 PD 衰减节奏要匹配训练长度。

---

## 4. 当前奖励引导链（2026-05-13 最新）

```
重置 → 机器人在默认站立 (~0.32m)
   ↓
height_tracking 目标 0.20m 持续拉低
maintain_contact 给 0.15 (脚着地)
   ↓
蹲下到 ≤ 0.32m (pushoff threshold)
   ↓ pushoff_phase 激活
   ↓
轻轻往上推 (vz > 0)
   ↓ takeoff_vertical_velocity (×4) 立刻发
   ↓ projected_peak (×5) 用抛体公式预测能跳多高
   ↓ 强信号告诉机器人"这个动作非常值钱"
   ↓
真正腾空 → has_taken_off
   ↓
飞行中:
   ↓ height_tracking 目标切换到跳跃命令 (0.40-0.70m)
   ↓ peak_height_progress / all_feet_airborne 持续奖励
   ↓ joint_angle_aerial 引导收腿姿势
   ↓
落地:
   ↓ joint_angle_landing 引导落地姿势
   ↓ successful_jump 20.0 大奖励 (容忍 10cm)
   ↓
回到蹲 (height_tracking 重新拉到 0.20m)
```

---

## 5. 已尝试但放弃的方向

- **加入 `task_max_height` sparse 奖励**: 与 `height_tracking` 功能重叠，对我们的密集梯度系统是冗余的
- **加 stable_stand 正奖励**: 仍然不解决 all_feet_contact 没有信号引导的问题，反而创造新局部最优
- **完全去掉 stable_stand**: 后来发现没必要完全删除，只需放宽触发条件即可
- **降低 maintain_contact 到 0**: 会失去对"四脚着地"的引导，机器人可能用奇怪姿势站立

---

## 6. 当前训练状态（iter ~171 of 5000）

- noise_std: 0.27 (健康)
- flight_rate: 22% (基本全是 RSI 20% + 极少量自主)
- successful_jump_rate: 3.5%
- mean_peak_height: 0.167m
- **takeoff_vertical_velocity: 0.0001** (起跳信号几乎为 0)
- **projected_peak: 0.0019**

刚改完起跳奖励权重 boost（1.5→4.0, 2.0→5.0），需要重启训练观察。

---

## 7. 待办与下一步

1. **重启训练验证新权重**: 重点观察 iter 500-1500 的 `takeoff_vertical_velocity` 是否上涨
2. **iter 1000-1500 关键节点**: PD 开始衰减，看是否出现自主跳跃
3. **如果还学不到**: 考虑
   - 进一步降低 `start_torque_scale` 或 PD 起始权重
   - 加 explicit "leg extension" 奖励（关节伸展速度）
   - 检查动作空间是否合理（torque 范围是否够）
4. **如果学到了**: 收紧 success_height_tolerance，逐步加入方向命令（lin_vel_x/y）

---

## 8. 关键文件路径

- 主环境: `legged_gym/envs/go2/go2_omnijump_torque/go2_omnijump_torque.py`
- 基类配置: `legged_gym/envs/go2/go2_omnijump_torque/go2_omnijump_torque_config.py`
- Curriculum 环境: `legged_gym/envs/go2/go2_omnijump_curriculum_torque/go2_omnijump_curriculum_torque.py`
- Curriculum 配置: `legged_gym/envs/go2/go2_omnijump_curriculum_torque/go2_omnijump_curriculum_torque_config.py`

## 9. 2026-05-13/14 续：Two-phase 姿态 + 简化 successful_jump + 升级到 200 权重

### 关键迭代节点

| 节点 | 状态 |
|---|---|
| iter 4999 (5000 iter run 终) | peak 0.448m, success 83%, flight 87% — 但 dof_acc -0.118、orientation -0.018、action_rate -0.040 都非常糟，"暴力换成功" |
| **重训 + successful_jump 100→200 + orientation -0.8→-1.6 + dof_acc 4× + action_rate 2.5× + collision 3× + joint_angle_extended 2× + 新 horizontal_drift -1.5** | |
| iter 1311 (新 run) | **peak 0.402m, success 88.3%, flight 89.2%** — noise_std 真收敛 0.26，action_rate -0.013（旧 run 同期 -0.04），完美质量提升 |
| iter 2481 | 退化到 success 76%, flight 77%, peak 0.37m — 我误判为崩盘 |
| **iter 3595** ✨ | **peak 0.463m (历史新高), success 92.2%, flight 93.4%, mean_reward +1.29** — 全面新高 |

### 关键发现

1. **iter 2481 的 dip 不是崩盘**——再训 1100 iter 全面恢复并新高。下次遇到 dip 至少等 1500 iter。
2. **successful_jump = 200 起决定性作用**——把稀疏奖励权重拉到能压过密集惩罚预算。raw 单次贡献 4.0，足以让 PPO 把"完成完整跳跃"当主目标。
3. **horizontal_drift -1.5 修复了后跳**——`(jumping_state & ~has_landed) × (vx² + vy²)` 直接抑制水平速度，cmd 是垂直跳就垂直跳。
4. **重训比续训干净**——iter 4999 的 policy 已经"暴力定型"，续训改不动；从 0 重训反而 1311 iter 就超越。

### 当前 run 的暴露问题（iter 3595 still 有的）

- `dof_acc -0.117`（占负向预算 46%）：policy 在花大量精力压"加速度"，但用户说**不在乎剧烈，只在乎抖动**
- `joint_angle_extended -0.083`（占 36%）：飞行姿态偏离 q_ground 还是有
- 视觉上：腿在空中有摆动（撞腿）、抽动、后跳

### 用户对奖励占比看不直观的反思

讨论了 share 日志方案。决定先不做日志，直接改 reward 设计：
- **dof_acc 惩罚被滥用了**——它惩罚"快但平滑的动作"和"抖动"双重，policy 为了降它直接放弃爆发力
- **action_rate 才是真正的"抖动"度量**（policy 输出步间差）
- **姿态引导用奖励比惩罚更好**——奖励 advantage 估计稳定（[0, +w]），惩罚是无下界（关节越歪罚越多）

### 当前最新改动（开始第 2 次重训前）

```python
# go2_omnijump_torque.py
def _reward_joint_angle_loaded(self):
    active = self.phase_loaded.float()
    pose_error = torch.sum(torch.abs(self.dof_pos - self.q_squat_target.unsqueeze(0)), dim=1)
    sigma = max(float(getattr(self.cfg.rewards, "pose_guidance_sigma", 1.5)), 1e-3)
    return active * torch.exp(-pose_error / sigma)  # POSITIVE bell curve

def _reward_joint_angle_extended(self):
    active = self.phase_extended.float()
    pose_error = torch.sum(torch.abs(self.dof_pos - self.q_ground_target.unsqueeze(0)), dim=1)
    sigma = max(float(getattr(self.cfg.rewards, "pose_guidance_sigma", 1.5)), 1e-3)
    return active * torch.exp(-pose_error / sigma)  # POSITIVE bell curve
```

权重变化：
- `joint_angle_loaded: -0.4 → +0.4`（正向）
- `joint_angle_extended: -0.8 → +0.8`（正向）
- `dof_acc: -1e-6 → -2.5e-7`（回到原始，释放爆发力预算）
- `action_rate: -0.025 → -0.08`（3.2× 加重，直击抖动）
- 新增配置 `pose_guidance_sigma = 1.5`

### iter 714（重训 2 次的早期）发现 sigma 配错了

- `joint_angle_extended` 加权 +0.0006（**几乎为 0**）
- 反推 raw exp 输出 ≈ 0.012 → pose_error ≈ 6.6 rad → 12 关节每个偏 0.55 rad（30°）
- **sigma=1.5 太尖**，远离 target 时梯度为 0（验证了之前担心）
- `action_rate -0.0498` 在 early stage 太猛，policy 还没学会跳就被锁死探索

**待修复**（明天接着做）：
- `pose_guidance_sigma: 1.5 → 5.0`（让 exp 钟形更宽，error 5 时仍有 0.37 信号）
- `action_rate: -0.08 → -0.04`（让 early policy 有探索空间）
- 重启训练

---

## 10. 参考论文

- **Atanassov 2025** "Curriculum-Based RL for Quadrupedal Jumping: A Reference-Free Design"
  - phase-aware rewards, RSI, multi-stage curriculum
  - PD 位置控制（与我们的力矩控制不同！）
- **Olsen 2025** "Towards Quadrupedal Jumping and Walking..."
  - projectile-motion reward densification (`projected_peak` 思想来源)
- **Soni 2023** "End-to-End RL for Torque Based Variable Height Hopping"
  - 力矩控制 + 短历史隐式 phase 推断
  - energy gain reward 思想
- **SATA Li 2025**: 力矩控制框架来源，PD 先验 + Gompertz 衰减
