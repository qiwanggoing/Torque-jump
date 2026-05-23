# Phase-Aware Jump Design — Notes

最后更新: 2026-05-23

记录围绕「按 jump 阶段 (phase) 设计 reward + obs」的思路演变和参考。

---

## 背景：为什么要分 phase

跳跃是 event-driven 的多阶段动作。理想 cycle：
```
stand → squat (loaded) → push-off → flight (ascent) → peak → prelanding (extend legs) → touchdown → landing → stand
```

每个 phase 想做的事不一样：
- **squat**：腿屈伸蓄力
- **push-off**：腿伸直发力
- **flight ascent**：维持垂直、收腿减小惯量
- **prelanding**：伸腿准备接触
- **landing**：吸收冲击
- **stand**：稳住

如果只用一个 reward 函数覆盖全部，policy 学不到「不同阶段做不同事」。所以参考论文都用 **phase-aware reward**。

---

## 已实现：状态驱动的 phase 检测

`go2_omnijump_torque.py` 里已经实现了一套完整的 phase 检测（`_update_jump_state`）：

| Phase flag | 触发条件 | 含义 |
|--|--|--|
| `phase_loaded` | `jumping_state & ~has_taken_off & vz ≤ 0` | 蹲下蓄力 |
| `phase_extended` | `jumping_state & ~phase_loaded` | 起跳后任何阶段 |
| `airborne` | `jumping_state & has_taken_off & ~has_landed & ~any_foot_contact` | 飞行 |
| `prelanding` | `airborne & descending & root_z ≤ prelanding_height` | 接近地面前的下降段 |
| `landing` | `jumping_state & has_landed` | 落地后 |

`prelanding_height = max(peak - margin, base_target + 0.04)` —— 用最高点向下减一段，能在 robot 真正接触前提前触发。

并且 env 里已经实现了对应的 IK pose reward：
- `_reward_joint_angle_aerial` 跟踪 `q_air_target`（腿收）
- `_reward_joint_angle_prelanding` 跟踪 `q_pre_target`（腿伸）
- `_reward_joint_angle_landing` 跟踪 `q_ground_target`（站立）

q_air_target / q_pre_target / q_ground_target 都用 analytical IK（`_solve_pose_from_foot_height`）解出来。

**当前 curriculum config 里这三个 reward 都 weight=0**，理由是当年被 `joint_angle_extended`（单 phase 通用 reward）替代。需要重新启用以解决落地不稳问题。

---

## 候选改进 1：启用 OmniNet 风格 phase pose reward（短期）

参照 OmniNet Table I 权重：
```python
joint_angle_aerial = -0.4       # 腿收（OmniNet）
joint_angle_prelanding = -0.6   # 腿伸（OmniNet）
joint_angle_landing = -0.12     # 站立（OmniNet）
joint_angle_extended = 0.0      # 关掉旧的
```

效果：policy 在不同阶段被引导往不同 pose target 走。**直接解决「落地腿不展开 → 硬冲击」问题。**

这是 **最近一次讨论中提议的下一步**。

---

## 候选改进 2：mygo2jump 的 sin/cos phase obs（长期、待评估）

mygo2jump 在 `compute_observations`：
```python
phase = episode_length_buf * dt / cycle_time   # 线性 0→1→2
sin_pos = sin(2π · phase)
cos_pos = cos(2π · phase)
obs ← [sin_pos, cos_pos, commands, ...]
```

policy 通过 sin/cos 在 obs 里**显式**知道当前在周期哪个位置。

**为什么适合 walking（mygo2jump 原本场景）**：步态 = 固定周期，时间驱动天然对应。

**直接套到 jumping 的问题**：
- 跳跃是 event-driven，不是周期性。cmd 来了才跳，跳完站着等
- 跳跃时长随 jump_height / cmd 变（飞 0.3s vs 0.6s）
- 时间驱动的"phase 0.5 是空中"假设可能 robot 还没起飞

**借鉴的可能做法**：
1. **从 cmd[4]=1 那一刻起**，启动一个内部 timer，不是从 episode 开始
2. timer 的 sin/cos 加入 obs
3. 给 policy 一个「预期 cycle 总时长」（基于 cmd[3] 计算飞行时长 + 蓄力时长）
4. policy 用 timer 知道"还有多久要落地"，提前做 prelanding 准备

**或者更简单**：直接把现有 `phase_loaded / airborne / prelanding / landing` 这些 bool 标志加进 obs。policy 显式知道当前 phase，决策更明确。

---

## 候选改进 3：Olsen-style 投影预测（已部分实现）

Olsen 2025 用抛物线方程 `ĥ = h + vz²/(2g)` 在飞行中估算 peak，给 dense reward。

我们已经在 `_reward_projected_peak` 里实现了这个（飞行中用 ĥ vs cmd[3] 跟踪），加上最近改的 Olsen φ + 3ψ 混合 reward shape。

可以扩展：用 projectile 同样**预测落地时刻**，提前触发 prelanding。但相比简单的 `vz < 0 & 接近 prelanding_height` 状态检测，复杂度上升较多，收益不确定。

---

## 决策记录

- **2026-05-23**: 用户提出「能不能借鉴 mygo2jump 的 sin/cos 周期」。讨论后认为：
  - mygo2jump 的固定周期不适合 event-driven 的 jumping
  - 但 phase obs 显式化的思想可借鉴 → 短期内先用状态 bool 加 obs（简单）
  - 长期可考虑「cmd 触发内部 timer」方案
  - **立即可做**：启用现有 `_reward_joint_angle_aerial / prelanding / landing`（候选 1），不需要新代码
