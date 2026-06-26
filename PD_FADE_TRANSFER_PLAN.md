# Control Responsibility Transfer (PD → Policy) — Plan & Rationale

Date started: 2026-06-26
Owner: go46vop / Claude
Scope: `go2_omnijump_landing_torque` (力矩控制 Go2 落点跳跃)

---

## 0. 为什么有这份文档

我们一直把问题当成「如何平滑地去掉 PD」。**真正的约束其实是「PD 必须在策略收敛之前退出」**：
如果等策略完全定型再撤 PD，策略已经收敛在一个「PD 存在」的局部最优附近，PPO 是局部的，几乎不可能再爬出来学纯力矩跳跃的技巧。

所以整个训练不是 *PD fadeout*，而是 **Control Responsibility Transfer（控制责任转移）**：

```
Teacher(PD)  50% → 30% → 10% → 0%
Policy       50% → 70% → 90% → 100%
关键问题 = 这个转移过程中 Jump Skill 不能丢。
```

阶段：
- **Stage 1**：PD 辅助，建立基本可行策略（discovery）。
- **Stage 2（关键）**：PD 退出，策略重新探索。**不能太晚**，否则锁死在 PD-shaped basin。
- **Stage 3**：纯力矩优化性能。

---

## 1. 一个必须记住的修正：这条线的 PD 不是「压制器」，是 motion prior

`_compute_torques`（`go2_omnijump_curriculum_torque.py`）里 PD =

```
PD_full = p_gains · (q_phase_target − q) − d_gains · q̇
```

`q_phase_target` 是**相位目标**（蓄力→q_squat、飞行→q_air、落地→q_ground）。
- **位置项**其实在**引导**机器人做相位动作（q_squat 就是深蹲），不是压制激进姿态。
- **真正压制「激进技巧」的是 `d_gains·q̇` 速度阻尼项** → 它压高关节速度 = 压爆发/弹道动作。

**含义**：撤 PD 真正解放的是**爆发速度**，不是整个姿态幅度。设计补偿时盯这一点。

---

## 2. 当前的力矩混合机制（基线，改之前的样子）

`go2_omnijump_curriculum_torque.py::_compute_torques`：

```python
pd_alpha = pd_prior_weight(0.5) · (1 − general_scale)     # 0.5 → 0 当 gscale 0→1
rl_alpha = 1 − pd_alpha
residual_torques        = actions[:, :12] · torque_limits           # 策略直接命令力矩
residual_torques_action = residual_torques · rl_alpha · torque_limit_scale   # ← 策略被 (1−pd_alpha) 压权
pd_prior_torques        = PD_full · pd_alpha
torques_action          = residual_torques_action + pd_prior_torques
# → total = residual·(1−pd_alpha) + PD_full·pd_alpha   （凸混合，老师-学生插值）
```

`general_scale` 由 `_update_growth_scale` 按 `step_count` 在 `[warmup_steps, x0]` 线性 0→1。
- 当前（慢）：`warmup_steps=19200(~iter200)`，`x0=115200(~iter1200)`。step≈69/iter。
- obs 里**已经有** `pd_prior_alpha`（fade 状态）和 `self.torques`（总力矩）——策略已知 fade 进度。

**已知毛病**：
1. **策略被压权**：pd_alpha=0.5 时策略输出被砍半 → PD 强时策略学得慢、权威小 → α 掉了才被迫猛追。
2. **dx 虚高**：慢淡出让 dx 课程在 PD 还开着(gscale~0.45)就升到 1.0 → 那个 reach 是 PD 撑的，不是纯力矩真实够程。

---

## 3. 当前已就位的前置修复（让下面的实验安全）

- **clean_takeoff 完全解耦**（`clean_takeoff_terminate=False`，检测器+没收已注释）→ iter714 那个 successful_jump 崩盘根治。验证 run `Jun26_01-38-47`：success 全程 0.77–0.93，dx 升到 1.0，不崩。
- **stand_no_takeoff 恢复**（-5.0，cmd4=0 腾空狠罚，gate 在 succ_ema≥0.80 latch 后）→ 站立修复，验证 run 里 `rew_stand_no_takeoff` 先咬(-0.147)后缩(-0.01)=策略学会站。
- **active-squat（stance_squat 主动蹲）**：治好了「PD→0 悬崖砸碎预载深蹲」的老崩因。

→ 这两个老崩因（没收 + 深蹲被砸）都已拆除，**所以现在重新试快淡出是安全的**。

---

## 4. 方法与步骤（按代价从低到高 / 按执行顺序）

### Step 0（必做，先于一切工程）：测快淡出
**做什么**：单变量，仅把 fade 调快：
- `warmup_steps 19200 → 7000`（~iter100 满 PD）
- `x0 115200 → 35000`（gscale=1 在 ~iter507，**iter800 前过渡完**）
（= 历史健康 run `Jun21_23-19-40` 的值。）

**为什么**：当前栈（解耦 + active-squat）的慢淡出已不崩。有没有「快淡出 cliff」根本还没验证。**别在不确定有没有问题前就造机器。** 顺便：快淡出让 dx 课程从纯力矩起步 → 拿到**诚实**的纯力矩 dx。

**盯什么（fresh run）**：
- PD→0 那段（gscale 在 iter~500 到 1）`squat_qualified_rate` / `successful_jump_rate` 崩不崩。
- 不崩 → 直接拿到诚实 dx，省掉全部补偿工程。
- 崩 → 才需要下面的补偿（Step L / Step H）。
- 对照基线 = 慢淡出解耦 run（`Jun26_01-38-47`）。

### Step L（轻量，一行）：策略停止被压权
**做什么**：`residual_torques_action = residual · rl_alpha · scale` → 去掉 `rl_alpha`：
`total = residual + pd_alpha · PD_full`（PD 当**叠加**脚手架，而非凸混合）。

**为什么**：直接兑现核心洞察——**策略全程满权威学跳**，不被 fade 拖；α→0 时 `total→residual` 平滑接管。这吃掉「策略被 fade 压制」这个最大结构问题。**没有第二个头、不动 rsl_rl。**

**代价/盯什么**：α 高时可能过驱动（`τ = residual + 0.5·PD`，Hill 曲线会 clamp）；策略可能赖 PD 叠加帮助而早期欠练（但有满权威=有梯度）。

### Step H（重型，若 Step 0/L 仍不够）：aux head + 稳定头 BC
**做什么**：策略出**双头**：
```
τ = τ_jump + (1−α)·τ_comp + α·τ_PD
```
- `τ_jump`：任务头，**永远满权**，自由探索跳跃技巧。
- `τ_comp`：稳定头，权重 `(1−α)`，对它做 BC：`L_bc = ‖τ_comp − PD_full‖²`。

**为什么**：
- aux head 光有结构**不可识别**（同一总和无穷多种拆法，架构不强制分工）→ **必须给 τ_comp 一个 target，最自然就是 PD_full（=对稳定头做 BC）**。
- 那时 `(1−α)·τ_comp + α·PD ≈ PD` 全程成立 = **无缝交接**；`τ_jump` 在上面自由探索。
- 价值 = **把「被 PD 锚住的稳定能力」和「自由探索的任务能力」在架构上拆开**。学的是「PD 承担的**职责**」而非「PD 的输出信号」。
- **注意**：这不是「比 BC 更好」，而是「**结构化的 BC**」——BC 是它的一部分。

**工程量**：改网络输出维度（12→24）+ `_compute_torques` 的混合 + rsl_rl PPO update 加 BC loss + env 每步存 `PD_full` target。最大。

---

## 5. 明确否决的想法

**`L = ‖τ_total(t) − τ_total(t−1)‖²`（总力矩时序连续性）— 否决**：
1. 跳跃本身=力矩剧烈突变（爆发蹬伸），这个 loss **直接罚掉你要的爆发**（dof_acc 当初就是被调小来解放爆发的）。
2. fade 的不连续是**跨训练迭代**的（α 每 iter 才变一点），不是 episode 内相邻步。这个 loss 是 episode 内平滑，**根本没碰到 fade 的悬崖**。概念错位。

---

## 6. 可用杠杆只有三个（设计纪律）

"capability transfer / 学职责不学信号" 是好的设计语言，但落地时**只有三种杠杆**：
① match 输出（BC）② reward 结果 ③ 架构先验。
aux-head = 架构先验 + (隐含) BC。别幻想有第四种"直接监督职责"的魔法。

---

## 7. 执行日志

- 2026-06-26：写下本计划。开始 **Step 0**（快淡出 `warmup 7000 / x0 35000`），单变量对照解耦慢淡出 run。
  - 改动文件：`go2_omnijump_landing_torque_config.py`（growth.warmup_steps / x0）。
  - 待 fresh 重训，盯 gscale→1 段（~iter500）squatQ/succ 崩不崩、纯力矩 dx 落点。
