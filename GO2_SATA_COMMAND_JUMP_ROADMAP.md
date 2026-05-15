# GO2 SATA Command Jump Roadmap

Date: 2026-05-10

这份记录用于固定我们下一阶段的跳跃训练方向。

核心目标：

```text
zero command 稳定站立
-> command 触发一次跳跃
-> 在 SATA 力矩控制框架下跳得更远
-> 落地后恢复站稳
```

我们不是单纯做 height tracking。高度只作为起跳、腾空、clearance 和安全落地的辅助信号。

## 1. 固定设计

必须保持：

- 使用 SATA torque-control 框架。
- 使用 `PD prior + residual torque`。
- SATA 的 torque-limit / frequency growth 思想要适配到我们的 residual torque 分支上。
- 最终训练和部署频率要和 SATA 一致：`200 Hz`。
- 每次跳跃都由 command 触发。
- command 为 0 时必须能稳定站立。
- 目标不必先固定为 velocity tracking 或 landing target；第一优先级是有效跳远。

我们的控制形式：

```text
torque = PD_prior_torque + residual_torque
```

其中：

```text
PD_prior_torque = Kp * (q_target - q) - Kd * qd
residual_torque = policy_action * action_scale
```

SATA 论文里的 torque-limit growth 不直接套在整条 torque 上，而是主要套在
`residual_torque` 上：

```text
residual_torque_raw = policy_action * action_scale
residual_torque = residual_limit_scale(t) * residual_torque_raw
torque = PD_prior_torque + residual_torque
torque = clip(torque, -hardware_torque_limit, hardware_torque_limit)
```

这样做的原因：

- `PD_prior_torque` 是早期站稳、蓄力、落地恢复的稳定扶手。
- `residual_torque` 是策略真正学习动态跳跃和爆发动作的通道。
- 训练初期限制 residual，可以避免策略一开始用大力矩乱探索。
- 随着跳跃链路学出来，再逐步放开 residual torque authority。

推荐同时调度：

```text
residual_limit_scale: small -> full
policy_frequency: 100 Hz -> 200 Hz
pd_prior_weight: high/medium -> medium/low
```

注意：PD prior 不应该永远压住 residual，否则策略可能只是在 PD pose
附近抖动，学不出真正的跳远爆发。

## 2. Command 接口

建议使用 5 维 command：

```text
commands[0] = target_x
commands[1] = target_y
commands[2] = target_yaw
commands[3] = target_aux
commands[4] = jump_command
```

语义：

```text
jump_command <= 0.5 -> stand
jump_command > 0.5  -> execute one jump
```

零 command 必须定义为：

```text
[0, 0, 0, 0, 0] = stand still
```

不要让策略在零 command 时自己跳。

## 3. Distance-First Objective

我们不需要先决定最终到底是 velocity tracking 还是 landing target。

当前最高优先级是：

```text
command 触发后跳得更远，并且能稳定落地
```

因此主任务应该是 distance-first jump，而不是纯 velocity tracking 或纯
landing-target tracking。

建议 command 语义：

```text
commands[0] = desired jump direction x / distance scale
commands[1] = desired jump direction y
commands[2] = desired yaw delta or yaw-rate hint
commands[3] = optional clearance / power hint
commands[4] = jump_command
```

其中 `commands[0:2]` 首先用于指定跳跃方向和强度，不必在第一版中严格追踪
某个速度或某个落点。

速度和落点可以作为训练扶手：

- launch velocity reward 用来教会策略把冲量打到正确方向。
- projectile estimated landing reward 用来在腾空中估计是否会跳远。
- final distance reward 用来奖励完成有效跳跃后的前向位移。
- landing target reward 只在需要精确落点时再加重。

主目标可以写成：

```text
valid_jump_distance = valid_jump_gate * stable_landing_gate * forward_distance_score
```

也就是说：

```text
跳得远
但必须是真跳
而且必须能落地恢复
```

## 4. Residual Growth And 100 Hz To 200 Hz Transition

SATA 当前控制频率：

```text
sim.dt = 0.005
control.decimation = 1
policy_dt = 0.005 s
policy_frequency = 200 Hz
```

我们的训练路线不是一开始就强行 200 Hz / full residual torque，而是：

```text
Phase A: 100 Hz bootstrap, low residual torque authority
  sim.dt = 0.005
  control.decimation = 2
  policy_dt = 0.010 s
  policy_frequency = 100 Hz
  residual_limit_scale = small
  pd_prior_weight = high / medium

Phase B: 100 Hz -> 200 Hz growth, residual torque opens gradually
  policy_frequency grows with curriculum / Gompertz-style schedule
  residual_limit_scale grows with jump metrics
  pd_prior_weight can decay if recovery remains stable

Phase C: 200 Hz SATA finetune, full residual torque authority
  sim.dt = 0.005
  control.decimation = 1
  policy_dt = 0.005 s
  policy_frequency = 200 Hz
  residual_limit_scale = full
```

原因：

- 100 Hz 更适合先学出站稳、触发、起跳、落地这些大结构。
- residual torque 从小到大，能避免早期大力矩探索破坏站立和落地。
- 200 Hz 用来和 SATA 最终框架对齐，并提升力矩控制细节。
- 频率切换后，动作保持时间、reward 累积、step gate 和 residual 作用强度都会变化，必须重新校准。

如果参考旧的 100 Hz 训练参数，不要直接复制 step 数。要按秒换算：

```text
steps_200hz = seconds / 0.005
steps_100hz = seconds / 0.010
```

从 100 Hz / low residual 迁移到 200 Hz / full residual 时，需要重点检查：

- first jump delay
- takeoff timeout
- grounded grace steps
- landing buffer steps
- post-jump stand steps
- command resampling time
- action rate penalty
- dof acceleration penalty
- residual action scale
- residual torque limit scale
- PD prior weight
- hardware torque clipping rate

迁移原则：

```text
先在 100 Hz + low residual 学会完整 jump chain
-> 逐步增加 residual torque limit
-> 逐步从 100 Hz 过渡到 200 Hz
-> 降低学习率或缩小更新幅度微调
-> 保持 command / observation / action 语义不变
-> 用秒而不是 step 对齐 phase gate
```

## 5. Curriculum

参考：

```text
SATA/legged_gym/legged_gym/envs/go2/go2_omnijump_curriculum_torque
```

这个任务的关键思想是：

- 先训练 zero-command stand。
- 再逐步打开 takeoff、flight、landing、motion rewards。
- 用 EMA 指标判断是否进入下一阶段。
- 同时逐步放开 residual torque limit 和 policy frequency。
- 不要手动每次训练改一堆 reward。

我们下一步应该做一个新的任务，例如：

```text
go2_sata_command_jump_torque
```

不要直接破坏已有 baseline。

## 6. 训练阶段

### Stage 0: zero-command stand

Command：

```text
[0, 0, 0, 0, 0]
```

目标：

- 四足稳定接触。
- base height 接近默认站立高度。
- 低 base linear/angular velocity。
- 关节接近 default pose。
- 无 base/thigh/calf/hip 非足端接触。

打开奖励：

- stand still
- zero linear velocity
- zero angular velocity
- base height
- default pose
- orientation
- collision / nonfoot contact
- torques
- action rate
- dof acceleration

这一阶段不要打开 jump reward。

### Stage 1: command-trigger preload

目标：

- `jump_command=1` 后进入可控下蹲/蓄力。
- 不摔倒，不贴地爬，不靠移动身体骗进度。

打开奖励：

- preload / squat pose
- contact symmetry
- anti-creeping
- stand stability

### Stage 2: takeoff

目标：

- 从支撑状态产生向上冲量。
- 出现自然腾空。

打开奖励：

- takeoff impulse
- takeoff vertical velocity
- release / all-feet airborne
- phase contact sync

关键指标：

- `natural_flight_rate`
- `takeoff_vertical_velocity`
- `all_feet_airborne_rate`
- `term_nonfoot_contact_rate`

### Stage 3: flight

目标：

- 保持真实腾空。
- 有足够 clearance。
- 机身姿态不乱翻。

打开奖励：

- clearance gate
- body attitude
- aerial joint pose
- feet clearance

高度在这里是 validity gate，不是最终主目标。

### Stage 4: landing and recovery

目标：

- 用足端落地。
- 落地冲击可控。
- 回到稳定站立。

打开奖励：

- landing contact
- landing stability
- prelanding pose
- landing pose
- impact penalty
- post-jump stand reward

### Stage 5: distance-first far jump

目标：

- command 控制跳跃方向和强度。
- 优先把有效前向距离推远。
- 速度和落点都只是辅助学习信号。

建议范围：

```text
direction_x: 1.0
direction_y: 0.0
yaw_hint: 0.0
power_or_distance_scale: curriculum from easy to hard
jump_command: 1.0
```

主奖励：

- valid jump distance
- completed jump distance
- stable landing after distance
- post-jump stand recovery

辅助奖励：

- launch velocity along command direction
- projectile estimated forward distance
- landing forward progress
- lateral drift penalty

注意：

- velocity reward 只在 takeoff / flight 阶段打开。
- distance reward 只在有效跳跃完成后打开。
- projectile reward 只在真实腾空后打开。
- 不能奖励地面跑动或地面爬行。
- landing target reward 不是第一目标，除非后面需要精确落点。

## 7. Command Sampling

训练中始终保留一部分 stand command：

```text
stand_command_prob = 0.20 ~ 0.30
```

单次跳跃逻辑：

```text
jump_command = 1
execute one jump
stable landing detected
command -> zero
require post-jump stand
```

连续跳应该作为后续模式，不应该早于单次跳跃稳定恢复。

## 8. 第一轮实验建议

新任务：

```text
go2_sata_command_jump_torque
```

基础设置：

```text
objective = "distance_first"
num_commands = 5
jump_command_threshold = 0.5
zero command = stand
control_type = "TG"
sim.dt = 0.005
control.decimation = 2 for 100 Hz bootstrap
control.decimation = 1 for 200 Hz SATA finetune
PD prior + residual torque
curriculum.enabled = True
```

训练顺序：

```text
Stage 0: zero-command stand
Stage 1: triggered preload
Stage 2: takeoff
Stage 3: flight
Stage 4: landing / recovery
Stage 5: distance-first far jump
```

对应 growth：

```text
Stage 0:
  low residual_limit_scale
  high PD prior
  100 Hz

Stage 1-2:
  residual_limit_scale starts increasing
  PD prior still protects posture
  100 Hz or early frequency growth

Stage 3-4:
  residual_limit_scale medium/high
  frequency grows toward 200 Hz
  PD prior starts decaying if landing is stable

Stage 5:
  residual_limit_scale high/full
  200 Hz
  residual torque carries most of the jump-distance improvement
```

## 9. 必看指标

站立：

- zero-command stand success
- base height
- four-feet contact rate
- nonfoot contact rate
- episode length

起跳：

- natural flight rate
- takeoff vertical velocity
- takeoff impulse reward
- all-feet airborne rate
- clearance gate

远跳：

- achieved forward distance
- forward velocity at takeoff
- flight time
- stable landing rate
- completed jump cycles

落点：

- estimated landing forward error
- stable landing forward error
- landing lateral error
- landing yaw error
- long jump success rate

安全：

- term nonfoot contact rate
- term rebound rate
- torque limit usage
- action rate
- dof acceleration

## 10. 下一步

下一步最应该验证的是：

```text
zero command 能站稳
command 能触发一次跳跃
先在 100 Hz 学会完整跳跃链路
再迁移到 200 Hz SATA torque + PD residual
落地后能恢复 zero-command stand
```

只有这件事成立后，再继续追求更远距离和更精确落点。
