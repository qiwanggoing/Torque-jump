# Commanded reference R(φ, c):从 14700 的平滑轨迹建命令条件参考 (2026-10-07)

对应答辩稿 Slide 16 "a motion reference for each jump command / begin with successful simulated
jumps as a reference library / weaker joint tracking around touchdown"。

产物:`paper_tools/commanded_ref_14700.npz`(0.29 MB)+ `build_commanded_ref.py` + `ref_query.py`。
上游是 `ref_traj_14700.npz`(见 `reference_trajectory_14700_2026-10-06.md`)。

## 1. 命令轴:5 个锚点 + 线性插值,已验证够用

锚点 c = 0.30 / 0.50 / 0.70 / 0.80 / 0.90(cmd 1.00 提不出合格轨迹,上端夹断)。

**留一法验证**(丢掉 0.70,用 0.50/0.80 插值再和实测的 0.70 比):

| 段 | q RMS 误差 | max | 同命令跨跳 1σ | 比值 |
| --- | --- | --- | --- | --- |
| 起跳前 | **0.0129 rad** | 0.0599 | 0.0343 | **0.38** |
| 起跳后 | **0.0328 rad** | 0.1226 | 0.0550 | **0.60** |

插值误差只有自然跳间噪声的 0.38–0.60 倍,**在噪声地板以下** → 插到没测过的命令是安全的。

支撑这一点的是所有量随命令单调:

| cmd | \|v\| | 仰角 | pitch | 峰值 z | **卸载→起跳** |
| --- | --- | --- | --- | --- | --- |
| 0.30 | 2.203 | 66.9° | +3.4° | 0.511 | 70.1 步 (350 ms) |
| 0.50 | 2.416 | 60.4° | −0.8° | 0.538 | 73.4 步 |
| 0.70 | 2.703 | 52.6° | −6.4° | 0.555 | 77.6 步 |
| 0.80 | 2.829 | 50.3° | −9.9° | 0.561 | 79.9 步 |
| 0.90 | 2.901 | 49.0° | −12.2° | 0.569 | 81.9 步 (410 ms) |

## 2. ⭐ 相位索引:时钟就够,不需要状态索引

原先担心的是"环境在起跳前不知道什么时候起跳,没法按时钟查表"。实测否掉了这个担心:

| cmd | n | 卸载→起跳 | 接地→起跳 | 全距 |
| --- | --- | --- | --- | --- |
| 0.80 | 128 | 79.9 ± **0.7** 步 (400 ± 3 ms) | 40.6 ± 0.5 步 | [79, 81] |
| 0.90 | 125 | 81.9 ± **0.8** 步 (410 ± 4 ms) | 42.0 ± 0.7 步 | [80, 83] |

**125–170 次跳,准备段时长的全距只有 2–3 步(10–15 ms)。** 所以:

- 参考时钟锚在**卸载事件**(足力合力首次跌破 50 N,环境直接看得到),不是锚在命令触发。
  (只验证了"卸载→起跳"是确定的;"触发→卸载"没测,所以不要锚在触发上。)
- **起跳前**用归一化进度 `φ = k / K(c)`,K 本身按命令插值(70.1 → 81.9 步)→ 起跳恰好落在 φ=1。
- **起跳后**用距起跳的绝对步数,因为滞空时长由起跳状态决定,本来就随命令变。

锚点处往返误差 max 0.012–0.057 rad(来自 φ 重采样的分数索引,出现在最陡的转折处)。

## 3. 存了什么

每个锚点 301 步 @200 Hz:`q`(12,已 clip)、`dq`、`tau`、`base`(z, vx, vy, vz, pitch, roll)、
`contact`(4,>20 N)、`q_std`(跨跳 1σ,可当自适应 σ 用)、`unload`(偏移均值/标准差/样本数)。

**clip**:小腿按 MJCF 硬限位 `[−2.7227, −0.83776]` 截断(每档 34–74 个采样越界,最多 0.082 rad),
髋按 `[−1.0472, 1.0472]`。不 clip 的话就是在要求策略去物理上到不了的地方。

## 4. 怎么用

```python
from ref_query import CommandedReference
ref = CommandedReference()
K = ref.takeoff_steps(c)                 # 这个命令下预计多少步后起跳
s = ref.query(c, k, after_takeoff=False) # k = 距卸载事件的步数
s = ref.query(c, k, after_takeoff=True)  # k = 距起跳的步数
obs = ref.preview(c, k, horizons=(0, 10, 25))   # 36 维预览,接 actor 输入
```

cmd 0.75(从没测过)的查询结果,物理上自洽:

| | z | vx | vz | pitch | calf FL/RL | 接触 |
| --- | --- | --- | --- | --- | --- | --- |
| 卸载 | 0.298 | +0.03 | −0.05 | +1.6° | −1.643 / −1.460 | 0100 |
| 准备中段 | 0.090 | −0.10 | −1.11 | −0.7° | −2.723 / −2.723 | 0011 |
| 起跳 | 0.345 | +1.73 | +2.16 | −8.1° | −1.154 / −1.277 | 0000 |
| 顶点 | 0.558 | +1.62 | −0.07 | −5.4° | −2.476 / −0.838 | 0000 |
| 触地 | 0.295 | +1.47 | −2.37 | +5.5° | −1.468 / −2.375 | 1000 |

注意"准备中段"只有**后两只脚**在接触 —— 和 `deploy_mujoco/ref_traj.py` 文档里记的那个结构缺陷一致
(蹲底时前脚已经离地,一半执行器对起跳不出力)。参考如实地把这个缺陷也带进来了。

## 5. 接进训练时的奖励形状(按文献,未实现)

照 Wang 2025 / Fuchioka 2023 的比例(见 `papers/REFERENCE_HOW_THEY_BUILD_IT_2026-10-07.md`):

| 项 | 目标 | 建议 σ | 说明 |
| --- | --- | --- | --- |
| 机身位置/高度 | `base[0]`, 水平位移 | 0.02 | 最紧 |
| 机身姿态 | `base[4:6]` | 0.01–0.14 | 次紧 |
| 关节位置(准备+飞行) | `q` | **0.2** | 比机身松 10× |
| 关节位置(落地后) | `q` | **2.0** | 再松 10×(答辩稿已写 "weaker joint tracking around touchdown") |

理由不是风格:实测这条参考已经吃掉 T-N 包络的 93–100%,**反馈没有力矩余量做紧的关节跟踪**。

另可用 `q_std_<c>`(跨跳 1σ)做逐时刻逐关节的自适应 σ —— 策略自己越不一致的地方,参考就越该放松。

---

# 已接进训练环境:`REF_PREVIEW_OBS`(2026-10-07,已冒烟,未 commit)

答辩稿四组对照里的 **"+trajectory input"** 那一组。默认关闭,关闭时与现有训练线逐字节相同。

## 改了什么

**`go2_omnijump_landing_torque_config.py`**
- env 块新增 `ref_preview_obs` / `ref_preview_h`(默认 `0,10,30` 步 = 0/50/150 ms)/ `ref_file`;
  `num_single_extras = 32 + 12·pd_target + 12·len(horizons)`,`num_observations` 与 critic 维度自动跟随。
- 镜像图 `_mirror_obs_permutation`:原来是 `if extras_dim >= 44: extras += legs(32)`,预览加进来之后这个
  条件会漏判。改成按 12 维块计数 `n_leg_blocks = (extras_dim-32)//12` 再逐块 `legs(32+12b)` ——
  pd_target 和每个 horizon 都是关节角,镜像奇偶完全相同。对 32(0 块)和 44(1 块)与原分支等价。
  **这一条必须改,否则 `sym_loss` 会用错位的镜像图训练。**

**`go2_omnijump_landing_torque.py`**
- `_build_ref_table()`:把 `commanded_ref_14700.npz` 预插值成 `(61, 371, 12)` 的稠密表(命令 0.30–0.90
  步长 0.01;时间轴 = 准备段 φ 栅格 200 点 + 起跳后 171 步),并**预先减掉 `default_dof_pos`**,
  运行时只剩一次 gather。在 `_init_buffers` 末尾、`_prime_history` 之前建。
- `_update_ref_clock()`:双锚点。`phase_loaded` 上升沿(锁存,`vz<=0` 抖动不会重复触发)启动准备段时钟;
  真实 `just_took_off` 把时钟重新对齐到起跳;`~jumping_state` 复位。
- `_post_physics_step_callback()`:新增 override,`super()` 末尾就是 `_update_jump_state()`,相位刚更新完。
  **⚠️ 时钟不能放在 `_get_single_extras`** —— 那个函数每步被调用两次(actor obs 一次、privileged frame
  一次),放那里计数会翻倍。
- `_ref_preview()`:查表并乘 `obs_scales.dof_pos`,表达方式与 `pd_target_obs` 完全一致。
  **预览窗口允许跨过参考的起跳点**(`k+h > K` 时落到起跳后分支的 `k+h−K` 步),这样准备段末尾能看到收腿;
  只有**时钟本身**夹在 φ=1 —— 比参考慢的策略停在"你该起跳了"那一帧,直到它真的离地。
- `resources/references/commanded_ref_14700.npz`(0.29 MB)。

## 维度

| 配置 | extras | actor obs | critic obs |
| --- | --- | --- | --- |
| 都关(论文基线) | 32 | **1012** | 363 |
| `PD_TARGET_OBS=1`(sstill6 线) | 44 | **1024** | 399 |
| `+REF_PREVIEW_OBS=1`(3 horizon) | 80 | **1060** | 507 |
| `REF_PREVIEW=0,15`(2 horizon) | 56 | 1036 | 435 |

## 验证

1. **四种配置全部建环境成功**,`len(obs_permutation) == num_observations`,镜像图通过
   **对合检验**(`idx[idx] == arange`)和**符号自逆检验**。
2. **查表对拍 `ref_query.py`**:起跳后分支逐位精确(0.000000);准备段最大 **0.0199 rad**,
   纯粹是 61 档命令栅格 + 200 点 φ 栅格的量化,**小于参考自身的跳间 1σ(0.034 rad)**。
   (第一版 φ 栅格 100 点 + 截断取整误差 0.052 rad,已加密到 200 点并改四舍五入。)
3. **关闭时的回归**:`REF_PREVIEW_OBS=0` 用 model_14700 跑 `meas_smooth`(NENV=256 / STEPS=900 / 全 DR),
   `hit 0.87 / dist 0.860 / err 0.067 / settled 208/256 (81%) / 0.56s` 以及九个平滑度数字
   **与改动前逐位相同**。

## 还没做

- 没有 commit、没有 push。(注意:工作区在我动手之前就已经有未提交的改动,
  `go2_omnijump_curriculum_torque.py` 里的 `PD_FADE_SCOPE=all` 就是其中之一,不是本次改的。)
- 没有训练。观测维度变了 → **不能从 14700/19500 续训,必须从头训**,这正是四组对照的要求。
- 跟踪奖励那一组(`+tracking reward`)没有实现,只在本文件第 5 节写了建议的 σ。
