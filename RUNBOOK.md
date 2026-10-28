# RUNBOOK — torque-controlled jumping on Go2

Reference for the task `go2_omnijump_landing_torque`. Control is pure torque, `decimation=1`,
`dt=0.005`, i.e. 200 Hz.

Every entry below is one of three things: **what the code does** (with the file and line to read),
**a measurement** (with the condition it was taken under), or **where a file is**. There are no
recommendations here.

Behaviour is set almost entirely by environment variables. `legged_gym/legged_gym/envs/go2/`
contains **134 distinct `os.environ.get` keys**, and nearly all of them default to the behaviour that
existed before the knob was added, so an unset knob produces a complete, error-free run of a
different configuration.

## Where things are

| Path | Contents |
|---|---|
| `legged_gym/legged_gym/envs/go2/go2_omnijump_landing_torque/` | the task: `..._torque.py` (rewards, reference clock, observation assembly), `..._torque_config.py` (all cfg + env knobs) |
| `legged_gym/legged_gym/envs/go2/go2_omnijump_torque/` | parent task: jump state machine, squat/takeoff gates |
| `legged_gym/legged_gym/envs/go2/go2_torque/` | `go2_torque.py` = PD scaffold, growth curriculum, domain randomisation, `update_inertia` |
| `legged_gym/legged_gym/scripts/eval_ckpt_scan_dr.py` | deterministic checkpoint scan with per-axis DR control |
| `legged_gym/legged_gym/scripts/play_landing.py` | single-env GUI play + per-episode trace |
| `legged_gym/resources/references/` | `commanded_ref_14700.npz` (default), `commanded_ref_sstill6.npz` |
| `deploy_mujoco/` | sim2sim |
| `checkpoints/` | three committed models, §5 |
| `README.md` | project + install pipeline |

---

## 0. Environment

The install pipeline is in [`README.md`](README.md) → "Pipeline to Install and Train ABS" and is not
duplicated here: conda env on python 3.8, torch, Isaac Gym Preview 4
(`cd isaacgym/python && pip install -e .`), `pip install -e rsl_rl`, `pip install -e legged_gym`,
`numpy` pinned `>1.20, <1.24`.

Versions every result in this file was produced under, read out of the live environments on
2026-10-28: **python 3.8.20, torch 2.0.1 (cu117 local / cu118 on the training box), numpy 1.21.0**.

Two facts not in `README.md`:

* `README.md` → "Observation layout (69 dims)" describes the flat-terrain ancestor. This task is
  `frame(49) × 20 + extras`; see §2.6.
* `deploy_mujoco/` needs **mujoco 3.2.3**, which is not part of the `README.md` pipeline.

Results in this file came from an environment with `PYTHONNOUSERSITE=1` set, i.e. with user
site-packages excluded. One earlier result taken under an environment that picked up ~169 extra
packages from user site-packages was initially attributed to a hardware difference.

---

## 1. Five calibration facts about the measurements

### 1.1 PD is re-applied unless three knobs are set

`go2_torque.py:107`: a fresh process sets `step_count = 0`, so `general_scale` re-ramps from 0 and
`pd_alpha = pd_prior_weight * max(PD_FLOOR, 1 - general_scale)`. For a policy trained to
`pd_alpha = 0`, this restores the PD scaffold.

`pd_alpha` is two-stage. Stage ① is the global fade above (with `PD_FLOOR=0.2` it is 0.100).
Stage ② is `_mult` in `go2_omnijump_curriculum_torque._compute_torques` (~line 352-382) =
`LOAD_PD_SCALE`'s closed-loop withdrawal × the `PD_OFF_PHASE` phase mask. `LOAD_PD_SCALE` defaults to
`"1.0"` at `go2_omnijump_landing_torque.py:244`, so stage ② resets to 1.0 in a new process.

* `PD_OFF_PHASE=all` → `pd_alpha_t = 0` and `_pd_w = 0`.
* `LOAD_PD_SCALE=0.0` pins `_load_pd_scale` at 0; the fade only decreases.
* `PD_FADE_SCOPE=all` extends the fade and the `PD_FLOOR` floor to every phase.
* On a resume, `SKIP_PD_FADE=1` starts `step_count` at `x0+1`, so `general_scale = 1` from step 0.

Measured: with PD silently restored, one run read −0.230 m and another −0.126 m against their own
pure-torque figures. Adding `alpha_PD = 0.100` onto a pure-torque checkpoint gave hit 0.00, takeoff
angle +12°, speed −14%. The comment at `go2_torque.py:98-105` records run `Aug15_19-09-47`:
`jump_flight_rate` 0.98 → 0.014.

`play_landing.py` prints
`pd_alpha = global_fade X x load_pd_scale Y x phase_off 'Z' -> APPLIED W`, appends `!! PD IS ON` when
`W > 0`, and after step 1 prints the value read back from `env.pd_prior_alpha`. An older print showed
stage ① only and was identical with and without `LOAD_PD_SCALE` set.
`deploy_mujoco/sim2sim_landing_history.py` omitted the `LOAD_PD_SCALE` factor twice.

### 1.2 What `meanHit` and `span` actually compute

`eval_ckpt_scan_dr.py:43-44,154`:

```python
DEAD_BAND  = [dx for dx in DX_SWEEP if 0.7 <= dx <= 0.9]
SCORE_BAND = [dx for dx in DX_SWEEP if 0.6 <= dx <= 1.2]
mean_hit   = float(np.nanmean(score_hits))
```

* The printed label `meanHit(0.6-1.2)` is a fixed string; the value averages over whatever
  `EVAL_DX_SWEEP` contains. One checkpoint read 0.814 on a 0.5–1.0 sweep and 0.704 on 0.5–1.2.
* `SCORE_BAND` excludes a 0.5 column, and the `7` in `span=X/7` is hard-coded. On a 0.5–1.0 sweep
  5 bins enter the score, so `span=5/7` is 5 of 5.
* `np.nanmean` drops a command bin that produced zero landings rather than scoring it 0.
  `hit[0.5:0.80 0.6:0.11 0.7:nan 0.8:0.71 0.9:0.87 1.0:0.45]` prints `meanHit=0.54`, which is
  `(0.11+0.71+0.87+0.45)/4`; the bin where all 256 envs failed to land is absent from the average.
  Scoring it 0 gives 0.428. `nMax` (`:159`) is `max(res[dx]['n'])` and `n` (`:86`) is the number of
  samples that produced a landing record.

### 1.3 The exact-nominal mass point

Run `Oct27_13-39-06_refprev2_25k_s1`, N=256, sweep 0.5–1.0, `PD_OFF_PHASE=all`:

| `EVAL_ADD_MASS` (kg) | −0.25 | −0.1 | 0.00 | +0.001 | +0.01 | +0.1 | +0.25 |
|---|---|---|---|---|---|---|---|
| ck22500 | 0.85 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ck23500 | 0.99 | 1.00 | **0.55** | **0.535** | 0.996 | 1.00 | 1.00 |
| ck24500 | 0.94 | 1.00 | **nan (nMax=1)** | — | — | 1.00 | 1.00 |

Unexplained as of 2026-10-28. The drop is checkpoint-specific, symmetric in `|x|`, and narrower than
0.01 kg. Three candidate mechanisms were tested and none survived:

1. *the COM offset is identically zero at x=0* — x=0.001 also drops, and the COM offset there is
   3.6 µm.
2. *all 256 envs become the identical body* — x=0.01 reads 0.996, and 36 µm is equally numerically
   zero, so the envs are equally identical there.
3. *x=0 takes a different code path* — `EVAL_ADD_MASS=0` sets `randomize_base_mass=True` with
   `range=[0,0]` (`eval_ckpt_scan_dr.py:116`), which does call `update_inertia`, unlike
   `EVAL_DR_MASS=0` which skips the block. But `go2_torque.py:27-38` at `mass_point=0` is exact
   identity: `new_com = com_box`, `displacement_box = 0`, `I_point = parallel_axis(0, 0, d) = 0`.
   The per-link term is `uniform(0,0) = 0`.

`added_mass_range` defaults to `[-1.0, +1.0]` (`go2_omnijump_landing_torque_config.py:265`), so x=0
is the centre of the training distribution. Each mass condition above is n=1; a seed-repeat control
at x=0 / 0.001 / 0.01 has not been run. `EVAL_DR_MASS=1` averages over the mass range and does not
sit on the point.

### 1.4 `EVAL_ADD_MASS=x` changes three things at once

`go2_torque.py:265-305`: `added_mass_range` is also read by the per-link loop
(`props[i].mass += uniform(rng[0]/16, rng[1]/16)`) and sets the magnitude of the COM point mass. So
`EVAL_ADD_MASS=x` adds `x` to the base, `x/16` to every link, and shifts the COM. No knob moves base
mass alone. At `x=0` the link term is `uniform(0,0)=0`, which is why pinning the base at nominal and
disabling randomisation entirely produce the same configuration.

`shifted_com_range_x` is set by `DR_COM_X` (default 0.05). `shifted_com_range_y/z` have no env knob
and inherit `[-0.1, 0.1]` from `go2_torque_config.py:157-158`.

### 1.5 Checkpoint-to-checkpoint and run-to-run comparability

* Adjacent checkpoints of one run, measured: `hit` 0.43 ↔ 0.81, `settled` 5% ↔ 100%.
* Isaac Gym: same machine, card, seed and code is bitwise identical for ~532 iterations, after which
  `atomicAdd` ordering crosses a rounding boundary and PPO amplifies the difference. Different
  hardware diverges at iteration 0; seeds do not control GPU atomic ordering. Measured spread across
  seeds at one config: 0.67–0.85 m.
* `landing_hit_rate` and the other tensorboard episode rates are not reproducible offline. Three
  caliper errors accounted for a 20× gap (`resampling_time` 60 against the training value 1.8,
  `episode_length_s` 20 against 10, and the script defaulting mass DR off); after all three were
  corrected an 8.5× gap remained, because the per-env adaptive state `_takeoff_omega_on`,
  `_load_hit_ema` and `_arm_hit_ema` is not stored in the checkpoint.
* `far = dx >= 0.6 * dx_max`, so the far-band metrics move with the command range.

---

## 2. Environment variables

Defaults are the `os.environ.get` fallbacks in the tree. A knob at its default reproduces the
behaviour from before it existed.

### 2.1 PD scaffold

| Knob | Default | What it does |
|---|---|---|
| `PD_OFF_PHASE` | `""` | `all` → `pd_alpha_t = 0`, `_pd_w = 0` |
| `LOAD_PD_SCALE` | `1.0` | `0.0` pins `_load_pd_scale` at 0 |
| `PD_FADE_SCOPE` | `load` | `all` extends the fade and the `PD_FLOOR` floor to every phase |
| `PD_FLOOR` | `0.0` | floor the PD ratio fades to; `0.2` leaves 20% PD |
| `SKIP_PD_FADE` | unset | starts `step_count` at `x0+1`, so `general_scale = 1` from step 0 |
| `LOAD_FADE`, `_TIMED`, `_STEP`, `_HOLD`, `_GUARD` | `0`, `0`, `0.05`, `20000`, `0.0` | the fade ladder; `_GUARD` latches once, then the ratio it multiplies makes it inert |
| `PD_TARGET_OBS` | `0` | adds the 12 PD targets to the observation (+12 extras) |
| `GROWTH_WARMUP`, `GROWTH_X0` | `19200`, `70400` | Gompertz torque/frequency ramp, counted in **steps** |

### 2.2 Command band and curricula

| Knob | Default | What it does |
|---|---|---|
| `DX_MIN`, `DX_MAX` | `0.0`, `1.0` | `landing_disp_x_stage2 = [DX_MIN, DX_MAX]`, a hard cap |
| `DY_MIN`, `DY_MAX` | `-0.30`, `0.30` | lateral command range; on by default, so aggregate hit rates average over oblique commands |
| `OMNI_DEG`, `OMNI_Y_SCALE` | `0`, `1.0` | polar command sampling |
| `DX_FLOOR_CURR`, `STANCE_CURR`, `STANCE_RAMP` | `1`, —, `60000` | curricula |

Measured interaction with the reference table: the table covers `cmd_x` 0.30–0.90 in 61 bins and
clamps outside. A run at `[0.5, 1.2]` draws a mismatched reference on 42.9% of samples against 14.3%
at `[0.3, 1.0]`; the `[0.5, 1.2]` run still re-lifted after landing on 100% of jumps at 20k
iterations with body yaw at +13.6° → +22.9° → +40.4° across three checkpoints, while the `[0.3, 1.0]`
run reached 14%. Separately, one run trained with commands beyond its reach read hit 0.00 at
`dx` 0.9 and 1.0 while a sibling at the same condition read 0.94.

### 2.3 Reward weights, by the phase the term is alive in

| Phase | Knobs (default) |
|---|---|
| Pre-jump stance | `STAND_POSE_W` (0), `STAND_STILL_W` (0), `STAND_TRACK_W` (0) + `STAND_TRACK_H` (0.303) / `_LAND` (1) / `_LSC` (30), `STAND_NO_TO_W` (0), `STAND_ANCHOR` (0) |
| Arming gate | `STAND_ARM_H0` (0), `_H` (0), `_HSTART` (0), `_STEP` (0.01), `_HOLD` (12000), `_GUARD` (0.85), `JUMP_DELAY` (120), `JUMP_SETTLE` (10) |
| Load / squat | `SQUAT_THR` (3.2), `SQUAT_HOLD` (25), `SQUAT_PAY_HELD` (1), `REQ_SQUAT` (1), `STANCE_SQUAT_W` (8.0) |
| Takeoff / flight | `PEAK_W` (20.0), `PEAK_ONESIDE` (0), `OMEGA_GATE` (0.80) |
| Landing | `SUCC_W` (1500), `SUCC_DIST_W` (1200), `SUCC_LAND_SIGMA` (0), `SUCCDIST_CAP` (0), `CLEAN_LAND_W` (0), `LAND_STAB_W` (1.0), `LAND_STAB_CONTACT` (0), `LAND_IMPACT_W` (0), `LAND_POSE_W` (0), `LAND_STAND_MIN_H` (0) |
| Post-landing settle | `SETTLE_DRIFT_W` (0), `SETTLE_POSE_W` (0), `SETTLE_ABSORB` (15), `DEFAULT_POS_REWARD` (0) + `_SIGMA` (3.0) + `_PEN_SCOPE` (`all`) |
| Yaw | `YAW_TRACK_W` (0), `YAW_TRACK_SIGMA` (0.25), `YAW_RATE_W` (0), `YAW_OBS` (0) |

### 2.4 Commanded reference

| Knob | Default | What it does |
|---|---|---|
| `REF_FILE` | `""` | falls back to `resources/references/commanded_ref_14700.npz` |
| `REF_PREVIEW_OBS` | `0` | adds `12 × len(REF_PREVIEW)` extras |
| `REF_PREVIEW` | `0,10,30` | preview horizons, in steps |
| `REF_PREVIEW_IDLE_MASK` | `0` | `1` multiplies the preview by `(ref_mode > 0)` |
| `REF_JOINT_W`, `REF_VEL_W`, `REF_ORI_W`, `REF_BASE_H_W` | all `0.0` | tracking rewards |
| `REF_SIG_*` | 0.02–2.0 | tracking sigmas: base 0.02, joints 0.2, joints-at-landing 2.0 |
| `REF_SLOT_RADIUS`, `REF_CMD_MAX`, `REF_EXT_FIT_SPAN` | `0`, `0`, `0.14` | table indexing |

`_update_ref_clock` (`go2_omnijump_landing_torque.py:1298-1301`) sets `ref_mode=0, ref_k=0` for
`idle = ~jumping_state`. `_ref_preview` (`:1330-1342`) applies the idle mask only when
`REF_PREVIEW_IDLE_MASK=1`. With the mask at its default 0, the three horizons stay frozen on
preparation slots 0 / 24-28 / 73-85 of 200 for the whole stance window, i.e. a constant
default-referenced vector lower than the default pose. The tracking rewards are masked by
`(ref_mode > 0)` (`:1352` and the comment above it); the observation is not.

Measured, pre-jump stance height across three checkpoints of each of two runs: with the mask
0.222 → 0.245 → 0.270; without it 0.291 → 0.274 → 0.219.

### 2.5 Domain randomisation

| Knob | Default | What it does |
|---|---|---|
| `DR_MASS_LO`, `DR_MASS_HI` | `-1.0`, `1.0` | `added_mass_range`; see §1.3 and §1.4 |
| `DR_LINK_MASS` | unset | `0.9,1.1` selects the multiplicative form and skips the legacy additive loop |
| `DR_COM_X` | `0.05` | `shifted_com_range_x`; Y and Z have no knob |
| `DR_FRIC_LO`, `DR_FRIC_HI` | `0.5`, `1.25` | friction range |
| `DR_PUSH` | `0` | pushes are off at the default, so they contribute nothing to results taken with it unset |
| `GRF_OBS` (+ `_CLIP_N`, `_NOISE`, `_BW_N`) | `0` | ground-reaction-force observation |

With `DR_LINK_MASS` unset, the legacy loop adds `uniform(added_mass_range/16)` kg to every link — the
same absolute amount regardless of link mass. Against the URDF masses at `added_mass_range=[-1,1]`
that is ±0.0625 kg, i.e. ±0.9% on the base (6.921 kg), ±9.2% on a hip (0.678), ±5.4% on a thigh
(1.152), ±40.6% on a calf (0.154), ±156% on a calflower (0.040) and ±6250% on each Head link (0.001).
A calflower therefore draws a negative mass about 18% of the time and each head link about 49%. The
loop also adds to `props[0]` after the composite inertia was computed, so that extra trunk mass is
absent from the inertia. The multiplicative form is relative and cannot go negative.

### 2.6 Observation geometry

`frame(49) × 20 = 980`, plus
`num_single_extras = 32 + 12·PD_TARGET_OBS + 12·len(REF_PREVIEW)`.

| Config | extras | actor | critic |
|---|---|---|---|
| neither | 32 | 1012 | 363 |
| `PD_TARGET_OBS=1` | 44 | 1024 | 399 |
| `PD_TARGET_OBS=1`, `REF_PREVIEW=0,10,30` | 80 | 1060 | 507 |

A checkpoint fails to load when these do not match.

---

## 3. Commands

### 3.1 Training

```bash
cd <repo>/legged_gym
export PYTHONNOUSERSITE=1
<the exports for the configuration being trained -- §2>
python legged_gym/scripts/train.py --task=go2_omnijump_landing_torque \
  --headless --max_iterations=<n> --seed=<s> --run_name=<name>
```

The exports in §2 are read at env construction, so they fix the configuration for the whole run and
are not recorded in the checkpoint. After launch, the curriculum scalars in tensorboard carry the
starting values the exports imply, and `Episode/curriculum_pd_alpha` carries the PD ratio of §1.1.

If `legged_gym` and `rsl_rl` are installed editable, `import rsl_rl` resolves by `PYTHONPATH` order,
and a wrong resolution produces a complete run with no error. One A/B whose two launch environments
differed by an omitted `LOAD_FADE` block ran 11.5 h before the difference was found, which is why a
comparison is set up as a comment-stripped diff of the two full environments rather than of the
experiment block alone.

### 3.2 Resuming

`on_policy_runner.py:372` restores `current_learning_iteration`; `:127` loops
`range(current, current + max_iterations)`; `:149` sets `_resumed=True`, so the entropy anneal fires
on the first iteration. `SKIP_PD_FADE=1` and `LOAD_PD_SCALE=0.0` apply as in §1.1.

* Measured: a child's `model_20000`, one PPO update past a parent that evaluated 0.814, evaluated
  0.36, and ~2000 iterations passed before it read as high as the parent again.
* The `iter` field does not survive a chain of resumes. A run resumed from iteration 20000 wrote
  `iter=20000` into every one of its own checkpoints, so resuming from that run's `model_23500`
  restarts the counter at 20000 and the new run writes `model_20000..25000` again. The weights are
  the ones loaded. A checkpoint is identified by run directory plus filename.

### 3.3 Deterministic evaluation

```bash
export PD_OFF_PHASE=all                        # §1.1
export EVAL_DX_SWEEP=0.5,0.6,0.7,0.8,0.9,1.0   # §1.2 -- the value meanHit averages over
export EVAL_HEIGHT=0.5 EVAL_N=256 EVAL_STEPS=400
export EVAL_DR_MASS=1                          # §1.3
export SCAN_START=... SCAN_END=... SCAN_STEP=...
SCAN_RUN=<run> python legged_gym/scripts/eval_ckpt_scan_dr.py \
  --task=go2_omnijump_landing_torque --headless --load_run=<run> --checkpoint=<n>
```

`EVAL_DR=1` keeps every DR axis; `EVAL_DR_<AXIS>=1` keeps one (`MASS`, `FRICTION`, …);
`EVAL_ADD_MASS=<kg>` pins the mass offset (§1.4). Without `--headless` the server run opens a viewer.

### 3.4 Play, local, with a GUI

`play_landing.py` overrides: `num_envs=1`, `episode_length_s=4`, `jump_command_range=[1,1]`,
`add_noise=False`, friction and pushes off, `landing_tilt_terminate=0.0`, `SPAWN_Z=0.35`.

| Knob | Default | What it does |
|---|---|---|
| `PLAY_LANDING_DX` | `1.2` | commanded forward displacement. Above `DX_MAX=1.0` and above the reference table's 0.90 clamp |
| `PLAY_LANDING_DY` | `0.0` | lateral |
| `PLAY_LANDING_HEIGHT` | `0.5` | height command; the file records the trained range as `[0.4, 0.6]` and 0.7 as an OOD false collapse |
| `PLAY_LANDING_ADDED_MASS` | `0.0` | `0.0` sets `randomize_base_mass = False` (`:131`); non-zero opens the whole DR block, i.e. COM point mass and per-link term as well (§1.4) |
| `PLAY_LANDING_TRACE` | `0` | `1` prints one line per episode, including episodes with no takeoff |
| `PLAY_LANDING_MONITOR_CONTACT` | unset | per-foot contact, height, drift, forward velocity |
| `PLAY_LANDING_STAND_ONLY` | `0` | commands cmd4 = 0.49, below the 0.5 threshold, so no jump fires |
| `PLAY_LANDING_NO_RSI` | `1` | forces `rsi_prob=0` |

The console hit counter is conditioned on having landed. In one session it printed `hit=0.75` while
the per-episode rate was 3/17 = 0.18: 9 episodes never armed and 4 never took off.

### 3.5 sim2sim

`deploy_mujoco/sim2sim_landing_history.py`. Needs mujoco 3.2.3 importable (§0).

Measured: for the history/stacked family, every interface item was checked item by item and a −46%
gap remains, traced to engine-level trajectory divergence — 18.6% apart within 50 ms under identical
torques. The flat `4600` policy matches to within 3.7% / 9%.

---

## 4. State machine

```
stand  --(first_jump_delay_steps=120, then contact4 & calm held jump_settle_steps=10)-->  armed
armed  --(cmd4 > 0.5)-->  loading     [ref_mode 0 -> 1, ref_k = 0]
loading --(_squat_pose_err() <= SQUAT_THR for SQUAT_HOLD consecutive steps)--> squat_qualified (latched)
       --(just_took_off & squat_qualified)-->  airborne  [ref_mode 2, ref_k = 0]
airborne --(descending below prelanding_height)-->  prelanding  -->  landing
landing  --(landing_step_counter >= landing_buffer_steps = 150, i.e. 0.75 s)-->  _finish_jump
_finish_jump: jumping_state = False  ->  stand   [ref_mode 0, ref_k 0, tracking rewards off]
```

`landing = jumping_state & has_landed` (`go2_omnijump_torque.py:684`).
`ready_to_finish = jumping_state & has_landed & (landing_step_counter >= landing_buffer_steps)`
(`:855`), and `landing_buffer_steps = 150` (`go2_omnijump_landing_torque_config.py:922`).
`_finish_jump` (`:494`) sets `jumping_state = False`.

### The takeoff gate is a 25-step consecutive run

`go2_omnijump_torque.py:667-673`:

```python
in_pose = loading & (self._squat_pose_err() <= _hold_thr)          # SQUAT_THR = 3.2
self.squat_hold_counter = torch.where(in_pose, counter + 1,
                              torch.where(loading, 0, counter))     # resets on any miss
self.squat_qualified |= (self.squat_hold_counter >= 25)             # SQUAT_HOLD
```

`_squat_pose_err` (`:1178-1181`) is `Σ|dof_pos − q_squat_target|`, the L1 distance to the squat
target, not to the default stance. Because the counter resets on any miss, the pass rate goes as
`p**25` for a per-step in-pose probability `p`: 0.90 → 7%, 0.95 → 28%, 0.99 → 78%.

### Pitch cost by phase

`_reward_pitch_level` (`go2_omnijump_landing_torque.py:2377-2389`):

```python
pg        = self.projected_gravity[:, 0]        # >0 = nose-down
nose_down = clamp(pg, min=0)**2                 # one-sided
level     = pg**2                               # symmetric
r = self.jumping_state.float() * nose_down
r = r + 5.0 * (self.prelanding | self.landing).float() * level      # landing_pitch_extra
r = r + 12.0 * self.airborne.float() * nose_down                    # jump_pitch_extra, latched
```

In the stance window `jumping_state` is False and none of `prelanding`/`landing`/`airborne` hold, so
the term is 0 there. Inside the landing window it is `5.0 × pg²`, symmetric.

### Termination

`terminate_after_contacts_on=["base"]` with `collision_cutoff = Σ|F| > 0.2 N`; `roll_cutoff` 2.4 rad;
`landing_tilt_terminate = 0.0` (off); `one_jump_reward_per_episode=True` with
`post_jump_stand_steps = 300`.

### Latched per-env state that is not in the checkpoint

`_takeoff_omega_on`, `_load_hit_ema`, `_arm_hit_ema`. Offline they initialise to `False`/0, so in play
and eval every term gated behind them is inactive: `stand_pose` (`:2008` returns zeros),
`land_stand_min_height` (`go2_omnijump_torque.py:881`), `squat_gate_floor` (`:1631`), and the
takeoff-omega damping. `LOAD_FADE_GUARD` and `STAND_ARM_GUARD` latch once and are then multiplied by
a ratio, which makes them inert from that point on.

---

## 5. `checkpoints/`

Three models, ~23 MB total. They are in the repository rather than as Release assets because a fork
copies repository contents and not Release assets. `.gitignore` excludes `*.pt`; these were added
with `git add -f`.

| File | Origin run | actor | critic |
|---|---|---|---|
| `refprev2_25k_s1_model_23500.pt` | `Oct27_13-39-06_refprev2_25k_s1` | 1060 | 507 |
| `sstill6_s1_model_19500.pt` | `Oct04_09-40-53_sstill6_s1` | 1024 | 399 |
| `inertiafix_s1_model_14700.pt` | `Sep22_04-02-15_inertiafix_s1` | 1012 | 363 |

### `refprev2_25k_s1_model_23500.pt`

A +5000-iteration continuation of `Oct07_20-24-02_refprev2_s1/model_20000`. `thesis-2026-10-28` is
tagged on the commit that carries it.

```bash
export GROWTH_WARMUP=96000 GROWTH_X0=192000 PD_FLOOR=0.2
export DX_MIN=0.3 DX_FLOOR_CURR=0 STAND_ANCHOR=1 STANCE_CURR=0
export PEAK_W=20 SUCC_LAND_SIGMA=0.15 SUCCDIST_CAP=1
export LOAD_FADE=1 PD_FADE_SCOPE=all LOAD_FADE_TIMED=1
export LOAD_FADE_STEP=0.1 LOAD_FADE_HOLD=12000 LOAD_FADE_GUARD=0.6
export PD_TARGET_OBS=1 STAND_POSE_W=3.0 STAND_STILL_W=6.0
export REF_PREVIEW_OBS=1 REF_PREVIEW_IDLE_MASK=0
export PD_OFF_PHASE=all          # evaluation only, §1.1
```

Measured, N=256, sweep 0.5–1.0, `PD_OFF_PHASE=all`:

| | mass DR | exact nominal | pre-jump stance |
|---|---|---|---|
| ck23500 | 0.954 (minDead 0.91, span 5/5, airPeak 0.947, nMax 256) | 0.55 (§1.3) | 0.203 m |
| ck22500 | 0.919 (minDead 0.87) | 1.00, and 1.00 at every mass point from −0.10 to +0.25 kg | 0.198 m |
| ck24500 | 0.935 (minDead 0.90) | nan, nMax 1 | 0.188 m |
| parent ck20000 | 0.814 (minDead 0.72, airPeak 0.934) | — | 0.219 m |

Default stance height is 0.303 m. `ck22500` and `ck24500` are `model_22500.pt` / `model_24500.pt` of
the same run and are not committed here.

### `sstill6_s1_model_19500.pt`

The block above minus `REF_PREVIEW_OBS` and `REF_PREVIEW_IDLE_MASK` (hence 1024), with
`STAND_STILL_W=6.0`.

Measured: `STAND_STILL_W=6` moved the pre-jump stance 0.228 → 0.301 m with hit 0.81 → 0.65. `W=2`
took late-run `settled` from 100% to 9%. Against the paper baseline, per-command net flight was
higher at every command, best 0.905 against 0.845 m (+7.1%), and `settled` went 82% → 98%; at
cmd 0.90 the baseline's `settled` was 0.72 and this run's 0.97.

`_reward_stand_still` is a velocity brake, with σ about 60× tighter than the landing term; it does not
read posture. §6 has the share arithmetic.

### `inertiafix_s1_model_14700.pt`

The published paper baseline, and the checkpoint `commanded_ref_14700.npz` was extracted from.
Geometry 1012 = 980 + 32, i.e. `PD_TARGET_OBS` unset and no ref preview; otherwise the
`..._19500.pt` block.

`commanded_ref_14700.npz` contains `cmd`, `fs`, `pre`, `calf_range` and per-command
`q / dq / tau / base / contact / q_std / unload` arrays. It carries **no source run or checkpoint
field**; the attribution above rests on the filename and on the file's
`calf_range = [-2.7227, -0.83776]` matching the measurement below. The extraction scripts that built
it are not in the repository (§7).

The extracted trajectory: a countermovement jump with a free-fall phase — unload, free fall 0.22 m,
squat bottom at z = 0.079 m against 786 N (5.3 BW), launch |v| = 2.90 m/s at 49°. Cross-jump aligned
and averaged, the joint trajectory has ρ₁ = +0.98 and is closer on every term than the medoid single
jump. The front calf sits at −2.7227 at the squat bottom and the rear calf at −0.8378 through flight,
penetrating the hard limit by 0.07 rad. The squat bottom has only the two rear feet in contact. No
admissible trajectory was extracted at cmd 1.00; the table covers 0.30–0.90 in 61 bins.

---

## 6. Measured outcomes of past changes

**Reward share of the stance terms.** `stand_pose` is 0.85% of the reward-term total and returns
zeros unless `_takeoff_omega_on` (`:2008`). `stand_still` is 9.21% and is a velocity brake, so a
droop that settles collects it in full. `orientation` is 1.2–1.7% and weights roll and pitch equally.
`pitch_level` is 0 in the stance window (§4). The code's own estimate is that ≤4% of the return is at
stake in the stance window. On the other side, `forward_reach` 15.2% + `projected_landing` 14.1% +
`landing_position` 13.7% + `successful_distance` 5.8% = 48.8%.
Shares are fractions of the reward-term total; `Episode/rew_*` are per-second rates, and the total
differs from `mean_reward` by about 10×.

**`STAND_ARM_H0`,** a height condition in the arming gate: stance 0.219 → 0.309 m, above the 0.303 m
default, with training-time hit −11 points. `stand_no_takeoff` was found to be dead code and was
already paid off (airborne 5 steps → 0.36 steps/episode).

**Weighting `landing_stability` up:** `settled` +3 points, far end −82%. The term saturates.

**Strengthening `takeoff_velocity_match`,** the only causal-side term: it is capped at 1.7% of the
return and differed by 0.04 between two runs. The limit measured at the same point is the knee
velocity wall. Separately, `forward_reach` flat between two runs does not imply equal distance — the
difference can fall entirely outside the scored band.

**Stacked observation history, trained from scratch under pure torque:** two independent
implementations did not find the jump (squat identically 0, reward stuck near 0.5 against 20 for
flat). The reward code was byte-identical, the observation content verified identical, and the critic
fitted normally. The paper the architecture came from uses PD position control.

**Penalising the landing harder:** per-command takeoff rate went `0.5: 128/128`, `0.7: 66/128`,
`0.8: 6/128`, while a sibling run on the same harness was full at every command. The training curves
for the two runs do not separate.

**The `main` branch.** Its tip `0157dfb` (separate jump/recovery episodes, RSI drill, combat squat,
2026-06-26/27) was compared against a re-train from `6c4032b`: squatQ 0.87 / dx 1.1 / hit 0.81 for
`6c4032b`, against 0.29 / 0.40 / 0.56 for `0157dfb`. The line of work carrying this file is on the
branch this file is committed to.

**Two sim artifacts.** The SATA URDF truncates the thigh range to `[0, 1.5]` front and `[0, 2.0]`
rear; the official ranges are `[-1.5708, 3.4907]` and `[-0.5236, 4.5379]`. On `model_4600`, FL_thigh
sat at exactly 1.500 and RL_thigh at −0.006 through the whole jump, i.e. against the truncated stop,
which does not exist in MuJoCo or on hardware (the same policy there swings to −0.64 / 1.61).
`cfg.asset.official_dof_pos_limits` overrides the limits in code. Separately, the stance-phase joint
chatter is a T-N centripetal/centrifugal branch-switching artifact: 15.8% hard jumps, a second
histogram peak at 0.158, 12% of steps, RMS matching within 10%. `action_rate` is 0.09% of the penalty
total.

**Jump distance, measured as net flight from the takeoff anchor.** refprev2 0.763 m (reached by
commanding cmd 1.7; cmd 1.0 gives 0.66), band512 0.654, omni180 0.630, landfix2 0.558. Measurement
noise ±0.06, and these are not directly comparable to the 0.905 figure quoted in §5, which uses a
different caliper.

---

## 7. Measurements whose raw records are not in this repository

The numbers in §1, §2 and §6 were produced by a set of measurement scripts and write-ups that were
removed from the repository on 2026-10-28. They are retrievable from git history:

    git show d81a246:paper_tools/<name>

and the directory listing of what existed is `git show d81a246 --stat -- paper_tools/`. It held the
reference-trajectory extraction chain (`extract_ref.py` -> `ref_traj_14700.npz` ->
`build_commanded_ref.py` -> `commanded_ref_14700.npz`, plus `ref_query.py` and `check_ref.py`), about
eighteen measurement scripts, and eleven dated write-ups from 2026-09-29 to 2026-10-07 that record
the protocol behind each figure -- among them the reward shares of §6, the smoothness figures, the
reference-trajectory description of §5, and the paper-baseline comparison.

Consequences for anything built from here:

* `resources/references/commanded_ref_14700.npz` is still present and is what the code loads, but
  **the chain that derives it is not**, so the table cannot be re-derived or extended from a
  checkpoint without recovering those scripts from history.
* The figures quoted in this file cannot be re-checked against their original protocol from the
  working tree alone.
