# RUNBOOK — torque-controlled jumping on Go2

Read this before running anything. This codebase is driven almost entirely by **environment
variables**: `grep -rc 'environ' legged_gym/legged_gym/envs/go2/` finds **134 distinct knobs**, and
nearly every one defaults to *the old behaviour*, so a run that forgets an export is not an error —
it is a silently different experiment that completes normally and produces a number you will
believe.

Everything below is either a **code fact** (file:line given, go read it) or a **measurement** (the
condition it was measured under is stated). Nothing here is a plan or an intention.

Task used throughout: `go2_omnijump_landing_torque`. Control is **pure torque**, `decimation=1`,
`dt=0.005` → **200 Hz**.

---

## 0. Environment setup

**Use the install pipeline in [`README.md`](README.md) → "Pipeline to Install and Train ABS".** It is
current and is not duplicated here: conda env on **python 3.8**, torch, Isaac Gym **Preview 4**
(`cd isaacgym/python && pip install -e .`), then `pip install -e rsl_rl` and
`pip install -e legged_gym`, and `numpy` pinned to `>1.20, <1.24`.

Versions this work actually ran on, verified in the live envs rather than taken from the doc:

| | local (`sata-clean`) | server |
|---|---|---|
| python | 3.8.20 | 3.8 |
| torch | **2.0.1**+cu117 | **2.0.1**+cu118 |
| numpy | 1.21.0 | 1.21 |

Three things the README does not tell you:

* **`README.md` → "Observation layout (69 dims)" is stale.** That was the flat-terrain ancestor. This
  task is `frame(49) × 20 + extras` → **actor 1060, critic 507** (asymmetric). See §2.6.
* **`PYTHONNOUSERSITE=1` for everything in Isaac Gym** — the clean env must not pick up the ~169
  packages in `~/.local`; a polluted env was once mistaken for a hardware difference.
  **The one exception is sim2sim:** mujoco 3.2.3 lives *only* in `~/.local`, so the MuJoCo scripts
  must run **without** it (§3).
* **Server runs additionally need `pathfix` on `PYTHONPATH`** (§3). This is not a convenience.

For the remote 4090 box specifically — connection, disk policy, tmux, directory layout, code sync,
wandb — see [`REMOTE_TRAINING.md`](REMOTE_TRAINING.md).

---

## 1. The five traps

These are not edge cases. Each one has silently invalidated real results.

### Trap 1 — pure torque is not the default

A fresh process sets `step_count = 0` (`go2_torque.py:107`), so `general_scale` re-ramps from 0 and
`pd_alpha = pd_prior_weight * max(PD_FLOOR, 1 - general_scale)`: **the full PD scaffold gets
re-applied to a policy that trained at `pd_alpha = 0`.** The comment at `go2_torque.py:98-105`
records that exact death (run `Aug15_19-09-47`: `jump_flight_rate` 0.98 → 0.014).

* **Evaluating** a pure-torque checkpoint: `PD_OFF_PHASE=all` (strongest pin), or at minimum
  `LOAD_PD_SCALE=0.0` **and** `PD_FADE_SCOPE=all`. Setting only one of the latter two is not enough.
* **Resuming** training: `SKIP_PD_FADE=1` (starts `step_count` at `x0+1` so `general_scale = 1` from
  step 0) **and** `LOAD_PD_SCALE=0.0`. Verify after launch: `Episode/curriculum_pd_alpha == 0`.

Measured cost of getting this wrong: −0.230 m on one run and −0.126 m on another — i.e. it
**fabricates differences between runs**. Adding `alpha_PD = 0.100` back onto a pure-torque
checkpoint took hit rate to **0.00** (takeoff angle +12°, speed −14%).

### Trap 2 — `meanHit` averages over whatever `EVAL_DX_SWEEP` is, and two of its labels are stale

`eval_ckpt_scan_dr.py:43-44,154`:

```python
DEAD_BAND  = [dx for dx in DX_SWEEP if 0.7 <= dx <= 0.9]
SCORE_BAND = [dx for dx in DX_SWEEP if 0.6 <= dx <= 1.2]
mean_hit   = float(np.nanmean(score_hits))
```

* The printed `meanHit(0.6-1.2)` is a **fixed string**. The same checkpoint reads **0.814** on a
  0.5–1.0 sweep and **0.704** on 0.5–1.2. Never compare numbers from different sweeps.
* `SCORE_BAND` **excludes the 0.5 column**, and the `/7` in `span=X/7` is hard-coded. On a 0.5–1.0
  sweep only 5 bins are scored, so `span=5/7` means "5 of 5".
* **`nanmean` DROPS a command bin that produced zero landings instead of scoring it 0.** Worked
  example: `hit[0.5:0.80 0.6:0.11 0.7:nan 0.8:0.71 0.9:0.87 1.0:0.45]` prints `meanHit=0.54`, which
  is `(0.11+0.71+0.87+0.45)/4` — the bin where **all 256 envs failed to land** was deleted from the
  score. Counting it as 0 gives 0.428. **`meanHit` therefore overstates any checkpoint with dead
  bins.** Always read `nMax` and look for `nan` in the per-bin row before quoting a `meanHit`.

### Trap 3 — the exact-nominal mass point is a per-checkpoint dead point, and the scan sits on it

Same run `Oct27_13-39-06_refprev2_25k_s1`, N=256, sweep 0.5–1.0, `PD_OFF_PHASE=all`:

| `EVAL_ADD_MASS` (kg) | −0.25 | −0.1 | **0.00** | +0.001 | +0.01 | +0.1 | +0.25 |
|---|---|---|---|---|---|---|---|
| ck22500 | 0.85 | 0.95 | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 |
| ck23500 | 0.99 | 1.00 | **0.55** | 0.535 | 0.996 | 1.00 | 1.00 |
| ck24500 | 0.94 | 1.00 | **nan (nMax=1)** | — | — | 1.00 | 1.00 |

**Unexplained as of 2026-10-28.** It is `run`- and `checkpoint`-specific, symmetric in `|x|`, and
narrower than 0.01 kg. Three hypotheses were tested and **all three failed**:

1. *"the COM offset is identically zero at x=0, so that point is special"* — refuted: x=0.001 still
   collapses (0.535) and the COM offset there is 3.6 µm, not 0.
2. *"the variable is that all 256 envs become the identical body"* — refuted: x=0.01 recovers fully
   (0.996) and 36 µm is equally numerically nothing, so the envs are equally identical there.
3. *"x=0 takes a different code path"* — refuted: `EVAL_ADD_MASS=0` is `randomize_base_mass=True`
   with `range=[0,0]` (`eval_ckpt_scan_dr.py:116`), which does still call `update_inertia`, but
   `go2_torque.py:27-38` at `mass_point=0` is exact identity (`new_com = com`,
   `displacement_box = 0`, `I_point = 0`), and the per-link term is `uniform(0,0) = 0`.

The remaining reading is that these checkpoints are **bimodal** and each mass condition was n=1.
`added_mass_range` defaults to `[-1.0, +1.0]` (`..._config.py:265`), i.e. **x=0 is the exact centre
of the training distribution**, so a systematic hole there is implausible on its face. **A
seed-repeat control at x=0 / 0.001 / 0.01 has not been run.** Until it is: rank with
`EVAL_DR_MASS=1` (averages the point away), and state the DR condition next to every hit rate.

### Trap 4 — `EVAL_ADD_MASS=x` does not move the base mass alone

`go2_torque.py:265-305`: `added_mass_range` is **reused** by the per-link loop
(`props[i].mass += uniform(rng[0]/16, rng[1]/16)`) and it also sets the magnitude of the COM point
mass. So every point adds `x` to the base, `x/16` to **every** link, and shifts the COM. There is no
knob that moves base mass alone, so such a sweep is a sweep of "mass offset", not of base mass.
At `x=0` the link term is `uniform(0,0)=0`, which is why "pin the base at nominal" and "no
randomisation at all" come out byte-identical.

`shifted_com_range_x` has the knob `DR_COM_X` (default 0.05). **`shifted_com_range_y/z` have no
knob** and inherit `[-0.1, 0.1]` from `go2_torque_config.py:157-158`, so the COM cannot be fully
zeroed from the environment.

### Trap 5 — adjacent checkpoints of one run disagree, and training-time rates are not comparable across runs

* Measured within one run, adjacent checkpoints: `hit` **0.43 ↔ 0.81**, `settled` **5% ↔ 100%**.
  A handful of points will give you the opposite conclusion. **Scan 10+ checkpoints.**
* `landing_hit_rate` and friends in tensorboard are **not reproducible offline and not comparable
  between runs**: three caliber errors accounted for a 20× gap (`resampling_time` 60 vs the training
  1.8, `episode_length_s` 20 vs 10, and the script defaulting mass DR off), and after fixing all
  three an **8.5× structural gap remains**, because per-env adaptive state (`_takeoff_omega_on`,
  `_load_hit_ema`, `_arm_hit_ema`) is **not stored in the checkpoint**. Use them for *trend within
  one run* only. Every cross-run verdict must come from `eval_ckpt_scan_dr.py`.
* `far-band` metrics are additionally unusable across runs: `far = dx >= 0.6 * dx_max`, so the band
  itself moves with the command range.

---

## 2. Environment variables that actually matter

Defaults are the code's own (`os.environ.get` fallbacks), extracted from the tree, not from memory.
A knob at its default reproduces the pre-knob behaviour.

### 2.1 Pure-torque / PD scaffold

| Knob | Default | Meaning |
|---|---|---|
| `PD_OFF_PHASE` | `""` | `all` → `pd_alpha_t = 0`, `_pd_w = 0`. The strongest pure-torque pin. |
| `LOAD_PD_SCALE` | `1.0` | `0.0` pins `_load_pd_scale` at 0. The fade only decreases, so it stays. |
| `PD_FADE_SCOPE` | `load` | `all` extends the fade (and the `PD_FLOOR` floor) to every phase. |
| `PD_FLOOR` | `0.0` | Floor the PD ratio fades to. `0.2` = keep 20% PD. |
| `SKIP_PD_FADE` | unset | **Mandatory on resume.** Starts `step_count` at `x0+1`. |
| `LOAD_FADE`, `_TIMED`, `_STEP`, `_HOLD`, `_GUARD` | `0,0,0.05,20000,0.0` | The fade ladder. `_GUARD` latches once and then goes inert. |
| `PD_TARGET_OBS` | `0` | Puts the 12 PD targets into the observation (+12 extras). |
| `GROWTH_WARMUP`, `GROWTH_X0` | `19200`, `70400` | Gompertz torque/frequency ramp, in **steps**, not iterations. |

### 2.2 Command band and curriculum

| Knob | Default | Note |
|---|---|---|
| `DX_MIN`, `DX_MAX` | `0.0`, `1.0` | `landing_disp_x_stage2 = [DX_MIN, DX_MAX]`, a hard cap. |
| `DY_MIN`, `DY_MAX` | `-0.30`, `0.30` | **Always on.** Every aggregate hit rate is an average over oblique commands. |
| `OMNI_DEG`, `OMNI_Y_SCALE` | `0`, `1.0` | Polar command sampling. |
| `DX_FLOOR_CURR`, `STANCE_CURR`, `STANCE_RAMP` | `1`, —, `60000` | Curricula. |

> **The command band is the biggest distance lever found so far** (confirmed three times), and the
> single most common way to break a run. The reference table covers `cmd_x` **0.30–0.90 in 61 bins
> and clamps outside it**: a run at `[0.5, 1.2]` draws a **mismatched reference on 42.9%** of its
> samples versus 14.3% at `[0.3, 1.0]`, and that is the measured cause of one run never learning to
> stop re-lifting after landing (100% re-lift at 20k iterations, body yaw drifting
> +13.6° → +22.9° → +40.4°). **Keep `DX_MAX` inside the table** unless you extend the table.
> Separately, feeding commands beyond actual reach teaches a conservative takeoff: one run lost
> `dx` 0.9 and 1.0 entirely (hit 0.00) while a sibling at the same condition did 0.94.

### 2.3 Reward weights, grouped by the phase they are alive in

| Phase | Knobs (default) |
|---|---|
| **Pre-jump stance** | `STAND_POSE_W` (0), `STAND_STILL_W` (0), `STAND_TRACK_W` (0) + `STAND_TRACK_H` (0.303) / `_LAND` (1) / `_LSC` (30), `STAND_NO_TO_W` (0), `STAND_ANCHOR` (0) |
| **Arming gate** | `STAND_ARM_H0` (0), `_H` (0), `_HSTART` (0), `_STEP` (0.01), `_HOLD` (12000), `_GUARD` (0.85), `JUMP_DELAY` (120), `JUMP_SETTLE` (10) |
| **Load / squat** | `SQUAT_THR` (3.2), `SQUAT_HOLD` (25), `SQUAT_PAY_HELD` (1), `REQ_SQUAT` (1), `STANCE_SQUAT_W` (8.0) |
| **Takeoff / flight** | `PEAK_W` (20.0), `PEAK_ONESIDE` (0), `OMEGA_GATE` (0.80) |
| **Landing** | `SUCC_W` (1500), `SUCC_DIST_W` (1200), `SUCC_LAND_SIGMA` (0), `SUCCDIST_CAP` (0), `CLEAN_LAND_W` (0), `LAND_STAB_W` (1.0), `LAND_STAB_CONTACT` (0), `LAND_IMPACT_W` (0), `LAND_POSE_W` (0), `LAND_STAND_MIN_H` (0) |
| **Post-landing settle** | `SETTLE_DRIFT_W` (0), `SETTLE_POSE_W` (0), `SETTLE_ABSORB` (15), `DEFAULT_POS_REWARD` (0) + `_SIGMA` (3.0) + `_PEN_SCOPE` (all) |
| **Yaw** | `YAW_TRACK_W` (0), `YAW_TRACK_SIGMA` (0.25), `YAW_RATE_W` (0), `YAW_OBS` (0) |

### 2.4 Commanded reference

| Knob | Default | Note |
|---|---|---|
| `REF_FILE` | `""` | Falls back to `resources/references/commanded_ref_14700.npz`. |
| `REF_PREVIEW_OBS` | `0` | Adds `12 × len(REF_PREVIEW)` extras. With `(0,10,30)`: +36 → actor input **1060**. |
| `REF_PREVIEW` | `0,10,30` | Preview horizons in steps. |
| `REF_PREVIEW_IDLE_MASK` | **`0`** | See below. |
| `REF_JOINT_W`, `REF_VEL_W`, `REF_ORI_W`, `REF_BASE_H_W` | all `0.0` | Tracking-reward arm. |
| `REF_SIG_*` | 0.02–2.0 | Tracking sigmas. Base tight, joints loose, landing loosest. |
| `REF_SLOT_RADIUS`, `REF_CMD_MAX`, `REF_EXT_FIT_SPAN` | `0`, `0`, `0.14` | Table indexing. |

> **`REF_PREVIEW_IDLE_MASK=0` is an asymmetry, not a neutral default.** `_update_ref_clock`
> (`..._torque.py:1298-1301`) sets `ref_mode=0, ref_k=0` for `idle = ~jumping_state`, so with the
> mask off the three horizons sit **frozen on preparation slots 0 / 24-28 / 73-85 of 200** — a
> constant "be lower than default" vector fed to the actor for the whole stance window. The
> **tracking rewards** have always been masked by `(ref_mode > 0)` ("a standing robot is never
> scored against a preparation pose"); the **observation** never was.
> Measured (training-level, two runs): stance height **0.222 → 0.245 → 0.270 ascending** with the
> mask, versus **0.291 → 0.274 → 0.219 descending** without it.

### 2.5 Domain randomisation

| Knob | Default | Note |
|---|---|---|
| `DR_MASS_LO`, `DR_MASS_HI` | `-1.0`, `1.0` | Symmetric about 0. See traps 3 and 4. |
| `DR_LINK_MASS` | unset | **Set it to `0.9,1.1`.** Unset selects the additive legacy loop, which draws **negative** masses: a `calflower` (0.040 kg) about 18% of the time, each `Head` link (0.001 kg) about 49%. |
| `DR_COM_X` | `0.05` | Y and Z have **no knob**; they inherit `[-0.1, 0.1]`. |
| `DR_FRIC_LO`, `DR_FRIC_HI` | `0.5`, `1.25` | |
| `DR_PUSH` | **`0`** | Pushes are **off** by default, so they contribute nothing to any historical result. |
| `GRF_OBS` (+`_CLIP_N`, `_NOISE`, `_BW_N`) | `0` | Ground-reaction-force observation. |

### 2.6 Observation geometry

`frame(49) × 20 + extras`, `num_single_extras = 32 + 12·PD_TARGET_OBS + 12·len(REF_PREVIEW)`.
With `PD_TARGET_OBS=1` and `REF_PREVIEW=0,10,30`: extras 80 → **actor 1060**. Critic is asymmetric
at **507**. A checkpoint will not load if these do not match, which is the fast way to notice a
missing export.

---

## 3. Running things

### Training (server)

Every launch script needs this preamble. `pathfix` is **not optional**: without it the shared env's
editable `legged_gym` / `rsl_rl` installs resolve to **another user's tree**, and the run completes
with no error and invalid results.

```bash
cd <repo>/legged_gym
source <env>/env.sh 2>/dev/null || true
P=<env>/envs/sata
export PATH=$P/bin:$PATH LD_LIBRARY_PATH=$P/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export PYTHONPATH=<env>/pathfix:$PYTHONPATH     # MANDATORY
export TORCH_EXTENSIONS_DIR=<env>/torch_extensions_sata
```

Then `wandb login`, and launch inside `tmux` (`ssh host 'cmd &'` hangs).

**A/B discipline, learned the expensive way** (one missing block wasted 11.5 h and produced a
reported result that was wrong): before launching a comparison, **diff the launch scripts
programmatically with comments stripped, including the preamble** — the preamble is not boilerplate.
Copy the experiment block **verbatim** from the control and take the preamble from the most recent
*successful* run. After launch, verify the curriculum scalars in tensorboard start where you expect.

### Resuming

`on_policy_runner.py:372` restores `current_learning_iteration`; `:127` loops
`range(current, current + max_iterations)`; `:149` sets `_resumed=True`, so the entropy anneal fires
on the **first** iteration. Plus `SKIP_PD_FADE=1` and `LOAD_PD_SCALE=0.0` from trap 1.

Two measured costs:
* **One PPO update past the parent already costs a lot.** A child's `model_20000` (one update past
  the parent's 0.814) evaluated **0.36**, and took ~2000 iterations to climb back.
* **Checkpoint `iter` fields do not survive a chain of resumes.** A run resumed from iteration 20000
  wrote `iter=20000` into *all* of its checkpoints, so resuming from its `model_23500` restarts the
  counter at 20000. The weights are correct; the filenames are not informative.
  **Always identify a checkpoint by run directory + filename, never by number alone.**

### Deterministic evaluation — the only thing a cross-run verdict may rest on

```bash
export PD_OFF_PHASE=all                      # trap 1
export EVAL_DX_SWEEP=0.5,0.6,0.7,0.8,0.9,1.0 # trap 2 -- record this with the result
export EVAL_HEIGHT=0.5 EVAL_N=256 EVAL_STEPS=400
export EVAL_DR_MASS=1                        # trap 3
export SCAN_START=... SCAN_END=... SCAN_STEP=...   # trap 5: 10+ points
SCAN_RUN=<run> python legged_gym/scripts/eval_ckpt_scan_dr.py \
  --task=go2_omnijump_landing_torque --headless --load_run=<run> --checkpoint=<n>
```

`--headless` always. `EVAL_DR=1` keeps every axis; `EVAL_DR_<AXIS>=1` keeps one
(`MASS`, `FRICTION`, …); `EVAL_ADD_MASS=<kg>` pins the mass offset (trap 4).

### Play (local, with a GUI)

`play_landing.py` overrides: `num_envs=1`, `episode_length_s=4`, `jump_command_range=[1,1]`,
`add_noise=False`, friction and push off, `landing_tilt_terminate=0.0`, `SPAWN_Z=0.35`.

| Knob | Default | Note |
|---|---|---|
| `PLAY_LANDING_DX` | **`1.2`** | Above both the usual `DX_MAX=1.0` **and** the reference table's 0.90 clamp. **Pin it explicitly.** |
| `PLAY_LANDING_HEIGHT` | `0.5` | Trained range `[0.4, 0.6]`; 0.7 is an OOD false collapse. |
| `PLAY_LANDING_ADDED_MASS` | `0.0` | `0.0` sets `randomize_base_mass = False`. Non-zero opens the **whole** DR block (COM point mass + per-link term), so it is not a clean mass ablation. |
| `PLAY_LANDING_TRACE` | `0` | **Use it.** One line per *episode*, including episodes that never took off. |

> The console's own hit counter is **conditioned on having landed**. It printed `hit=0.75` for a
> session whose true per-episode rate was **3/17 = 0.18** (9 never armed, 4 never took off).
> `PLAY_LANDING_TRACE=1` exists because of this.

### sim2sim (MuJoCo)

Run **without** `PYTHONNOUSERSITE=1`: mujoco 3.2.3 lives only in `~/.local`.

**Not a valid acceptance gate for the history/stacked family.** Every interface item was verified
item-by-item; the residual **−46%** is engine-level trajectory divergence (18.6% apart within 50 ms
under identical torques). The flat `4600` policy does transfer and matches to within 3.7% / 9%.

---

## 4. State machine, in code

```
stand  --(first_jump_delay_steps=120, then contact4 & calm held jump_settle_steps=10)-->  armed
armed  --(cmd4 > 0.5)-->  loading     [ref_mode 0 -> 1, ref_k = 0]
loading --(_squat_pose_err() <= SQUAT_THR for SQUAT_HOLD CONSECUTIVE steps)--> squat_qualified (latched)
       --(just_took_off & squat_qualified)-->  airborne  [ref_mode 2, ref_k = 0]
airborne --(descending below prelanding_height)-->  prelanding  -->  landing
landing  --(landing_step_counter >= landing_buffer_steps = 150, i.e. 0.75 s)-->  _finish_jump
_finish_jump: jumping_state = False  ->  stand   [ref_mode 0, ref_k 0, tracking rewards off]
```

Three things worth knowing before you tune anything:

**The takeoff gate is a 25-fold product, not a threshold.** `go2_omnijump_torque.py:667-673`:

```python
in_pose = loading & (self._squat_pose_err() <= _hold_thr)          # SQUAT_THR = 3.2
self.squat_hold_counter = torch.where(in_pose, counter + 1,
                              torch.where(loading, 0, counter))     # resets the instant it pops out
self.squat_qualified |= (self.squat_hold_counter >= 25)             # SQUAT_HOLD
```

`_squat_pose_err` (`:1181`) is `Σ|dof_pos − q_squat_target|`, the L1 distance to the **squat** target,
not to the default stance. Because the counter resets on any miss, the pass rate behaves like
`p**25`: `p = 0.90` → 7%, `0.95` → 28%, `0.99` → 78%. **A few percent of margin changes the pass
rate several-fold**, so this gate is the sharpest discontinuity in the whole system and the first
suspect whenever a result is bimodal or a command bin reads `nan`.

**Posture pressure exists only inside the landing window.** `_reward_pitch_level` (landing task
`:2377-2389`):

```python
r = self.jumping_state.float() * nose_down                                  # one-sided
r = r + 5.0 * (self.prelanding | self.landing).float() * level              # SYMMETRIC, 5x
r = r + 12.0 * self.airborne.float() * nose_down                            # latched
```

In the stance window `jumping_state` is False and none of `prelanding`/`landing`/`airborne` hold, so
this term is **exactly 0 there** and `5.0 ×` and **symmetric** inside the landing window. That is why
the same policy straightens itself out after touchdown and does not stand up before the jump.

**Termination:** `terminate_after_contacts_on=["base"]` with `collision_cutoff = Σ|F| > 0.2 N`;
`roll_cutoff = 2.4 rad`; `landing_tilt_terminate = 0.0` (off); `one_jump_reward_per_episode=True`
with `post_jump_stand_steps = 300`.

**Latched per-env state that is NOT in the checkpoint** — `_takeoff_omega_on`, `_load_hit_ema`,
`_arm_hit_ema`. Offline it comes back `False`, which (a) is the head suspect for play ≠ train, and
(b) means every reward gated behind it (`stand_pose`, `land_stand_min_height`, `squat_gate_floor`,
the takeoff-omega damping) is **inert in play and eval**. A guard that latches once and is then
multiplied by a ratio goes permanently inert the same way (`LOAD_FADE_GUARD`, `STAND_ARM_GUARD`).

---

## 5. The checkpoints in `checkpoints/`

Three checkpoints are committed to the repo (~23 MB total, so a fork gets them automatically —
Release assets would not be copied by a fork). **A checkpoint will not load unless the observation
geometry matches**, so the required exports are listed with each one. `num_single_extras =
32 + 12·PD_TARGET_OBS + 12·len(REF_PREVIEW)`, and `frame(49) × 20 = 980`, hence the actor sizes
below.

### `refprev2_25k_s1_model_23500.pt` — the current best, actor **1060**

From `Oct27_13-39-06_refprev2_25k_s1`, itself a +5000-iteration continuation of
`Oct07_20-24-02_refprev2_s1` `model_20000`. This is what `thesis-2026-10-28` is tagged for.

```bash
export GROWTH_WARMUP=96000 GROWTH_X0=192000 PD_FLOOR=0.2
export DX_MIN=0.3 DX_FLOOR_CURR=0 STAND_ANCHOR=1 STANCE_CURR=0
export PEAK_W=20 SUCC_LAND_SIGMA=0.15 SUCCDIST_CAP=1
export LOAD_FADE=1 PD_FADE_SCOPE=all LOAD_FADE_TIMED=1
export LOAD_FADE_STEP=0.1 LOAD_FADE_HOLD=12000 LOAD_FADE_GUARD=0.6
export PD_TARGET_OBS=1 STAND_POSE_W=3.0 STAND_STILL_W=6.0
export REF_PREVIEW_OBS=1 REF_PREVIEW_IDLE_MASK=0
export PD_OFF_PHASE=all          # evaluation only -- trap 1
```

Measured (N=256, sweep 0.5–1.0, `PD_OFF_PHASE=all`):

| | mass DR | exact nominal | pre-jump stance |
|---|---|---|---|
| ck23500 | **0.954** (minDead 0.91, span 5/5, airPeak 0.947) | **0.55** ← trap 3 | 0.203 m |
| ck22500 | 0.919 | **1.00**, and 1.00 at every mass point from −0.10 to +0.25 kg | 0.198 m |
| parent ck20000 | 0.814 (minDead 0.72) | — | 0.219 m |

**For hardware, prefer `ck22500`** — it has no dead point at exact nominal, which is where a real
robot sits. ck23500 is the better number; ck22500 is the safer one. (ck22500 is not committed here;
it is `model_22500.pt` of the same run.)

### `sstill6_s1_model_19500.pt` — the `stand_still` working point, actor **1024**

From `Oct04_09-40-53_sstill6_s1`. Same block as above **minus `REF_PREVIEW_OBS`/`_IDLE_MASK`**
(hence 1024 = 980 + 32 + 12), with `STAND_STILL_W=6.0`.

`stand_still` is the term that first moved the pre-jump stance at all: **0.228 → 0.301 m**, at the
cost of hit 0.81 → 0.65, and `W=6` at iteration 19500 was the best compromise found (`W=2` collapsed
late-run `settled` from 100% to 9%). Against the paper baseline, per-command net flight was better at
**every** command, best 0.905 vs 0.845 m (+7.1%), and `settled` went 82% → 98%; the baseline's
"stable landing frontier at 0.80–0.90" disappears (cmd 0.90: 0.72 → 0.97).
Note `stand_still` is a **velocity brake**, not a posture target — see §5's first dead end.

### `sstill6_s1_model_14700.pt` — the reference-trajectory source, actor **1024**

Same exports as `..._19500.pt`. This is the checkpoint
`resources/references/commanded_ref_14700.npz` was extracted from, i.e. the default reference table
every `REF_*` run tracks. Keep it: without it the table cannot be re-derived or extended.

What the extracted trajectory is: a **countermovement jump with a free-fall phase** — unload, free
fall 0.22 m, squat bottom at z = 0.079 m against 786 N (5.3 BW), launch |v| = 2.90 m/s at 49°.
Cross-jump aligned and averaged, the joint trajectory has ρ₁ = +0.98 (analytically smooth) and beats
the medoid single jump on every term.

⚠️ **Two structural flaws in that table, recorded so nobody rediscovers them.** It **leans on
mechanical limits** — the front calf is pinned at −2.7227 at the squat bottom and the rear calf at
−0.8378 through flight, penetrating the hard limit by 0.07 rad — so it is **not directly
reproducible on hardware**. And the squat bottom has only the **two rear feet** in contact. It also
yields no admissible trajectory at cmd 1.00, which is why the table stops at 0.90 (trap 2 / §2.2).

---

## 6. Dead ends — verified, do not redo these

Each line cost days. The point of this section is that you spend yours elsewhere.

**Stance posture cannot be bought with reward weight.** The arithmetic: `stand_pose` is **0.85%** of
the reward-term total and is latched behind `_takeoff_omega_on`; `stand_still` is 9.21% but is a
**velocity brake**, so a slow droop that settles collects it in full; `orientation` (1.2–1.7%) is the
only live pitch pressure and weights roll and pitch equally. The code's own estimate is that
**≤ 4% of the return is at stake** in the stance window, against `forward_reach` 15.2% +
`projected_landing` 14.1% + `landing_position` 13.7% + `successful_distance` 5.8% = **48.8%** that a
forward lean helps. Posture is the cheapest currency the policy has.
*What did work:* the **hard gate** `STAND_ARM_H0` — stance 0.219 → **0.309 m**, above the 0.303
default, at the cost of 11 points of training-time hit. And `REF_PREVIEW_IDLE_MASK=1` (§2.4).

**Weighting `landing_stability` up failed** — it saturates. `settled` gained 3 points while the far
end lost **82%**.

**Strengthening `takeoff_velocity_match`, the only causal-side term, failed.** It is capped at 1.7%
of the return and differs by only 0.04 between two runs; the limit is the **knee velocity wall**,
not the reward. Relatedly: `forward_reach` staying flat does **not** mean distance did not improve —
the gain can be entirely outside the scored band.

**Stacked observation history kills from-scratch discovery in pure torque.** Two independent
implementations both failed to find the jump (squat identically 0, reward stuck at 0.5 vs 20 for
flat). Reward was byte-identical, observation content verified identical, the critic fitted fine.
The paper this came from uses **PD position control**, where discovery is easy. History is useful for
*refinement*, not for discovery — warm-start it.

**Penalising the landing hard buys "don't jump".** Takeoff rate by command went
`0.5: 128/128 → 0.7: 66/128 → 0.8: **6/128**` while a sibling run on the same harness was full at
every command. **The training curve does not show this.** Any landing-penalty change must report
per-command takeoff rate alongside hit rate.

**The `main` branch ends on a measured regression.** Its tip `0157dfb` (separate jump/recovery
episodes + RSI drill + combat squat, 2026-06-26/27) was diagnosed by re-training from `6c4032b`:
squatQ **0.87** / dx 1.1 / hit **0.81** against `0157dfb`'s **0.29 / 0.40 / 0.56**. **Do not build on
`main` and do not merge it.** The line of work is on the branch that carries this file.

**Two sim artifacts that look like policy problems.** The SATA URDF truncates the thigh range to
`[0, 1.5]` (official: `[-1.5708, 3.4907]`) and the policy **leans on the fake mechanical stop** for
free reaction force — overridden in code via `cfg.asset.official_dof_pos_limits`, but any older
checkpoint learned against the wall. And the stance-phase joint chatter is a T-N
centripetal/centrifugal branch-switching artifact (15.8% hard jumps, a second histogram peak at
0.158, 12% of steps), **not** policy noise; `action_rate` is 0.09% of the penalty total and is not
buying smoothness.

**Isaac Gym is not reproducible run to run.** Same machine, card, seed and code: bitwise identical
for ~532 iterations, then `atomicAdd` ordering crosses a rounding boundary and PPO amplifies it.
Different hardware diverges at iteration 0. Seeds do not control GPU atomic ordering. Measured
same-config seed spread: **0.67–0.85 m**. **A single run never decides anything** — 2–3 seeds for a
reward change, and judge on deterministic eval, never on `dx_max` from the training curve.

---

## 7. Before you change a reward

In this task's history, the measured outcome of weight changes was negative or null far more often
than positive, while **structural changes (hard gates, command band, observation masking) carried
the wins.** Before touching a weight:

1. Measure the term's **share** of the reward-term total (not of `mean_reward` — they differ by ~10×)
   and the fraction of steps it is alive. If the ceiling is a few percent, weight cannot fix it.
2. Check what it is competing with. A term that must outbid ~49% of the return will not.
3. Check whether it **saturates** — `landing_stability` did.
4. Check what the policy's cheapest route to the new reward is. Usually it is "do less of the task"
   (don't jump, don't move) rather than "do the task better".
5. Prefer a **gate** over a weight: a gate puts the whole downstream return behind a condition
   without bidding against anything and without saturating.
6. Run 2–3 seeds, then a 10+ point deterministic scan. Report the DR condition and the sweep.
