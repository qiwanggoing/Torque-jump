"""Clean landing-point play — mirrors the VALIDATED batch eval.

Background: the legacy play.py drives the jump task through a TEST-mode state machine
(forces commands[:] = test.vel every step, KILLS cmd[4] at takeoff, single_jump_play
re-arm, etc.). That path DIVERGES from how the policy was trained -> in play the robot
lands short / topples, even though the SAME checkpoint scores landing_hit_rate ~0.98 at
dx=0.40 in a faithful, normal-mode rollout (256-env batch eval).

This script drives the env exactly the way training / the batch eval do:
  - NORMAL command flow (use_test = False), env fires the jump itself (first_jump_delay,
    single-jump logic) and holds cmd[4] through the jump,
  - fixed forward command via the command ranges,
  - PD prior fully faded (pd_alpha = 0, the iter-10000 trained condition),
  - deterministic (mean) policy.
plus a viewer + per-landing readout (fwd displacement, landing error vs target, hit/miss).

Edit DX / DY / HEIGHT below to test different commands. Run:
  python legged_gym/scripts/play_landing.py --task=go2_omnijump_landing_torque
"""
import isaacgym
from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        print(f"[play_landing] invalid {name}={value!r}; using {default}", flush=True)
        return float(default)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def train_step_count_at_iter(target_iter, W, X, sf, mf, dt, nspe):
    """Reproduce TRAINING's step_count at iteration N (for faithful PD/general_scale replay).

    step() runs ~1/(dt*freq) physics substeps per env.step, each doing step_count += 1, and
    current_freq ramps sf->mf as general_scale goes 0->1 LINEARLY over step_count in [W, X]
    (warmup_steps W, x0 X). So step/iter = nspe/(dt*freq) is NON-LINEAR: 96 during warmup (freq=sf)
    down to 48 at pure torque (freq=mf). The old play used LINEAR nspe*N -> ~2x off in warmup/fade
    -> replayed a wrong (PD-heavier) general_scale for fade-era ckpts. This closed form integrates
    the ramp; verified vs the measured x0=48000 -> fade completes ~iter650. Pure-torque ckpts (past x0)
    were already correct (both linear and this land past x0 -> general_scale=1), so only fade ckpts change.
    """
    r0 = nspe / (dt * sf)                 # step/iter during warmup (freq = sf), e.g. 48/(0.005*100)=96
    iw = W / r0                           # iteration at which warmup ends (step_count reaches W)
    if target_iter <= iw:
        return r0 * target_iter
    Delta = X - W
    a = (mf - sf) / (2.0 * Delta)
    b = sf
    iter_fade_end = iw + (dt / nspe) * (a * Delta * Delta + b * Delta)   # step_count reaches X here
    if target_iter <= iter_fade_end:
        c = (nspe / dt) * (target_iter - iw)
        u = (-b + math.sqrt(b * b + 4.0 * a * c)) / (2.0 * a)
        return W + u
    r_end = nspe / (dt * mf)              # step/iter during pure torque (freq = mf) = 48
    return X + r_end * (target_iter - iter_fade_end)

# === command to visualise =====================================================
# Defaults are conservative for deterministic/no-noise play. Override without
# editing this file, e.g.:
#   PLAY_LANDING_DX=0.7 PLAY_LANDING_HEIGHT=0.7 python legged_gym/scripts/play_landing.py --task=go2_omnijump_landing_torque
DX = _env_float("PLAY_LANDING_DX", 1.2)       # forward landing displacement (m). 新模型准确到 0.8; 0.9+ 够不到. 改数字试别的.
DY = _env_float("PLAY_LANDING_DY", 0.0)       # lateral landing displacement (m)
HEIGHT = _env_float("PLAY_LANDING_HEIGHT", 0.5)  # jump-height command. ⚠️ 新模型训练范围 [0.4,0.6], 别超 0.6 (0.7=OOD假崩)
ADDED_MASS = _env_float("PLAY_LANDING_ADDED_MASS", 0.0)  # kg added to BASE. 0.0 = URDF标称 = 真机 = 新模型([-1,+1])训练中心.
                    # ⚠️ 新模型必须 0.0! 给 +2kg 是 OOD 会原地蹦假崩. (只有旧 [-1,+5] 模型才该设 2.0.)
STAND_ONLY = _env_bool("PLAY_LANDING_STAND_ONLY", False)  # PURE-STAND test: command cmd4=0.45 (<=0.5 threshold -> NEVER jumps) + zero displacement,
                    # so the robot is told to just stand quietly at spawn (landing_target=spawn -> err=0).
                    # Watch how stable the stand is / how often it twitches a foot off. Set False for normal jumps.
NO_RSI = _env_bool("PLAY_LANDING_NO_RSI", True)  # force rsi_prob=0 during play. RSI is a TRAINING exploration
                    # mechanism (air-drops a fraction of resets into a squat/mid-launch state); in play it makes the
                    # single viewed robot RANDOMLY spawn in flight -> confusing. Default True = clean deterministic
                    # spawns (normal stand->jump). Set PLAY_LANDING_NO_RSI=0 to SEE the RSI air-drops (e.g. far task).
# ====================================================================================


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 4          # ~1 jump per episode, paced for viewing

    # Fixed forward landing command — fed through the env's NORMAL command ranges so the
    # jump fires and evolves exactly as in training (no test-mode override / state machine).
    env_cfg.commands.resampling_time = 20.0   # hold the FIRST command for the whole episode -> no mid-episode resample
                                              # that flips STAND->JUMP without updating landing_target (= the in-place-hop bug)
    env_cfg.commands.landing_dx_curriculum = False
    env_cfg.commands.landing_disp_x_stage2 = [DX, DX]
    env_cfg.commands.landing_disp_y_stage2 = [DY, DY]
    env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
    env_cfg.commands.ranges.jump_command = [1.0, 1.0]    # (legacy key)
    if STAND_ONLY:
        env_cfg.commands.jump_command_range = [0.49, 0.49]   # cmd4 always 0.45 <= 0.5 -> NEVER fires a jump (pure stand)
        env_cfg.commands.landing_disp_x_stage2 = [0.0, 0.0]  # landing_target = spawn -> zero landing error
        env_cfg.commands.landing_disp_y_stage2 = [0.0, 0.0]
    else:
        # play: EVERY episode is a JUMP (cmd4=1), NO stand episodes -> strictly follow the DX jump command.
        # The old warning about [1,1] "never stops -> drifts" NO LONGER applies now the single-jump setup is
        # restored: disable_jump_on_landing=True force-stands cmd4=0 in the landing buffer + single_jump_play=True
        # resets after ONE jump -> jump once -> land -> stand -> reset -> repeat (every play episode is a jump).
        env_cfg.commands.jump_command_range = [1.0, 1.0]
    # Clean conditions (the eval that scored 0.98 used these; isolates the policy).
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # Base mass: default nominal (URDF = real robot). PLAY_LANDING_ADDED_MASS=2.0 pins +2kg to reproduce
    # an OLD-DR ([-1,+5], mean +2kg) model's train performance (it undershoots at nominal). See mass diag.
    if abs(ADDED_MASS) > 1e-6:
        env_cfg.domain_rand.randomize_base_mass = True
        env_cfg.domain_rand.added_mass_range = [ADDED_MASS, ADDED_MASS]
    else:
        env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.rewards.landing_tilt_terminate = 0.0   # MEASURE mode: disable the tilt-reset so we see the TRUE max nose-down
    train_cfg.runner.resume = True

    # PD-replay: auto-read the loaded checkpoint's iter N so the env replays THAT iter's PD ratio
    # (general_scale -> pd_alpha), instead of forcing pure torque. We pin step_count = num_steps_per_env
    # * N EVERY STEP (in the rollout below) -> same mechanism as the old BIG override, but at the
    # checkpoint's own general_scale. We do NOT set test.use_test: with control_type='TG' that's the only
    # thing that would trigger the growth-replay, BUT it ALSO force-overwrites commands[:]=test.vel every
    # step -> that's what killed the DX command (REALcmd0 came out 0).
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    _load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    _checkpoint = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    _resume_path = get_load_path(log_root, load_run=_load_run, checkpoint=_checkpoint)
    ckpt_iter = int(os.path.basename(_resume_path).replace('model_', '').replace('.pt', ''))
    # REPLAY_STEP (= train's step_count at ckpt_iter) computed AFTER make_env — needs env.dt/start_freq/max_freq.

    # ARCHITECTURE FROM THE CHECKPOINT (2026-09-24). The layer widths and the critic's history length are
    # written into the weight shapes, so read them from the file instead of relying on the caller to pass
    # the matching NET / C_STACK. Getting either wrong used to fail deep inside load_state_dict with a wall
    # of size-mismatch errors; now a variant checkpoint just plays.
    try:
        import torch as _t
        _sd = _t.load(_resume_path, map_location='cpu')['model_state_dict']

        def _dims(prefix):
            idx = sorted({int(k.split('.')[1]) for k in _sd if k.startswith(prefix + '.') and k.endswith('.weight')})
            return [int(_sd['%s.%d.weight' % (prefix, i)].shape[0]) for i in idx[:-1]], \
                   int(_sd['%s.%d.weight' % (prefix, idx[0])].shape[1])
        _a_dims, _a_in = _dims('actor')
        _c_dims, _c_in = _dims('critic')
        if list(train_cfg.policy.actor_hidden_dims) != _a_dims:
            print(f"[play] actor hidden dims {list(train_cfg.policy.actor_hidden_dims)} -> {_a_dims} (from checkpoint)")
            train_cfg.policy.actor_hidden_dims = _a_dims
        if list(train_cfg.policy.critic_hidden_dims) != _c_dims:
            print(f"[play] critic hidden dims {list(train_cfg.policy.critic_hidden_dims)} -> {_c_dims} (from checkpoint)")
            train_cfg.policy.critic_hidden_dims = _c_dims
        _per = int(getattr(env_cfg.env, 'single_num_privileged_obs', 0))
        if _per > 0 and _c_in % _per == 0:
            _cs = _c_in // _per
            if int(getattr(env_cfg.env, 'c_frame_stack', 0)) != _cs:
                print(f"[play] c_frame_stack {env_cfg.env.c_frame_stack} -> {_cs} (from checkpoint)")
                env_cfg.env.c_frame_stack = _cs
                env_cfg.env.num_privileged_obs = _cs * _per
        if _a_in != int(env_cfg.env.num_observations):
            print(f"[play] WARNING: checkpoint actor expects {_a_in} observations, env builds "
                  f"{env_cfg.env.num_observations} -- this one really is a different task/layout")
    except Exception as _e:
        print(f"[play] could not read the architecture from the checkpoint ({_e}); using the config as-is")

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic   # Step H: need comp_forward() to feed env.comp_torque (τ_comp)

    # PD-replay step_count: reproduce TRAINING's NON-LINEAR step_count at ckpt_iter (freq ramp 96->48/iter),
    # NOT the old linear 48*iter (which replayed too-much PD for fade-era ckpts). Pinned every rollout step below.
    REPLAY_STEP = int(round(train_step_count_at_iter(
        ckpt_iter,
        float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
        float(env.start_freq), float(env.max_freq),
        float(env.dt), int(train_cfg.runner.num_steps_per_env))))
    # MUST match the env's own formula, INCLUDING pd_fade_floor (PD_FLOOR): without the floor term this
    # print said "pd_alpha=0.000" for a PD_FLOOR=0.2 model that is actually running at 0.100, which reads
    # exactly like "the PD scaffold is off" when it is not.
    _gs = min(1.0, max(0.0, (REPLAY_STEP - float(env.cfg.growth.warmup_steps))
                       / max(1.0, float(env.cfg.growth.x0) - float(env.cfg.growth.warmup_steps))))
    _floor = max(0.0, min(1.0, float(getattr(env.cfg.growth, "pd_fade_floor", 0.0))))
    _pd_a = float(env.cfg.control.pd_prior_weight) * max(_floor, 1.0 - _gs)
    # pd_alpha is TWO-STAGE and printing only the first stage is the single most expensive trap in
    # this project (hit 5 times). Stage 1 = pd_prior_weight * max(PD_FLOOR, 1-general_scale), the
    # global fade -- that is _pd_a above. Stage 2 = _mult in go2_omnijump_curriculum_torque's
    # _compute_torques: LOAD_PD_SCALE (the closed-loop load-phase withdrawal) composed with the
    # PD_OFF_PHASE mask. LOAD_PD_SCALE DEFAULTS TO 1.0 in a fresh process, so a run trained with
    # LOAD_FADE + PD_FADE_SCOPE=all (whose endpoint really is pd_alpha=0.000) gets the full stage-1
    # PD handed back unless the eval sets it to 0 -- and the old one-number print was identical
    # either way, so it could not tell the two apart. Print the composition instead.
    _lf = float(os.environ.get("LOAD_PD_SCALE", "1.0"))
    _offp = str(os.environ.get("PD_OFF_PHASE", "")).strip().lower() or "none"
    _applied = _pd_a * _lf * (0.0 if _offp == "all" else 1.0)
    _flag = ("" if _applied <= 1e-9 else
             "  !! PD IS ON -- set LOAD_PD_SCALE=0.0 PD_OFF_PHASE=all for pure torque")
    print(f"[play] {os.path.basename(_resume_path)} -> iter {ckpt_iter}: step_count={REPLAY_STEP} "
          f"(non-linear fade-correct)")
    print(f"[play] pd_alpha = global_fade {_pd_a:.3f} x load_pd_scale {_lf:.2f} x phase_off '{_offp}'"
          f"  ->  APPLIED {_applied:.3f}" + ("  (pure torque)" if _applied <= 1e-9 else "") + _flag)

    # (removed BIG=500000 pure-torque override — env now replays the checkpoint's own general_scale/pd_alpha)
    init = torch.tensor([float(env.cfg.init_state.pos[0]), float(env.cfg.init_state.pos[1])], device=env.device)
    spawn = env.env_origins[:, :2] + init

    # EPISODE-0 fix: reset once up front so the FIRST episode runs from a CLEAN reset state (not the raw
    # post-build state) -> without this the first jump fails and only after the first auto-reset is it normal.
    env.reset()

    # FIX: the env's __init__ reset drew commands[0]=0 (command_ranges wasn't [DX,DX] yet at that early
    # reset) and resampling_time=20 holds it ALL episode -> the robot was getting an in-place (dx=0)
    # command, not DX. Force the viewing command now + refresh landing_target so it truly jumps DX forward.
    if not STAND_ONLY:
        env.commands[:, 0] = DX
        env.commands[:, 1] = DY
        env.landing_target[:, 0] = env.root_states[:, 0] + DX
        env.landing_target[:, 1] = env.root_states[:, 1] + DY

    # Side view of the spawn + landing strip.
    if getattr(env, "viewer", None) is not None:
        sx, sy = float(spawn[0, 0]), float(spawn[0, 1])
        env.gym.viewer_camera_look_at(
            env.viewer, None,
            gymapi.Vec3(sx - 1.6, sy - 1.6, 1.1),
            gymapi.Vec3(sx + 0.4, sy, 0.25),
        )

    # The command/target override above changes observation slots. Refresh once so
    # the very first policy action sees the requested DX instead of reset-time obs.
    env.compute_observations()
    obs = env.get_observations()
    _mass_note = f"added_mass={ADDED_MASS:+.1f}kg" if abs(ADDED_MASS) > 1e-6 else "nominal mass (URDF)"
    print(f"[play_landing] cmd dx={DX} dy={DY} height={HEIGHT} | {_mass_note} | normal jump flow", flush=True)

    import math
    pg2deg = lambda v: math.degrees(math.asin(max(-1.0, min(1.0, float(v)))))  # projected_gravity_x -> tilt deg
    n_jump = n_hit = 0
    max_nd = 0.0   # worst NOSE-DOWN pitch (projected_gravity[:,0], >0 = nose-down) during this jump's flight+landing
    resets_since = 0   # episode resets since the last logged landing (so we know if a reset happened between jumps)
    STAND_CMD0 = False      # while STANDING (not jumping, cmd4<=0.5), force cmd4 to a CLEAR 0.0 (vs the 0.45 the env
                            # pins, which sits just under the 0.5 jump threshold). Tests if the stand-hop is the
                            # policy hesitating at a near-threshold command. Does NOT touch landing_target.
    DISABLE_JUMP_ON_LAND = False  # 单跳只练跳(user): 落地不 force cmd4=0, 跟训练 disable_jump_on_landing=False 一致.
    DEBUG_HOP = False      # True = print the per-step feet-off / per-episode reset debug (hop diagnosis).
                           # False = clean output: just the per-jump [land] line (peak height + forward reach).
    # LEVEL, not a flag. 0 = off (default) = clean output, just the per-jump [land] line.
    # 1 = phase transitions + the only post-landing events that carry information: the FIRST
    #     all-four-planted and EVERY all-feet-off (relift, the thing landing cleanliness is
    #     actually read from). The raw per-step contact permutation flickers 25+ times while
    #     landing, which buried the phase line it was there to annotate.
    # 2 = every contact-permutation change (the old level-1 behaviour).
    _mc = os.environ.get("PLAY_LANDING_MONITOR_CONTACT", "").strip().lower()
    MONITOR_CONTACT = 2 if _mc == "2" else (1 if _mc in ("1", "true", "yes", "on") else 0)
                           # Set PLAY_LANDING_MONITOR_CONTACT=1 to also dump the FULL per-foot contact + height + drift + fwd vel
                           # EVERY step after landing (the post-landing slide/hop diagnosis).
    n_flights = 0          # all-feet-off phases this episode (1 = clean single jump; 2+ = extra in-place hop)
    feet_off_prev = False
    ep_len = 0             # python-side per-episode step counter (env.episode_length_buf is already 0 by reset-print time)
    drift_dx = drift_dy = max_drift = 0.0   # POST-LANDING DRIFT from touchdown point (this episode)
    landed_ep = False      # did the robot land this episode (so the drift line is meaningful)
    pl_steps = pl_all4 = 0   # post-landing: total steps + steps with ALL 4 feet planted (all4_frac~1 => SLIDE,
    pl_min_feet = 4          #   maintain_contact can't catch it; all4_frac low / min_feet<4 => STEP-creep, it can)
    prev_cs = None
    prev_phase = None
    prev_nc = None
    shown_all4 = False
    # ---- PLAY_LANDING_TRACE=1: one line per EPISODE, including the ones that never took off --------
    # WHY. The default output prints ONLY inside `if just_landed`, so n_jump / n_hit count LANDINGS,
    # not episodes: an episode that dies in the stand phase leaves NO line at all and the printed
    # `hit=` is conditioned on having landed. With terminate_after_contacts_on=["base"] (inherited
    # from go2_omnijump_torque_config.py:77) and collision_cutoff = sum|F| > 0.2 N, a nose-down dip
    # that brushes the trunk ends the episode silently -- exactly the failure being looked at, and
    # exactly the one the old accounting could not see. Nothing logged the TAKEOFF process either
    # (arm step, squat bottom, takeoff pitch/velocity), and `max_nd` was computed and then dropped.
    # Default OFF -> the existing output is unchanged.
    TRACE = _env_bool("PLAY_LANDING_TRACE", False)
    n_ep = n_ep_hit = n_ep_noarm = n_ep_noto = n_ep_noland = 0
    tr = {}

    def _tr_reset():
        tr.clear()
        tr.update(armed=False, tookoff=False, squatQ=False, hit=False,
                  arm_step=-1, to_step=-1, bottom=9.9,
                  to_pitch=0.0, to_vx=0.0, to_vz=0.0,
                  stand_nd=-9.9, ep_nd=-9.9)

    _tr_reset()
    _pd_probe = True        # one-shot: report the pd_alpha the env ACTUALLY applied
    # ---- Isaac->MuJoCo PAIRING DUMP (PLAY_LANDING_DUMP=path.npz) ----------------------------------
    # Records the (obs, action) pair the policy ACTUALLY produced, so deploy_mujoco/*.py --replay can
    # feed these exact obs vectors through the same checkpoint and compare actions. A match proves the
    # checkpoint loading AND the observation layout/ordering are right in the MuJoCo port; any jump
    # difference is then physics or the drive logic, not wiring. Without this, the two cannot be told
    # apart -- which is why the Aug 1.9e-6 pairing could not be reproduced (no dump producer existed).
    # obs is taken BEFORE env.step() = exactly what policy() consumed. Single env (idx 0) only.
    DUMP_PATH = os.environ.get("PLAY_LANDING_DUMP", "").strip()
    DUMP_STEPS = int(_env_float("PLAY_LANDING_DUMP_STEPS", 1200))   # 1200 @200Hz = 6 s = a few jumps
    import numpy as _np0
    _dump_obs, _dump_act, _dump_st = [], [], []
    _dump_rs, _dump_dq, _dump_tq = [], [], []
    if DUMP_PATH:
        print(f"[dump] recording {DUMP_STEPS} (obs, action) pairs -> {DUMP_PATH}", flush=True)
    for _ in range(100000):
        # PD-replay: pin step_count to the checkpoint's iter so general_scale/pd_alpha match training.
        # (control_type='TG' -> _update_growth_scale won't override this; only use_test would, which we avoid.)
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        with torch.no_grad():
            actions = policy(obs.detach())
            comp = actor_critic.comp_forward(obs.detach())   # Step H: deterministic τ_comp (PD-mimic)
            if comp is not None:
                env.comp_torque = comp                        # feed env so the stabiliser runs after PD fades
        if DUMP_PATH and len(_dump_obs) < DUMP_STEPS:
            # obs BEFORE the step = the vector policy() just consumed; actions = what it answered.
            _dump_obs.append(obs[0].detach().cpu().numpy().copy())
            _dump_act.append(actions[0].detach().cpu().numpy().copy())
            # PHYSICAL state for phase alignment. Step index is NOT comparable across the two sims
            # (play_landing calls compute_observations() once before the loop, so Isaac step 0 has
            # already rolled the history ring and carries reset randomisation, while the MuJoCo port
            # fills all 20 history slots with one frame). Comparing obs blocks only makes sense at a
            # milestone both sims can label from physics, so record what the MuJoCo --dump-obs marks
            # are derived from and let the offline comparator find arm/bottom/squatQ/takeoff itself.
            # Contact test is >1.0 N on fz, identical to the MuJoCo port's C.CONTACT_THRESH.
            _nc = float((env.contact_forces[0, env.feet_indices, 2] > 1.0).sum().item())
            # Full rigid-body state + the torque Isaac actually applied last step. With these the
            # MuJoCo port can be put into Isaac's exact state and driven OPEN-LOOP from the recorded
            # actions (tests the torque law + contacts) or the recorded torques (tests contacts only),
            # which is the only way to separate "the observations are wired wrong" from "the physics
            # differ" once the two trajectories have diverged.
            _dump_rs.append(env.root_states[0].detach().cpu().numpy().copy())   # 13: xyz quat v w
            _dump_dq.append(_np0.concatenate([env.dof_pos[0].detach().cpu().numpy(),
                                              env.dof_vel[0].detach().cpu().numpy()]))
            _dump_tq.append(env.torques[0].detach().cpu().numpy().copy())
            _dump_st.append([
                float(env.root_states[0, 2].item()),                    # base z
                float(env.root_states[0, 7].item()),                    # world vx
                float(env.root_states[0, 9].item()),                    # world vz
                float(env.projected_gravity[0, 0].item()),              # -sin(pitch)
                _nc,                                                    # feet in contact (0..4)
                float(env.jumping_state[0].item()),
                float(env.squat_qualified[0].item()),
                float(env.has_landed[0].item()),
            ])
            if len(_dump_obs) >= DUMP_STEPS:
                import numpy as _np
                _o = _np.asarray(_dump_obs, dtype=_np.float32)
                _a = _np.asarray(_dump_act, dtype=_np.float32)
                _s = _np.asarray(_dump_st, dtype=_np.float32)
                _np.savez(DUMP_PATH, obs=_o, action=_a, state=_s,
                          root_states=_np.asarray(_dump_rs, dtype=_np.float64),
                          dof=_np.asarray(_dump_dq, dtype=_np.float64),
                          torque=_np.asarray(_dump_tq, dtype=_np.float64),
                          state_cols=_np.asarray(["z", "vx", "vz", "projg_x",
                                                  "n_contact", "jumping", "squatQ", "landed"]),
                          rl_driven=_np.ones(len(_o), dtype=_np.float32))
                print(f"[dump] wrote {DUMP_PATH}: obs{_o.shape} action{_a.shape} "
                      f"(every step is policy-driven in play_landing)", flush=True)
                return
        obs, _, _, dones, infos = env.step(actions.detach())
        if _pd_probe:
            _pd_probe = False
            _meas = float(env.pd_prior_alpha.float().mean())
            print(f"[play] pd_alpha MEASURED off env.pd_prior_alpha after step 1: {_meas:.4f}"
                  + ("" if abs(_meas - _applied) < 5e-3 else
                     f"   !! does not match the predicted {_applied:.3f} -- trust this number"),
                  flush=True)
        # TEST: while standing, force cmd4 to a CLEAR 0.0 (the env otherwise pins it to 0.45, just under the 0.5
        # jump threshold). If the stand-hop STOPS -> it was near-threshold command hesitation (cheap fix);
        # if it PERSISTS -> the policy has no stable stand fixed-point (needs a stand-hold reward / retrain).
        if STAND_CMD0 and (not bool(env.jumping_state[0])) and float(env.commands[0, 4]) <= 0.5:
            env.commands[0, 4] = 0.0
        # TEST Option-1: once landed, force the jump command OFF (the env otherwise holds cmd4=1 for the whole
        # 0.75s buffer). If the post-landing chase-hop stops -> disabling the jump command AT TOUCHDOWN is the fix.
        if DISABLE_JUMP_ON_LAND and bool(env.has_landed[0]):
            env.commands[0, 4] = 0.0
        # POST-LANDING DRIFT: how far the base has slid from its touchdown point (landing_root_xy, locked at
        # first touchdown). Tracked on live (non-reset) steps; reported at episode end. +dx = creep toward
        # target (residual chase); large |dy| = lateral/yaw drift (the Stage-1 stand-drift).
        if bool(env.has_landed[0]):
            landed_ep = True
            _dv = env.root_states[0, :2] - env.landing_root_xy[0]
            drift_dx, drift_dy = float(_dv[0]), float(_dv[1])
            max_drift = max(max_drift, (drift_dx ** 2 + drift_dy ** 2) ** 0.5)
            # is the creep a STEP (a foot lifts) or a SLIDE (4 planted)? -> tells us if maintain_contact can catch it
            _cs = env._get_contact_state()[0].int().tolist()   # per-foot contact [FL, FR, RL, RR] (1=down)
            _nc = sum(_cs)
            pl_steps += 1
            pl_all4 += int(_nc >= 4)
            pl_min_feet = min(pl_min_feet, _nc)
        if MONITOR_CONTACT:
            _cs = env._get_contact_state()[0].int().tolist()
            _nc = sum(_cs)
            if not bool(env.jumping_state[0]):
                _phase = "STAND"
            elif not bool(env.has_taken_off[0]):
                _phase = "SQUAT/TAKEOFF"
            elif not bool(env.has_landed[0]):
                _phase = "AIRBORNE"
            else:
                _phase = "LANDED"
            if MONITOR_CONTACT >= 2:
                _show = (_cs != prev_cs or _phase != prev_phase)
            elif _phase == "LANDED":
                _show = (_phase != prev_phase                      # the touchdown line
                         or (_nc == 0 and prev_nc != 0)            # relift: all four feet left again
                         or (_nc >= 4 and not shown_all4))         # first time it is fully planted
                shown_all4 = shown_all4 or _nc >= 4
            else:
                _show = (_phase != prev_phase)
            if _show:
                _h = float(env.root_states[0, 2])
                _vx = float(env.root_states[0, 7])
                _vz = float(env.root_states[0, 9])
                print(f"  [step {ep_len:3d}] phase={_phase:13s} | contact[FL,FR,RL,RR]={_cs} (n={_nc}) | h={_h:.3f}m | vx={_vx:+.2f}m/s vz={_vz:+.2f}m/s", flush=True)
            prev_cs = _cs
            prev_phase = _phase
            prev_nc = _nc
        # Count distinct FLIGHT phases per episode: a clean single jump = 1 all-feet-off period; an EXTRA
        # in-place HOP after landing = a 2nd all-feet-off period. Catches the post-landing hop that the
        # land-event print misses (just_landed fires only on the FIRST touchdown).
        feet_off = not bool(env._get_contact_state()[0].any())
        if feet_off and not feet_off_prev:
            n_flights += 1
            # PROOF: at every all-feet-off transition, dump the state machine. jump_starts = how many times
            # _start_jump fired this episode. If jump_starts stays 1 while n_flights climbs, the extra
            # "flights" are POST-LANDING PHYSICAL BOUNCES (landed latched True), NOT commanded 2nd jumps.
            if DEBUG_HOP:
                print(f"  [feet-off #{n_flights}] step={int(env.episode_length_buf[0])} "
                      f"jump_starts={float(env.jump_starts[0]):.0f} "
                      f"jumping={int(env.jumping_state[0])} took_off={int(env.has_taken_off[0])} "
                      f"landed={int(env.has_landed[0])} cmd4={float(env.commands[0,4]):.2f} "
                      f"h={float(env.root_states[0,2]):.2f}", flush=True)
        feet_off_prev = feet_off
        ep_len += 1
        if TRACE:
            # post-step state: projected_gravity[:,0] > 0 is nose-DOWN, and just_took_off fired during
            # THIS step, so the takeoff snapshot is taken at the takeoff instant, not one step late.
            _nd = float(env.projected_gravity[0, 0])
            tr["ep_nd"] = max(tr["ep_nd"], _nd)
            _armed_now = bool(env.jumping_state[0])
            if not _armed_now:
                # the stand phase is the one with NO pitch penalty at all (_reward_pitch_level
                # multiplies nose_down by jumping_state), so track its worst nose-down separately
                tr["stand_nd"] = max(tr["stand_nd"], _nd)
            else:
                if not tr["armed"]:
                    tr["armed"] = True
                    tr["arm_step"] = ep_len
                if not bool(env.has_taken_off[0]):
                    tr["bottom"] = min(tr["bottom"], float(env.root_states[0, 2]))
            if bool(env.squat_qualified[0]):
                tr["squatQ"] = True
            if bool(env.just_took_off[0]) and not tr["tookoff"]:
                tr["tookoff"] = True
                tr["to_step"] = ep_len
                tr["to_pitch"] = _nd
                tr["to_vx"] = float(env.root_states[0, 7])
                tr["to_vz"] = float(env.root_states[0, 9])
        if bool(dones.any()):
            if TRACE:
                # time_out_buf is set fresh in check_termination and NOT cleared by reset_idx, and
                # reset_idx does not touch contact_forces either, so both still describe the step that
                # ended the episode. Everything is guarded: a diagnostic must never kill the viewer.
                n_ep += 1
                try:
                    _timeout = bool(env.time_out_buf[0])
                except Exception:
                    _timeout = False
                try:
                    _bf = float(torch.norm(env.contact_forces[0, env.termination_contact_indices, :],
                                           dim=-1).sum())
                except Exception:
                    _bf = -1.0
                if _timeout:
                    _cause = "timeout"
                elif _bf > 0.2:
                    _cause = f"base-contact {_bf:.0f}N"
                else:
                    _cause = "postjump/other"
                if not tr["armed"]:
                    _out = "NO-ARM    "
                    n_ep_noarm += 1
                elif not tr["tookoff"]:
                    _out = "NO-TAKEOFF"
                    n_ep_noto += 1
                elif not landed_ep:
                    _out = "NO-LANDING"
                    n_ep_noland += 1
                else:
                    _out = "LAND-HIT  " if tr["hit"] else "LAND-MISS "
                n_ep_hit += int(tr["hit"])
                _bot = tr["bottom"] if tr["bottom"] < 9.0 else float("nan")
                if tr["tookoff"]:
                    _tos = (f"to@{tr['to_step']:3d} toPitch={pg2deg(tr['to_pitch']):+5.1f}d "
                            f"toV=({tr['to_vx']:+.2f},{tr['to_vz']:+.2f})")
                else:
                    _tos = "to@ --  toPitch=   --  toV=(  --  ,  --  )"
                print(f"[ep #{n_ep:3d}] {_out} len={ep_len:3d} reset={_cause:16s} "
                      f"arm@{tr['arm_step']:3d} squatQ={int(tr['squatQ'])} bottom={_bot:.3f} {_tos} "
                      f"| maxNoseDown stand={pg2deg(tr['stand_nd']):+5.1f}d ep={pg2deg(tr['ep_nd']):+5.1f}d "
                      f"| epHit {n_ep_hit}/{n_ep}={n_ep_hit / n_ep:.2f} "
                      f"[noArm {n_ep_noarm} noTakeoff {n_ep_noto} noLand {n_ep_noland}]", flush=True)
                _tr_reset()
            if DEBUG_HOP:
                tilt = pg2deg(env.projected_gravity[0, 0])   # nose-down tilt at reset (>0 nose-down); large => tipped over
                print(f"[reset] ep_len={ep_len} tilt={tilt:.0f}deg h={float(env.root_states[0,2]):.2f} "
                      f"flight_phases={n_flights}  {'(clean: 1 jump)' if n_flights <= 1 else '<-- EXTRA HOP(s) after landing'}", flush=True)
            if landed_ep and MONITOR_CONTACT:
                _all4 = pl_all4 / max(pl_steps, 1)
                print(f"  [post-land drift] dx={drift_dx:+.3f} dy={drift_dy:+.3f} "
                      f"|d|={(drift_dx ** 2 + drift_dy ** 2) ** 0.5:.3f}m  max|d|={max_drift:.3f}m  "
                      f"| all4_planted={_all4*100:.0f}% min_feet={pl_min_feet}  "
                      f"({'SLIDE: maintain_contact wont catch' if _all4 > 0.9 else 'STEP-creep: maintain_contact can catch'})", flush=True)
            resets_since += 1
            n_flights = 0
            feet_off_prev = False
            ep_len = 0
            drift_dx = drift_dy = max_drift = 0.0
            landed_ep = False
            pl_steps = pl_all4 = 0
            pl_min_feet = 4
            prev_cs = None
            prev_phase = None
            prev_nc = None
            shown_all4 = False
        # track the worst nose-down tilt while airborne/landing (same units as cfg.rewards.landing_tilt_terminate)
        if bool(env.airborne[0]) or bool(env.prelanding[0]) or bool(env.landing[0]):
            max_nd = max(max_nd, float(env.projected_gravity[0, 0]))
        if hasattr(env, "just_landed") and bool(env.just_landed.any()):
            for idx in env.just_landed.nonzero(as_tuple=False).flatten().tolist():
                land = env.root_states[idx, :2]
                err = torch.norm(land - env.landing_target[idx, :2]).item()
                fwd = float(land[0] - spawn[idx, 0])
                lat = float(land[1] - spawn[idx, 1])
                peak = float(env.peak_base_height[idx])
                td = float(env.projected_gravity[idx, 0])   # pitch tilt AT touchdown (>0 = nose-down)
                hit = (err <= 0.10) and (peak >= 0.40)
                n_jump += 1
                n_hit += int(hit)
                if TRACE:
                    tr["hit"] = tr["hit"] or bool(hit)
                # DIAG: real dx command, how far AHEAD of the TAKEOFF point the target actually is, and the
                # forward reach measured FROM THE TAKEOFF POINT (= this jump's TRUE distance, not cumulative).
                cmd0 = float(env.commands[idx, 0])
                tgt_ahead = float(env.landing_target[idx, 0] - env.takeoff_root_xy[idx, 0])
                fwd_tk = float(land[0] - env.takeoff_root_xy[idx, 0])
                # RESULT ONLY: peak height, net flight from the TAKEOFF anchor (fwd -- the only
                # distance this project counts), total from spawn (tot, includes creep), landing
                # error, hit, running hit rate. REALcmd0 / tgt_ahead used to sit on every line as a
                # guard against the "DX command came out 0" bug (note at the top of this file); they
                # are now appended ONLY when they disagree with the requested command, so the guard
                # still fires and costs nothing when everything is fine.
                _cmd_bad = abs(cmd0 - DX) > 0.02 or abs(tgt_ahead - DX) > 0.05
                _warn = f"  !! REALcmd0={cmd0:+.2f} tgt_ahead={tgt_ahead:+.2f} (asked {DX:+.2f})" if _cmd_bad else ""
                print(f"[land #{n_jump}] peak={peak:.3f} fwd={fwd_tk:+.3f} tot={fwd:+.3f} "
                      f"err={err:.3f} {'HIT ' if hit else 'miss'} hit={n_hit/n_jump:.2f}{_warn}", flush=True)
                max_nd = 0.0
                resets_since = 0


if __name__ == "__main__":
    main()
