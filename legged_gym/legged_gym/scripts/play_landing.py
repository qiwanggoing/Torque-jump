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
DX = _env_float("PLAY_LANDING_DX", 0.9)       # forward landing displacement (m). 新模型准确到 0.8; 0.9+ 够不到. 改数字试别的.
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
    _pd_a = float(env.cfg.control.pd_prior_weight) * max(0.0, 1.0 - min(1.0, max(0.0,
        (REPLAY_STEP - float(env.cfg.growth.warmup_steps)) / max(1.0, float(env.cfg.growth.x0) - float(env.cfg.growth.warmup_steps)))))
    print(f"[play] {os.path.basename(_resume_path)} -> iter {ckpt_iter}: step_count={REPLAY_STEP} "
          f"(non-linear fade-correct), replay pd_alpha={_pd_a:.3f}")

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
    print(f"[play_landing] cmd dx={DX} dy={DY} height={HEIGHT} | {_mass_note} | normal jump flow, replay pd_alpha={_pd_a:.3f}", flush=True)

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
    MONITOR_CONTACT = _env_bool("PLAY_LANDING_MONITOR_CONTACT", False) # DEFAULT OFF = clean output (just the per-jump [land] line).
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
        obs, _, _, dones, infos = env.step(actions.detach())
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
            if _cs != prev_cs or _phase != prev_phase:
                _h = float(env.root_states[0, 2])
                _vx = float(env.root_states[0, 7])
                _vz = float(env.root_states[0, 9])
                print(f"  [step {ep_len:3d}] phase={_phase:13s} | contact[FL,FR,RL,RR]={_cs} (n={_nc}) | h={_h:.3f}m | vx={_vx:+.2f}m/s vz={_vz:+.2f}m/s", flush=True)
                prev_cs = _cs
                prev_phase = _phase
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
        if bool(dones.any()):
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
                # DIAG: real dx command, how far AHEAD of the TAKEOFF point the target actually is, and the
                # forward reach measured FROM THE TAKEOFF POINT (= this jump's TRUE distance, not cumulative).
                cmd0 = float(env.commands[idx, 0])
                tgt_ahead = float(env.landing_target[idx, 0] - env.takeoff_root_xy[idx, 0])
                fwd_tk = float(land[0] - env.takeoff_root_xy[idx, 0])
                print(f"[land #{n_jump}] REALcmd0={cmd0:+.2f} tgt_ahead={tgt_ahead:+.2f}m | peak={peak:.3f}m  "
                      f"fwd_from_takeoff={fwd_tk:+.3f}m fwd_from_spawn={fwd:+.3f}m | land_err={err:.3f} "
                      f"{'HIT ' if hit else 'miss'} hit_rate={n_hit/n_jump:.2f}", flush=True)
                max_nd = 0.0
                resets_since = 0


if __name__ == "__main__":
    main()
