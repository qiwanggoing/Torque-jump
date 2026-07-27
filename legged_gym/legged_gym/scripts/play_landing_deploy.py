"""Deploy-faithful play: STAND until settled, THEN command the jump (mirrors real-robot use).

The normal play_landing.py resets and jumps almost immediately (first_jump_delay ~55 steps).
On the REAL robot you don't do that -- you place the robot, let it STAND and stabilise, and
only THEN send the jump command. This script reproduces that protocol per trial:

    reset  ->  STAND (cmd4=0, in-place)  ->  [wait until env.stand_step_counter >= STAND_HOLD]
           ->  TRIGGER JUMP (cmd4=1, dx=DX)  ->  observe takeoff/flight/landing
           ->  POST-LAND settle (cmd4=0)  ->  report stability  ->  next trial

Everything else matches the VALIDATED play_landing.py rollout: NORMAL command flow (no test-mode
state machine), faithful PD-replay (step_count pinned to the checkpoint's training iter so
general_scale/pd_alpha match), Step-H comp_torque fed, deterministic (mean) policy, nominal
mass (URDF = real robot), no noise / no DR / rsi_prob=0.

CAVEAT (read the printout accordingly): if the loaded checkpoint was trained JUMP-ONLY
(jump_command_range=[1,1], no stand episodes), cmd4=0 is OUT OF DISTRIBUTION and the robot may
NOT hold a stand -- that is itself the finding (the policy is not deploy-ready for stand->jump).
Try PLAY_DEPLOY_STAND_CMD4=0.49 (just under the 0.5 jump threshold) to probe a near-in-dist stand.

RUN (viewer):
  python legged_gym/scripts/play_landing_deploy.py --task=go2_omnijump_landing_torque \
      --load_run=Jul16_12-07-41_stage1_landing --checkpoint=4600
Override the command / timings without editing:
  PLAY_DEPLOY_DX=1.0 PLAY_DEPLOY_HEIGHT=0.5 PLAY_DEPLOY_STAND_HOLD=80 PLAY_DEPLOY_TRIALS=5 python ...
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


def _envf(name, default):
    v = os.environ.get(name)
    if v is None:
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)


def _envi(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else int(default)


def _envb(name, default):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def train_step_count_at_iter(target_iter, W, X, sf, mf, dt, nspe):
    """Reproduce TRAINING's non-linear step_count at iter N (freq ramp 96->48/iter) for faithful
    PD/general_scale replay. Same closed form as play_landing.py."""
    r0 = nspe / (dt * sf)
    iw = W / r0
    if target_iter <= iw:
        return r0 * target_iter
    Delta = X - W
    a = (mf - sf) / (2.0 * Delta)
    b = sf
    iter_fade_end = iw + (dt / nspe) * (a * Delta * Delta + b * Delta)
    if target_iter <= iter_fade_end:
        c = (nspe / dt) * (target_iter - iw)
        u = (-b + math.sqrt(b * b + 4.0 * a * c)) / (2.0 * a)
        return W + u
    r_end = nspe / (dt * mf)
    return X + r_end * (target_iter - iter_fade_end)


# === deploy protocol knobs (env-overridable) =================================
DX = _envf("PLAY_DEPLOY_DX", 1.2)              # forward landing displacement (m) commanded at the jump
DY = _envf("PLAY_DEPLOY_DY", 0.0)              # lateral landing displacement (m)
HEIGHT = _envf("PLAY_DEPLOY_HEIGHT", 0.5)      # jump-height command (train range [0.4,0.6]; don't exceed 0.6)
PD_STAND = _envb("PLAY_DEPLOY_PD_STAND", True)  # TRUE = hybrid deploy architecture: a PURE PD controller holds the
                                               # robot at the DEFAULT pose during stand/post-land (pd_alpha=1, rl_alpha=0),
                                               # and control is handed to the RL policy ONLY for the jump. This mimics a
                                               # real-robot default-stand controller + RL jump, AND it hands the policy the
                                               # familiar reset pose at takeoff. FALSE = let the RL policy itself stand (cmd4=0).
STAND_CMD4 = _envf("PLAY_DEPLOY_STAND_CMD4", 0.0)  # (RL-stand mode only) cmd4 during the STAND phase. 0.0 = clear stand.
STAND_PD_WEIGHT = _envf("PLAY_DEPLOY_STAND_PD_WEIGHT", 1.0)  # PRE-JUMP stand stiffness. FIRM (1.0) so the PD holds the TRUE
                                               # default pose (h~0.30) the policy trained to jump from -> full trained reach. Soft PD
                                               # (0.5) sags the pose (~0.27) -> the jump lands ~0.05m short. Robot is already at default
                                               # here, so firm = no snap (zero initial error).
PD_WEIGHT = _envf("PLAY_DEPLOY_PD_WEIGHT", 0.5)    # POST-LAND catch stiffness. SOFT (0.5) so taking the stand back after the jump is
                                               # gentle (a firm snap onto a landed/bent pose is violent). Lower = softer.
PD_ENGAGE_MIN = _envi("PLAY_DEPLOY_PD_ENGAGE_MIN", 12)  # RL keeps control through touchdown; PD may take over only after this
                                               # many post-land steps (lets RL absorb the landing bounce first) AND once the base has
                                               # gone QUIESCENT (|v_xy|,|v_z| < ~0.4 m/s). Engaging after the dynamics die = gentle, no snap.
PD_ENGAGE_MAX = _envi("PLAY_DEPLOY_PD_ENGAGE_MAX", 45)  # hard cap: force PD to take over by this many post-land steps even if the
                                               # RL stand is still creeping (jump-only policy can't hold a clean stand -> PD must stop the drift).
WINDUP = _envb("PLAY_DEPLOY_WINDUP", False)        # Default OFF: verified INERT -- windup on vs off gives byte-identical jumps
                                               # (same 0.52m run-up, same reach); it only shifts WHEN jumping_state flips, not the
                                               # policy's actions. TRUE resets the jump timer at handoff so jumping_state flips ~55
                                               # steps later (cosmetic; makes h_at_start read the mid-squat ~0.14m instead of ~0.30m).
STAND_HOLD = _envi("PLAY_DEPLOY_STAND_HOLD", 60)   # consecutive STABLE-stand steps required before we trigger the jump
STAND_MAX = _envi("PLAY_DEPLOY_STAND_MAX", 400)    # give up settling after this many stand steps (report "did not settle")
JUMP_MAX = _envi("PLAY_DEPLOY_JUMP_MAX", 400)      # max steps to watch the jump+landing before moving on
POST_LAND = _envi("PLAY_DEPLOY_POST_LAND", 120)    # steps to keep standing after touchdown (post-landing stability watch)
N_TRIALS = _envi("PLAY_DEPLOY_TRIALS", 6)          # how many stand->jump trials to run
# =============================================================================


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 12         # long enough for stand + jump + settle within one episode
    env_cfg.commands.resampling_time = 40.0   # never auto-resample mid-trial (we drive cmd4 by hand)
    env_cfg.commands.landing_dx_curriculum = False
    # Reset draws a STAND command (cmd4<=0.5) and an in-place target, so a fresh reset = stand, not jump.
    env_cfg.commands.jump_command_range = [0.0, 0.0]
    env_cfg.commands.landing_disp_x_stage2 = [0.0, 0.0]
    env_cfg.commands.landing_disp_y_stage2 = [0.0, 0.0]
    env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
    env_cfg.commands.ranges.jump_command = [1.0, 1.0]
    # Clean, real-robot-like conditions.
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.rewards.rsi_prob = 0.0
    env_cfg.rewards.landing_tilt_terminate = 0.0   # measure mode: don't tilt-reset, so we see the true attitude
    train_cfg.runner.resume = True

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    _load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    _checkpoint = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    _resume_path = get_load_path(log_root, load_run=_load_run, checkpoint=_checkpoint)
    ckpt_iter = int(os.path.basename(_resume_path).replace('model_', '').replace('.pt', ''))

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic

    REPLAY_STEP = int(round(train_step_count_at_iter(
        ckpt_iter,
        float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
        float(env.start_freq), float(env.max_freq),
        float(env.dt), int(train_cfg.runner.num_steps_per_env))))
    _pd_a = float(env.cfg.control.pd_prior_weight) * max(0.0, 1.0 - min(1.0, max(0.0,
        (REPLAY_STEP - float(env.cfg.growth.warmup_steps)) / max(1.0, float(env.cfg.growth.x0) - float(env.cfg.growth.warmup_steps)))))

    pg2deg = lambda v: math.degrees(math.asin(max(-1.0, min(1.0, float(v)))))   # projected_gravity_x -> tilt deg (>0 nose-down)
    PD_W_ORIG = float(env.cfg.control.pd_prior_weight)   # restore this for the RL jump phase
    _zero_act = torch.zeros((env.num_envs, env.num_actions), device=env.device)

    def act_step():
        """One faithful RL-policy step (pd_alpha=0 pure-torque replay, comp_torque fed). Returns dones."""
        env.cfg.control.pd_prior_weight = PD_W_ORIG      # RL authority (undo any PD-hold override)
        env.step_count = REPLAY_STEP                     # general_scale=1 -> pd_alpha=0, rl_alpha=1
        env.common_step_counter = REPLAY_STEP
        with torch.no_grad():
            actions = policy(env.get_observations().detach())
            comp = actor_critic.comp_forward(env.get_observations().detach())
            if comp is not None:
                env.comp_torque = comp
        _, _, _, dones, _ = env.step(actions.detach())
        return bool(dones[0])

    def pd_hold_step(weight):
        """One PURE-PD step: hold the DEFAULT pose with the given stiffness `weight`. Mimics a real-robot
        default-stand controller. pd_prior_weight=weight + general_scale=0 (step_count pinned low) ->
        torque = weight * (p_gains*(default_dof_pos - q) - d_gains*qd), RL residual and comp fully off.
        Lower weight = softer (gentler catch); ramp it up over a few steps to avoid a violent snap."""
        env.cfg.control.pd_prior_weight = float(weight)
        env.step_count = 0                                # general_scale=0 -> pd_alpha=weight, rl_alpha=1-weight (residual=0)
        env.common_step_counter = 0
        _, _, _, dones, _ = env.step(_zero_act)
        return bool(dones[0])

    print(f"\n[deploy] {os.path.basename(_resume_path)} -> iter {ckpt_iter}: step_count={REPLAY_STEP} "
          f"replay pd_alpha={_pd_a:.3f}", flush=True)
    _stand_desc = f"PURE-PD(w={STAND_PD_WEIGHT}) holds default pose" if PD_STAND else f"RL policy at cmd4={STAND_CMD4}"
    _settle_desc = f"RL lands, then PD(w={PD_WEIGHT}) takes over once base quiescent ({PD_ENGAGE_MIN}-{PD_ENGAGE_MAX} steps)" if PD_STAND else "RL"
    print(f"[deploy] protocol: reset -> STAND ({_stand_desc}) until {STAND_HOLD} stable steps "
          f"-> hand to RL, JUMP(cmd4=1, dx={DX}, h={HEIGHT}) -> {_settle_desc} "
          f"| nominal mass, no noise/DR", flush=True)

    n_hit = n_done = 0
    for trial in range(1, N_TRIALS + 1):
        env.reset()
        env.commands[:, 0] = 0.0
        env.commands[:, 1] = 0.0
        env.commands[:, 4] = STAND_CMD4
        spawn_xy = env.root_states[0, :2].clone()
        # in-place stand target so obs / any target term reads "stay here"
        env.landing_target[:, 0] = spawn_xy[0]
        env.landing_target[:, 1] = spawn_xy[1]
        env.compute_observations()

        if getattr(env, "viewer", None) is not None:
            sx, sy = float(spawn_xy[0]), float(spawn_xy[1])
            env.gym.viewer_camera_look_at(env.viewer, None,
                                          gymapi.Vec3(sx - 1.7, sy - 1.7, 1.1),
                                          gymapi.Vec3(sx + 0.5, sy, 0.25))

        print(f"\n=== TRIAL {trial}/{N_TRIALS} ===", flush=True)

        # ---- STAND phase: PD holds default pose (PD_STAND) or RL stands at cmd4 (else) ----------
        settled = False
        fell_standing = False
        max_tilt = 0.0
        max_drift = 0.0
        min_feet = 4
        for st in range(STAND_MAX):
            env.commands[:, 4] = (0.0 if PD_STAND else STAND_CMD4)   # cmd4<=0.5 -> stand; PD target = default pose
            fell = pd_hold_step(STAND_PD_WEIGHT) if PD_STAND else act_step()   # firm: hold the TRUE default pose for full reach
            if fell:
                fell_standing = True
                break
            tilt = pg2deg(env.projected_gravity[0, 0])
            drift = float(torch.norm(env.root_states[0, :2] - spawn_xy))
            feet = int(env._get_contact_state()[0].int().sum())
            max_tilt = max(max_tilt, abs(tilt))
            max_drift = max(max_drift, drift)
            min_feet = min(min_feet, feet)
            if int(env.stand_step_counter[0]) >= STAND_HOLD:
                settled = True
                print(f"[stand] SETTLED after {st + 1} steps | h={float(env.root_states[0, 2]):.3f}m "
                      f"tilt={tilt:+.1f}deg drift={drift:.3f}m feet={feet}/4 "
                      f"(held {int(env.stand_step_counter[0])} stable steps)", flush=True)
                break
        if not settled:
            why = "FELL while standing" if fell_standing else f"did NOT settle in {STAND_MAX} steps"
            _who = "PURE-PD" if PD_STAND else f"RL(cmd4={STAND_CMD4})"
            print(f"[stand] {why} | maxTilt={max_tilt:.1f}deg maxDrift={max_drift:.3f}m minFeet={min_feet}/4 "
                  f"-> {_who} could not hold the stand"
                  f"{'' if PD_STAND else ' (jump-only training?)'}", flush=True)
            if fell_standing:
                continue   # can't jump from a collapsed stand; next trial

        # ---- TRIGGER the jump: flip to the jump command; target auto-locks at takeoff ----------
        env.commands[:, 0] = DX
        env.commands[:, 1] = DY
        env.commands[:, 4] = 1.0
        env.landing_target[:, 0] = env.root_states[0, 0] + DX     # provisional (env re-locks to squat_xy+DX at takeoff)
        env.landing_target[:, 1] = env.root_states[0, 1] + DY
        if WINDUP:
            env.episode_length_buf[:] = 0        # restart the jump timer -> RL winds up cmd4=1 for first_jump_delay_steps,
            env.jump_starts[:] = 0.0             #    exactly as in a fresh training episode, before the jump fires
        env.compute_observations()
        x_handoff = float(env.root_states[0, 0])   # base x when the jump is commanded -> measures the pre-takeoff run-up creep
        _fjd = int(getattr(env.cfg.rewards, "first_jump_delay_steps", 55))
        print(f"[jump>] {'from settled stand' if settled else 'from UNSETTLED stand'}: cmd4=1, dx={DX}"
              f"{f' (wind-up {_fjd} steps)' if WINDUP else ' (immediate)'}", flush=True)

        # ---- JUMP + LANDING watch ---------------------------------------------------------------
        took_off = False
        landed = False
        pd_engaged = False
        to_pitch = None
        post = 0
        fell = False
        jump_started = False
        for jt in range(JUMP_MAX):
            # RL drives the jump AND the landing + re-stabilization (it was trained to land and settle). We hand
            # the stand back to the PD controller only AFTER the robot is stably standing again (pd_engaged) --
            # engaging PD from an already-stable near-default stand is gentle, vs snapping mid-landing (violent)
            # or catching a dynamic touchdown with weak PD (falls).
            if PD_STAND and pd_engaged:
                if pd_hold_step(PD_WEIGHT):
                    fell = True
                    break
            elif act_step():
                fell = True
                break
            if (not jump_started) and bool(env.jumping_state[0]):
                jump_started = True
                print(f"[trig ] jump STARTED {jt + 1} steps after cmd4=1 "
                      f"(jumping_state=1, jump_starts={float(env.jump_starts[0]):.0f}, "
                      f"h_at_start={float(env.root_states[0, 2]):.3f}m)", flush=True)
            if (not took_off) and bool(env.has_taken_off[0]):
                took_off = True
                env.commands[:, 4] = 0.0             # ONE jump only: drop cmd so it re-arms into a stand, not a 2nd jump
                to_pitch = pg2deg(env.projected_gravity[0, 0])
            if (not landed) and hasattr(env, "just_landed") and bool(env.just_landed[0]):
                landed = True
                land = env.root_states[0, :2]
                err = float(torch.norm(land - env.landing_target[0, :2]))
                peak = float(env.peak_base_height[0])
                fwd_tk = float(land[0] - env.takeoff_root_xy[0, 0])
                runup = float(env.takeoff_root_xy[0, 0] - x_handoff)   # forward creep from jump-command to takeoff = run-up
                td_pitch = pg2deg(env.projected_gravity[0, 0])
                hit = (err <= 0.10) and (peak >= 0.40)
                n_hit += int(hit)
                print(f"[land ] runup={runup:+.3f}m fwd_from_takeoff={fwd_tk:+.3f}m peak={peak:.3f}m land_err={err:.3f} "
                      f"{'HIT ' if hit else 'miss'} | takeoff_pitch={to_pitch:+.1f}deg touchdown_pitch={td_pitch:+.1f}deg",
                      flush=True)
                td_xy = land.clone()
            if landed:
                post += 1
                if PD_STAND and not pd_engaged:
                    vxy = float(torch.norm(env.root_states[0, 7:9]))
                    vz = abs(float(env.root_states[0, 9]))
                    feet = int(env._get_contact_state()[0].int().sum())
                    # Engage as soon as the VERTICAL landing bounce is absorbed (vz small) and the feet are down,
                    # arresting the horizontal drift EARLY before it builds speed (a loose v_xy guard just avoids
                    # grabbing mid-lunge). vz-first because the drift itself is what PD is there to stop.
                    quiescent = (post >= PD_ENGAGE_MIN) and (vz < 0.4) and (vxy < 0.8) and (feet >= 3)
                    if quiescent or post >= PD_ENGAGE_MAX:
                        pd_engaged = True
                        why = "base quiescent" if quiescent else "engage cap"
                        print(f"[pd>  ] {post} steps post-land, {why} (|vxy|={vxy:.2f} |vz|={vz:.2f}) "
                              f"-> PD(w={PD_WEIGHT}) takes over the stand", flush=True)
                if post >= POST_LAND:
                    tilt = pg2deg(env.projected_gravity[0, 0])
                    drift = float(torch.norm(env.root_states[0, :2] - td_xy))
                    feet = int(env._get_contact_state()[0].int().sum())
                    stable = (feet == 4) and (abs(tilt) < 12.0) and (drift < 0.20)
                    _held = "PD-held" if pd_engaged else "RL-held (PD never engaged: RL stand stayed jittery)"
                    print(f"[post ] after {POST_LAND} steps ({_held}): {'STABLE stand' if stable else 'UNSTABLE'} | "
                          f"tilt={tilt:+.1f}deg drift_from_touchdown={drift:.3f}m feet={feet}/4", flush=True)
                    break
        if fell:
            n_done += 1
            stage = "in flight/landing" if took_off else "before takeoff"
            print(f"[fell ] episode terminated {stage} (robot fell)", flush=True)

    total = N_TRIALS
    print(f"\n[deploy summary] {N_TRIALS} trials | landings hit={n_hit} | terminations(fell)={n_done}", flush=True)


if __name__ == "__main__":
    main()
