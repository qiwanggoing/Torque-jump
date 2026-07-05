"""Reach-ceiling health check — the STANDARD post-training deterministic eval.

WHY THIS EXISTS
---------------
TensorBoard's `landing_dx_max` (max over envs) and `landing_hit_rate` (uniform over mostly-easy
near commands) are ILLUSIONS: on Jul04_20-48-40/model_3000 they read 0.82 / 0.97, yet the
deterministic policy tops out at ~0.6 total displacement. This script is the honest ground truth:
it drives the env exactly like play_landing.py (normal command flow, deterministic MEAN policy,
faithful PD/general_scale replay via step_count pin, comp_torque fed) but BATCH (256 envs) and
HEADLESS, sweeping the forward command and reporting the REAL capability curve.

WHAT IT ANSWERS — H1 (physical wall) vs H2 (optimization local optimum):
  * air reach (land_x - takeoff_x)          true in-flight distance (ground creep excluded)
  * total disp (land_x - spawn_x)           includes pre-takeoff push-off creep
  * hit rate (err<=0.10 & peak>=0.40)       honest accuracy at each commanded distance
  * takeoff vx / vz                         the launch velocity split
  * PEAK-during-push joint speed vs limit   calf 15.65 / thigh 30 / hip 30 rad/s
      -> if calf pins at ~100%+ while thigh has headroom AND reach stalls => calf velocity WALL
         (loosening curriculum won't add real distance; the lever is physical / thigh recruitment)

FINDING (model_3000): wall at cmd 0.6->0.7. calf pinned ~130% (torque-speed curve zeros torque at
15.65; the >100% is post-unload inertial free-wheel), thigh only ~70% and DROPS with distance.
=> 0.7-0.8 is NOT reachable by loosening curriculum/reward; it's the calf reduction-geared limit.

RUN
---
  export PATH=/home/qiwang/miniconda3/envs/sata/bin:$PATH
  cd ~/torque_jump2/SATA/legged_gym
  python legged_gym/scripts/eval_reach_ceiling.py --task=go2_omnijump_landing_torque --headless
  # optional overrides (env vars):
  #   EVAL_DX_SWEEP=0.4,0.5,0.6,0.7,0.8  EVAL_N=256  EVAL_HEIGHT=0.7  EVAL_STEPS=700
  # loads the latest run by default; pin one with --load_run=<dir> --checkpoint=<N>
"""
import isaacgym
from isaacgym import gymapi
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
import os, math


def _env_floats(name, default):
    v = os.environ.get(name)
    if not v:
        return list(default)
    try:
        return [float(x) for x in v.split(",") if x.strip() != ""]
    except ValueError:
        print(f"[eval] invalid {name}={v!r}; using {default}", flush=True)
        return list(default)


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else int(default)


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v else float(default)


DX_SWEEP = _env_floats("EVAL_DX_SWEEP", [0.4, 0.5, 0.6, 0.7, 0.8])
HEIGHT = _env_float("EVAL_HEIGHT", 0.7)
N_ENVS = _env_int("EVAL_N", 256)
STEPS_PER_DX = _env_int("EVAL_STEPS", 700)
CALF_IDX, THIGH_IDX, HIP_IDX = [2, 5, 8, 11], [1, 4, 7, 10], [0, 3, 6, 9]
CALF_LIMIT, THIGH_LIMIT, HIP_LIMIT = 15.649452, 30.0, 30.0


def train_step_count_at_iter(target_iter, W, X, sf, mf, dt, nspe):
    """Reproduce TRAINING's NON-LINEAR step_count at iter N -> faithful PD/general_scale replay
    (copied from play_landing.py; freq ramps sf->mf as general_scale 0->1 over step_count in [W,X])."""
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


def main():
    args = get_args()
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    env_cfg.env.num_envs = N_ENVS
    env_cfg.env.episode_length_s = 4
    env_cfg.commands.resampling_time = 20.0
    env_cfg.commands.landing_dx_curriculum = False        # fixed command, no curriculum drift
    env_cfg.commands.landing_disp_x_stage2 = [DX_SWEEP[0], DX_SWEEP[0]]
    env_cfg.commands.landing_disp_y_stage2 = [0.0, 0.0]
    env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
    env_cfg.commands.ranges.jump_command = [1.0, 1.0]
    env_cfg.commands.jump_command_range = [1.0, 1.0]      # every episode is a jump
    env_cfg.noise.add_noise = False                       # clean, deterministic-capability conditions
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.rewards.landing_tilt_terminate = 0.0          # MEASURE mode: don't reset on tilt -> see true outcome
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
    print(f"\n[eval] {os.path.basename(_resume_path)} iter={ckpt_iter} step_count={REPLAY_STEP} "
          f"replay_pd_alpha={_pd_a:.3f}  N={N_ENVS}  height_cmd={HEIGHT}  sweep={DX_SWEEP}", flush=True)

    init = torch.tensor([float(env.cfg.init_state.pos[0]), float(env.cfg.init_state.pos[1])], device=env.device)
    spawn = env.env_origins[:, :2] + init

    rows = []
    for DX in DX_SWEEP:
        env.command_ranges["lin_vel_x"] = [DX, DX]
        env.reset()
        to_vx = torch.zeros(env.num_envs, device=env.device)
        to_vz = torch.zeros(env.num_envs, device=env.device)
        c_air, c_tot, c_err, c_peak, c_hit = [], [], [], [], []
        c_tovx, c_tovz, c_calf, c_thigh, c_hip = [], [], [], [], []
        # PEAK joint speed over the whole PUSH phase (loading = jumping & not-yet-taken-off), so the
        # proximo-distal timing (thigh fires early, calf late) can't hide thigh usage behind a snapshot.
        pk_calf = torch.zeros(env.num_envs, device=env.device)
        pk_thigh = torch.zeros(env.num_envs, device=env.device)
        pk_hip = torch.zeros(env.num_envs, device=env.device)
        env.compute_observations()
        obs = env.get_observations()
        for _ in range(STEPS_PER_DX):
            env.step_count = REPLAY_STEP
            env.common_step_counter = REPLAY_STEP
            with torch.no_grad():
                actions = policy(obs.detach())
                comp = actor_critic.comp_forward(obs.detach())
                if comp is not None:
                    env.comp_torque = comp
            obs, _, _, dones, infos = env.step(actions.detach())

            loading = env.jumping_state & (~env.has_taken_off)
            cv = env.dof_vel[:, CALF_IDX].abs().max(dim=1).values
            tv = env.dof_vel[:, THIGH_IDX].abs().max(dim=1).values
            hv = env.dof_vel[:, HIP_IDX].abs().max(dim=1).values
            pk_calf = torch.where(loading, torch.maximum(pk_calf, cv), pk_calf)
            pk_thigh = torch.where(loading, torch.maximum(pk_thigh, tv), pk_thigh)
            pk_hip = torch.where(loading, torch.maximum(pk_hip, hv), pk_hip)

            if bool(env.just_took_off.any()):
                m = env.just_took_off
                to_vx[m] = env.root_states[m, 7]
                to_vz[m] = env.root_states[m, 9]
                c_tovx += env.root_states[m, 7].cpu().tolist()
                c_tovz += env.root_states[m, 9].cpu().tolist()
                c_calf += pk_calf[m].cpu().tolist()
                c_thigh += pk_thigh[m].cpu().tolist()
                c_hip += pk_hip[m].cpu().tolist()
                pk_calf[m] = 0.0; pk_thigh[m] = 0.0; pk_hip[m] = 0.0

            if bool(dones.any()):
                pk_calf[dones] = 0.0; pk_thigh[dones] = 0.0; pk_hip[dones] = 0.0

            if bool(env.just_landed.any()):
                for i in env.just_landed.nonzero(as_tuple=False).flatten().tolist():
                    land_x = float(env.root_states[i, 0])
                    to_x = float(env.takeoff_root_xy[i, 0])
                    err = float(torch.norm(env.root_states[i, :2] - env.landing_target[i, :2]))
                    peak = float(env.peak_base_height[i])
                    c_air.append(land_x - to_x)
                    c_tot.append(land_x - float(spawn[i, 0]))
                    c_err.append(err); c_peak.append(peak)
                    c_hit.append(1.0 if (err <= 0.10 and peak >= 0.40) else 0.0)

        m = lambda a: float(np.nanmean(a)) if len(a) else float('nan')
        row = dict(DX=DX, n=len(c_air),
                   air=m(c_air), air_md=(float(np.nanmedian(c_air)) if c_air else float('nan')),
                   tot=m(c_tot), err=m(c_err), peak=m(c_peak), hit=m(c_hit),
                   tovx=m(c_tovx), tovz=m(c_tovz),
                   calf=m(c_calf) / CALF_LIMIT * 100.0,
                   thigh=m(c_thigh) / THIGH_LIMIT * 100.0,
                   hip=m(c_hip) / HIP_LIMIT * 100.0)
        rows.append(row)
        print(f"  DX={DX:.2f} | n={row['n']:4d} | air={row['air']:.3f}(md{row['air_md']:.3f}) tot={row['tot']:.3f} "
              f"| hit={row['hit']:.2f} err={row['err']:.3f} peak={row['peak']:.3f} | TOvx={row['tovx']:.2f} "
              f"TOvz={row['tovz']:.2f} | PEAKpush%lim calf={row['calf']:.0f} thigh={row['thigh']:.0f} hip={row['hip']:.0f}",
              flush=True)

    print("\n================ SUMMARY (deterministic reach; H1 wall vs H2 local optimum) ================")
    print(f"{'DX':>5} {'air':>7} {'total':>7} {'hit':>6} {'TOvx':>7} {'TOvz':>7} {'calf%':>6} {'thigh%':>7} {'hip%':>5}")
    for r in rows:
        print(f"{r['DX']:5.2f} {r['air']:7.3f} {r['tot']:7.3f} {r['hit']:6.2f} {r['tovx']:7.2f} {r['tovz']:7.2f} "
              f"{r['calf']:5.0f} {r['thigh']:6.0f} {r['hip']:4.0f}")
    airs = [r['air'] for r in rows]; vxs = [r['tovx'] for r in rows]; calfs = [r['calf'] for r in rows]
    print(f"\n[read] Δair({DX_SWEEP[0]}->{DX_SWEEP[-1]})={airs[-1]-airs[0]:+.3f}m  Δvx={vxs[-1]-vxs[0]:+.2f}m/s  calf@far={calfs[-1]:.0f}%")
    print("  -> air stalls/drops while calf pinned ~100%+ => H1 calf velocity WALL (curriculum/reward won't add reach)")
    print("  -> air keeps rising with DX               => H2, policy reaches farther when commanded (curriculum/probe help)")


if __name__ == "__main__":
    main()
