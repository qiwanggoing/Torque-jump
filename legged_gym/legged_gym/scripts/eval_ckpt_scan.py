"""Multi-checkpoint deterministic reach scan — build the env ONCE, loop checkpoints,
reload only the policy weights each time (env-build is the expensive fixed cost).

Mirrors eval_reach_ceiling.py's env setup + metrics EXACTLY (headless, 256 env, deterministic
mean policy, comp_torque fed, pure-torque replay since all scanned ckpts are past PD fade).
Per checkpoint it sweeps DX and prints one compact line; at the end it RANKS by reliability.

RUN
  export PATH=/home/qiwang/miniconda3/envs/sata/bin:$PATH
  cd ~/torque_jump2/SATA/legged_gym
  SCAN_RUN=Jul16_12-07-41_stage1_landing SCAN_START=3000 SCAN_END=5000 SCAN_STEP=100 \
  EVAL_DX_SWEEP=0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2 EVAL_HEIGHT=0.5 EVAL_N=256 EVAL_STEPS=700 \
  python legged_gym/scripts/eval_ckpt_scan.py --task=go2_omnijump_landing_torque --headless
"""
import isaacgym
from isaacgym import gymapi
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym import LEGGED_GYM_ROOT_DIR
import os, math


def _floats(name, d):
    v = os.environ.get(name)
    return [float(x) for x in v.split(",") if x.strip()] if v else list(d)
def _int(name, d):
    v = os.environ.get(name); return int(v) if v else int(d)
def _flt(name, d):
    v = os.environ.get(name); return float(v) if v else float(d)


DX_SWEEP = _floats("EVAL_DX_SWEEP", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
HEIGHT = _flt("EVAL_HEIGHT", 0.5)
N_ENVS = _int("EVAL_N", 256)
STEPS_PER_DX = _int("EVAL_STEPS", 700)
SCAN_RUN = os.environ.get("SCAN_RUN", "")
SCAN_START = _int("SCAN_START", 3000)
SCAN_END = _int("SCAN_END", 5000)
SCAN_STEP = _int("SCAN_STEP", 100)
# DX band used to score "dead zone" (the mid range where the bimodal split appears)
DEAD_BAND = [dx for dx in DX_SWEEP if 0.7 <= dx <= 0.9 + 1e-6]
SCORE_BAND = [dx for dx in DX_SWEEP if 0.6 - 1e-6 <= dx <= 1.2 + 1e-6]


def replay_step_at(ck, W, X, sf, mf, dt, nspe):
    r0 = nspe / (dt * sf); iw = W / r0
    if ck <= iw:
        return r0 * ck
    Delta = X - W; a = (mf - sf) / (2.0 * Delta); b = sf
    iter_fade_end = iw + (dt / nspe) * (a * Delta * Delta + b * Delta)
    if ck <= iter_fade_end:
        c = (nspe / dt) * (ck - iw); u = (-b + math.sqrt(b * b + 4 * a * c)) / (2 * a); return W + u
    return X + (nspe / (dt * mf)) * (ck - iter_fade_end)


def sweep_one(env, policy, actor_critic, REPLAY_STEP, spawn):
    """Return dict DX -> (hit, air, tot, err, nmax, tovx)."""
    out = {}
    for DX in DX_SWEEP:
        env.command_ranges["lin_vel_x"] = [DX, DX]
        env.reset()
        env.compute_observations(); obs = env.get_observations()
        c_air, c_tot, c_err, c_hit, c_tovx = [], [], [], [], []
        for _ in range(STEPS_PER_DX):
            env.step_count = REPLAY_STEP; env.common_step_counter = REPLAY_STEP
            with torch.no_grad():
                actions = policy(obs.detach())
                comp = actor_critic.comp_forward(obs.detach())
                if comp is not None:
                    env.comp_torque = comp
            obs, _, _, dones, infos = env.step(actions.detach())
            if bool(env.just_took_off.any()):
                m = env.just_took_off
                c_tovx += env.root_states[m, 7].cpu().tolist()
            if bool(env.just_landed.any()):
                for i in env.just_landed.nonzero(as_tuple=False).flatten().tolist():
                    lx = float(env.root_states[i, 0]); tx = float(env.takeoff_root_xy[i, 0])
                    err = float(torch.norm(env.root_states[i, :2] - env.landing_target[i, :2]))
                    peak = float(env.peak_base_height[i])
                    c_air.append(lx - tx); c_tot.append(lx - float(spawn[i, 0]))
                    c_err.append(err); c_hit.append(1.0 if (err <= 0.10 and peak >= 0.40) else 0.0)
        m = lambda a: float(np.nanmean(a)) if len(a) else float('nan')
        out[DX] = dict(hit=m(c_hit), air=m(c_air), tot=m(c_tot), err=m(c_err),
                       n=len(c_air), tovx=m(c_tovx))
    return out


def main():
    args = get_args(); args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = N_ENVS
    env_cfg.env.episode_length_s = 4
    env_cfg.commands.resampling_time = 20.0
    env_cfg.commands.landing_dx_curriculum = False
    env_cfg.commands.landing_disp_x_stage2 = [DX_SWEEP[0], DX_SWEEP[0]]
    env_cfg.commands.landing_disp_y_stage2 = [0.0, 0.0]
    env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
    env_cfg.commands.ranges.jump_command = [1.0, 1.0]
    env_cfg.commands.jump_command_range = [1.0, 1.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.rewards.landing_tilt_terminate = 0.0
    train_cfg.runner.resume = True

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    run_dir = SCAN_RUN if SCAN_RUN else (train_cfg.runner.load_run)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    actor_critic = ppo_runner.alg.actor_critic

    init = torch.tensor([float(env.cfg.init_state.pos[0]), float(env.cfg.init_state.pos[1])], device=env.device)
    spawn = env.env_origins[:, :2] + init

    ckpts = list(range(SCAN_START, SCAN_END + 1, SCAN_STEP))
    print(f"\n[scan] run={run_dir}  ckpts={ckpts[0]}..{ckpts[-1]}@{SCAN_STEP} ({len(ckpts)}) "
          f"N={N_ENVS} height={HEIGHT} sweep={DX_SWEEP}\n", flush=True)

    summary = []
    for ck in ckpts:
        path = os.path.join(log_root, run_dir, f'model_{ck}.pt')
        if not os.path.isfile(path):
            print(f"  ckpt {ck}: MISSING, skip", flush=True); continue
        ppo_runner.load(path)
        policy = ppo_runner.get_inference_policy(device=env.device)
        actor_critic = ppo_runner.alg.actor_critic
        RS = int(round(replay_step_at(ck, float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
                                      float(env.start_freq), float(env.max_freq),
                                      float(env.dt), int(train_cfg.runner.num_steps_per_env))))
        res = sweep_one(env, policy, actor_critic, RS, spawn)
        hit_str = " ".join(f"{dx:.1f}:{res[dx]['hit']:.2f}" for dx in DX_SWEEP)
        score_hits = [res[dx]['hit'] for dx in SCORE_BAND]
        dead_hits = [res[dx]['hit'] for dx in DEAD_BAND]
        mean_hit = float(np.nanmean(score_hits))
        min_dead = float(np.nanmin(dead_hits)) if dead_hits else float('nan')
        # reliable_span = number of SCORE_BAND bins with hit>=0.9
        span = int(sum(1 for h in score_hits if h >= 0.9))
        air_peak = max(res[dx]['air'] for dx in DX_SWEEP)
        n_max = max(res[dx]['n'] for dx in DX_SWEEP)
        summary.append(dict(ck=ck, mean_hit=mean_hit, min_dead=min_dead, span=span,
                            air_peak=air_peak, n_max=n_max))
        print(f"  ckpt {ck:4d} | hit[{hit_str}] | meanHit(0.6-1.2)={mean_hit:.2f} "
              f"minDead(0.7-0.9)={min_dead:.2f} span>=.9={span}/7 | airPeak={air_peak:.3f} nMax={n_max}", flush=True)

    print("\n================= RANKING (best reliable checkpoint) =================", flush=True)
    # primary: meanHit desc; tiebreak: min_dead desc, then air_peak desc, then lower n_max
    ranked = sorted(summary, key=lambda r: (-r['mean_hit'], -r['min_dead'], -r['air_peak'], r['n_max']))
    for i, r in enumerate(ranked[:8]):
        tag = "  <== BEST" if i == 0 else ""
        print(f"  #{i+1} ckpt {r['ck']:4d} | meanHit={r['mean_hit']:.3f} minDead={r['min_dead']:.2f} "
              f"span={r['span']}/7 airPeak={r['air_peak']:.3f} nMax={r['n_max']}{tag}", flush=True)
    print("\n[note] meanHit=avg hit over DX0.6-1.2; minDead=worst hit in bimodal band 0.7-0.9 "
          "(high=no dead zone); span=#DX bins with hit>=0.9; airPeak=max true-air reach.", flush=True)


if __name__ == "__main__":
    main()
