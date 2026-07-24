"""Empirically nail the sign convention: base_ang_vel[:,1] vs d(pitch)/dt, and dump the
takeoff-instant + flight pitch dynamics for ONE jump. Removes ambiguity before touching
_reward_base_ang_vel_xy / _reward_pitch_level.

RUN
  export PATH=/home/qiwang/miniconda3/envs/sata-clean/bin:$PATH ; export PYTHONNOUSERSITE=1
  cd ~/torque_jump2/SATA/legged_gym
  CURVE_DX=1.0 CURVE_HEIGHT=0.5 python legged_gym/scripts/pitch_sign_check.py \
      --task=go2_omnijump_landing_torque --headless --load_run=<run> --checkpoint=<N>
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

DX = float(os.environ.get("CURVE_DX", 1.0))
HEIGHT = float(os.environ.get("CURVE_HEIGHT", 0.5))
MAXSTEPS = int(os.environ.get("CURVE_MAXSTEPS", 700))


def train_step_count_at_iter(ck, W, X, sf, mf, dt, nspe):
    r0 = nspe / (dt * sf); iw = W / r0
    if ck <= iw: return r0 * ck
    Delta = X - W; a = (mf - sf) / (2.0 * Delta); b = sf
    ife = iw + (dt / nspe) * (a * Delta * Delta + b * Delta)
    if ck <= ife:
        c = (nspe / dt) * (ck - iw); u = (-b + math.sqrt(b * b + 4 * a * c)) / (2 * a); return W + u
    return X + (nspe / (dt * mf)) * (ck - ife)


def main():
    args = get_args(); args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 4
    env_cfg.commands.resampling_time = 20.0
    env_cfg.commands.landing_dx_curriculum = False
    env_cfg.commands.landing_disp_x_stage2 = [DX, DX]
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

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic

    ck = int(args.checkpoint) if getattr(args, "checkpoint", None) else 4600
    RS = int(round(train_step_count_at_iter(ck, float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
                                            float(env.start_freq), float(env.max_freq),
                                            float(env.dt), int(train_cfg.runner.num_steps_per_env))))
    env.reset(); env.compute_observations(); obs = env.get_observations()

    PITCH, PRATE, PHASE = [], [], []
    took_off_step = None; landed_step = None
    for t in range(MAXSTEPS):
        env.step_count = RS; env.common_step_counter = RS
        with torch.no_grad():
            act = policy(obs.detach())
            comp = actor_critic.comp_forward(obs.detach())
            if comp is not None: env.comp_torque = comp
        obs, _, _, dones, infos = env.step(act.detach())
        pg_x = float(env.projected_gravity[0, 0])
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, pg_x))))
        PITCH.append(pitch); PRATE.append(float(env.base_ang_vel[0, 1]))
        ph = "push" if bool(env.jumping_state[0] and not env.has_taken_off[0]) else \
             ("air" if bool(env.airborne[0]) else ("land" if bool(env.landing[0]) else "-"))
        PHASE.append(ph)
        if took_off_step is None and bool(env.just_took_off[0]): took_off_step = t
        if landed_step is None and bool(env.just_landed[0]): landed_step = t; break

    P = np.array(PITCH); R = np.array(PRATE)
    dP = np.gradient(P)  # finite-diff of pitch (deg/step); sign vs R is what we want
    # correlation over the whole recorded window
    mask = np.ones_like(P, dtype=bool)
    corr = float(np.corrcoef(R[mask], dP[mask])[0, 1])
    print("\n================= SIGN CHECK =================")
    print(f"ckpt={ck} dx={DX}  took_off@{took_off_step} landed@{landed_step}  steps={len(P)}")
    print(f"corr( base_ang_vel[1] , d(pitch_deg)/dt ) = {corr:+.3f}")
    print("  >0  => base_ang_vel[1]>0 means pitch INCREASING (nose-DOWN grows)")
    print("  <0  => base_ang_vel[1]>0 means pitch DECREASING (nose-UP)  <-- comment would be WRONG")
    print(f"\npitch convention: >0 = nose-DOWN (projected_gravity[0]>0)")
    if took_off_step is not None:
        i = took_off_step
        print(f"\n@TAKEOFF (step {i}): pitch={P[i]:+.1f}deg  base_ang_vel[1]={R[i]:+.3f}  d(pitch)/dt={dP[i]:+.2f}deg/step")
    print(f"@LAND    (step {len(P)-1}): pitch={P[-1]:+.1f}deg  base_ang_vel[1]={R[-1]:+.3f}")
    # a few rows around takeoff and mid-flight
    print("\n step  phase   pitch   angvel[1]  dpitch")
    idxs = list(range(max(0, (took_off_step or 5) - 3), min(len(P), (took_off_step or 5) + 6)))
    if landed_step: idxs += list(range((took_off_step or 0) + (landed_step - (took_off_step or 0)) // 2,
                                       (took_off_step or 0) + (landed_step - (took_off_step or 0)) // 2 + 3))
    for i in sorted(set(idxs)):
        print(f" {i:4d}  {PHASE[i]:5s}  {P[i]:+6.1f}   {R[i]:+7.3f}   {dP[i]:+6.2f}")


if __name__ == "__main__":
    main()
