"""Per-joint torque/velocity/power TIME curves + saturation/jitter/flight analysis over ONE full jump.

HEADLESS, deterministic (mean policy), nominal mass, no noise/DR, faithful PD-replay (step_count pin).
Answers: (1) which joints saturate (velocity AND torque) in the push, (2) how jittery each thigh is,
(3) how much each leg swings during FLIGHT (rear vs front). Saves a PNG + prints analysis.

RUN
  export PATH=/home/qiwang/miniconda3/envs/sata/bin:$PATH ; unset PYTHONNOUSERSITE
  cd ~/torque_jump2/SATA/legged_gym
  CURVE_DX=1.2 CURVE_HEIGHT=0.5 python legged_gym/scripts/joint_time_curves.py \
      --task=go2_omnijump_landing_torque --headless \
      --load_run=Jul16_12-07-41_stage1_landing --checkpoint=4600
"""
import isaacgym
from isaacgym import gymapi
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
import os, math

DX = float(os.environ.get("CURVE_DX", 1.2))
HEIGHT = float(os.environ.get("CURVE_HEIGHT", 0.5))
OUT = os.environ.get("CURVE_OUT", "/tmp/joint_time_curves.png")
MAXSTEPS = int(os.environ.get("CURVE_MAXSTEPS", 700))
LEG_NAME = ["FL", "FR", "RL", "RR"]
HIP = [0, 3, 6, 9]; THIGH = [1, 4, 7, 10]; CALF = [2, 5, 8, 11]
FRONT = [0, 1]; REAR = [2, 3]


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

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    lr = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    ckp = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    path = get_load_path(log_root, load_run=lr, checkpoint=ckp)
    ck_iter = int(os.path.basename(path).replace('model_', '').replace('.pt', ''))

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo.get_inference_policy(device=env.device); ac = ppo.alg.actor_critic
    RS = int(round(train_step_count_at_iter(ck_iter, float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
                   float(env.start_freq), float(env.max_freq), float(env.dt), int(train_cfg.runner.num_steps_per_env))))
    tl = env.torque_limits.detach().float().cpu().numpy()
    vl = env.dof_vel_limits.detach().float().cpu().numpy()
    dt = float(env.dt)
    print(f"[curves] {os.path.basename(path)} iter{ck_iter} dx={DX} h={HEIGHT} dt={dt*1000:.1f}ms")
    print(f"[limits] hip: tau{tl[0]:.1f} vel{vl[0]:.1f} | thigh: tau{tl[1]:.1f} vel{vl[1]:.1f} | calf: tau{tl[2]:.1f} vel{vl[2]:.1f}")

    env.reset()
    env.commands[:, 0] = DX; env.commands[:, 1] = 0.0
    env.landing_target[:, 0] = env.root_states[:, 0] + DX; env.landing_target[:, 1] = env.root_states[:, 1]
    env.compute_observations(); obs = env.get_observations()

    TAU = []; VEL = []; POS = []; CON = []; H = []; VX = []; VZ = []; PITCH = []; PRATE = []
    t_takeoff = t_land = t_squat = None
    for step in range(MAXSTEPS):
        env.step_count = RS; env.common_step_counter = RS
        with torch.no_grad():
            a = policy(obs.detach()); comp = ac.comp_forward(obs.detach())
            if comp is not None: env.comp_torque = comp
        obs, _, _, dones, _ = env.step(a.detach())
        TAU.append(env.torques[0].cpu().numpy().copy()); VEL.append(env.dof_vel[0].cpu().numpy().copy())
        POS.append(env.dof_pos[0].cpu().numpy().copy()); CON.append(env._get_contact_state()[0].cpu().numpy().astype(float).copy())
        H.append(float(env.root_states[0, 2])); VX.append(float(env.root_states[0, 7])); VZ.append(float(env.root_states[0, 9]))
        PITCH.append(math.degrees(math.asin(max(-1.0, min(1.0, float(env.projected_gravity[0, 0]))))))  # >0 = nose-down
        PRATE.append(float(env.base_ang_vel[0, 1]))   # wy = pitch rate (rad/s)
        if t_squat is None and bool(env.jumping_state[0]): t_squat = step
        if t_takeoff is None and bool(env.just_took_off[0]): t_takeoff = step
        if t_takeoff is not None and t_land is None and bool(env.just_landed[0]): t_land = step
        if t_land is not None and step > t_land + 25: break
        if bool(dones[0]) and t_land is not None: break

    tau = np.array(TAU); vel = np.array(VEL); pos = np.array(POS); con = np.array(CON)
    H = np.array(H); VX = np.array(VX); pitch = np.array(PITCH); prate = np.array(PRATE); n = len(tau)
    if t_takeoff is None: print("!! never took off"); return
    t = (np.arange(n) - t_takeoff) * dt * 1000.0
    push = slice(t_squat if t_squat else 0, t_takeoff + 1)
    flight = slice(t_takeoff, (t_land if t_land else n))
    vr = np.abs(vel) / vl; tr = np.abs(tau) / tl   # ratios [n,12]

    # ================= SATURATION (push) =================
    print(f"\n================ SATURATION during PUSH (squat->takeoff, {(t_takeoff-(t_squat or t_takeoff))*dt*1000:.0f}ms) ================")
    print(f"{'joint':6s} {'peakVEL/lim':>11s} {'%push>0.9v':>10s} | {'peakTAU/lim':>11s} {'%push>0.9t':>10s}   (per-leg peak in [])")
    for name, idx in [("hip", HIP), ("thigh", THIGH), ("calf", CALF)]:
        vseg = vr[push][:, idx]; tseg = tr[push][:, idx]
        pv = vseg.max(); pt = tseg.max()
        pctv = float((vseg.max(1) >= 0.9).mean() * 100); pctt = float((tseg.max(1) >= 0.9).mean() * 100)
        perleg_v = " ".join(f"{LEG_NAME[l]}{vseg[:, k].max():.2f}" for k, l in enumerate(range(4)))
        print(f"{name:6s} {pv:11.2f} {pctv:10.0f} | {pt:11.2f} {pctt:10.0f}   [{perleg_v}]")

    # ================= THIGH JITTER (push) =================
    print(f"\n================ THIGH JITTER (push) — reversals & Δτ ================")
    print(f"{'leg':4s} {'#tau_reversals':>14s} {'mean|Δτ|/step':>14s} {'osc_period_ms':>13s}")
    for l in range(4):
        th = tau[push, 3*l+1]
        d = np.diff(th)
        sig = np.abs(d) > 2.0                     # significant moves (>2Nm/step)
        rev = int((np.diff(np.sign(d[sig if sig.any() else slice(0)])) != 0).sum()) if sig.any() else 0
        # simpler robust reversal count on smoothed sign
        s = np.sign(d); s = s[s != 0]
        rev = int((np.diff(s) != 0).sum())
        mdt = float(np.abs(d).mean())
        per = (2 * len(th) / max(rev, 1)) * dt * 1000
        print(f"{LEG_NAME[l]:4s} {rev:14d} {mdt:14.2f} {per:13.0f}")

    # ================= FLIGHT leg motion (rear vs front) =================
    print(f"\n================ FLIGHT leg motion (takeoff->land, {(t_land-t_takeoff)*dt*1000 if t_land else 0:.0f}ms) ================")
    print(f"{'leg':4s} {'hip_ROM':>8s} {'thigh_ROM':>9s} {'calf_ROM':>9s} {'mean|ω| hip/thi/calf (rad/s)':>30s}   (ROM=rad range)")
    rom = {}
    for l in range(4):
        hr = pos[flight, 3*l].ptp(); tr_ = pos[flight, 3*l+1].ptp(); cr = pos[flight, 3*l+2].ptp()
        wv = [np.abs(vel[flight, 3*l+j]).mean() for j in range(3)]
        rom[l] = (hr, tr_, cr)
        print(f"{LEG_NAME[l]:4s} {hr:8.2f} {tr_:9.2f} {cr:9.2f}   {wv[0]:8.1f}/{wv[1]:6.1f}/{wv[2]:6.1f}")
    fr_rom = np.mean([[rom[l][j] for j in range(3)] for l in FRONT], 0)
    rr_rom = np.mean([[rom[l][j] for j in range(3)] for l in REAR], 0)
    print(f">> FLIGHT ROM avg  front(hip/thi/calf)={fr_rom[0]:.2f}/{fr_rom[1]:.2f}/{fr_rom[2]:.2f}  rear={rr_rom[0]:.2f}/{rr_rom[1]:.2f}/{rr_rom[2]:.2f}  rear/front={rr_rom.sum()/max(fr_rom.sum(),1e-6):.2f}x")

    # ================= ATTITUDE: is the leg-swing compensating a bad-takeoff rotation? =================
    print(f"\n================ BODY PITCH / TAKEOFF ROTATION (>0 pitch = nose-down) ================")
    print(f"pitch@takeoff   = {pitch[t_takeoff]:+.1f} deg   | PITCH-RATE@takeoff = {prate[t_takeoff]:+.2f} rad/s  (this is imparted AT takeoff, cannot be undone by ballistics)")
    if t_land is not None:
        fl = slice(t_takeoff, t_land)
        print(f"pitch@land      = {pitch[t_land]:+.1f} deg   | flight pitch range = [{pitch[fl].min():+.1f}, {pitch[fl].max():+.1f}] deg  (max nose-down {pitch[fl].max():+.1f})")
        print(f"pitch-rate flight range = [{prate[fl].min():+.2f}, {prate[fl].max():+.2f}] rad/s  (if it swings a lot in flight = legs are actively counter-rotating the body)")
        # correlation: does rear-thigh angular velocity oppose base pitch-rate in flight? (compensation signature)
        rear_thigh_w = np.mean([vel[fl, 3*l+1] for l in REAR], 0)
        if len(rear_thigh_w) > 3 and np.std(rear_thigh_w) > 1e-6 and np.std(prate[fl]) > 1e-6:
            cc = float(np.corrcoef(rear_thigh_w, prate[fl])[0, 1])
            print(f"corr(rear-thigh ω , body pitch-rate) in flight = {cc:+.2f}  (strong |corr| => rear legs are steering pitch)")

    # ================= PLOT =================
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    w0 = max(0, (t_squat or t_takeoff) - 20); w1 = min(n, (t_land or t_takeoff) + 25)
    tc = {"hip": "#2ca02c", "thigh": "#1f77b4", "calf": "#d62728"}
    for name, idx in [("hip", HIP), ("thigh", THIGH), ("calf", CALF)]:
        ax[0].plot(t[w0:w1], vr[w0:w1][:, idx].mean(1), color=tc[name], label=f"{name} |ω|/lim", lw=1.8)
        ax[1].plot(t[w0:w1], tr[w0:w1][:, idx].mean(1), color=tc[name], label=f"{name} |τ|/lim", lw=1.8)
    ax[0].axhline(1, color="k", ls=":", lw=0.8); ax[0].set_ylabel("velocity /limit\n(mean 4 legs)"); ax[0].legend(fontsize=8); ax[0].set_title(f"model_{ck_iter} dx={DX}: which joints saturate (t=0 takeoff)")
    ax[1].axhline(1, color="k", ls=":", lw=0.8); ax[1].set_ylabel("torque /limit\n(mean 4 legs)"); ax[1].legend(fontsize=8)
    ax[2].plot(t[w0:w1], pitch[w0:w1], "k-", lw=2.0, label="body pitch (deg, +=nose-down)")
    ax[2].axhline(0, color="k", ls=":", lw=0.6)
    ax2b = ax[2].twinx()
    ax2b.plot(t[w0:w1], prate[w0:w1], color="orange", lw=1.4, label="pitch rate (rad/s)")
    rt = np.mean([pos[:, 3*l+1] for l in REAR], 0)
    ax2b.plot(t[w0:w1], rt[w0:w1], color="purple", lw=1.0, ls="--", alpha=0.6, label="rear-thigh angle (rad)")
    ax2b.axhline(0, color="orange", ls=":", lw=0.5)
    ax[2].set_ylabel("body pitch (deg)"); ax2b.set_ylabel("pitch rate / rear-thigh", color="orange")
    ax[2].legend(fontsize=8, loc="upper left"); ax2b.legend(fontsize=7, loc="upper right")
    ax[3].plot(t[w0:w1], H[w0:w1], "k-", label="height"); ax[3].plot(t[w0:w1], VX[w0:w1], "g-", label="vx")
    ax[3].plot(t[w0:w1], con.sum(1)[w0:w1]*0.1, color="gray", lw=1, label="#feet x0.1")
    ax[3].set_ylabel("base"); ax[3].set_xlabel("time rel takeoff (ms)"); ax[3].legend(fontsize=8, ncol=3)
    for a_ in ax:
        a_.axvline(0, color="purple", lw=1.2, alpha=0.7)
        if t_squat is not None: a_.axvline((t_squat-t_takeoff)*dt*1000, color="brown", ls="--", lw=1, alpha=0.6)
        if t_land is not None: a_.axvline((t_land-t_takeoff)*dt*1000, color="green", lw=1.2, alpha=0.6)
        a_.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT, dpi=110)
    print(f"\n[curves] saved -> {OUT}  (purple=takeoff brown=squat green=land)")


if __name__ == "__main__":
    main()
