"""torque_diag.py — Torque-speed (Hill / T-N) diagnostic DURING the jump push, PER-LEG (all 12 joints).

Rolls out the landing jump (HEADLESS, NOMINAL mass, deterministic, no noise) and records, over the
PUSH phase (jumping_state & not-yet-taken-off), EACH of the 12 leg joints' (|speed|, |torque|, power).
Reports every joint's saturation vs the actuator's REAL torque-speed envelope — the SAME curve
go2_torque._compute_torques uses:
    flat peak Y1 up to X1, then linear decay to 0 at X2 (no-load speed);  eccentric branch = Y2>Y1.
Because `torques = activation_sign * max_effort`, |tau|/cap == |act| (0..1): a point ON the envelope
means fully commanded; BELOW it = torque left on the table.

Per-joint HEADROOM = how much of that joint's T-N ceiling is UNUSED:
  - |v|/vlim  (velocity axis)   - the binding wall for the calf (knee).
  - |tau|/Y1  (torque axis).
  - propulsion power |tau*w|+ vs Pmax (same motor on all 12 joints => same ~Pmax; gearing trades
    torque<->speed, not power).

Run (nominal mass, headless):
  python legged_gym/scripts/torque_diag.py --task=go2_omnijump_landing_torque \
         --load_run=Jul05_17-47-45_stage1_landing --checkpoint=3000
  TQ_DX=0.9 TQ_HEIGHT=0.7 python legged_gym/scripts/torque_diag.py --task=... --load_run=... --checkpoint=N
Output PNG: /tmp/torque_speed_curve.png   (override with TQ_OUT=...)
"""
import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math
import numpy as np

DX = float(os.environ.get("TQ_DX", 0.7))
HEIGHT = float(os.environ.get("TQ_HEIGHT", 0.7))
N_ENVS = int(os.environ.get("TQ_N", 64))
STEPS = int(os.environ.get("TQ_STEPS", 500))
OUT = os.environ.get("TQ_OUT", "/tmp/torque_speed_curve.png")

# 12 dofs are ordered per-leg in blocks of 3: [hip, thigh, calf] x 4 legs. (CALF=3l+2, THIGH=3l+1, HIP=3l)
JTYPE = ["hip", "thigh", "calf"]     # dof (3l + k) -> JTYPE[k]


def train_step_count_at_iter(target_iter, W, X, sf, mf, dt, nspe):
    """TRAINING's NON-LINEAR step_count at iter N (freq ramp sf->mf) -> faithful PD replay. From play_landing.py."""
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
    return X + nspe / (dt * mf) * (target_iter - iter_fade_end)


def _envelope(vgrid, X1, X2, Y):
    """Max |tau| available at |v| for peak torque Y: flat to X1, linear to 0 at X2."""
    e = np.where(vgrid < X1, Y, Y - Y / (X2 - X1) * (vgrid - X1))
    return np.clip(e, 0.0, None)


def _env_max_power(X1, X2, Y1):
    """Peak mechanical power (W) of the concentric envelope."""
    v_star = X2 / 2.0
    p_dec = Y1 * v_star * (X2 - v_star) / (X2 - X1) if v_star > X1 else Y1 * X1
    return max(Y1 * X1, p_dec)


def main():
    args = get_args()
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    env_cfg.env.num_envs = N_ENVS
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
    env_cfg.domain_rand.randomize_base_mass = False      # NOMINAL mass = deploy condition
    env_cfg.rewards.landing_tilt_terminate = 0.0
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

    # ---- T-N envelope params, per-dof, straight from the env (exact) ----
    C = env.cfg.control
    tn_knee = float(getattr(C, 'tn_knee_speed_ratio', 0.45))
    tn_max = float(getattr(C, 'tn_max_speed_ratio', 1.0))
    tn_ecc = float(getattr(C, 'tn_peak_eccentric_ratio', 1.1584))
    ndof = env.num_dof
    dof_names = list(getattr(env, 'dof_names', [f"dof{i}" for i in range(ndof)]))
    leg_of = [dof_names[3 * l].split('_')[0] if 3 * l < len(dof_names) else f"L{l}" for l in range(ndof // 3)]
    vl = env.dof_vel_limits.detach().float()               # [ndof]
    tl = env.torque_limits.detach().float()                # [ndof] (calf already 1.917x-overridden, scale=1)
    X1 = (vl * tn_knee).cpu().numpy()
    X2 = (vl * tn_max).cpu().numpy()
    Y1 = tl.cpu().numpy()
    print(f"\n[torque_diag] {os.path.basename(_resume_path)} iter={ckpt_iter} step_count={REPLAY_STEP} | "
          f"N={N_ENVS} dx={DX} h={HEIGHT} | NOMINAL mass | tn(knee={tn_knee:.3f},max={tn_max:.3f},ecc={tn_ecc:.4f})",
          flush=True)

    # ---- per-dof storage (all 12 joints) ----
    pk_v = torch.zeros(ndof, device=env.device)            # peak |v|
    pk_tau = torch.zeros(ndof, device=env.device)          # peak |tau|
    pk_pow = torch.zeros(ndof, device=env.device)          # peak POSITIVE (propulsion) power tau*w
    va_at_pv = torch.zeros(ndof, device=env.device)        # |act| at the peak-|v| sample (per dof)
    pk_cmd = torch.zeros(ndof, device=env.device)          # peak |torques_action|/Y1 = |tanh INPUT| (raw UNCLAMPED intent)
    pk_act = torch.zeros(ndof, device=env.device)          # peak |activation_sign| = tanh OUTPUT (delivered fraction)
    traj = [[[], [], []] for _ in range(ndof)]             # env0 per-dof: |v|, |tau|, step
    traj_done = False
    # ---- STROKE view: bin push samples by BASE HEIGHT (squat bottom -> leg extension), per-dof utilization ----
    band_edges = torch.tensor([0.20, 0.25, 0.30, 0.35], device=env.device)
    band_lbl = ["<0.20", "0.20-.25", "0.25-.30", "0.30-.35", ">0.35"]
    nb = band_edges.numel() + 1
    b_act = torch.zeros(nb, ndof, device=env.device)       # sum |act| per (band,dof) -> mean = utilization in that band
    b_vr = torch.zeros(nb, ndof, device=env.device)        # sum |v|/vlim
    b_tr = torch.zeros(nb, ndof, device=env.device)        # sum |tau|/Y1
    b_cnt = torch.zeros(nb, device=env.device)             # samples per band
    vlim_t = (vl * tn_max).float()                         # [ndof] no-load speed on device
    # ---- STABILITY test: does driving the REAR THIGH correlate with body PITCH RATE? (per band) ----
    rt_idx = [i for i, n in enumerate(dof_names) if 'thigh' in n.lower() and ('rl' in n.lower() or 'rr' in n.lower())]
    if len(rt_idx) == 0:
        rt_idx = [7, 10]
    c_a = torch.zeros(nb, device=env.device)   # sum rear-thigh act
    c_p = torch.zeros(nb, device=env.device)   # sum pitch rate
    c_aa = torch.zeros(nb, device=env.device)  # sum act^2
    c_pp = torch.zeros(nb, device=env.device)  # sum pitchrate^2
    c_ap = torch.zeros(nb, device=env.device)  # sum act*pitchrate
    c_pg = torch.zeros(nb, device=env.device)  # sum forward-tilt (proj grav x)
    # ---- per-band REWARD TERM values (scaled/step) through the push stroke ----
    track_rew = ["four_leg_push", "forward_reach", "takeoff_velocity_match", "projected_peak",
                 "stance_squat", "base_ang_vel_xy", "pitch_level", "clean_takeoff_bonus"]
    rfmap = {n: f for n, f in zip(getattr(env, "reward_names", []), getattr(env, "reward_functions", []))}
    rscale = getattr(env, "reward_scales", {})
    track_rew = [r for r in track_rew if r in rfmap]
    b_rew = {r: torch.zeros(nb, device=env.device) for r in track_rew}

    env.command_ranges["lin_vel_x"] = [DX, DX]
    env.reset()
    env.compute_observations()
    obs = env.get_observations()

    for t in range(STEPS):
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        env._takeoff_omega_on = True    # force the succ-latched gate ON (as in late training) so four_leg_push +
                                        # the push-phase ω penalty actually fire -> their real per-band values show
        with torch.no_grad():
            actions = policy(obs.detach())
            comp = actor_critic.comp_forward(obs.detach())
            if comp is not None:
                env.comp_torque = comp
        obs, _, _, dones, infos = env.step(actions.detach())

        loading = env.jumping_state & (~env.has_taken_off)      # [N]
        if bool(loading.any()):
            vabs = env.dof_vel[loading].abs()                   # [n,ndof]
            tabs = env.torques[loading].abs()
            aabs = env.activation_sign[loading].abs()
            pw = (env.torques[loading] * env.dof_vel[loading]).clamp(min=0.0)   # propulsion power only
            vmax, vidx = vabs.max(dim=0)                        # per-dof peak |v| + which row
            newpk = vmax > pk_v
            va_at_pv = torch.where(newpk, aabs[vidx, torch.arange(ndof, device=env.device)], va_at_pv)
            pk_v = torch.maximum(pk_v, vmax)
            pk_tau = torch.maximum(pk_tau, tabs.max(dim=0).values)
            pk_pow = torch.maximum(pk_pow, pw.max(dim=0).values)
            cmdn = (env.torques_action[loading].abs() / tl)     # [n,ndof] |raw command|/Y1 = |tanh input|
            pk_cmd = torch.maximum(pk_cmd, cmdn.max(dim=0).values)
            pk_act = torch.maximum(pk_act, aabs.max(dim=0).values)
            # STROKE binning: which base-height band is each loading env in right now
            bh = env.root_states[loading, 2]                    # [n] base height
            bidx = torch.bucketize(bh, band_edges)              # [n] -> 0..nb-1
            b_act.index_add_(0, bidx, aabs)                     # aabs [n,ndof]
            b_vr.index_add_(0, bidx, vabs / vlim_t)
            b_tr.index_add_(0, bidx, tabs / tl)
            b_cnt.index_add_(0, bidx, torch.ones_like(bh))
            ra = aabs[:, rt_idx].mean(dim=1)                    # [n] rear-thigh act this sample
            pr = env.base_ang_vel[loading, 1]                   # [n] pitch RATE (base frame y)
            pg = env.projected_gravity[loading, 0]              # [n] forward-tilt proxy
            c_a.index_add_(0, bidx, ra); c_p.index_add_(0, bidx, pr)
            c_aa.index_add_(0, bidx, ra * ra); c_pp.index_add_(0, bidx, pr * pr)
            c_ap.index_add_(0, bidx, ra * pr); c_pg.index_add_(0, bidx, pg)
            for r in track_rew:
                try:
                    rv = rfmap[r]() * float(rscale.get(r, 1.0))   # [N] scaled per-step reward
                    b_rew[r].index_add_(0, bidx, rv[loading])
                except Exception:
                    pass
            if (not traj_done) and bool(loading[0]):
                for i in range(ndof):
                    traj[i][0].append(float(env.dof_vel[0, i].abs()))
                    traj[i][1].append(float(env.torques[0, i].abs()))
                    traj[i][2].append(t)
        if (not traj_done) and bool(env.has_taken_off[0]) and len(traj[2][0]) > 0:
            traj_done = True

    pk_v = pk_v.cpu().numpy(); pk_tau = pk_tau.cpu().numpy()
    pk_pow = pk_pow.cpu().numpy(); va_at_pv = va_at_pv.cpu().numpy()
    pk_cmd = pk_cmd.cpu().numpy(); pk_act = pk_act.cpu().numpy()

    # ---- per-leg text table (all 12 joints) ----
    print("\n================ PER-LEG torque-speed usage during PUSH (nominal, all envs) ================")
    print(f"{'leg':>4} {'joint':>6} | {'vlim':>6} {'pk|v|':>6} {'|v|/vlim':>8} {'v_SPARE':>8} | "
          f"{'Y1':>5} {'pk|tau|':>7} {'tau/Y1':>7} | {'cmd/Y1':>7} {'act':>5} | {'Pmax':>5} {'Ppk':>5} {'P/Pmax':>7} | verdict")
    for l in range(ndof // 3):
        for k in range(3):
            i = 3 * l + k
            Pmax = _env_max_power(X1[i], X2[i], Y1[i])
            vr = pk_v[i] / X2[i] * 100.0
            tr = pk_tau[i] / Y1[i] * 100.0
            pr = pk_pow[i] / Pmax * 100.0
            spare = max(0.0, 100.0 - vr)
            vd = "VEL WALL" if vr >= 100 else ("near vlim" if vr >= 90 else "headroom")
            note = "  (P>Pmax: past-wall/StepH)" if pr > 105 else ""
            print(f"{leg_of[l]:>4} {JTYPE[k]:>6} | {X2[i]:6.2f} {pk_v[i]:6.2f} {vr:7.0f}% {spare:7.0f}% | "
                  f"{Y1[i]:5.1f} {pk_tau[i]:7.1f} {tr:6.0f}% | {pk_cmd[i]:6.2f} {pk_act[i]:5.2f} | "
                  f"{Pmax:5.0f} {pk_pow[i]:5.0f} {pr:6.0f}% | {vd}{note}")
        print(f"{'':>4} {'':>6} |" + "-" * 92)
    print("  v_SPARE = unused speed ceiling. calf ~0 (wall) ; hip/thigh spare = idle speed axis.")
    print("  cmd/Y1 = |raw commanded torque|/Y1 = the tanh INPUT (policy's UNCLAMPED intent): >~2 => tanh-saturated")
    print("           (flat gradient, 'maxed & blocked') ; <1 => headroom left (has gradient, policy not pushing).")
    print("  act    = delivered |activation| = tanh(cmd) = fraction of the T-N ceiling actually applied (->1 = maxed).")
    print("  same motor everywhere => Pmax~275W all joints. P>100% on calf = artifact of |v|>X2 + StepH comp.\n")

    # ---- STROKE view: is the leg maxed THROUGHOUT the extension, or only for one instant? ----
    denom = b_cnt.clamp(min=1).unsqueeze(1)
    m_act = (b_act / denom).cpu().numpy()
    m_vr = (b_vr / denom).cpu().numpy() * 100.0
    m_tr = (b_tr / denom).cpu().numpy() * 100.0
    b_cnt_np = b_cnt.cpu().numpy()
    print("============ UTILIZATION THROUGH THE PUSH STROKE (squat bottom -> leg extension) ============")
    print("  cell = act (mean T-N utilization, ->1.0 = delivering the FULL ceiling for its speed) | |v|/vlim% (velocity wall).")
    print("  READ: act ~1 across ALL bands = maxed the WHOLE stroke (used up). High in only one band = momentary (capacity left).")
    for k in range(3):   # hip, thigh, calf
        print(f"\n  --- {JTYPE[k]} ---   (band = base height m; n = push samples)")
        print(f"  {'band(m)':>9} {'n':>6} | " + " ".join(f"{leg_of[l]:>11}" for l in range(ndof // 3)) + "   [act | v/vlim%]")
        for bnd in range(nb):
            cells = []
            for l in range(ndof // 3):
                i = 3 * l + k
                cells.append(f"{m_act[bnd, i]:4.2f}|{m_vr[bnd, i]:3.0f}")
            print(f"  {band_lbl[bnd]:>9} {int(b_cnt_np[bnd]):6d} | " + " ".join(f"{c:>11}" for c in cells))
    print("")

    # ---- Is STABILITY (body pitch) the reason the rear thigh stays idle? ----
    n_ = b_cnt.clamp(min=1)
    corr = (c_ap * b_cnt - c_a * c_p) / torch.sqrt(
        (c_aa * b_cnt - c_a ** 2).clamp(min=1e-9) * (c_pp * b_cnt - c_p ** 2).clamp(min=1e-9))
    rta = (c_a / n_).cpu().numpy()
    mean_pr = (c_p / n_).cpu().numpy()
    mean_pg = (c_pg / n_).cpu().numpy()
    corr = corr.cpu().numpy()
    print("============ REAR-THIGH drive vs body PITCH per band (stability suppressor test) ============")
    print(f"  {'band(m)':>9} {'n':>6} {'rearThAct':>10} {'pitchRate':>10} {'fwdTilt':>8} {'corr(act,pitchRate)':>20}")
    for bnd in range(nb):
        print(f"  {band_lbl[bnd]:>9} {int(b_cnt_np[bnd]):6d} {rta[bnd]:10.2f} {mean_pr[bnd]:10.2f} {mean_pg[bnd]:8.3f} {corr[bnd]:20.2f}")
    print("  corr >0 & rising = MORE rear-thigh drive <-> MORE forward pitch rate => stability holds it back (needs pitch-safe recruit).")
    print("  corr ~0 = rear thigh idle for another reason (a FREE lever). pitchRate sign is base-frame; magnitude = 'rotating'.\n")

    # ---- which REWARDS drive / oppose the push, through the stroke? (scaled per-step means) ----
    if track_rew:
        print("============ REWARD TERMS (scaled / step) THROUGH THE PUSH STROKE ============")
        print("  +driver  -penalty ; watch which terms rise as the legs extend (0.20 -> >0.35).")
        print(f"  {'term':>22} | " + " ".join(f"{band_lbl[b]:>9}" for b in range(nb)))
        for r in track_rew:
            m = (b_rew[r] / n_).cpu().numpy()
            print(f"  {r:>22} | " + " ".join(f"{m[b]:9.4f}" for b in range(nb)))
        print("")

    # ---- plot: 3 subplots (hip/thigh/calf), each with all 4 legs' env0 push trajectory in its own color ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        legcolors = ['C0', 'C1', 'C2', 'C3']
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
        for ax, k in zip(axes, range(3)):          # k: 0 hip,1 thigh,2 calf
            i0 = k                                  # representative dof for envelope (all legs same type share limits)
            e_X1, e_X2, e_Y1 = X1[i0], X2[i0], Y1[i0]
            vg = np.linspace(0, e_X2 * 1.35, 240)
            ax.plot(vg, _envelope(vg, e_X1, e_X2, e_Y1), color='k', lw=2.0, label='cap Y1')
            ax.plot(vg, _envelope(vg, e_X1, e_X2, e_Y1 * tn_ecc), color='k', lw=1.0, ls='--', alpha=0.5, label='cap Y2')
            ax.axvline(e_X2, color='r', ls=':', lw=1.3, label=f'vlim={e_X2:.1f}')
            for l in range(ndof // 3):
                i = 3 * l + k
                if traj[i][0]:
                    tv = np.array(traj[i][0]); tt = np.array(traj[i][1])
                    ax.plot(tv, tt, '-o', color=legcolors[l % 4], ms=4, lw=0.8, alpha=0.85, label=leg_of[l])
            ax.set_title(f"{JTYPE[k]}", fontsize=12)
            ax.set_xlabel("|joint speed| (rad/s)"); ax.set_ylabel("|torque| (Nm)")
            ax.set_xlim(0, e_X2 * 1.35); ax.set_ylim(0, e_Y1 * tn_ecc * 1.15)
            ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2, loc='upper right')
        fig.suptitle(f"Per-leg torque-speed during PUSH | {os.path.basename(_resume_path)} | dx={DX} h={HEIGHT} | nominal",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUT, dpi=110)
        print(f"[torque_diag] saved plot -> {OUT}", flush=True)
    except Exception as e:
        print(f"[torque_diag] plot skipped ({type(e).__name__}: {e}); text table above is the result.", flush=True)


if __name__ == '__main__':
    main()
