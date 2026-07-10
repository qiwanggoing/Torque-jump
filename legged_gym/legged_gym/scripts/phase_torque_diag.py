"""phase_torque_diag.py — PER-LEG, PER-JOINT torque-speed utilization vs the T-N envelope,
segmented by JUMP PHASE (squat -> push -> flight -> preland -> land), for a DOCUMENT.

Rolls out the landing jump HEADLESS, NOMINAL mass, DETERMINISTIC (no noise / no DR), and for every
one of the 12 leg joints records, IN EACH PHASE, how hard the joint was driven relative to the
actuator's REAL torque-speed (Hill / T-N) ceiling -- the SAME envelope go2_torque._compute_torques uses
(flat peak Y1 up to X1, linear decay to 0 at the no-load speed X2).

Because torques = activation_sign * max_effort(|v|),  |tau|/ceiling(|v|) == |activation| in [0,1]:
  act = 1.00  -> the joint delivered the FULL torque available AT ITS CURRENT SPEED (maxed the wall).
  act < 1     -> torque left on the table (headroom; the policy is not pushing that joint that phase).
Also reported per phase/joint:
  |v|/vlim%  : how close to the VELOCITY wall (100% = torque derated to ~0, the calf/knee's binding limit)
  |tau|/Y1%  : torque-axis usage (100% = at the flat peak torque)
  P/Pmax%    : mechanical propulsion power vs the motor's peak (~275W, same motor all 12 joints)

Phases (mutually exclusive, priority land>preland>flight>push>squat):
  SQUAT   = loading (jumping, not taken off) & base descending (vz<=0)  -- the wind-up
  PUSH    = loading & base ascending (vz>0)                             -- the propulsive extension
  FLIGHT  = airborne                                                    -- ballistic (legs shouldn't push)
  PRELAND = prelanding                                                  -- reaching for the ground
  LAND    = landing                                                     -- impact absorption

Run (nominal mass, headless):
  TQ_DX=0.8 TQ_HEIGHT=0.5 python legged_gym/scripts/phase_torque_diag.py \
      --task=go2_omnijump_landing_torque --load_run=RUN --checkpoint=N
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

DX = float(os.environ.get("TQ_DX", 0.8))
HEIGHT = float(os.environ.get("TQ_HEIGHT", 0.5))
N_ENVS = int(os.environ.get("TQ_N", 128))
STEPS = int(os.environ.get("TQ_STEPS", 900))

JTYPE = ["hip", "thigh", "calf"]     # dof (3l + k) -> JTYPE[k]
PHASES = ["SQUAT", "PUSH", "FLIGHT", "PRELAND", "LAND"]
NPH = len(PHASES)


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
    env_cfg.env.episode_length_s = 5
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
    ndof = env.num_dof
    dof_names = list(getattr(env, 'dof_names', [f"dof{i}" for i in range(ndof)]))
    leg_of = [dof_names[3 * l].split('_')[0] if 3 * l < len(dof_names) else f"L{l}" for l in range(ndof // 3)]
    vl = env.dof_vel_limits.detach().float()               # [ndof]
    tl = env.torque_limits.detach().float()                # [ndof] (calf already 1.917x-overridden)
    X1 = (vl * tn_knee).cpu().numpy()
    X2 = (vl * tn_max).cpu().numpy()
    Y1 = tl.cpu().numpy()
    Pmax = np.array([_env_max_power(X1[i], X2[i], Y1[i]) for i in range(ndof)])
    vlim_t = (vl * tn_max).float()                         # no-load speed on device
    Pmax_t = torch.tensor(Pmax, device=env.device, dtype=torch.float)

    print(f"\n[phase_torque_diag] {os.path.basename(_resume_path)} iter={ckpt_iter} step_count={REPLAY_STEP} | "
          f"N={N_ENVS} dx={DX} h={HEIGHT} | NOMINAL mass | tn(knee={tn_knee:.3f},max={tn_max:.3f})", flush=True)

    # ---- per (phase, dof) accumulators ----
    s_act = torch.zeros(NPH, ndof, device=env.device)   # sum |activation| (delivered fraction of T-N ceiling)
    s_vr = torch.zeros(NPH, ndof, device=env.device)    # sum |v|/vlim
    s_tr = torch.zeros(NPH, ndof, device=env.device)    # sum |tau|/Y1
    s_pr = torch.zeros(NPH, ndof, device=env.device)    # sum propulsion P/Pmax
    pk_act = torch.zeros(NPH, ndof, device=env.device)  # peak |activation|
    pk_tr = torch.zeros(NPH, ndof, device=env.device)   # peak |tau|/Y1
    cnt = torch.zeros(NPH, device=env.device)           # samples per phase

    # ---- env0 per-frame trajectory (for the TIME-SERIES curves) ----
    X1_t = (vl * tn_knee).float(); X2_t = (vl * tn_max).float(); Y1_t = tl.float()
    rec = []   # list of dict per step: t_ms, phase, v[ndof], tau[ndof], ceil[ndof]

    env.command_ranges["lin_vel_x"] = [DX, DX]
    env.reset()
    env.compute_observations()
    obs = env.get_observations()

    for t in range(STEPS):
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        env._takeoff_omega_on = True
        with torch.no_grad():
            actions = policy(obs.detach())
            comp = actor_critic.comp_forward(obs.detach())
            if comp is not None:
                env.comp_torque = comp
        obs, _, _, dones, infos = env.step(actions.detach())

        # ---- phase id per env (mutually exclusive; priority land>preland>flight>push>squat) ----
        vz = env.root_states[:, 9]
        loading = env.jumping_state & (~env.has_taken_off)
        phase = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        phase = torch.where(loading & (vz <= 0), torch.zeros_like(phase), phase)   # SQUAT
        phase = torch.where(loading & (vz > 0), torch.ones_like(phase), phase)     # PUSH
        phase = torch.where(env.airborne, torch.full_like(phase, 2), phase)        # FLIGHT
        if hasattr(env, "prelanding"):
            phase = torch.where(env.prelanding, torch.full_like(phase, 3), phase)  # PRELAND
        if hasattr(env, "landing"):
            phase = torch.where(env.landing, torch.full_like(phase, 4), phase)     # LAND

        vabs = env.dof_vel.abs()                                   # [N,ndof]
        aabs = env.activation_sign.abs()                          # [N,ndof] delivered fraction
        tr = (env.torques.abs() / tl)                             # [N,ndof] |tau|/Y1
        vr = vabs / vlim_t                                        # [N,ndof] |v|/vlim
        pr = (env.torques * env.dof_vel).clamp(min=0.0) / Pmax_t  # [N,ndof] P/Pmax (propulsion)

        for p in range(NPH):
            m = (phase == p)
            n = int(m.sum())
            if n == 0:
                continue
            cnt[p] += n
            s_act[p] += aabs[m].sum(dim=0)
            s_vr[p] += vr[m].sum(dim=0)
            s_tr[p] += tr[m].sum(dim=0)
            s_pr[p] += pr[m].sum(dim=0)
            pk_act[p] = torch.maximum(pk_act[p], aabs[m].max(dim=0).values)
            pk_tr[p] = torch.maximum(pk_tr[p], tr[m].max(dim=0).values)

        # ---- env0 frame for the time-series curves ----
        v0 = env.dof_vel[0].abs()
        ceil0 = torch.where(v0 < X1_t, Y1_t, Y1_t - Y1_t / (X2_t - X1_t) * (v0 - X1_t)).clamp(min=0.0)
        grf0 = env.contact_forces[0, env.feet_indices, 2].cpu().tolist()   # per-foot world-vertical GRF
        rec.append(dict(
            t=float(t * env.dt * 1000.0),
            phase=int(phase[0].item()),
            v=v0.cpu().tolist(),
            tau=env.torques[0].abs().cpu().tolist(),
            ceil=ceil0.cpu().tolist(),
            grf=grf0,
        ))

        # ---- per-phase per-FOOT GRF accumulation (all envs) : are rear feet loaded during push? ----
        if t == 0:
            fnames = [env.dof_names[3 * l].split('_')[0] for l in range(env.num_dof // 3)]
            s_grf = torch.zeros(NPH, len(env.feet_indices), device=env.device)
            grf_cnt = torch.zeros(NPH, device=env.device)
        fz = env.contact_forces[:, env.feet_indices, 2]                    # [N,4]
        for p in range(NPH):
            m = (phase == p)
            if bool(m.any()):
                s_grf[p] += fz[m].sum(dim=0)
                grf_cnt[p] += int(m.sum())

    denom = cnt.clamp(min=1).unsqueeze(1)
    m_act = (s_act / denom).cpu().numpy()
    m_vr = (s_vr / denom).cpu().numpy() * 100.0
    m_tr = (s_tr / denom).cpu().numpy() * 100.0
    m_pr = (s_pr / denom).cpu().numpy() * 100.0
    pk_act = pk_act.cpu().numpy()
    pk_tr = pk_tr.cpu().numpy() * 100.0
    cnt_np = cnt.cpu().numpy()

    def tbl(title, mat, fmt="{:6.2f}", note=""):
        print(f"\n=== {title} ===")
        if note:
            print("  " + note)
        print(f"  {'leg':>4} {'joint':>6} | " + " ".join(f"{PHASES[p]:>8}" for p in range(NPH)))
        for l in range(ndof // 3):
            for k in range(3):
                i = 3 * l + k
                cells = " ".join(fmt.format(mat[p, i]).rjust(8) for p in range(NPH))
                print(f"  {leg_of[l]:>4} {JTYPE[k]:>6} | {cells}")
            print(f"  {'':>4} {'':>6} |" + "-" * (9 * NPH))

    print(f"\n samples per phase: " + "  ".join(f"{PHASES[p]}={int(cnt_np[p])}" for p in range(NPH)))

    tbl("FORCE USED — mean act (delivered fraction of the T-N ceiling; 1.00=maxed the wall for its speed)",
        m_act, "{:6.2f}",
        "how much of the torque AVAILABLE AT ITS SPEED the joint actually delivered, averaged over the phase.")
    tbl("FORCE PEAK — peak act in the phase (the hardest single instant)",
        pk_act, "{:6.2f}")
    tbl("VELOCITY WALL — mean |v|/vlim %  (100% = at the speed wall, torque derated to ~0)",
        m_vr, "{:5.0f}%")
    tbl("TORQUE AXIS — mean |tau|/Y1 %  (100% = at the flat peak torque)",
        m_tr, "{:5.0f}%")
    tbl("TORQUE PEAK — peak |tau|/Y1 %  (max torque reached in the phase)",
        pk_tr, "{:5.0f}%")
    tbl("POWER — mean propulsion P/Pmax %  (mechanical output vs ~275W motor peak)",
        m_pr, "{:5.0f}%")

    print("\n  READ: PUSH is the propulsive phase — act~1 & |v|/vlim~100 on a joint = that joint is MAXED (used up);")
    print("        act<1 with |v|/vlim<100 = that joint has HEADROOM the policy isn't using (a free lever).")
    print("        FLIGHT/LAND torque is posture/absorption, not propulsion.\n")

    # ---- per-phase per-FOOT GRF table: is the REAR planted during the push? ----
    fden = grf_cnt.clamp(min=1).unsqueeze(1)
    mgrf = (s_grf / fden).cpu().numpy()
    print("============ per-FOOT vertical GRF (N) by phase — is the REAR loaded during PUSH? ============")
    print(f"  {'phase':>8} | " + " ".join(f"{fnames[l]:>8}" for l in range(len(fnames))) + "   front/rear")
    for p in range(NPH):
        fr_front = mgrf[p, 0] + mgrf[p, 1]
        fr_rear = mgrf[p, 2] + mgrf[p, 3]
        ratio = fr_front / max(fr_rear, 1e-3)
        print(f"  {PHASES[p]:>8} | " + " ".join(f"{mgrf[p, l]:8.0f}" for l in range(len(fnames))) +
              f"   {ratio:5.1f}x  (F={fr_front:.0f} R={fr_rear:.0f})")
    print("  if REAR GRF ~0 during PUSH => rear feet unloaded => rear thigh has nothing to push on (idle by kinematics).\n")

    # ---- dump env0 FIRST-JUMP trajectory as JSON (for the time-series curve chart) ----
    ph = [r["phase"] for r in rec]
    start = next((i for i, p in enumerate(ph) if p in (0, 1)), 0)          # first SQUAT/PUSH
    land_seen = False
    end = len(rec)
    for i in range(start, len(rec)):
        if ph[i] == 4:
            land_seen = True
        elif land_seen and ph[i] in (0, 1, -1):   # next jump begins / idles -> stop after this landing
            end = i
            break
    s = max(0, start - 4)
    window = rec[s:end]
    out = dict(
        run=os.path.basename(os.path.dirname(_resume_path)), ckpt=os.path.basename(_resume_path),
        dx=DX, height=HEIGHT, dt_ms=float(env.dt * 1000.0),
        phase_names=PHASES,
        leg=leg_of, jtype=JTYPE,
        vlim=(vl * tn_max).cpu().tolist(), Y1=Y1.tolist(), X1=X1.tolist(), X2=X2.tolist(),
        frames=window,
    )
    traj_out = os.environ.get("TQ_TRAJ_OUT", "/tmp/phase_traj.json")
    import json
    with open(traj_out, "w") as f:
        json.dump(out, f)
    print(f"[phase_torque_diag] env0 first-jump trajectory ({len(window)} frames, "
          f"{window[0]['t']:.0f}-{window[-1]['t']:.0f} ms) -> {traj_out}\n", flush=True)


if __name__ == '__main__':
    main()
