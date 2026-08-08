"""Torque-speed (T-N) audit of a rollout dump -- did any joint leave the actuator envelope?

Works on BOTH dump formats (they carry the same `tau` / `dq` arrays):
  * MuJoCo : deploy_mujoco/sim2sim_landing_torque.py --dump rollout.npz
  * Isaac  : legged_gym/scripts/dump_landing_rollout.py            (DUMP_OUT=...)
so the same audit can be run on either engine and compared side by side.

WHAT IT CHECKS (three different questions -- don't confuse them):

  1. ENVELOPE VIOLATION -- |tau| above what the T-N curve allows at that |omega|.
     This is a BUG signal: the training env clamps every torque to the curve
     (go2_torque.py:237-257 / go2_omnijump_torque.py:883-905), so a rollout that
     exceeds it means the port applied a torque the trained actuator model never could.
     Envelope:  X1 = 0.45*v_lim, X2 = v_lim, Y1 = tau_lim, Y2 = 1.158*tau_lim (eccentric)
                flat at Y (= Y1 when tau and omega push the same way, else Y2) up to X1,
                then linear down to 0 at X2, and 0 beyond X2.

  2. AT THE TORQUE WALL -- |tau| >= 95% of the envelope. NOT a bug: it means the
     actuator is giving everything it has at that speed. High share = the motion is
     torque-limited.

  3. PAST THE SPEED LIMIT -- |omega| > v_lim (calf 15.65 rad/s, hip/thigh 30.0).
     NOT a violation either: the T-N curve gives ZERO torque there, so the joint is
     coasting/being driven by the dynamics, not by the motor. This is the "knee speed
     wall" that caps this robot's jump distance -- worth measuring, not fixing.

USAGE
  python deploy_mujoco/tn_check.py rollout.npz [more.npz ...] [--plot out.png]
  python deploy_mujoco/tn_check.py mj.npz isaac.npz --plot /tmp/tn.png
"""
import argparse
import os

import numpy as np


KNEE_REDUCTION = 1.917
TAU_LIM = np.array([20.2, 20.2, 20.2 * KNEE_REDUCTION] * 4)
VEL_LIM = np.array([30.0, 30.0, 30.0 / KNEE_REDUCTION] * 4)
TN_KNEE_SPEED_RATIO = 13.5 / 30.0
TN_PEAK_ECC_RATIO = 23.4 / 20.2
GROUPS = (("hip", [0, 3, 6, 9]), ("thigh", [1, 4, 7, 10]), ("calf", [2, 5, 8, 11]))
JOINT_NAMES = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
               "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]


def envelope(dq, tau):
    """Max |torque| the T-N model allows at this speed, given which way the torque pushes."""
    x1 = VEL_LIM * TN_KNEE_SPEED_RATIO
    x2 = VEL_LIM
    y1 = TAU_LIM
    y2 = TAU_LIM * TN_PEAK_ECC_RATIO
    # concentric (motor and motion the same way) -> Y1; eccentric (braking) -> the higher Y2
    same = (dq * tau) > 0
    y = np.where(same, y1, y2)
    k = -y / (x2 - x1)
    decayed = np.clip(k * (np.abs(dq) - x1) + y, 0.0, None)
    return np.where(np.abs(dq) < x1, y, decayed)


def audit(path):
    d = np.load(path)
    tau, dq = d["tau"].astype(np.float64), d["dq"].astype(np.float64)
    if "done" in d:                      # Isaac teleports the base on a done step
        keep = d["done"].reshape(-1) < 0.5
        tau, dq = tau[keep], dq[keep]
    # TIMING: both dumps record the torque APPLIED during step t together with the velocity
    # measured AFTER that step. The envelope that produced tau[t] was evaluated on the velocity the
    # joint had BEFORE the step = dq[t-1]. Checking tau[t] against dq[t] instead reports huge phantom
    # violations (a knee changes tens of rad/s in one 5 ms step) -- it flagged Isaac itself, which
    # clamps to the curve by construction, which is how this timing bug was caught.
    tau, dq_used, dq = tau[1:], dq[:-1], dq[1:]
    env = envelope(dq_used, tau)
    over = np.abs(tau) - env
    wall = np.abs(tau) >= 0.95 * np.maximum(env, 1e-9)
    past = np.abs(dq_used) > VEL_LIM

    print(f"\n=== {os.path.basename(path)}  ({len(tau)} steps) ===")
    print(f"  {'joint group':11s} {'max|tau|':>9s} {'lim':>7s} {'max|w|':>8s} {'lim':>7s} "
          f"{'%w>lim':>7s} {'%at wall':>9s} {'envelope violation':>19s}")
    worst = 0.0
    for name, idx in GROUPS:
        t, w, o = np.abs(tau[:, idx]), np.abs(dq_used[:, idx]), over[:, idx]
        worst = max(worst, float(o.max()))
        viol = (o > 1e-3)
        print(f"  {name:11s} {t.max():9.1f} {TAU_LIM[idx[0]]:7.1f} {w.max():8.1f} "
              f"{VEL_LIM[idx[0]]:7.2f} {100 * past[:, idx].mean():6.1f}% "
              f"{100 * wall[:, idx].mean():8.1f}% "
              f"{('%d steps, max +%.3f Nm' % (viol.sum(), o.max())) if viol.any() else 'none':>19s}")
    per_joint = np.abs(dq_used).max(axis=0) / VEL_LIM
    top = int(np.argmax(per_joint))
    print(f"  fastest joint: {JOINT_NAMES[top]} at {100 * per_joint[top]:.0f}% of its speed limit")
    if worst > 1e-3:
        print(f"  !! ENVELOPE VIOLATED by up to {worst:.3f} Nm -- the applied torque is outside the "
              f"trained actuator model; check the T-N transcription")
    else:
        print(f"  OK: every applied torque is inside the T-N envelope (max slack "
              f"{-float(over.max()):.3f} Nm)")
    return dq_used, tau, env


def plot(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (name, idx) in zip(axes, GROUPS):
        j = idx[0]
        v = np.linspace(0, VEL_LIM[j] * 1.6, 400)
        x1 = VEL_LIM[j] * TN_KNEE_SPEED_RATIO
        for y, lab, ls in ((TAU_LIM[j], "concentric limit (Y1)", "-"),
                           (TAU_LIM[j] * TN_PEAK_ECC_RATIO, "eccentric limit (Y2)", "--")):
            env_curve = np.where(v < x1, y, np.clip(-y / (VEL_LIM[j] - x1) * (v - x1) + y, 0, None))
            ax.plot(v, env_curve, ls, color="k", lw=1.3, label=lab)
        for (label, dq, tau, _), color in zip(runs, ("tab:blue", "tab:red", "tab:green")):
            ax.scatter(np.abs(dq[:, idx]).ravel(), np.abs(tau[:, idx]).ravel(), s=3, alpha=0.35,
                       color=color, label=label)
        ax.axvline(VEL_LIM[j], color="gray", lw=1, ls=":")
        ax.set_title(f"{name}  (tau_lim {TAU_LIM[j]:.1f} Nm, v_lim {VEL_LIM[j]:.2f} rad/s)")
        ax.set_xlabel("|joint speed| [rad/s]")
        ax.set_ylabel("|torque| [Nm]")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\n[plot] torque-speed scatter vs the T-N envelope -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dumps", nargs="+", help="rollout .npz files (MuJoCo and/or Isaac)")
    p.add_argument("--plot", default=None, help="write a torque-speed scatter PNG here")
    args = p.parse_args()

    runs = []
    for path in args.dumps:
        dq, tau, env = audit(path)
        runs.append((os.path.basename(path), dq, tau, env))
    if args.plot:
        plot(runs, args.plot)


if __name__ == "__main__":
    main()
