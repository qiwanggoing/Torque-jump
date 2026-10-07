"""Phase A: build a push-off REFERENCE TRAJECTORY and check it is physically achievable.

WHY A REFERENCE AT ALL
----------------------
Measured on the current best policy (Aug18 history, model_5000), at the squat bottom only 2 of 4
feet are on the ground -- the front feet have already left. Both the near-standing jump (dx=0.5) and
the bounding one (dx=1.0) are REAR-ONLY pushes, so half the actuators contribute nothing to the
launch. That is the single biggest structural defect left, and it is exactly the thing a scalar
reward cannot express: `four_leg_push` was tried at weight 5 and 10 and had to be killed (it demands
>=90 N of VERTICAL force from all four feet SIMULTANEOUSLY, which fights a forward launch, and the
policy learned to jump sideways). A trajectory does not ask for simultaneity -- it says WHEN each
leg extends. Front-then-rear, both loaded at the start.

WHAT THIS SCRIPT DOES (offline, no training, no Isaac)
  --check    replay q_ref through a stiff PD under the REAL T-N torque envelope and report whether
             it actually leaves the ground, at what speed/angle, how far, and whether any joint was
             asked for more torque than the actuator can give. This is the go/no-go: if a reference
             cannot jump when tracked perfectly, no amount of RL will make it jump.
  --viewer   watch it
  --compare  overlay the push measured from an Isaac dump (dump_landing_rollout.py .npz)

The reference is hand-authored on top of the existing 2-link IK, exactly the way Li 2023 hand-made
their Cassie animation ("only kinematically feasible, not dynamically feasible") -- the difference is
that --check then measures the dynamics, which is the part Fuchioka gets from TO constraints.

USAGE  (needs the sata env's python, WITHOUT PYTHONNOUSERSITE -- mujoco lives in ~/.local)
  python deploy_mujoco/ref_traj.py --check
  python deploy_mujoco/ref_traj.py --check --t-push 0.20 --h-ext 0.37 --rear-delay 0.30
  python deploy_mujoco/ref_traj.py --viewer
  python deploy_mujoco/ref_traj.py --compare /path/to/push_dx05.npz
"""
import argparse
import math
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.normpath(os.path.join(HERE, "..", "..", "my_go2_jump", "resources", "robots", "go2", "go2", "scene.xml"))  # scene.xml, NOT go2.xml: go2.xml has no floor

# --- geometry / actuator constants: same numbers the env uses -------------------------------
L1 = L2 = 0.213                  # rewards.ik_thigh_length / ik_calf_length
FOOT_X = 0.02                    # rewards.ik_nominal_foot_x
KP, KD = 40.0, 1.2               # control.stiffness / damping  ("joint")
KNEE_RED = 1.917
TAU_LIM = np.array([20.2, 20.2, 20.2 * KNEE_RED] * 4)
VEL_LIM = np.array([30.0, 30.0, 30.0 / KNEE_RED] * 4)
TN_X1_RATIO, TN_Y2_RATIO = 13.5 / 30.0, 23.4 / 20.2
CALF_MIN, CALF_MAX = -2.7227, -0.83776
DT = 0.005
FEET = ["FL", "FR", "RL", "RR"]
HIP_DEFAULT = np.array([0.1, -0.1, 0.1, -0.1])          # FL FR RL RR (init_state.default_joint_angles)


def ik(foot_height, foot_x=FOOT_X):
    """2-link sagittal IK -> (thigh, calf), same convention as
    GO2OmniJumpTorque._solve_pose_from_foot_height, but with the foot's FORE-AFT position exposed.

    That extra argument is the whole difference between a jump and a hop. The env's keyframes fix
    foot_x at the nominal 0.02 m, i.e. the leg only ever extends STRAIGHT DOWN, so the ground
    reaction is purely vertical and there is no forward impulse at all -- the first version of this
    reference measured vx = -1.41 m/s (it jumped BACKWARDS). Forward momentum comes from sweeping the
    foot BACKWARD under the body during the push, so foot_x has to be part of the trajectory.

    !! SIGN, measured against the MJCF, not assumed: foot_x is the offset BEHIND the hip.
       foot_x = +0.02 puts the foot 0.02 m behind the thigh root; foot_x = -0.10 puts it 0.10 m in
       FRONT. So a backward foot sweep (= forward thrust) means foot_x INCREASING over the push.
    """
    z = -foot_height
    reach = np.clip(math.hypot(foot_x, z), 1e-4, L1 + L2 - 1e-4)
    cos_knee = np.clip((reach * reach - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0)
    calf = -math.acos(cos_knee)
    thigh = math.atan2(z, foot_x) - math.atan2(L2 * math.sin(calf), L1 + L2 * math.cos(calf)) + math.pi / 2.0
    return thigh, calf


def max_foot_height():
    """Longest the leg can get: the calf cannot straighten past its hard stop (-48 deg)."""
    reach2 = L1 * L1 + L2 * L2 + 2 * L1 * L2 * math.cos(CALF_MAX)
    return math.sqrt(max(reach2 - FOOT_X * FOOT_X, 0.0))





R_MAX = max_foot_height()   # 0.389 m -- the calf stop, not L1+L2


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def polar_to_hx(r, theta_deg):
    """Foot target in POLAR form -> (height, foot_x). r = leg extension, theta = swing from vertical
    (positive = foot swept BACKWARD, which is the direction that produces forward thrust).

    Cartesian (h, x) was the first parameterisation and it was a trap: nothing stopped a target from
    leaving the leg's workspace. h=0.384 with x=0.32 needs a 0.50 m leg -- the leg is 0.426 m -- so
    the IK silently clipped, the trajectory got a kink, and it ended up demanding knee speeds of
    110 rad/s (700% of the 15.65 rad/s limit). In polar form r is the leg length itself, so capping
    r at max_foot_height() makes every target reachable by construction.
    """
    th = math.radians(theta_deg)
    return r * math.cos(th), r * math.sin(th)


def build_reference(t_push=0.18, h_squat=0.10, h_ext=None, front_win=(0.00, 0.60),
                    rear_win=(0.30, 1.00), x_start=FOOT_X, x_end=FOOT_X, n=None):
    """q_ref(phi) for a FOUR-LEG, front-then-rear sequenced push.

    Each leg's foot height goes h_squat -> h_ext on its OWN smoothstep window, so the front pair
    finishes extending while the rear pair is still driving. Both pairs start LOADED (phi=0 is the
    four-feet-down squat), which is the whole point -- the current policy has the front feet already
    airborne at the squat bottom, so their extension does no work on the ground.

    Returns (phi[N], q_ref[N,12]) with the env's joint order FL/FR/RL/RR x (hip, thigh, calf).
    """
    if h_ext is None:
        h_ext = max_foot_height() - 0.005          # just short of the calf stop
    n = n or int(round(t_push / DT)) + 1
    phi = np.linspace(0.0, 1.0, n)
    q = np.zeros((n, 12))
    for leg, name in enumerate(FEET):
        a, b = front_win if name[0] == "F" else rear_win
        s = smoothstep((phi - a) / max(b - a, 1e-6))
        r0 = min(math.hypot(h_squat, x_start), R_MAX)
        r1 = min(math.hypot(h_ext, x_end), R_MAX)
        t0 = math.degrees(math.atan2(x_start, h_squat))
        t1 = math.degrees(math.atan2(x_end, h_ext))
        for k in range(n):
            hh, xx = polar_to_hx(r0 + (r1 - r0) * s[k], t0 + (t1 - t0) * s[k])
            th, cf = ik(hh, xx)
            q[k, 3 * leg + 0] = HIP_DEFAULT[leg]
            q[k, 3 * leg + 1] = th
            q[k, 3 * leg + 2] = cf
    return phi, q


# ------------------------------------------------------------------------------------------
def tn_clamp(tau_cmd, dq):
    """Clip a commanded torque to the real Go2 torque-speed envelope (same law as the env)."""
    s = np.sign(tau_cmd)
    x1, x2 = VEL_LIM * TN_X1_RATIO, VEL_LIM
    y1, y2 = TAU_LIM, TAU_LIM * TN_Y2_RATIO
    same_dir = (dq * s) > 0
    peak = np.where(same_dir, y1, y2)
    va = np.abs(dq)
    decay = np.clip(-peak / (x2 - x1) * (va - x1) + peak, 0.0, None)
    envelope = np.where(va < x1, peak, decay)
    return np.clip(tau_cmd, -envelope, envelope), envelope


def _load_robot(xml):
    """Reuse the sim2sim MjRobot: it already resolves joint/actuator/foot-geom indices, matches
    Isaac's pyramidal contact cone (MuJoCo's default elliptic + impratio 100 makes the push slip)
    and trims the mass to the URDF nominal."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s2s", os.path.join(HERE, "sim2sim_landing_torque.py"))
    s2s = importlib.util.module_from_spec(spec)
    sys.modules["s2s"] = s2s
    spec.loader.exec_module(s2s)
    rb = s2s.MjRobot(xml, friction=1.0)
    rb.match_isaac_contacts()
    rb.match_urdf_mass()
    return rb


def replay(xml, phi, q_ref, settle=0.6, viewer=False, verbose=True):
    rb = _load_robot(xml)
    model, data = rb.model, rb.data
    model.opt.timestep = DT

    # start in the reference's phi=0 squat and let it settle so all four feet are genuinely loaded
    data.qpos[:3] = [0, 0, 0.30]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[rb.q_ids] = q_ref[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    v = None
    if viewer:
        from mujoco import viewer as mj_viewer
        v = mj_viewer.launch_passive(model, data)

    def step_to(target):
        qj = data.qpos[rb.q_ids]
        dqj = data.qvel[rb.dq_ids]
        tau_cmd = KP * (target - qj) - KD * dqj
        tau, env_lim = tn_clamp(tau_cmd, dqj)
        rb.apply(tau)
        if v is not None:
            v.sync()
            import time
            time.sleep(DT)
        return tau_cmd, tau, env_lim

    for _ in range(int(settle / DT)):
        step_to(q_ref[0])
    z_squat = data.qpos[2]
    contacts0 = int((rb.foot_fz() > 1.0).sum())

    sat = np.zeros(12)
    demand = np.zeros(12)
    track_err = []
    took_off_at = None
    for k in range(len(phi)):
        tau_cmd, tau, env_lim = step_to(q_ref[k])
        sat += (np.abs(tau_cmd) > env_lim + 1e-6).astype(float)
        demand = np.maximum(demand, np.abs(tau_cmd))
        # how far the ACHIEVED pose is from the reference: if this is large the check is measuring
        # the PD, not the trajectory, and the result says nothing about whether the reference is good
        track_err.append(np.abs(data.qpos[rb.q_ids] - q_ref[k]))
        if (rb.foot_fz() > 1.0).sum() == 0 and took_off_at is None:
            took_off_at = k
    track_err = np.array(track_err)

    vx, vz = data.qvel[0], data.qvel[2]
    z_to, x_to = data.qpos[2], data.qpos[0]
    # let it fly (holding the last reference pose) until a foot touches down again
    peak, x_land = z_to, None
    for _ in range(400):
        step_to(q_ref[-1])
        peak = max(peak, data.qpos[2])
        if data.qvel[2] < 0 and (rb.foot_fz() > 1.0).sum() > 0:
            x_land = data.qpos[0]
            break
    if v is not None:
        v.close()

    r = dict(z_squat=z_squat, contacts_at_squat=contacts0, took_off=took_off_at is not None,
             takeoff_phi=(phi[took_off_at] if took_off_at is not None else None),
             vx=vx, vz=vz, speed=math.hypot(vx, vz),
             angle=math.degrees(math.atan2(vz, vx)) if vx != 0 else 90.0,
             z_takeoff=z_to, peak=peak, flight=(x_land - x_to) if x_land is not None else None,
             sat=sat, demand=demand, n=len(phi),
             track_rms=float(np.sqrt((track_err ** 2).mean())), track_max=float(track_err.max()))
    if verbose:
        report(r)
    return r


def report(r):
    print("\n" + "=" * 78)
    print("REFERENCE FEASIBILITY CHECK (stiff PD tracking, real T-N envelope)")
    print("=" * 78)
    print(f"  tracking     : rms {r['track_rms']:.3f} rad, max {r['track_max']:.3f} rad"
          f"   {'(PD follows the reference)' if r['track_rms'] < 0.10 else '<-- PD CANNOT follow: this measures the PD, not the reference'}")
    print(f"  squat settle : base z {r['z_squat']:.3f} m, {r['contacts_at_squat']}/4 feet loaded"
          f"   {'OK' if r['contacts_at_squat'] == 4 else '<-- NOT a four-leg squat!'}")
    if not r["took_off"]:
        print("  RESULT       : ** never left the ground **")
    else:
        print(f"  takeoff      : at phi={r['takeoff_phi']:.2f}, base z {r['z_takeoff']:.3f} m")
        print(f"  launch       : vx={r['vx']:+.2f}  vz={r['vz']:+.2f}  |v|={r['speed']:.2f} m/s"
              f"   angle={r['angle']:.1f} deg")
        print(f"  flight       : {'%.3f m' % r['flight'] if r['flight'] is not None else 'n/a'}"
              f"   peak {r['peak']:.3f} m")
    jn = [f"{l}_{j}" for l in FEET for j in ("hip", "thigh", "calf")]
    over = [(jn[i], r["sat"][i] / r["n"], r["demand"][i], TAU_LIM[i]) for i in range(12) if r["sat"][i] > 0]
    if over:
        print(f"  torque       : {len(over)}/12 joints asked for more than the envelope can give")
        for n, frac, dmd, lim in sorted(over, key=lambda x: -x[1])[:6]:
            print(f"      {n:<9} saturated {100 * frac:>5.1f}% of the push   peak demand "
                  f"{dmd:6.1f} Nm (limit {lim:.1f})")
        print("      -> the reference is too aggressive here: slow it down (--t-push) or shorten"
              " the extension (--h-ext)")
    else:
        print("  torque       : never exceeded the envelope -- the reference is achievable as-is")
    print()


def compare(npz, phi, q_ref):
    d = np.load(npz)
    q, c = d["q"], d["contact"]
    jn = [f"{l}_{j}" for l in FEET for j in ("hip", "thigh", "calf")]
    print("\n" + "=" * 78)
    print(f"MEASURED push ({os.path.basename(npz)}, {len(q)} frames)  vs  REFERENCE")
    print("=" * 78)
    print(f"  feet loaded at the squat bottom: measured {int(c[0].sum())}/4   reference 4/4")
    print(f"\n  {'joint':<10}{'measured start':>15}{'measured end':>14}{'ref start':>12}{'ref end':>10}"
          f"{'meas travel':>13}{'ref travel':>12}")
    for i, n in enumerate(jn):
        print(f"  {n:<10}{q[0, i]:>15.3f}{q[-1, i]:>14.3f}{q_ref[0, i]:>12.3f}{q_ref[-1, i]:>10.3f}"
              f"{q[-1, i] - q[0, i]:>13.3f}{q_ref[-1, i] - q_ref[0, i]:>12.3f}")
    print("\n  per-foot liftoff phase (measured):  " +
          "  ".join(f"{f}={(np.argmax(c[:, k] == 0) / max(len(c) - 1, 1)) if (c[:, k] == 0).any() else 1.0:.2f}"
                    for k, f in enumerate(FEET)))
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=DEFAULT_XML)
    p.add_argument("--t-push", type=float, default=0.18, help="push duration [s]")
    p.add_argument("--h-squat", type=float, default=0.10)
    p.add_argument("--h-ext", type=float, default=None, help="default = just short of the calf stop")
    p.add_argument("--rear-delay", type=float, default=0.30, help="phase at which the rear pair starts extending")
    p.add_argument("--front-end", type=float, default=0.60, help="phase at which the front pair finishes")
    p.add_argument("--check", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--compare", default=None)
    p.add_argument("--save", default=None, help="write phi/q_ref to this .npz")
    a = p.parse_args()

    phi, q_ref = build_reference(t_push=a.t_push, h_squat=a.h_squat, h_ext=a.h_ext,
                                 front_win=(0.0, a.front_end), rear_win=(a.rear_delay, 1.0))
    print(f"[ref] {len(phi)} frames, T_push={a.t_push * 1000:.0f} ms, foot height "
          f"{a.h_squat:.3f} -> {(a.h_ext if a.h_ext else max_foot_height() - 0.005):.3f} m "
          f"(calf-limited max {max_foot_height():.3f})")
    print(f"[ref] front extends over phi [0.00, {a.front_end:.2f}]   rear over "
          f"[{a.rear_delay:.2f}, 1.00]   -- both pairs loaded at phi=0")
    if a.compare:
        compare(a.compare, phi, q_ref)
    if a.save:
        np.savez(a.save, phi=phi, q_ref=q_ref, t_push=a.t_push)
        print(f"[ref] saved -> {a.save}")
    if a.check or a.viewer:
        replay(a.xml, phi, q_ref, viewer=a.viewer)


if __name__ == "__main__":
    main()
