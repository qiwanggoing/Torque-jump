"""Show the Go2 joint limits -- the table, and the robot actually moving through them.

WHY
---
The SATA URDF ships a TRUNCATED thigh range ([0, 1.5] front / [0, 2.0] rear) and model_4600 was
trained sitting on those fake stops. `GO2Torque.OFFICIAL_DOF_POS_LIMITS` overrides them at env-build
time, so what Isaac actually uses is no longer what the URDF says -- and neither number is visible
anywhere while training. This prints all of it side by side and, with --viewer, drives each joint
between its real limits so you can SEE what the robot can and cannot do.

Runs on MuJoCo (CPU, no Isaac needed). The MJCF ranges come straight from Unitree's Menagerie model,
which is one of the three official sources that agree with the override -- so watching this IS
watching our trained limits.

  hard      the physical stop (official Unitree). PhysX will not let the joint past it.
  soft      OUR band where the `dof_pos_limits` reward (-5.0) starts pushing back. Not hardware.
  terminate hard +/- 0.05 -- overshooting it ends the episode.

USAGE
  python deploy_mujoco/joint_limit_viewer.py                     # print the table and exit
  python deploy_mujoco/joint_limit_viewer.py --viewer            # + sweep every joint in a window
  python deploy_mujoco/joint_limit_viewer.py --viewer --joint thigh   # sweep one joint group only
  python deploy_mujoco/joint_limit_viewer.py --used <model.pt>   # overlay the range a policy USES

Keys in the viewer: it cycles joint groups automatically (SECONDS_PER_JOINT each). Close the window
or Ctrl-C to stop.
"""
import argparse
import math
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "..", "..", "my_go2_jump", "resources", "robots", "go2", "go2", "go2.xml")

# Mirrors GO2Torque.OFFICIAL_DOF_POS_LIMITS / SOFT_DOF_POS_BAND (legged_gym/envs/go2/go2_torque).
# Kept as a literal copy on purpose: this script must run without importing isaacgym.
OFFICIAL = {
    "hip":         (-1.0472, 1.0472),
    "thigh_front": (-1.5708, 3.4907),
    "thigh_rear":  (-0.5236, 4.5379),
    "calf":        (-2.7227, -0.83776),
}
SOFT = {
    "hip":         (-0.9425, 0.9425),
    "thigh_front": (-1.20, 2.40),
    "thigh_rear":  (-0.40, 2.60),
    "calf":        (-2.6285, -0.9320),
}
TERM_MARGIN = 0.05

# The SATA URDF's truncated values, for the "what we used to train against" column.
SATA_URDF = {
    "hip":         (-1.0472, 1.0472),
    "thigh_front": (0.0, 1.5),
    "thigh_rear":  (0.0, 2.0),
    "calf":        (-2.7227, -0.8378),
}

JOINTS = [f"{leg}_{j}_joint" for leg in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
# Neutral stance to hold the joints that are not being swept (the training default_joint_angles).
DEFAULT_POSE = {"hip": 0.1, "thigh": 0.8, "calf": -1.5}


def key_of(name):
    if "hip" in name:
        return "hip"
    if "calf" in name:
        return "calf"
    return "thigh_front" if name.startswith("F") else "thigh_rear"


def deg(x):
    return math.degrees(x)


def print_table(model, used=None):
    print("\n" + "=" * 108)
    print("GO2 JOINT LIMITS -- what the policy is actually trained against")
    print("=" * 108)
    hdr = f"{'group':<13}{'HARD (official)':>26}{'SOFT (our -5.0 band)':>26}{'terminate outside':>26}"
    print(hdr)
    for k in ("hip", "thigh_front", "thigh_rear", "calf"):
        lo, hi = OFFICIAL[k]
        slo, shi = SOFT[k]
        print(f"{k:<13}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>26}"
              f"{f'[{slo:+.4f}, {shi:+.4f}]':>26}"
              f"{f'[{lo - TERM_MARGIN:+.4f}, {hi + TERM_MARGIN:+.4f}]':>26}")
        print(f"{'':<13}{f'[{deg(lo):+7.1f}, {deg(hi):+7.1f}] deg':>26}"
              f"{f'[{deg(slo):+7.1f}, {deg(shi):+7.1f}] deg':>26}")

    print("\n" + "-" * 108)
    print("CROSS-CHECK -- three independent official sources vs the SATA URDF we override")
    print("-" * 108)
    print(f"{'group':<13}{'official (3 sources agree)':>30}{'SATA URDF (TRUNCATED)':>28}{'lost range':>16}")
    for k in ("hip", "thigh_front", "thigh_rear", "calf"):
        lo, hi = OFFICIAL[k]
        ulo, uhi = SATA_URDF[k]
        lost = (hi - lo) - (uhi - ulo)
        flag = "  <-- truncated" if lost > 1e-3 else ""     # 1e-3: the calf differs only by rounding
        print(f"{k:<13}{f'[{lo:+.4f}, {hi:+.4f}]':>30}{f'[{ulo:+.4f}, {uhi:+.4f}]':>28}"
              f"{f'{lost:+.3f} rad':>16}{flag}")
    print("  sources: unitree_ros go2_description.urdf | MuJoCo Menagerie go2.xml (x2) | our override")

    # what MuJoCo itself has loaded, straight from the model (catches an xml that disagrees)
    print("\n" + "-" * 108)
    print("LOADED MJCF (this is what the viewer below will enforce)")
    print("-" * 108)
    for name in JOINTS[:6]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        k = key_of(name)
        ok = abs(lo - OFFICIAL[k][0]) < 1e-3 and abs(hi - OFFICIAL[k][1]) < 1e-3
        print(f"  {name:<18}[{lo:+.4f}, {hi:+.4f}]   {'== official' if ok else '!! DIFFERS from official'}")
    print("  (FR/RR mirror FL/RL)")

    if used:
        print("\n" + "-" * 108)
        print("RANGE THE POLICY ACTUALLY USES (one jump) -- headroom left at each stop")
        print("-" * 108)
        print(f"{'joint':<18}{'used min':>11}{'used max':>11}{'room @lo':>11}{'room @hi':>11}{'% of range':>12}")
        for i, name in enumerate(JOINTS):
            lo, hi = OFFICIAL[key_of(name)]
            umin, umax = used[i]
            print(f"  {name:<16}{umin:>11.3f}{umax:>11.3f}{umin - lo:>11.3f}{hi - umax:>11.3f}"
                  f"{100 * (umax - umin) / (hi - lo):>11.1f}%")
    print()


def measure_used(ckpt, xml, dx, height):
    """Run one jump with the real policy and report the min/max each joint reaches."""
    sys.path.insert(0, HERE)
    import importlib.util
    spec = importlib.util.spec_from_file_location("s2s", os.path.join(HERE, "sim2sim_landing_torque.py"))
    s2s = importlib.util.module_from_spec(spec)
    sys.modules["s2s"] = s2s
    spec.loader.exec_module(s2s)

    actor, comp = s2s.load_policy(ckpt)
    rb = s2s.MjRobot(xml, friction=1.0)
    rb.match_isaac_contacts()
    rb.match_urdf_mass()
    args = argparse.Namespace(**{k: v for k, v in dict(
        dx=dx, dy=0.0, height=height, z0=0.33, stand_pd_weight=1.0, pd_weight=0.5, pd_stand=1,
        stand_hold=60, stand_max=400, jump_max=600, post_land=120, pd_engage_min=12,
        pd_engage_max=45, load_mass=0.0, friction=1.0, isaac_contacts=1, match_mass=1,
        isaac_joints=0, fresh_jump=0, fresh_settle=150, viewer=False, dump=None).items()})
    rec = []
    s2s.run_trial(rb, actor, comp, args, 1, recorder=rec)
    q = np.stack([r["q"] for r in rec])
    return list(zip(q.min(axis=0), q.max(axis=0)))


SECONDS_PER_JOINT = 5.0


def sweep(model, data, only=None):
    from mujoco import viewer as mj_viewer

    groups = [g for g in ("hip", "thigh", "calf") if only is None or g == only]
    jids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS}
    qadr = {name: model.jnt_qposadr[jids[name]] for name in JOINTS}

    def set_neutral():
        data.qpos[:3] = [0.0, 0.0, 0.55]          # hold the base up: pure kinematic display
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for name in JOINTS:
            data.qpos[qadr[name]] = DEFAULT_POSE[name.split("_")[1]]

    print(f"\n[viewer] kinematic display (no physics) -- base pinned at 0.55 m so the legs are visible.")
    print(f"[viewer] sweeping: {', '.join(groups)}   ({SECONDS_PER_JOINT:.0f}s each)   close the window to stop\n")
    dt = 0.01
    with mj_viewer.launch_passive(model, data) as v:
        for g in groups:
            names = [n for n in JOINTS if n.split("_")[1] == g]
            lo = min(model.jnt_range[jids[n]][0] for n in names)
            hi = max(model.jnt_range[jids[n]][1] for n in names)
            print(f"  >> {g:<6} sweeping [{lo:+.4f}, {hi:+.4f}] rad = [{deg(lo):+.1f}, {deg(hi):+.1f}] deg")
            n_steps = int(SECONDS_PER_JOINT / dt)
            for i in range(n_steps):
                if not v.is_running():
                    return
                phase = i / n_steps
                # triangle lo -> hi -> lo so both stops are held in view briefly
                u = 2 * phase if phase < 0.5 else 2 * (1 - phase)
                set_neutral()
                for name in names:
                    jlo, jhi = model.jnt_range[jids[name]]
                    data.qpos[qadr[name]] = jlo + u * (jhi - jlo)
                mujoco.mj_forward(model, data)
                v.sync()
                if i % 50 == 0:
                    ang = data.qpos[qadr[names[0]]]
                    print(f"     {names[0]:<16} {ang:+.3f} rad  {deg(ang):+7.1f} deg", flush=True)
                import time
                time.sleep(dt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=os.path.normpath(DEFAULT_XML))
    p.add_argument("--viewer", action="store_true", help="open a window and sweep each joint to its stops")
    p.add_argument("--joint", choices=["hip", "thigh", "calf"], default=None, help="sweep only this group")
    p.add_argument("--used", default=None, help="path to model_XXXX.pt: also report the range one jump USES")
    p.add_argument("--used-dx", type=float, default=1.0)
    p.add_argument("--used-height", type=float, default=0.45)
    a = p.parse_args()

    model = mujoco.MjModel.from_xml_path(a.xml)
    data = mujoco.MjData(model)
    used = measure_used(a.used, a.xml, a.used_dx, a.used_height) if a.used else None
    print_table(model, used)
    if a.viewer:
        sweep(model, data, a.joint)


if __name__ == "__main__":
    main()
