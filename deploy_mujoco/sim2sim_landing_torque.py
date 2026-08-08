"""Sim2Sim (Isaac Gym -> MuJoCo) for the LANDING-TORQUE jump policy (model_4600 family).

The old deploy_mujoco scripts (torque_jump/SATA/.../sim2sim_sata_jump.py and
my_go2_jump/.../sim2sim_GO2_jump_torque_final.py) target a DIFFERENT task (velocity-command
jumping, 62-dim phase obs, tau = PD + action*scale). This policy is the landing-point task
`go2_omnijump_landing_torque`: 69-dim obs with a yaw-frame landing error, a two-head network,
and the SATA torque pipeline (activation smoothing -> Hill/T-N curve -> clip -> fatigue).
Nothing of the old control path applies, so this file re-implements the training-time
physics-facing code exactly.

WHAT IS REPRODUCED FROM TRAINING (verified against the wandb code snapshot of the run that
produced model_4600, Jul16_12-07-41 -- its compute_observations is byte-identical to the
reference builder kept at go2_omnijump_landing_torque.py `_compute_observations_flat_UNUSED`):

  observation (69) =
      base_lin_vel(body) * 2.0                    [0:3]
      base_ang_vel(body) * 0.25                   [3:6]
      projected_gravity                           [6:9]
      yaw-frame landing error (fwd, lat, 0)       [9:12]   <- NOT a velocity command
      cmd_jump_height * 2.0                       [12:13]
      cmd4 (jump/stand, 0 or 1)                   [13:14]
      [base_z*2, (cmd_h - base_z)*2]              [14:16]
      (q - q_default)                             [16:28]
      dq * 0.05                                   [28:40]
      foot contact (fz > 1.0N)                    [40:44]
      torques APPLIED last step (raw Nm)          [44:56]
      motor_fatigue                               [56:68]
      pd_prior_alpha                              [68:69]
    then clipped to +-100.

  control (GO2OmniJumpCurriculumTorque._compute_torques, aux_stabilizer_head=True):
      pd_alpha  = pd_prior_weight * (1 - general_scale)
      rl_alpha  = 1 - pd_alpha
      scale     = start_torque_scale + general_scale*(max - start)      (0.3 -> 1.0)
      comp_w    = max(0, pd_prior_weight - pd_alpha)                    (0 -> 0.5)
      tau_act   = action*tau_lim*rl_alpha*scale
                  + pd_alpha*(kp*(q_def - q) - kd*dq)
                  + comp_w*(comp_head_out*tau_lim*scale)
      activation: s <- s + 0.6*(tanh(tau_act/tau_lim) - s)              (first-order lag)
      Hill/T-N :  X1 = 0.45*v_lim, X2 = v_lim, Y1 = tau_lim, Y2 = 1.158*tau_lim
                  effort = Y1 if dq and s same sign else Y2, linearly decayed to 0 from X1 to X2
      tau       = clip(s * effort, +-tau_lim)
      fatigue  <- (fatigue + |tau|*dt) * 0.9
  model_4600 is at the PURE-TORQUE endpoint: general_scale = 1 -> pd_alpha = 0, rl_alpha = 1,
  scale = 1.0, comp_w = 0.5. NOTE the weights are NOT a convex combination there: 1.0 + 0 + 0.5
  = 1.5. rl_alpha = 1 - pd_alpha means the RESIDUAL reabsorbs the share the PD gives up, and the
  comp head is added on top as a third term (the env comment calling it "takes over PD's share"
  is wrong about the mechanism). Measured on a dx=1.0 rollout: the comp term is 32-40% of the
  commanded torque and OPPOSES the residual 75% of the time -- the actor is trained against a
  posture-restoring partner, so dropping the head would leave its residual unopposed. A JIT export
  of the actor alone is therefore wrong, which is why this script loads the raw checkpoint and
  runs both heads.

  rate: decimation=1 and general_scale=1 -> current_freq = 200Hz -> ONE policy step per 0.005s
  physics step (no decimation loop).

  protocol (mirrors legged_gym/scripts/play_landing_deploy.py, the validated deploy play):
      reset -> PURE-PD holds the DEFAULT pose until the stand is stable
            -> hand to RL, cmd4=1, dx=DX (landing_target locks to squat-bottom xy + dx at takeoff)
            -> RL flies and lands -> PD takes the stand back once the base is quiescent.
      The PD stand is NOT cosmetic: this policy was trained jump-only and cannot jump out of its
      own settled stand (see the 2026-07-24 finding); PD holding the default pose restores the
      takeoff initial condition it trained from.

KNOWN SIM2SIM GAPS (print at startup, tune/inspect before blaming the policy):
  * joint damping/armature/frictionloss come from go2.xml (0.05 / 0.005 / 0.05); Isaac takes its
    values from the SATA URDF. Different -> different effective damping.
  * ground friction 0.6 (MuJoCo foot class, condim=6) vs Isaac's terrain friction.
  * MuJoCo elliptic cone + impratio 100 vs Isaac's PhysX contact model.
  * total body mass may differ between go2.xml and the SATA URDF -- printed at startup, and
    --load-mass lets you match it.
  * base_lin_vel is the base-body frame velocity in both, but the frame origins can differ
    slightly between the URDF and the MJCF.

USAGE
  # numbers only (headless), 3 trials at dx=1.0, dump the trajectory for cross-checking:
  python deploy_mujoco/sim2sim_landing_torque.py \
      --ckpt legged_gym/logs/go2_omnijump_landing_torque/Jul16_12-07-41_stage1_landing/model_4600.pt \
      --dx 1.0 --trials 3 --dump /tmp/mj_rollout.npz

  # watch it:
  ... --viewer

  # CROSS-CHECK the network+obs wiring against an Isaac rollout (no physics involved):
  python legged_gym/scripts/dump_landing_rollout.py --task=go2_omnijump_landing_torque \
      --load_run=Jul16_12-07-41_stage1_landing --checkpoint=4600   # writes /tmp/isaac_rollout.npz
  python deploy_mujoco/sim2sim_landing_torque.py --ckpt <same> --replay /tmp/isaac_rollout.npz
  -> feeds Isaac's recorded observations through THIS script's network and compares the actions.
     A mismatch there is a loading/layout bug; a match means any jump difference is real physics.
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn

import mujoco


# ============================================================================
# Constants mirrored from the training config (file:line given for each source)
# ============================================================================
class C:
    # go2_omnijump_torque_config.py:45  (Isaac DOF order: FL, FR, RL, RR x hip, thigh, calf)
    JOINT_NAMES = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    FOOT_GEOMS = ["FL", "FR", "RL", "RR"]          # go2.xml foot-class geoms, same order
    DEFAULT_DOF_POS = np.array([
        0.1,  0.8, -1.5,      # FL
        -0.1, 0.8, -1.5,      # FR
        0.1,  1.0, -1.5,      # RL (rear thigh 1.0)
        -0.1, 1.0, -1.5,      # RR
    ], dtype=np.float64)

    # go2_omnijump_torque_config.py:67-68
    P_GAIN = 40.0
    D_GAIN = 1.2
    PD_PRIOR_WEIGHT = 0.5                          # go2_omnijump_torque_config.py:70

    # go2_torque.py:344-349 (code OVERRIDES the URDF; official Go2 T-N numbers)
    KNEE_REDUCTION = 1.917
    TAU_LIM = np.array([20.2, 20.2, 20.2 * KNEE_REDUCTION] * 4, dtype=np.float64)
    VEL_LIM = np.array([30.0, 30.0, 30.0 / KNEE_REDUCTION] * 4, dtype=np.float64)

    # go2_torque_config.py:104-112, 163-169
    ACTIVATION_SMOOTH = 0.6
    TN_KNEE_SPEED_RATIO = 13.5 / 30.0
    TN_MAX_SPEED_RATIO = 1.0
    TN_PEAK_ECC_RATIO = 23.4 / 20.2
    FATIGUE_DECAY = 0.9
    START_TORQUE_SCALE = 0.3
    MAX_TORQUE_SCALE = 1.0

    # legged_robot_config.py:160-168 / go2_omnijump_torque_config.py:175-183
    S_LIN_VEL = 2.0
    S_ANG_VEL = 0.25
    S_DOF_POS = 1.0
    S_DOF_VEL = 0.05
    CLIP_OBS = 100.0
    CLIP_ACT = 100.0

    DT = 0.005                                     # legged_robot_config.py:189, decimation=1
    CONTACT_THRESH = 1.0                           # go2_omnijump_torque_config.py:95
    NUM_OBS = 69


# ============================================================================
# Policy: rebuild both heads straight from the checkpoint (no JIT, no isaacgym)
# ============================================================================
def _build_mlp(state_dict, prefix):
    """Rebuild an rsl_rl MLP (Linear/ELU/.../Linear) from its state_dict entries."""
    idxs = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith(prefix + ".")})
    layers = []
    for n, i in enumerate(idxs):
        w = state_dict[f"{prefix}.{i}.weight"]
        b = state_dict[f"{prefix}.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        layers.append(lin)
        if n < len(idxs) - 1:
            layers.append(nn.ELU())                # go2_omnijump_torque_config.py:230
    return nn.Sequential(*layers).eval()


def load_policy(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model_state_dict"]
    actor = _build_mlp(sd, "actor")
    comp = _build_mlp(sd, "comp_head") if any(k.startswith("comp_head.") for k in sd) else None
    in_dim = actor[0].in_features
    if in_dim != C.NUM_OBS:
        raise SystemExit(
            f"checkpoint actor expects {in_dim}-dim obs, this script builds {C.NUM_OBS}. "
            f"A {in_dim}-dim actor is an OBS-HISTORY checkpoint (1012 = 20x49 stacked + extras); "
            f"this sim2sim only implements the FLAT 69-dim landing policy (the 4600 family)."
        )
    return actor, comp


# ============================================================================
# Torque law: byte-for-byte the training pipeline
# ============================================================================
class TorqueLaw:
    def __init__(self):
        self.activation_sign = np.zeros(12)
        self.fatigue = np.zeros(12)
        self.last_torque = np.zeros(12)

    def reset(self):
        self.activation_sign[:] = 0.0
        self.fatigue[:] = 0.0
        self.last_torque[:] = 0.0

    def compute(self, action, comp, q, dq, general_scale, pd_prior_weight):
        pd_alpha = pd_prior_weight * max(0.0, 1.0 - general_scale)
        rl_alpha = 1.0 - pd_alpha
        scale = C.START_TORQUE_SCALE + general_scale * (C.MAX_TORQUE_SCALE - C.START_TORQUE_SCALE)
        comp_w = max(0.0, pd_prior_weight - pd_alpha)

        pd_full = C.P_GAIN * (C.DEFAULT_DOF_POS - q) - C.D_GAIN * dq
        tau_jump = action * C.TAU_LIM * rl_alpha * scale
        tau_comp = (comp * C.TAU_LIM * scale) if comp is not None else 0.0
        tau_action = tau_jump + pd_alpha * pd_full + comp_w * tau_comp

        # muscle-activation first-order lag on the NORMALISED command
        target_sign = np.tanh(tau_action / C.TAU_LIM)
        self.activation_sign += (target_sign - self.activation_sign) * C.ACTIVATION_SMOOTH
        s = self.activation_sign

        # Hill / torque-speed curve
        x1 = C.VEL_LIM * C.TN_KNEE_SPEED_RATIO
        x2 = C.VEL_LIM * C.TN_MAX_SPEED_RATIO
        y1 = C.TAU_LIM
        y2 = C.TAU_LIM * C.TN_PEAK_ECC_RATIO
        max_effort = np.where((dq * s) > 0, y1, y2)
        vel_abs = np.abs(dq)
        k = -max_effort / (x2 - x1)
        decayed = np.clip(k * (vel_abs - x1) + max_effort, 0.0, None)
        effort = np.where(vel_abs < x1, max_effort, decayed)

        tau = np.clip(s * effort, -C.TAU_LIM, C.TAU_LIM)
        self.fatigue = (self.fatigue + np.abs(tau) * C.DT) * C.FATIGUE_DECAY
        self.last_torque = tau
        return tau, pd_alpha


class TNMonitor:
    """Live torque-speed watch: how hard is each joint group pushed, and how often does it run past
    its speed limit (where the T-N curve allows ZERO torque = the knee-speed wall that caps this
    robot's reach). For the full audit incl. envelope-violation checking use deploy_mujoco/tn_check.py
    on a --dump file."""
    GROUPS = (("hip", [0, 3, 6, 9]), ("thigh", [1, 4, 7, 10]), ("calf", [2, 5, 8, 11]))

    def __init__(self):
        self.reset()

    def reset(self):
        self.max_w = np.zeros(12)
        self.max_t = np.zeros(12)
        self.past = np.zeros(12)
        self.n = 0

    def update(self, dq, tau):
        self.n += 1
        self.max_w = np.maximum(self.max_w, np.abs(dq))
        self.max_t = np.maximum(self.max_t, np.abs(tau))
        self.past += (np.abs(dq) > C.VEL_LIM).astype(float)

    def line(self):
        if self.n == 0:
            return "tn: (no steps)"
        out = []
        for name, idx in self.GROUPS:
            w = 100 * (self.max_w[idx] / C.VEL_LIM[idx]).max()
            t = 100 * (self.max_t[idx] / C.TAU_LIM[idx]).max()
            past = 100 * self.past[idx].sum() / (self.n * len(idx))
            out.append(f"{name} w{w:3.0f}%/tau{t:3.0f}%" + (f"/past{past:.0f}%" if past > 0.5 else ""))
        return "tn: " + "  ".join(out)


# ============================================================================
# MuJoCo wrapper
# ============================================================================
def _quat_wxyz_to_rot(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


class MjRobot:
    def __init__(self, xml_path, add_mass=0.0, friction=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = C.DT
        self.data = mujoco.MjData(self.model)

        self.q_ids, self.dq_ids, self.act_ids = [], [], []
        for name in C.JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise SystemExit(f"joint {name} not in the MJCF")
            self.q_ids.append(self.model.jnt_qposadr[jid])
            self.dq_ids.append(self.model.jnt_dofadr[jid])
            aid = next(a for a in range(self.model.nu)
                       if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                       and self.model.actuator_trnid[a, 0] == jid)
            self.act_ids.append(aid)
        self.q_ids = np.array(self.q_ids)
        self.dq_ids = np.array(self.dq_ids)
        self.act_ids = np.array(self.act_ids)

        self.foot_gids = []
        for g in C.FOOT_GEOMS:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if gid < 0:
                raise SystemExit(f"foot geom {g} not in the MJCF")
            self.foot_gids.append(gid)

        self.base_bid = next(
            (b for b in ("base_link", "base", "trunk")
             if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0), None)
        if self.base_bid is None:
            raise SystemExit("no base body (base_link/base/trunk) in the MJCF")
        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_bid)

        if add_mass:
            self.model.body_mass[self.base_bid] += add_mass
        self.total_mass = float(self.model.body_mass.sum())

        # Ground friction. go2.xml ships mu=0.6 on the foot class; TRAINING ran on mu=1.0
        # (go2_omnijump_torque_config.py:20-21 static/dynamic_friction, DR disabled here). A 0.6
        # ground is a different push-off surface, so default to matching Isaac.
        self.friction = friction
        if friction is not None:
            floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            for gid in list(self.foot_gids) + ([floor_gid] if floor_gid >= 0 else []):
                self.model.geom_friction[gid, 0] = friction

    def match_isaac_contacts(self):
        """Use the friction-cone model Isaac Gym actually uses.

        go2.xml ships `cone="elliptic" impratio="100"` -- MuJoCo's high-fidelity friction setting,
        which makes the contact problem far stiffer. Isaac Gym/PhysX approximates the cone as a
        PYRAMID with no such impedance ratio, so the policy was trained against pyramidal contacts.
        Measured over 20 trigger times (stand 0.3-3.5 s, dx=1.0), fraction of commands that produce
        a real flight phase:
            elliptic + impratio 100 (as shipped) ..... 20%   <- the "it usually doesn't jump" report
            pyramidal + impratio 1 ................... 90%
            pyramidal + impratio 1 + mass matched .... 95%
        This is engine matching, not knob-fitting: an over-stiff tangential solver was eating the
        push-off. (Making the NORMAL contact stiffer, solref 0.005, goes the wrong way: 45%.)
        """
        before = (int(self.model.opt.cone), float(self.model.opt.impratio))
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.model.opt.impratio = 1.0
        return before

    def match_urdf_mass(self, nominal=15.019):
        """Trim the base so the total matches the SATA URDF the policy trained on (15.019 kg)."""
        delta = self.total_mass - nominal
        self.model.body_mass[self.base_bid] -= delta
        self.total_mass = float(self.model.body_mass.sum())
        return delta

    def match_isaac_joints(self):
        """Zero the MJCF joint damping / armature / frictionloss.

        The SATA URDF declares NO joint damping or friction and legged_gym sets asset.armature = 0
        (legged_robot_config.py:121), so Isaac's joints are ideal. go2.xml ships damping=0.05,
        armature=0.005, frictionloss=0.05 -- real hardware properties that Isaac never had. Keeping
        them makes MuJoCo a HARDER world than training; zeroing them isolates "does the policy
        transfer across the contact solver" from "does it survive joint drag it never saw".
        """
        before = (float(self.model.dof_damping.max()), float(self.model.dof_armature.max()),
                  float(self.model.dof_frictionloss.max()))
        self.model.dof_damping[6:] = 0.0        # first 6 dofs = free joint
        self.model.dof_armature[6:] = 0.0
        self.model.dof_frictionloss[6:] = 0.0
        return before

    def reset(self, z0=0.33):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, z0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.q_ids] = C.DEFAULT_DOF_POS
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # --- state ------------------------------------------------------------
    def state(self):
        d = self.data
        q = d.qpos[self.q_ids].copy()
        dq = d.qvel[self.dq_ids].copy()
        base_xyz = d.qpos[0:3].copy()
        qw, qx, qy, qz = d.qpos[3:7]
        rot = _quat_wxyz_to_rot(qw, qx, qy, qz)
        lin_world = d.qvel[0:3].copy()
        ang_world = d.qvel[3:6].copy()
        lin_body = rot.T @ lin_world
        ang_body = rot.T @ ang_world
        proj_grav = rot.T @ np.array([0.0, 0.0, -1.0])
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return q, dq, base_xyz, lin_body, ang_body, proj_grav, yaw

    def foot_fz(self):
        """|world-z| contact force per foot, Isaac's contact_forces[:, feet, 2] analogue."""
        fz = np.zeros(4)
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for k, gid in enumerate(self.foot_gids):
                if con.geom1 == gid or con.geom2 == gid:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    world = con.frame.reshape(3, 3).T @ f6[:3]
                    fz[k] += abs(world[2])
        return fz

    def apply(self, tau):
        self.data.ctrl[self.act_ids] = tau
        mujoco.mj_step(self.model, self.data)


# ============================================================================
# Observation (69) -- see the module docstring for the layout and its source
# ============================================================================
def build_obs(q, dq, base_xyz, lin_body, ang_body, proj_grav, yaw,
              landing_target, cmd_height, cmd4, contact, torques, fatigue, pd_alpha):
    err_w = landing_target - base_xyz[:2]
    cy, sy = math.cos(yaw), math.sin(yaw)
    err_fwd = cy * err_w[0] + sy * err_w[1]
    err_lat = -sy * err_w[0] + cy * err_w[1]
    z = base_xyz[2]
    obs = np.concatenate([
        lin_body * C.S_LIN_VEL,
        ang_body * C.S_ANG_VEL,
        proj_grav,
        [err_fwd, err_lat, 0.0],
        [cmd_height * 2.0],
        [cmd4],
        [z * 2.0, (cmd_height - z) * 2.0],
        (q - C.DEFAULT_DOF_POS) * C.S_DOF_POS,
        dq * C.S_DOF_VEL,
        contact.astype(np.float64),
        torques,
        fatigue,
        [pd_alpha],
    ])
    obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
    return np.clip(obs, -C.CLIP_OBS, C.CLIP_OBS).astype(np.float32)


# ============================================================================
# One stand -> jump -> land trial
# ============================================================================
def run_trial(rb, actor, comp_head, args, trial_idx, recorder=None, viewer=None):
    law = TorqueLaw()
    law.reset()
    rb.reset(z0=args.z0)

    def policy_step(obs):
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0)
            a = actor(t)[0].numpy().astype(np.float64)
            c = comp_head(t)[0].numpy().astype(np.float64) if comp_head is not None else None
        return np.clip(a, -C.CLIP_ACT, C.CLIP_ACT), c

    # ---------------- phase 1: PURE-PD stand at the default pose -------------
    stand_stable = 0
    settled = False
    for st in range(args.stand_max):
        q, dq, base, lin, ang, pg, yaw = rb.state()
        tau, pd_alpha = law.compute(np.zeros(12), None, q, dq,
                                    general_scale=0.0, pd_prior_weight=args.stand_pd_weight)
        rb.apply(tau)
        if viewer is not None:
            viewer.sync()
        fz = rb.foot_fz()
        contact = fz > C.CONTACT_THRESH
        upright = abs(pg[0]) < 0.1 and abs(pg[1]) < 0.1
        stand_stable = stand_stable + 1 if (contact.all() and upright) else 0
        if stand_stable >= args.stand_hold:
            settled = True
            break
    q, dq, base, lin, ang, pg, yaw = rb.state()
    stand_h = base[2]
    print(f"  [stand] {'SETTLED' if settled else 'NOT settled'} after {st + 1} steps | "
          f"h={stand_h:.3f}m tilt={math.degrees(math.asin(np.clip(pg[0], -1, 1))):+.1f}deg "
          f"feet={int((rb.foot_fz() > C.CONTACT_THRESH).sum())}/4")
    if not settled:
        return None

    # ---------------- phase 2: hand to RL and command the jump ---------------
    handoff_xy = base[:2].copy()
    squat_xy = base[:2].copy()          # squat_root_xy (landing env _start_jump / _update_jump_state)
    jump_min_z = base[2]
    landing_target = handoff_xy + np.array([args.dx, args.dy])   # provisional; re-locked at takeoff
    cmd4 = 1.0

    took_off = False
    landed = False
    takeoff_xy = None
    land_xy = None
    peak_z = base[2]
    air_steps = 0
    post_land = 0
    pd_engaged = False
    fell = False
    # Contact debouncing. Isaac uses contact_filt = contact | last_contacts; MuJoCo needs MORE than
    # that here: during the fast downward LOAD phase the feet genuinely unload and the normal force
    # dips under the 1.0N threshold for several consecutive steps (measured: front feet read 0N for
    # ~5 steps mid-squat). Taking that as "all feet off" fires a phantom takeoff, and the phantom
    # landing that follows hands control to the PD mid-squat -- i.e. the jump never happens. So a
    # takeoff must be SUSTAINED for TAKEOFF_STREAK steps AND the base must be RISING (a load-phase
    # unload dips the force but the base is going DOWN), a landing for LANDING_STREAK.
    TAKEOFF_STREAK = 8      # 40 ms of all-feet-off; real flight lasts 100ms+, load dips are shorter
    TAKEOFF_MIN_VZ = 0.2    # m/s, base must be moving UP when the all-off streak starts
    LANDING_STREAK = 2
    last_contact = contact.copy()
    air_streak = 0
    ground_streak = 0
    takeoff_cand_xy = None
    traj_max_z = base[2]
    traj_max_x = 0.0
    lin_world_z = 0.0
    tn = TNMonitor()

    # obs is computed AFTER the physics step in training -> build the first one from the
    # post-PD-stand state with the PD torques that were just applied.
    torques = law.last_torque.copy()
    fatigue = law.fatigue.copy()
    # obs slot 68 = pd_prior_alpha as set by the LAST _compute_torques call, i.e. the PD-stand
    # weight on this first RL observation and 0 from the next step on (verified against an Isaac
    # dump: obs[:,68] = [1, 0, 0, ...]). Hardcoding 0 here made step 0 disagree with Isaac.
    pd_alpha_obs = args.stand_pd_weight
    contact = rb.foot_fz() > C.CONTACT_THRESH

    for jt in range(args.jump_max):
        q, dq, base, lin, ang, pg, yaw = rb.state()
        obs = build_obs(q, dq, base, lin, ang, pg, yaw, landing_target,
                        args.height, cmd4, contact, torques, fatigue, pd_alpha_obs)

        if pd_engaged:
            tau, pd_a = law.compute(np.zeros(12), None, q, dq,
                                    general_scale=0.0, pd_prior_weight=args.pd_weight)
            action = np.zeros(12)
            comp = None
        else:
            action, comp = policy_step(obs)
            tau, pd_a = law.compute(action, comp, q, dq,
                                    general_scale=1.0, pd_prior_weight=C.PD_PRIOR_WEIGHT)
        pd_alpha_obs = pd_a

        rb.apply(tau)
        if viewer is not None:
            viewer.sync()

        q, dq, base, lin, ang, pg, yaw = rb.state()
        lin_world_z = float(rb.data.qvel[2])
        fz = rb.foot_fz()
        contact = fz > C.CONTACT_THRESH
        torques = law.last_torque.copy()
        fatigue = law.fatigue.copy()
        # RAW trajectory extremes, independent of the jump state machine: if the detector ever gets
        # confused again these still say what the body actually did.
        traj_max_z = max(traj_max_z, base[2])
        traj_max_x = max(traj_max_x, base[0] - handoff_xy[0])
        tn.update(dq, torques)

        if recorder is not None:
            recorder.append(dict(trial=trial_idx, step=jt, obs=obs, action=action,
                                 comp=(comp if comp is not None else np.zeros(12)),
                                 tau=tau, q=q, dq=dq, base=base.copy(), fz=fz,
                                 # 0 on PD-engaged steps, where `action`/`comp` are zeros by design
                                 # and must NOT be compared against a policy forward pass.
                                 rl_driven=np.array([0.0 if pd_engaged else 1.0])))

        # --- jump state machine (mirrors _update_jump_state / landing override) ---
        contact_filt = contact | last_contact
        last_contact = contact.copy()
        if not took_off:
            if base[2] < jump_min_z:                    # track the squat bottom
                jump_min_z = base[2]
                squat_xy = base[:2].copy()
            if not contact_filt.any() and (air_streak > 0 or lin_world_z > TAKEOFF_MIN_VZ):
                if air_streak == 0:
                    takeoff_cand_xy = base[:2].copy()
                air_streak += 1
                if air_streak >= TAKEOFF_STREAK:        # just_took_off (debounced)
                    took_off = True
                    takeoff_xy = takeoff_cand_xy
                    landing_target = squat_xy + np.array([args.dx, args.dy])   # ANTI-CHEAT lock
            else:
                air_streak = 0
        elif not landed:
            air_steps += 1
            peak_z = max(peak_z, base[2])
            if contact_filt.any():
                if ground_streak == 0:
                    touchdown_xy = base[:2].copy()   # FIRST contact, not the confirming step
                ground_streak += 1
                if ground_streak >= LANDING_STREAK:
                    landed = True
                    land_xy = touchdown_xy
            else:
                ground_streak = 0
        else:
            post_land += 1
            quiescent = abs(lin[0]) < 0.4 and abs(lin[1]) < 0.4 and abs(lin[2]) < 0.4
            if args.pd_stand and not pd_engaged and (
                    (post_land >= args.pd_engage_min and quiescent) or post_land >= args.pd_engage_max):
                pd_engaged = True
            if post_land >= args.post_land:
                break

        if abs(pg[0]) > 0.85 or abs(pg[1]) > 0.85 or base[2] < 0.08:
            fell = True
            break

    raw = (f"raw traj: max_z={traj_max_z:.3f}m squat_bottom={jump_min_z:.3f}m "
           f"max_fwd={traj_max_x:+.3f}m\n         {tn.line()}")
    if not took_off:
        print(f"  [jump ] NEVER LEFT THE GROUND (no sustained all-feet-off) | {raw}")
        return dict(ok=False, reason="no_takeoff", stand_h=stand_h,
                    traj_max_z=traj_max_z, traj_max_x=traj_max_x)

    creep = float(takeoff_xy[0] - handoff_xy[0])
    if land_xy is None:
        print(f"  [jump ] took off (peak {peak_z:.3f}m) but never landed within {args.jump_max} steps | {raw}")
        return dict(ok=False, reason="no_landing", stand_h=stand_h, peak=peak_z, creep=creep,
                    traj_max_z=traj_max_z, traj_max_x=traj_max_x)

    fwd_takeoff = float(land_xy[0] - takeoff_xy[0])
    fwd_squat = float(land_xy[0] - squat_xy[0])
    land_err = float(np.linalg.norm(land_xy - landing_target))
    air_s = air_steps * C.DT
    print(f"  [jump ] {'JUMP' if air_s >= 0.15 else 'NO FLIGHT (legs extended, never left the ground)'}"
          f"{' FELL' if fell else ''} | air={air_s:.3f}s peak={peak_z:.3f}m "
          f"| fwd_from_takeoff={fwd_takeoff:.3f}m fwd_from_squat={fwd_squat:.3f}m "
          f"(cmd dx={args.dx:.2f}) land_err={land_err:.3f}m "
          f"({'HIT' if land_err <= 0.10 else 'miss'}, threshold 0.10) | run-up creep={creep:+.3f}m")
    print(f"         {raw}")
    return dict(ok=True, fell=fell, stand_h=stand_h, peak=peak_z, air=air_steps * C.DT,
                fwd_takeoff=fwd_takeoff, fwd_squat=fwd_squat, land_err=land_err, creep=creep,
                traj_max_z=traj_max_z, traj_max_x=traj_max_x)


# ============================================================================
# Interactive mode: viewer window + keyboard, drive the robot yourself
# ============================================================================
class _FakeViewer:
    """Headless stand-in so the interactive loop can be smoke-tested without a display
    (S2S_HEADLESS_TEST=<n steps>). Not used in normal operation."""
    def __init__(self, n):
        self.left = int(n)

    def is_running(self):
        self.left -= 1
        return self.left > 0

    def sync(self):
        pass

    def close(self):
        pass


HELP = """
  ------------------------------------------------------------------
   SPACE  jump now (uses the current dx / height)
   R      reset the robot back to the origin, standing
   I / K  commanded distance dx  +/- 0.1 m
   U / J  commanded jump height  +/- 0.05 m
   P      print the full status line
   ESC    quit (or just close the window)
  ------------------------------------------------------------------
  >> For a repeatable GOOD jump: press R, then SPACE right away.
     In MuJoCo this policy has two stable outcomes -- a good jump (peak
     ~0.48-0.58, 0.35 s of flight) and a collapsed one (~0.35, 0.09 s) --
     and which one you get depends on the exact pre-jump state. Measured
     on the spawn stand: settle <=150 steps -> 0.48, >=300 steps -> 0.35,
     from a stand that differs by 2 mm and 0.4 deg. ISAAC SHOWS NONE OF
     THIS (peak 0.420 whether it stands 60 or 1000 steps), so it is a
     sim2sim gap of the 4600 policy -- not a bug in this script.
  While STANDING a pure PD controller holds the default pose (this is the
  deploy architecture: the policy is jump-only and cannot start a jump from
  its own settled stand). SPACE hands control to the RL policy for one jump,
  then the PD takes the stand back after touchdown.
"""


def interactive_loop(rb, actor, comp_head, args):
    import time

    law = TorqueLaw()
    law.reset()
    rb.reset(z0=args.z0)

    st = dict(dx=args.dx, height=args.height, jump_req=False, reset_req=False, print_req=False)

    def key_callback(keycode):
        if keycode == 32:                      # SPACE
            st["jump_req"] = True
        elif keycode in (ord('R'), ord('r')):
            st["reset_req"] = True
        elif keycode in (ord('I'), ord('i'), 265):     # 265 = up arrow
            st["dx"] = min(st["dx"] + 0.1, 2.0)
            print(f"  [cmd] dx = {st['dx']:.2f} m")
        elif keycode in (ord('K'), ord('k'), 264):     # 264 = down arrow
            st["dx"] = max(st["dx"] - 0.1, 0.0)
            print(f"  [cmd] dx = {st['dx']:.2f} m")
        elif keycode in (ord('U'), ord('u'), 262):     # 262 = right arrow
            st["height"] = min(st["height"] + 0.05, 0.6)
            print(f"  [cmd] jump height = {st['height']:.2f} m")
        elif keycode in (ord('J'), ord('j'), 263):     # 263 = left arrow
            st["height"] = max(st["height"] - 0.05, 0.3)
            print(f"  [cmd] jump height = {st['height']:.2f} m")
        elif keycode in (ord('P'), ord('p')):
            st["print_req"] = True

    test_steps = os.environ.get("S2S_HEADLESS_TEST")
    if test_steps:
        viewer = _FakeViewer(test_steps)
    else:
        from mujoco import viewer as mj_viewer   # NOT `import mujoco.viewer`: that binds a LOCAL
        viewer = mj_viewer.launch_passive(         # name `mujoco` and shadows the module below
            rb.model, rb.data, key_callback=key_callback)

    print(HELP)
    print(f"  ready: dx={st['dx']:.2f}m  height={st['height']:.2f}m  -- press SPACE in the viewer "
          f"window to jump\n", flush=True)

    mode = "STAND"                     # STAND -> JUMP -> CATCH -> STAND
    loop_i = 0
    # self-test hook: fire "SPACE" at these loop steps (S2S_AUTO_JUMP="150,900,1600") so the
    # multi-jump behaviour can be reproduced headlessly.
    auto_jump_at = set()
    if test_steps:
        auto_jump_at = {int(v) for v in os.environ.get("S2S_AUTO_JUMP", "150").split(",")}
    trigger_obs = []
    contact = rb.foot_fz() > C.CONTACT_THRESH
    torques = law.last_torque.copy()
    fatigue = law.fatigue.copy()
    pd_alpha_obs = args.stand_pd_weight
    # per-jump bookkeeping
    jd = {}
    catch_left = 0
    last_status = 0.0
    tn = TNMonitor()

    try:
        while viewer.is_running():
            t0 = time.perf_counter()
            loop_i += 1
            if loop_i in auto_jump_at:
                st["jump_req"] = True

            if st["reset_req"]:
                st["reset_req"] = False
                law.reset()
                rb.reset(z0=args.z0)
                mode = "STAND"
                print("  [reset] robot back at the origin", flush=True)

            q, dq, base, lin, ang, pg, yaw = rb.state()

            if mode in ("STAND", "DOWN"):
                if mode == "STAND" and (base[2] < 0.15 or abs(pg[0]) > 0.65 or abs(pg[1]) > 0.65):
                    mode = "DOWN"
                    print("  [DOWN ] robot is on the ground -- a posture PD cannot get up from "
                          "here. Press R to reset.\n", flush=True)
                if st["jump_req"]:
                    st["jump_req"] = False
                    stable = (mode == "STAND" and contact.all()
                              and abs(pg[0]) < 0.15 and abs(pg[1]) < 0.15)
                    if not stable:
                        print(f"  [jump ] refused: {'robot is DOWN' if mode == 'DOWN' else 'not standing cleanly yet'} "
                              f"-- press R to reset", flush=True)
                    else:
                        if args.fresh_jump:
                            # Re-place the robot at the exact post-reset condition (default pose,
                            # zero velocity, cleared actuator state) at its CURRENT x/y, then settle
                            # for a fixed number of steps. Without this the jump outcome depends on
                            # how long you happened to stand: measured peak 0.48-0.51 after a short
                            # settle vs 0.35 after >=300 steps, from a stand that differs by 2 mm of
                            # height and 0.4 deg of joint angle. That bimodality is the POLICY's, not
                            # the sim's -- this flag only makes the demo repeatable. On hardware the
                            # same fragility is real and has to be fixed in training.
                            # Keep only WHERE it is and WHICH WAY it faces; level the body. Carrying
                            # the previous landing's pitch/roll over is enough to flip the jump into
                            # the collapsing mode (measured: jump 2 peak 0.42 vs 0.48 with the
                            # attitude levelled), because that tilt is part of the takeoff state.
                            xy = rb.data.qpos[0:3].copy()
                            half = 0.5 * yaw
                            mujoco.mj_resetData(rb.model, rb.data)
                            rb.data.qpos[0:3] = [xy[0], xy[1], args.z0]
                            rb.data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
                            rb.data.qpos[rb.q_ids] = C.DEFAULT_DOF_POS
                            rb.data.qvel[:] = 0.0
                            mujoco.mj_forward(rb.model, rb.data)
                            law.reset()
                            for _ in range(args.fresh_settle):
                                _q, _dq, *_ = rb.state()
                                _tau, _ = law.compute(np.zeros(12), None, _q, _dq,
                                                      general_scale=0.0,
                                                      pd_prior_weight=args.stand_pd_weight)
                                rb.apply(_tau)
                                viewer.sync()
                            q, dq, base, lin, ang, pg, yaw = rb.state()
                            contact = rb.foot_fz() > C.CONTACT_THRESH
                            torques = law.last_torque.copy()
                            fatigue = law.fatigue.copy()
                        # Command in the robot's HEADING frame: "dx = 1 m" means 1 m the way it is
                        # facing, not world +x. After a jump or two the base has yawed a few degrees;
                        # a world-frame target then arrives off-axis, the policy sees a lateral error
                        # it was never trained on (deploy uses dy=0) and the jump degrades. The
                        # displacement is frozen here and re-used at the takeoff re-lock, which is
                        # what Isaac does (commands stay world-fixed for the duration of the jump).
                        cy, sy = math.cos(yaw), math.sin(yaw)
                        disp = np.array([cy * st["dx"] - sy * args.dy,
                                         sy * st["dx"] + cy * args.dy])
                        jd = dict(handoff=base[:2].copy(), squat_xy=base[:2].copy(), min_z=base[2],
                                  target=base[:2] + disp, disp=disp, took_off=False,
                                  landed=False, takeoff_xy=None, cand_xy=None, air=0, streak=0,
                                  ground=0, peak=base[2], last_contact=contact.copy(), steps=0,
                                  post=0, dx=st["dx"], height=st["height"])
                        tn.reset()
                        pd_alpha_obs = args.stand_pd_weight
                        mode = "JUMP"
                        print(f"  [jump ] GO  dx={jd['dx']:.2f}m height={jd['height']:.2f}m",
                              flush=True)
                # DOWN: a soft hold so the robot rests in a sane pose instead of fighting the floor.
                tau, _ = law.compute(
                    np.zeros(12), None, q, dq, general_scale=0.0,
                    pd_prior_weight=(0.3 if mode == "DOWN" else args.stand_pd_weight))

            elif mode == "JUMP":
                obs = build_obs(q, dq, base, lin, ang, pg, yaw, jd["target"], jd["height"],
                                1.0, contact, torques, fatigue, pd_alpha_obs)
                if test_steps and not jd.get("captured"):
                    jd["captured"] = True
                    trigger_obs.append(obs.copy())
                with torch.no_grad():
                    t = torch.from_numpy(obs).unsqueeze(0)
                    action = np.clip(actor(t)[0].numpy().astype(np.float64), -C.CLIP_ACT, C.CLIP_ACT)
                    comp = comp_head(t)[0].numpy().astype(np.float64) if comp_head is not None else None
                tau, pd_alpha_obs = law.compute(action, comp, q, dq,
                                                general_scale=1.0,
                                                pd_prior_weight=C.PD_PRIOR_WEIGHT)
            else:                                    # CATCH
                tau, pd_alpha_obs = law.compute(np.zeros(12), None, q, dq,
                                                general_scale=0.0, pd_prior_weight=args.pd_weight)
                catch_left -= 1
                if catch_left <= 0:
                    mode = "STAND"

            rb.apply(tau)
            viewer.sync()

            q, dq, base, lin, ang, pg, yaw = rb.state()
            vz = float(rb.data.qvel[2])
            contact = rb.foot_fz() > C.CONTACT_THRESH
            torques = law.last_torque.copy()
            fatigue = law.fatigue.copy()

            if mode == "JUMP":
                jd["steps"] += 1
                tn.update(dq, torques)
                cf = contact | jd["last_contact"]
                jd["last_contact"] = contact.copy()
                if not jd["took_off"]:
                    if base[2] < jd["min_z"]:
                        jd["min_z"] = base[2]
                        jd["squat_xy"] = base[:2].copy()
                    if not cf.any() and (jd["streak"] > 0 or vz > 0.2):
                        if jd["streak"] == 0:
                            jd["cand_xy"] = base[:2].copy()
                        jd["streak"] += 1
                        if jd["streak"] >= 8:
                            jd["took_off"] = True
                            jd["takeoff_xy"] = jd["cand_xy"]
                            jd["target"] = jd["squat_xy"] + jd["disp"]
                    else:
                        jd["streak"] = 0
                elif not jd["landed"]:
                    jd["air"] += 1
                    jd["peak"] = max(jd["peak"], base[2])
                    if cf.any():
                        if jd["ground"] == 0:
                            jd["touchdown"] = base[:2].copy()   # FIRST contact
                        jd["ground"] += 1
                        if jd["ground"] >= 2:
                            jd["landed"] = True
                            td = jd["touchdown"]
                            err = float(np.linalg.norm(td - jd["target"]))
                            air = jd["air"] * C.DT
                            # A "peak" of ~0.43 with 0.01 s of air is NOT flight -- it is the robot
                            # standing up on fully extended legs (ballistic rise in 10 ms is ~1 mm).
                            # Air time is the only clean separator, so say the verdict out loud.
                            verdict = ("JUMP" if air >= 0.15 else
                                       "NO FLIGHT (legs extended, never left the ground)")
                            print(f"  [land ] {verdict} | "
                                  f"air={air:.3f}s peak={jd['peak']:.3f}m "
                                  f"fwd_from_takeoff={float(td[0] - jd['takeoff_xy'][0]):.3f}m "
                                  f"fwd_from_squat={float(td[0] - jd['squat_xy'][0]):.3f}m "
                                  f"land_err={err:.3f}m ({'HIT' if err <= 0.10 else 'miss'}, "
                                  f"threshold 0.10) | run-up="
                                  f"{float(jd['takeoff_xy'][0] - jd['handoff'][0]):+.3f}m",
                                  flush=True)
                            print(f"         {tn.line()}", flush=True)
                    else:
                        jd["ground"] = 0
                else:
                    jd["post"] += 1
                    quiet = abs(lin[0]) < 0.4 and abs(lin[1]) < 0.4 and abs(lin[2]) < 0.4
                    # NEVER hand the stand to the PD while the robot is still tumbling: a posture PD
                    # snapping to the default pose from a 27-deg nose-down attitude with no feet down
                    # face-plants it (seen in a user run). Require an upright, planted body first;
                    # if that never happens the landing failed -> DOWN, which needs a reset.
                    safe = (abs(pg[0]) < 0.4 and abs(pg[1]) < 0.4 and int(contact.sum()) >= 2)
                    if safe and ((jd["post"] >= args.pd_engage_min and quiet)
                                 or jd["post"] >= args.pd_engage_max):
                        mode = "CATCH"
                        catch_left = 120
                        print("  [stand] PD has the stand back\n", flush=True)
                    elif jd["post"] >= 3 * args.pd_engage_max:
                        mode = "DOWN"
                        print(f"  [DOWN ] landing failed -- robot is not upright "
                              f"(tilt {math.degrees(math.asin(np.clip(pg[0], -1, 1))):+.0f}deg, "
                              f"{int(contact.sum())}/4 feet). Press R to reset.\n", flush=True)
                if mode == "JUMP" and jd["steps"] > args.jump_max:
                    mode = "CATCH"
                    catch_left = 120
                    print(f"  [jump ] gave up after {args.jump_max} steps "
                          f"({'never took off' if not jd['took_off'] else 'never landed'}) -- PD "
                          f"taking over\n", flush=True)

            now = time.perf_counter()
            if st["print_req"] or now - last_status > 2.0:
                st["print_req"] = False
                last_status = now
                print(f"  [{mode:5s}] z={base[2]:.3f}m x={base[0]:+.3f}m tilt="
                      f"{math.degrees(math.asin(np.clip(pg[0], -1, 1))):+5.1f}deg feet="
                      f"{int(contact.sum())}/4 | cmd dx={st['dx']:.2f} h={st['height']:.2f}",
                      flush=True)

            # real-time pacing (200 Hz)
            sleep = C.DT - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        viewer.close()
        if trigger_obs and os.environ.get("S2S_TRIGGER_OBS"):
            np.save(os.environ["S2S_TRIGGER_OBS"], np.stack(trigger_obs))
            print(f"[test] {len(trigger_obs)} trigger observations -> "
                  f"{os.environ['S2S_TRIGGER_OBS']}", flush=True)


# ============================================================================
# Cross-check mode: replay Isaac observations through this script's network
# ============================================================================
def replay_check(actor, comp_head, npz_path):
    d = np.load(npz_path)
    obs = d["obs"].astype(np.float32)
    if obs.shape[1] != C.NUM_OBS:
        raise SystemExit(f"dump has {obs.shape[1]}-dim obs, expected {C.NUM_OBS}")
    # Only steps the POLICY actually drove are comparable: on PD-engaged steps this script records
    # action/comp as zeros by design (Isaac dumps contain only policy steps and carry no mask).
    mask = (d["rl_driven"].reshape(-1) > 0.5) if "rl_driven" in d else np.ones(len(obs), bool)
    with torch.no_grad():
        t = torch.from_numpy(obs[mask])
        a = actor(t).numpy()
        c = comp_head(t).numpy() if comp_head is not None else None
    print(f"[replay] {int(mask.sum())}/{obs.shape[0]} policy-driven steps from {npz_path}")
    da = np.abs(a - d["action"][mask]).max()
    print(f"[replay] actor    max|diff| = {da:.3e}   {'OK' if da < 1e-4 else 'MISMATCH'}")
    if c is not None and "comp" in d:
        dc = np.abs(c - d["comp"][mask]).max()
        print(f"[replay] comp_head max|diff| = {dc:.3e}   {'OK' if dc < 1e-4 else 'MISMATCH'}")
    print("[replay] a match proves the checkpoint loading + obs ORDER are right; any jump "
          "difference in MuJoCo is then physics, not wiring.")


# ============================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_xml = os.path.abspath(os.path.join(
        here, "..", "..", "my_go2_jump", "resources", "robots", "go2", "go2", "scene.xml"))

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="path to model_XXXX.pt (flat 69-dim landing policy)")
    p.add_argument("--xml", default=default_xml, help="MuJoCo scene xml")
    p.add_argument("--dx", type=float, default=1.0, help="commanded forward landing displacement (m)")
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--height", type=float, default=0.5, help="jump-height command (train range 0.4-0.6)")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--z0", type=float, default=0.33, help="spawn base height before the PD stand settles")
    p.add_argument("--stand-pd-weight", type=float, default=1.0,
                   help="PRE-jump PD stiffness; 1.0 holds the TRUE default pose = full trained reach")
    p.add_argument("--pd-weight", type=float, default=0.5, help="POST-land PD catch stiffness")
    p.add_argument("--pd-stand", type=int, default=1, help="1 = PD takes the stand back after landing")
    p.add_argument("--stand-hold", type=int, default=60)
    p.add_argument("--stand-max", type=int, default=400)
    p.add_argument("--jump-max", type=int, default=600)
    p.add_argument("--post-land", type=int, default=120)
    p.add_argument("--pd-engage-min", type=int, default=12)
    p.add_argument("--pd-engage-max", type=int, default=45)
    p.add_argument("--load-mass", type=float, default=0.0, help="kg added to the base (mass matching)")
    p.add_argument("--friction", type=float, default=1.0,
                   help="sliding friction for feet+floor; 1.0 = Isaac's training value "
                        "(go2.xml ships 0.6). Pass -1 to keep the MJCF value.")
    p.add_argument("--isaac-contacts", type=int, default=1,
                   help="1 (default) = pyramidal friction cone + impratio 1, the contact model Isaac "
                        "Gym/PhysX uses and this policy trained against. go2.xml's shipped "
                        "elliptic/impratio-100 is far stiffer and drops the jump success rate from "
                        "95%% to 20%%. 0 = keep go2.xml as shipped.")
    p.add_argument("--match-mass", type=int, default=1,
                   help="1 (default) = trim the base so the total matches the SATA URDF (15.019 kg); "
                        "go2.xml is +0.19 kg. --load-mass is applied on top as a real payload.")
    p.add_argument("--isaac-joints", type=int, default=0,
                   help="0 (default) = keep go2.xml's joint damping/armature/frictionloss. Matching "
                        "Isaac here (1, zero them) makes things WORSE -- 35%% jump success vs 95%% -- "
                        "because armature is real rotor inertia that Isaac simply omits "
                        "(asset.armature=0). Contacts are the part worth matching, not the joints.")
    p.add_argument("--dx-sweep", default=None,
                   help="comma-separated dx values to sweep instead of --dx, e.g. 0.6,0.8,1.0,1.2")
    p.add_argument("--viewer", action="store_true", help="show the first trial in a viewer window")
    p.add_argument("--fresh-jump", type=int, default=0,
                   help="interactive: re-place the robot (level, default pose, zero velocity) at its "
                        "current spot before each jump, then settle --fresh-settle steps. Measured: "
                        "it does NOT reliably buy a good jump -- in MuJoCo this policy has a good "
                        "mode (peak ~0.48, short settle) and a collapsed mode (~0.35, settle >=300 "
                        "steps) and the re-placement can land in either. Default OFF; to get the "
                        "good mode press R and then SPACE right away.")
    p.add_argument("--fresh-settle", type=int, default=150,
                   help="steps of PD settling after the re-placement (150 = the good regime)")
    p.add_argument("--interactive", action="store_true",
                   help="viewer + keyboard: you trigger the jumps yourself (SPACE), change dx/height "
                        "live, reset with R. Runs in real time at 200Hz.")
    p.add_argument("--dump", default=None, help="write the rollout to this .npz")
    p.add_argument("--replay", default=None, help="cross-check against an Isaac rollout .npz and exit")
    args = p.parse_args()

    actor, comp_head = load_policy(args.ckpt)
    print(f"[policy] {args.ckpt}")
    print(f"[policy] actor {actor[0].in_features}->{actor[-1].out_features}, "
          f"comp_head {'present (carries 0.5 of the torque at pure-torque)' if comp_head else 'ABSENT'}")

    if args.replay:
        replay_check(actor, comp_head, args.replay)
        return

    rb = MjRobot(args.xml, friction=(None if args.friction < 0 else args.friction))
    if args.isaac_contacts:
        cone, imp = rb.match_isaac_contacts()
        print(f"[model ] contacts matched to Isaac/PhysX: cone "
              f"{'elliptic' if cone == 0 else 'pyramidal'}->pyramidal, impratio {imp:.0f}->1 "
              f"(as shipped: 20% of commands produce a jump; matched: 95%)")
    if args.match_mass:
        print(f"[model ] mass trimmed by {rb.match_urdf_mass():+.3f} kg to the URDF nominal")
    if args.load_mass:
        rb.model.body_mass[rb.base_bid] += args.load_mass
        rb.total_mass = float(rb.model.body_mass.sum())
    print(f"[model ] {args.xml}")
    print(f"[model ] total mass {rb.total_mass:.3f} kg (payload {args.load_mass:+.2f}), "
          f"URDF nominal is 15.019 kg | friction "
          f"{'MJCF default' if rb.friction is None else f'{rb.friction:.2f} (Isaac trains at 1.0)'} | "
          f"dt {C.DT}s = 200Hz, one policy step per physics step")
    if args.isaac_joints:
        d, a, f = rb.match_isaac_joints()
        print(f"[model ] joints matched to Isaac: damping {d}->0, armature {a}->0, frictionloss {f}->0 "
              f"(SATA URDF declares none, asset.armature=0)")
    else:
        print(f"[model ] keeping go2.xml joint drag (damping/armature/frictionloss) -- harsher than "
              f"training, closer to hardware")
    print(f"[gaps  ] remaining engine differences: contact solver (MuJoCo elliptic cone/impratio 100 "
          f"vs PhysX) and a {rb.total_mass - 15.019:+.2f} kg mass offset -- see the module docstring")
    print(f"[proto ] PD stand (w={args.stand_pd_weight}) -> RL jump cmd4=1 dx={args.dx} h={args.height} "
          f"-> {'PD catch (w=%.2f)' % args.pd_weight if args.pd_stand else 'RL keeps the stand'}")

    if args.interactive:
        interactive_loop(rb, actor, comp_head, args)
        return

    recorder = [] if args.dump else None
    # MuJoCo + a deterministic (mean) policy is fully deterministic, so repeated trials at the SAME
    # dx are identical by construction -- sweep dx to actually learn something.
    dx_list = ([float(v) for v in args.dx_sweep.split(",")] if args.dx_sweep
               else [args.dx] * args.trials)
    results = []
    for i, dx in enumerate(dx_list, start=1):
        args.dx = dx
        print(f"\n=== TRIAL {i}/{len(dx_list)}  dx={dx:.2f} ===")
        viewer = None
        if args.viewer and i == 1:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(rb.model, rb.data)
        try:
            r = run_trial(rb, actor, comp_head, args, i, recorder=recorder, viewer=viewer)
        finally:
            if viewer is not None:
                viewer.close()
        if r:
            r["dx"] = dx
            results.append(r)

    good = [r for r in results if r.get("ok")]
    if good:
        print(f"\n=== SUMMARY ({len(good)} completed jumps) ===")
        print(f"  {'cmd dx':>7} {'peak':>7} {'air':>6} {'fwd(TO)':>8} {'fwd(squat)':>11} "
              f"{'land_err':>9} {'run-up':>8}  hit")
        for r in good:
            print(f"  {r['dx']:7.2f} {r['peak']:7.3f} {r['air']:6.3f} {r['fwd_takeoff']:8.3f} "
                  f"{r['fwd_squat']:11.3f} {r['land_err']:9.3f} {r['creep']:+8.3f}  "
                  f"{'HIT' if r['land_err'] <= 0.10 else 'miss'}{' FELL' if r['fell'] else ''}")
        hits = sum(1 for r in good if r["land_err"] <= 0.10)
        print(f"  hit rate (err<=0.10m) {hits}/{len(good)} | PD stand height "
              f"{np.mean([r['stand_h'] for r in good]):.3f}m (Isaac: 0.303m)")
    else:
        print("\n=== SUMMARY: no completed jump ===")

    if recorder:
        keys = ("obs", "action", "comp", "tau", "q", "dq", "base", "fz")
        np.savez(args.dump,
                 trial=np.array([r["trial"] for r in recorder], dtype=np.int32),
                 step=np.array([r["step"] for r in recorder], dtype=np.int32),
                 **{k: np.stack([r[k] for r in recorder]).astype(np.float32) for k in keys})
        print(f"\n[dump  ] {len(recorder)} steps -> {args.dump}")


if __name__ == "__main__":
    main()
