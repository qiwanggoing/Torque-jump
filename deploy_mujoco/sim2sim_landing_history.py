"""Sim2Sim (Isaac Gym -> MuJoCo) for the OBS-HISTORY landing-torque policy (1012-dim actor).

The older `sim2sim_landing_torque.py` targets the model_4600 family: a 69-dim flat observation,
a two-head network, and a PD prior that anneals to exactly zero. None of that matches the
policies trained after 2026-09-19, so this file re-implements the physics-facing path for them:

  observation (1012) = [ frame(49) x 20 history | extras(32) ]
      frame  = base_lin_vel*2.0 (3) | base_ang_vel*0.25 (3) | projected_gravity (3)
               | (q - q_default) (12) | dq*0.05 (12) | foot_contact (4) | previous action (12)
      extras = yaw-frame landing error [fwd, lat, 0] (3) | cmd_height*2 (1) | cmd4 (1)
               | [z*2, (cmd_h - z)*2] (2) | applied torques (12) | motor fatigue (12)
               | pd_prior_alpha (1)
      the ring is oldest-first / newest-last, exactly as compute_observations() rolls it.

  control   = tau_res * rl_alpha * scale + pd_alpha * (kp*(q_phase - q) - kd*dq)
      pd_alpha = pd_prior_weight * max(PD_FLOOR, 1 - general_scale)      <- the FLOOR, not 0
      q_phase  = the phase-scheduled keyframe the PD prior tracks:
                 not jumping -> q_default, loaded -> q_squat, push/landing -> q_ground,
                 airborne -> q_air, prelanding -> q_pre
      There is no comp head on this line (aux_head_dim = 0).

⚠️ Trunk mass: these policies have a narrow dead zone at EXACTLY the nominal mass (measured:
they fail within about +-12 g of nominal and work everywhere else), so this script adds a small
offset by default, the same way the Isaac evaluations do.

Run:
  python deploy_mujoco/sim2sim_landing_history.py --ckpt <model_N.pt> --dx 0.85 [--viewer]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim_landing_torque import C, MjRobot, _build_mlp, TNMonitor  # noqa: E402

import mujoco  # noqa: E402


class H:
    """Everything this policy family needs that the flat script does not have."""
    # PD_TARGET_OBS=1 appends the 12-dim PD phase-target reference to the extras block, exactly as
    # go2_omnijump_landing_torque._get_single_extras does:
    #     pd_ref = (default_joint_pd_target - default_dof_pos) * obs_scales.dof_pos
    # 1012 -> 1024 actor input.
    PD_TGT_OBS = bool(int(os.environ.get("PD_TARGET_OBS", "0")))
    HIST = 20
    FRAME = 49
    EXTRAS = 32 + (12 if PD_TGT_OBS else 0)
    NUM_OBS = HIST * FRAME + EXTRAS

    PD_W = 0.5
    PD_FLOOR = 0.2                      # pd_alpha ends at 0.5*0.2 = 0.100, never 0
    # PD_FADE_SCOPE=all + the closed-loop load fade (_update_load_pd_fade) multiply pd_alpha by this.
    # The allfade3_* endpoint is 0.0 -> pd_alpha = 0.5*0.2*0.0 = 0.000, pure torque in EVERY phase.
    # NOTE the env recomputes rl_alpha from the MULTIPLIED pd_alpha (`rl_alpha_t = 1.0 - pd_alpha_t`,
    # go2_omnijump_curriculum_torque._compute_torques), so `rl_alpha = 1 - pd_alpha` below is already
    # the trained relation -- do NOT "fix" it to use the unmultiplied 0.100.
    LOAD_PD_SCALE = float(os.environ.get("LOAD_PD_SCALE", "1.0"))

    # phase keyframes, dumped from the training env (_solve_pose_from_foot_height)
    Q_SQUAT = np.array([0.1, 1.52645, -2.65812, -0.1, 1.52645, -2.65812,
                        0.1, 1.52645, -2.65812, -0.1, 1.52645, -2.65812])
    Q_GROUND = np.array([0.1, 0.85383, -1.57452, -0.1, 0.85383, -1.57452,
                         0.1, 0.85383, -1.57452, -0.1, 0.85383, -1.57452])
    Q_AIR = np.array([0.1, 1.24234, -2.26337, -0.1, 1.24234, -2.26337,
                      0.1, 1.24234, -2.26337, -0.1, 1.24234, -2.26337])
    Q_PRE = np.array([0.1, 1.02114, -1.88262, -0.1, 1.02114, -1.88262,
                      0.1, 1.02114, -1.88262, -0.1, 1.02114, -1.88262])

    SQUAT_THR = 3.2                     # squat_pose_threshold: |q - q_squat|_1 below this = in pose
    SQUAT_HOLD = 25                     # substeps the pose must be held to qualify
    FIRST_DELAY = 120                   # first_jump_delay_steps
    SETTLE = 10                         # jump_settle_steps
    LANDING_BUFFER = 150                # substeps of landing before the jump counts as finished
    PRELAND_MARGIN = 0.06
    BASE_H_TARGET = 0.42                # cfg.rewards.base_height_target (was wrongly 0.30 here)
    SPAWN_Z = 0.35


class TorqueLawH:
    """Training torque pipeline for this policy family: PD floor, phase target, no comp head."""

    def __init__(self):
        self.activation_sign = np.zeros(12)
        self.fatigue = np.zeros(12)
        self.last_torque = np.zeros(12)

    def reset(self):
        self.activation_sign[:] = 0.0
        self.fatigue[:] = 0.0
        self.last_torque[:] = 0.0

    def compute(self, action, q, dq, q_phase, general_scale):
        pd_alpha = H.PD_W * max(H.PD_FLOOR, 1.0 - general_scale) * H.LOAD_PD_SCALE
        rl_alpha = 1.0 - pd_alpha
        scale = C.START_TORQUE_SCALE + general_scale * (C.MAX_TORQUE_SCALE - C.START_TORQUE_SCALE)

        pd_full = C.P_GAIN * (q_phase - q) - C.D_GAIN * dq
        tau_action = action * C.TAU_LIM * rl_alpha * scale + pd_alpha * pd_full

        target_sign = np.tanh(tau_action / C.TAU_LIM)
        self.activation_sign += (target_sign - self.activation_sign) * C.ACTIVATION_SMOOTH
        s = self.activation_sign

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


def load_actor(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    actor = _build_mlp(ck["model_state_dict"], "actor")
    n = actor[0].in_features
    if n != H.NUM_OBS:
        raise SystemExit("checkpoint actor expects %d-dim obs; this script builds %d "
                         "(use sim2sim_landing_torque.py for the flat 69-dim family)" % (n, H.NUM_OBS))
    return actor


def replay_check(actor, npz_path):
    """Feed Isaac's OWN obs vectors through this script's actor and compare actions.

    This is the one measurement that separates WIRING from PHYSICS. The obs vectors come from
    legged_gym/scripts/play_landing.py run with PLAY_LANDING_DUMP=<npz>, i.e. exactly what the
    policy consumed in Isaac. Therefore:
      match (max|diff| < 1e-4) -> the checkpoint loading and the 1012/1024-dim obs LAYOUT are
                                  correct here; any MuJoCo jump difference is physics or the drive
                                  logic (state machine / keyframes / timing), NOT the observation.
      mismatch                 -> the layout or the ckpt load is wrong, and the per-block report
                                  below says which block, because the obs is a fixed concatenation:
                                  [frame(49) x 20 | extras(32) (| pd_target(12))].
    It does NOT validate how this script BUILDS obs from MuJoCo state -- only that the network and
    the vector layout it expects are right. Building is validated by the jump itself once this passes.
    """
    d = np.load(npz_path)
    obs = d["obs"].astype(np.float32)
    if obs.shape[1] != H.NUM_OBS:
        raise SystemExit("dump has %d-dim obs, this script expects %d. The dump must come from a run "
                         "with the SAME obs flags (PD_TARGET_OBS / REF_PREVIEW_OBS) as this script."
                         % (obs.shape[1], H.NUM_OBS))
    mask = (d["rl_driven"].reshape(-1) > 0.5) if "rl_driven" in d else np.ones(len(obs), bool)
    with torch.no_grad():
        a = actor(torch.from_numpy(obs[mask])).numpy()
    ref = d["action"][mask]
    da = np.abs(a - ref).max()
    print("[replay] %d/%d policy-driven steps from %s" % (int(mask.sum()), obs.shape[0], npz_path))
    print("[replay] actor max|diff| = %.3e   %s" % (da, "OK" if da < 1e-4 else "MISMATCH"))
    if da >= 1e-4:
        # Localise it: per-step error vs which obs block differs is not computable from one dump, but
        # the WORST step and the action-component spread say whether it is a global scale/offset error
        # (all 12 components off by a similar factor) or a layout scramble (components off at random).
        per_comp = np.abs(a - ref).max(axis=0)
        worst = int(np.abs(a - ref).max(axis=1).argmax())
        print("[replay] worst step %d | per-action-component max|diff|:" % worst)
        print("[replay]   %s" % np.array2string(per_comp, precision=4, max_line_width=200))
        print("[replay] a uniform spread suggests a SCALE/OFFSET error in one obs block; a ragged one "
              "suggests a LAYOUT scramble. Compare obs[worst] against this script's build_obs blocks.")
    else:
        print("[replay] layout + checkpoint load PROVEN correct -> the jump gap is physics or drive "
              "logic (keyframes / state machine / torque law), not the observation vector.")
    return da


class Trial:
    """One stand -> settle -> jump -> land episode, driven exactly as the env drives it."""

    def __init__(self, rb, actor, dx, dy, cmd_h):
        self.rb, self.actor = rb, actor
        self.dx, self.dy, self.cmd_h = dx, dy, cmd_h
        # Interactive mode sets these to keep the robot standing until the user presses SPACE.
        # reset() deliberately leaves them alone so they survive a re-arm. False = scripted behaviour.
        self.hold = False
        # stand_pd: hold the standing pose with a full PD instead of letting the policy drive.
        # The policy is jump-only -- it was trained to stand for 120 substeps before the jump fires
        # and collapses if it is asked to hold much longer -- so an open-ended wait needs the PD,
        # exactly as the deploy architecture does. The policy still runs, it just does not drive.
        self.stand_pd = False
        self.law = TorqueLawH()
        # --dump-obs: record THIS script's own obs vectors so they can be compared BLOCK BY BLOCK
        # against an Isaac dump. --replay proves the LAYOUT is right; this proves whether each block
        # is FILLED right. The honest comparison point is step 0 (both sides just reset: q=default,
        # dq=0, prev_action=0, torques=0, fatigue=0, four feet down, err_fwd==dx), because the two
        # trajectories diverge immediately afterwards and step k is then not the same state.
        self.obs_log = []
        self.state_log = []     # same 8 columns as the Isaac dump, so phases can be matched offline
        self.ring = np.zeros((H.HIST, H.FRAME), dtype=np.float64)
        self.prev_action = np.zeros(12)
        # reset_idx leaves default_joint_pd_target at default_dof_pos, so the first obs sees zeros
        self.q_phase_prev = C.DEFAULT_DOF_POS.copy()
        # jumps completed in THIS episode. The env's stand-phase PD target is
        # default_dof_pos only while jump_starts <= 0; after the first jump it is
        # q_ground_target (go2_omnijump_torque._update_default_joint_pd_target).
        self.jumps_done = 0
        self.reset()

    def reset(self):
        """EPISODE reset: teleport home and wipe everything (the env's reset_idx)."""
        self.rb.reset(z0=H.SPAWN_Z)
        self.law.reset()
        self.prev_action[:] = 0.0
        self.ring[:] = 0.0
        self.first_frame = True
        self.jumps_done = 0
        self.q_phase_prev = C.DEFAULT_DOF_POS.copy()
        self.rearm()

    def rearm(self):
        """WITHIN-episode re-arm: the robot stays where it landed and can jump again.

        This is what the env does between jumps of one episode -- the jump state machine goes
        back to stand, but the body, the obs history ring, the actuator activation and the
        fatigue all CARRY OVER (reset_idx is what clears those, and it is not called here).
        """
        self.t = 0
        self.settle_run = 0
        self.jumping = False
        self.squat_hold = 0
        self.squat_qualified = False
        self.took_off = False
        self.landed = False
        self.peak_h = 0.0
        self.takeoff_xy = None
        self.land_xy = None
        self.target = None
        self.mark = {}          # milestone snapshots for --trace
        self.min_h = 9.9
        self.arm_t = None

    # -- phases ---------------------------------------------------------
    def phase_target(self, z, vz, contact_any):
        if not self.jumping:
            # default_dof_pos only before the FIRST jump; afterwards the env's stand keyframe is
            # q_ground_target, because use_default_pose requires jump_starts <= 0.
            return H.Q_GROUND if self.jumps_done > 0 else C.DEFAULT_DOF_POS
        if not self.took_off:
            return H.Q_SQUAT if not self.squat_qualified else H.Q_GROUND
        if self.landed:
            # The landing task OVERRIDES the PD target for the whole landing buffer with the STANDING
            # pose, not q_ground (go2_omnijump_landing_torque.py::_update_default_joint_pd_target).
            return C.DEFAULT_DOF_POS
        preland_h = max(self.peak_h - H.PRELAND_MARGIN, H.BASE_H_TARGET + 0.04)
        if vz < -0.05 and z <= preland_h:
            return H.Q_PRE
        return H.Q_AIR

    def build_obs(self, q, dq, base_xyz, lin_body, ang_body, proj_grav, yaw, contact, pd_alpha):
        frame = np.concatenate([
            lin_body * C.S_LIN_VEL,
            ang_body * C.S_ANG_VEL,
            proj_grav,
            (q - C.DEFAULT_DOF_POS) * C.S_DOF_POS,
            dq * C.S_DOF_VEL,
            contact.astype(np.float64),
            self.prev_action,
        ])
        frame = np.nan_to_num(frame, nan=0.0, posinf=100.0, neginf=-100.0)
        if self.first_frame:                       # reset_idx fills the whole ring with one frame
            self.ring[:] = frame
            self.first_frame = False
        else:
            self.ring[:-1] = self.ring[1:]
            self.ring[-1] = frame

        tgt = self.target if self.target is not None else (base_xyz[:2] + np.array([self.dx, self.dy]))
        err_w = tgt - base_xyz[:2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        z = base_xyz[2]
        extras = np.concatenate([
            [cy * err_w[0] + sy * err_w[1], -sy * err_w[0] + cy * err_w[1], 0.0],
            [self.cmd_h * 2.0],
            [1.0 if self.jumping or not self.landed else 0.0],
            [z * 2.0, (self.cmd_h - z) * 2.0],
            self.law.last_torque,
            self.law.fatigue,
            [pd_alpha],
        ])
        if H.PD_TGT_OBS:
            # The PREVIOUS step's keyframe, matching the env's ordering exactly:
            # _update_default_joint_pd_target() runs inside _compute_torques (the decimation loop),
            # compute_observations() runs later in post_physics_step, and the policy consumes that obs
            # on the NEXT step -- the same one-step lag the pd_alpha slot already has here.
            extras = np.concatenate([
                extras, (self.q_phase_prev - C.DEFAULT_DOF_POS) * C.S_DOF_POS])
        extras = np.nan_to_num(extras, nan=0.0, posinf=100.0, neginf=-100.0)
        obs = np.concatenate([self.ring.reshape(-1), extras])
        return np.clip(obs, -C.CLIP_OBS, C.CLIP_OBS).astype(np.float32)

    # -- one control step ------------------------------------------------
    def step(self, pd_alpha_prev):
        rb = self.rb
        q, dq, base_xyz, lin_body, ang_body, proj_grav, yaw = rb.state()
        fz = rb.foot_fz()
        contact = fz > C.CONTACT_THRESH
        any_contact = bool(contact.any())
        all_contact = bool(contact.all())
        z, vz = base_xyz[2], rb.data.qvel[2]

        # arming: delay + settled (four feet, calm) for SETTLE consecutive substeps
        if not self.jumping:
            calm = all_contact and abs(vz) < 0.2 and np.linalg.norm(rb.data.qvel[3:6]) < 0.5
            self.settle_run = self.settle_run + 1 if calm else 0
            if self.t >= H.FIRST_DELAY and self.settle_run >= H.SETTLE and not self.hold:
                self.jumping = True
                self.arm_t = self.t
                self.target = base_xyz[:2] + np.array([self.dx, self.dy])
                self.mark["arm"] = dict(t=self.t, z=z, q=q.copy(),
                                        vxy=float(np.linalg.norm(rb.data.qvel[0:2])),
                                        wxy=float(np.linalg.norm(rb.data.qvel[3:5])),
                                        pitch=math.asin(max(-1.0, min(1.0, -proj_grav[0]))))

        # squat qualification gates takeoff
        if self.jumping and not self.took_off:
            pose_err = float(np.abs(q - H.Q_SQUAT).sum())
            self.squat_hold = self.squat_hold + 1 if pose_err <= H.SQUAT_THR else 0
            if self.squat_hold >= H.SQUAT_HOLD:
                self.squat_qualified = True
            if z < self.min_h:
                self.min_h = z
                self.mark["bottom"] = dict(t=self.t, z=z, q=q.copy())
            if self.squat_qualified and "squatQ" not in self.mark:
                self.mark["squatQ"] = dict(t=self.t, z=z)
            if self.squat_qualified and not any_contact:
                self.took_off = True
                self.mark["takeoff"] = dict(
                    t=self.t, z=z, vx=float(rb.data.qvel[0]), vz=float(rb.data.qvel[2]),
                    q=q.copy(), pitch=math.asin(max(-1.0, min(1.0, -proj_grav[0]))),
                    wy=float(rb.data.qvel[4]))
                self.takeoff_xy = base_xyz[:2].copy()
                self.target = self.takeoff_xy + np.array([self.dx, self.dy])   # takeoff anchor

        if self.took_off and not self.landed:
            self.peak_h = max(self.peak_h, z)
            if any_contact:
                self.landed = True
                self.land_xy = base_xyz[:2].copy()
        elif not self.took_off and self.target is not None:
            self.target = base_xyz[:2] + np.array([self.dx, self.dy])          # slides before takeoff

        obs = self.build_obs(q, dq, base_xyz, lin_body, ang_body, proj_grav, yaw, contact, pd_alpha_prev)
        if self.obs_log is not None and len(self.obs_log) < 2000:
            self.obs_log.append(obs.copy())
            # Mirror of play_landing's dump: same columns, same instant (obs just built, before action).
            self.state_log.append([z, float(rb.data.qvel[0]), float(rb.data.qvel[2]),
                                   float(proj_grav[0]), float(contact.sum()),
                                   float(self.jumping), float(self.squat_qualified),
                                   float(self.landed)])
        with torch.no_grad():
            action = self.actor(torch.from_numpy(obs)).numpy()
        action = np.clip(action, -C.CLIP_ACT, C.CLIP_ACT)
        self.prev_action = action.copy()

        q_phase = self.phase_target(z, vz, any_contact)
        self.q_phase_prev = np.asarray(q_phase, dtype=np.float64).copy()
        tau, pd_alpha = self.law.compute(action, q, dq, q_phase, general_scale=1.0)
        if self.stand_pd and not self.jumping:
            tau = np.clip(C.P_GAIN * (C.DEFAULT_DOF_POS - q) - C.D_GAIN * dq, -C.TAU_LIM, C.TAU_LIM)
        rb.apply(tau)
        self.t += 1
        return pd_alpha


HELP_I = """
  ------------------------------------------------------------------
   SPACE  jump now (uses the current dx / height)
   R      reset the robot to the origin, standing
   I / K  commanded distance dx   +/- 0.05 m
   U / J  commanded jump height   +/- 0.05 m
   P      print the current state
   ESC    quit (or just close the window)
  ------------------------------------------------------------------
  The robot stands and holds until you press SPACE, then runs exactly one
  jump and stands again, so you can watch a single jump as many times as
  you like. Everything else -- the torque blend, the phase postures, the
  observation -- is identical to the scripted trials.
"""


def interactive_loop(rb, actor, args):
    import time
    import mujoco.viewer

    st = dict(dx=args.dx, height=args.height, jump=False, reset=False, prt=False)

    def key_callback(keycode):
        if keycode == 32:                                   # SPACE
            st["jump"] = True
        elif keycode in (ord("R"), ord("r")):
            st["reset"] = True
        elif keycode in (ord("I"), ord("i"), 265):          # 265 = up arrow
            st["dx"] = min(st["dx"] + 0.05, 2.0); print("  [cmd] dx = %.2f m" % st["dx"], flush=True)
        elif keycode in (ord("K"), ord("k"), 264):          # 264 = down arrow
            st["dx"] = max(st["dx"] - 0.05, 0.0); print("  [cmd] dx = %.2f m" % st["dx"], flush=True)
        elif keycode in (ord("U"), ord("u"), 262):          # 262 = right arrow
            st["height"] = min(st["height"] + 0.05, 0.60); print("  [cmd] height = %.2f m" % st["height"], flush=True)
        elif keycode in (ord("J"), ord("j"), 263):          # 263 = left arrow
            st["height"] = max(st["height"] - 0.05, 0.30); print("  [cmd] height = %.2f m" % st["height"], flush=True)
        elif keycode in (ord("P"), ord("p")):
            st["prt"] = True

    viewer = mujoco.viewer.launch_passive(rb.model, rb.data, key_callback=key_callback)
    print(HELP_I)
    print("  ready: dx=%.2f m  height=%.2f m -- press SPACE in the viewer window\n"
          % (st["dx"], st["height"]), flush=True)

    tr = Trial(rb, actor, st["dx"], args.dy, st["height"])
    tr.hold = True
    tr.stand_pd = True
    spawn = rb.data.qpos[0:2].copy()
    pd_a = H.PD_W * H.PD_FLOOR * H.LOAD_PD_SCALE   # the OBS slot must report the ACTUAL pd_alpha: TorqueLawH.compute multiplies by LOAD_PD_SCALE, so omitting it here told a pure-torque policy it had 10% PD help that the torque law was not applying
    n_jump, after_land = 0, 0

    while viewer.is_running():
        t0 = time.perf_counter()
        if st["reset"]:
            st["reset"] = False
            tr.reset(); tr.hold = True; tr.stand_pd = True
            tr.dx, tr.cmd_h = st["dx"], st["height"]
            spawn = rb.data.qpos[0:2].copy()
            pd_a = H.PD_W * H.PD_FLOOR * H.LOAD_PD_SCALE   # the OBS slot must report the ACTUAL pd_alpha: TorqueLawH.compute multiplies by LOAD_PD_SCALE, so omitting it here told a pure-torque policy it had 10% PD help that the torque law was not applying
            after_land = 0
            print("  [reset]", flush=True)
        if st["jump"]:
            st["jump"] = False
            if tr.hold:
                tr.dx, tr.cmd_h = st["dx"], st["height"]
                # hand the stand back to the policy and restart the clock, so the last 0.6 s before
                # take-off is the same policy-driven settle the scripted trials run.
                tr.hold = False
                tr.stand_pd = False
                tr.t = 0
                tr.settle_run = 0
                print("  [jump] dx=%.2f height=%.2f" % (tr.dx, tr.cmd_h), flush=True)
        if st["prt"]:
            st["prt"] = False
            z = float(rb.data.qpos[2])
            print("  [state] t=%d z=%.3f jumping=%s squatQ=%s took_off=%s landed=%s"
                  % (tr.t, z, tr.jumping, tr.squat_qualified, tr.took_off, tr.landed), flush=True)

        pd_a = tr.step(pd_a)
        viewer.sync()
        # real-time pacing at 200 Hz. Without it the loop runs as fast as the CPU allows and the
        # motion looks far faster and jerkier than it is; Isaac's viewer paces itself.
        _sleep = C.DT - (time.perf_counter() - t0)
        if _sleep > 0:
            time.sleep(_sleep)

        if tr.landed:
            after_land += 1
            if after_land == 1:
                n_jump += 1
                a = float(np.linalg.norm(tr.land_xy - tr.takeoff_xy)) if tr.takeoff_xy is not None else float("nan")
                t = float(np.linalg.norm(tr.land_xy - spawn))
                e = float(np.linalg.norm(tr.land_xy - tr.target)) if tr.target is not None else float("nan")
                print("  [jump %d] peak=%.3f  fwd_from_takeoff=%+.3f  fwd_from_spawn=%+.3f  land_err=%.3f  %s"
                      % (n_jump, tr.peak_h, a, t, e, "HIT" if e <= 0.10 else "miss"), flush=True)
            if after_land > 150:                 # settled: hold the stance HERE and wait for SPACE
                # NO reset: the robot keeps the pose and the place it landed in, so successive
                # jumps chain the way they do inside one training episode. R still teleports home.
                tr.jumps_done += 1
                tr.rearm()
                tr.hold = True
                tr.stand_pd = True
                tr.dx, tr.cmd_h = st["dx"], st["height"]
                spawn = rb.data.qpos[0:2].copy()   # measure the next jump from HERE
                after_land = 0
                print("  ready at x=%+.3f y=%+.3f (jump %d done) -- SPACE for the next jump"
                      % (spawn[0], spawn[1], tr.jumps_done), flush=True)
    viewer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    _here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--xml", default=os.path.abspath(os.path.join(
        _here, "..", "..", "my_go2_jump", "resources", "robots", "go2", "go2", "scene.xml")),
        help="MuJoCo scene xml (same default as sim2sim_landing_torque.py)")
    ap.add_argument("--dx", type=float, default=0.85)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=0.45)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--add-mass", type=float, default=0.1,
                    help="trunk mass offset; keep non-zero, these policies fail at exactly nominal")
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--match-urdf", action="store_true",
                    help="scale the calf bodies to the URDF mass Isaac trains with")
    ap.add_argument("--isaac-joints", action="store_true",
                    help="zero the MJCF joint damping/armature/frictionloss (Isaac has none)")
    ap.add_argument("--trace", action="store_true",
                    help="print the state at arming / squat bottom / takeoff for Isaac comparison")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--interactive", action="store_true",
                    help="viewer + keyboard: you trigger each jump yourself (SPACE), change dx/height "
                         "with the arrow keys. Implies --viewer.")
    ap.add_argument("--pyramidal", action="store_true", default=True)
    ap.add_argument("--replay", default=None,
                    help="npz from legged_gym play_landing.py run with PLAY_LANDING_DUMP=<npz>: feed "
                         "Isaac's own obs through this actor and compare actions. Run this BEFORE "
                         "trusting any jump number -- it separates obs/ckpt wiring from physics.")
    ap.add_argument("--dump-obs", default=None,
                    help="write THIS script's own obs vectors (trial 1) to an npz, for a block-by-block "
                         "comparison against an Isaac dump. --replay checks the layout; this checks the "
                         "FILLING of each block.")
    args = ap.parse_args()

    # --replay needs only the checkpoint, not the MJCF or MuJoCo: do it before anything is built so it
    # still works when the scene/model is unavailable.
    if args.replay:
        replay_check(load_actor(args.ckpt), args.replay)
        return

    xml = os.path.abspath(args.xml)
    if not os.path.exists(xml):
        raise SystemExit("MJCF not found: %s (pass --xml)" % xml)
    rb = MjRobot(xml, add_mass=args.add_mass, friction=args.friction)
    if args.match_urdf:
        # The MJCF lumps the foot into the calf but at 0.2414 kg, while the URDF Isaac loads has
        # calf 0.1540 + foot 0.0400 = 0.1940 kg. That is +0.047 kg per leg at the far end of the
        # limb -- the mass that is swung fastest during the push, right at the knee-speed wall --
        # and it makes the MuJoCo robot 15.206 kg against Isaac's 15.019 kg. Scale the four calf
        # bodies (mass and inertia together) back to the URDF value.
        import mujoco as _mj
        tgt, fixed = 0.1540 + 0.0400, []
        for nm in ("FL_calf", "FR_calf", "RL_calf", "RR_calf"):
            bid = _mj.mj_name2id(rb.model, _mj.mjtObj.mjOBJ_BODY, nm)
            if bid < 0:
                continue
            f = tgt / float(rb.model.body_mass[bid])
            rb.model.body_mass[bid] *= f
            rb.model.body_inertia[bid] *= f
            fixed.append(nm)
        rb.total_mass = float(rb.model.body_mass.sum())
        print("[match-urdf] scaled %s to %.4f kg each -> total %.4f kg"
              % (",".join(fixed), tgt, rb.total_mass))
    if hasattr(rb, "match_isaac_contacts"):
        rb.match_isaac_contacts()
    if args.isaac_joints:
        # The SATA URDF declares no joint damping/friction and legged_gym sets armature = 0, so Isaac's
        # joints are ideal. go2.xml ships damping 0.05, armature 0.005, frictionloss 0.05.
        d, a, f = rb.match_isaac_joints()
        print("[isaac-joints] zeroed damping %.3f / armature %.3f / frictionloss %.3f" % (d, a, f))
    actor = load_actor(args.ckpt)
    mon = TNMonitor() if "TNMonitor" in globals() else None

    if args.interactive:
        print("[sim2sim-h] %s | interactive | add_mass=%+.2fkg friction=%.2f | total mass %.3f kg"
              % (os.path.basename(args.ckpt), args.add_mass, args.friction, rb.total_mass))
        interactive_loop(rb, actor, args)
        return

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(rb.model, rb.data)

    print("[sim2sim-h] %s | dx=%.2f dy=%.2f h=%.2f | add_mass=%+.2fkg friction=%.2f | total mass %.3f kg"
          % (os.path.basename(args.ckpt), args.dx, args.dy, args.height,
             args.add_mass, args.friction, rb.total_mass))
    air, tot, errs, ok = [], [], [], 0
    for k in range(args.trials):
        tr = Trial(rb, actor, args.dx, args.dy, args.height)
        spawn = rb.data.qpos[0:2].copy()
        pd_a = H.PD_W * H.PD_FLOOR * H.LOAD_PD_SCALE   # the OBS slot must report the ACTUAL pd_alpha: TorqueLawH.compute multiplies by LOAD_PD_SCALE, so omitting it here told a pure-torque policy it had 10% PD help that the torque law was not applying
        for _ in range(args.steps):
            pd_a = tr.step(pd_a)
            if viewer is not None:
                viewer.sync()
            if tr.landed and tr.t > 0:
                break
        if args.dump_obs and k == 0:
            _o = np.asarray(tr.obs_log, dtype=np.float32)
            np.savez(args.dump_obs, obs=_o,
                     state=np.asarray(tr.state_log, dtype=np.float32),
                     state_cols=np.asarray(["z", "vx", "vz", "projg_x",
                                            "n_contact", "jumping", "squatQ", "landed"]),
                     marks=np.asarray([tr.mark.get(m, {}).get("t", -1)
                                       for m in ("arm", "bottom", "squatQ", "takeoff")], dtype=np.int32))
            print("[dump-obs] wrote %s: obs%s | marks(arm,bottom,squatQ,takeoff)=%s"
                  % (args.dump_obs, _o.shape,
                     [tr.mark.get(m, {}).get("t", -1) for m in ("arm", "bottom", "squatQ", "takeoff")]))
        if tr.landed and tr.takeoff_xy is not None:
            a = float(np.linalg.norm(tr.land_xy - tr.takeoff_xy))
            t = float(np.linalg.norm(tr.land_xy - spawn))
            e = float(np.linalg.norm(tr.land_xy - tr.target))
            air.append(a); tot.append(t); errs.append(e); ok += e <= 0.10
            print("  trial %d | peak=%.3f  fwd_from_takeoff=%+.3f  fwd_from_spawn=%+.3f  land_err=%.3f  %s"
                  % (k + 1, tr.peak_h, a, t, e, "HIT" if e <= 0.10 else "miss"))
        else:
            print("  trial %d | NO JUMP (took_off=%s squat_qualified=%s)"
                  % (k + 1, tr.took_off, tr.squat_qualified))
        if args.trace:
            HIP, THI, CAL = [0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]
            for key in ("arm", "bottom", "squatQ", "takeoff"):
                m = tr.mark.get(key)
                if m is None:
                    print("    %-8s -- not reached" % key)
                    continue
                line = "    %-8s t=%4d z=%.3f" % (key, m["t"], m["z"])
                if "q" in m:
                    q = m["q"]
                    line += "  |hip|=%.3f thigh=%.3f calf=%.3f (F %.2f/%.2f  R %.2f/%.2f)" % (
                        np.abs(q[HIP]).mean(), q[THI].mean(), q[CAL].mean(),
                        q[1], q[2], q[7], q[8])
                if "vx" in m:
                    line += "  vx=%.3f vz=%.3f pitch=%+.3f wy=%+.2f" % (
                        m["vx"], m["vz"], m["pitch"], m["wy"])
                elif "pitch" in m:
                    line += "  |v_xy|=%.3f |w_xy|=%.3f pitch=%+.3f" % (m["vxy"], m["wxy"], m["pitch"])
                print(line)
            print("    peak    z=%.3f   crouch %s substeps" % (
                tr.peak_h,
                (tr.mark["takeoff"]["t"] - tr.arm_t) if ("takeoff" in tr.mark and tr.arm_t is not None) else "-"))
    if air:
        print("  mean: air=%.3f total=%.3f land_err=%.3f | hit %d/%d"
              % (np.mean(air), np.mean(tot), np.mean(errs), ok, len(air)))


if __name__ == "__main__":
    main()
