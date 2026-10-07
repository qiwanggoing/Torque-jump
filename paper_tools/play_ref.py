"""Play back the Isaac-extracted reference trajectory in MuJoCo.

MODE=kin     (default) pure KINEMATIC playback: qpos is set straight from the reference.
             No controller, no tracking error -- you see exactly the motion the policy produced.
             This is the right mode for "is the motion what we want", because the trajectory is
             dynamically feasible BY CONSTRUCTION (it came out of a physics rollout).
MODE=torque  open-loop replay of the RECORDED torques, no feedback at all. Tests the torque
             trajectory itself; it will drift, because nothing corrects it.
MODE=pd      closed-loop PD on the reference, KP configurable (KP=40 is the task's own gain).

RUN (needs a display for the viewer; mujoco lives in ~/.local so do NOT set PYTHONNOUSERSITE)
  cd ~/torque_jump2/SATA/deploy_mujoco
  CMD=0.90 MODE=kin LOOP=5 SLOW=3 ~/miniconda3/envs/sata-clean/bin/python ../paper_tools/play_ref.py
  CMD=0.90 MODE=kin GIF=/tmp/ref.gif ~/miniconda3/envs/sata-clean/bin/python ../paper_tools/play_ref.py
"""
import os, sys, math, time, importlib.util
import numpy as np
GIF = os.environ.get('GIF', '')
if GIF: os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
HERE = '/home/qiwang/torque_jump2/SATA/deploy_mujoco'
PT = '/home/qiwang/torque_jump2/SATA/paper_tools'
spec = importlib.util.spec_from_file_location('ref_traj', os.path.join(HERE, 'ref_traj.py'))
rt = importlib.util.module_from_spec(spec); sys.modules['ref_traj'] = rt; spec.loader.exec_module(rt)
CMD = os.environ.get('CMD', '0.90'); MODE = os.environ.get('MODE', 'kin')
KP = float(os.environ.get('KP', '40')); KD = float(os.environ.get('KD', '0')) or 1.2 * math.sqrt(KP / 40.0)
LOOP = int(os.environ.get('LOOP', '3')); SLOW = float(os.environ.get('SLOW', '3'))
D = np.load(os.path.join(PT, 'ref_traj_14700.npz'))
PRE = int(D['pre'][0]); FS = float(D['fs'][0]); dt = 1.0 / FS
q = D['q_%s' % CMD]; tau = D['tau_%s' % CMD]; base = D['base_%s' % CMD]
W = len(q)
# base pose: z and orientation are measured; x,y are integrated from the measured velocity
# (the stored reference is takeoff-aligned, so absolute x carries no meaning).
x = np.concatenate([[0.0], np.cumsum(base[:-1, 1]) * dt]); x -= x[PRE]
y = np.concatenate([[0.0], np.cumsum(base[:-1, 2]) * dt]); y -= y[PRE]
z = base[:, 0]
# stored convention: pitch = asin(g_x) > 0 = NOSE-DOWN = +rotation about +y ; roll = asin(g_y) = -phi
pit = base[:, 4]; rol = -base[:, 5]
cp, sp = np.cos(pit / 2), np.sin(pit / 2); cr, sr = np.cos(rol / 2), np.sin(rol / 2)
quat = np.stack([cp * cr, cp * sr, sp * cr, -sp * sr], axis=1)
rb = rt._load_robot(rt.DEFAULT_XML)
model, data = rb.model, rb.data
model.opt.timestep = rt.DT
print('[play] cmd %s  mode %s  %d frames @%.0f Hz (%.2f s)  takeoff at frame %d'
      % (CMD, MODE, W, FS, W * dt, PRE))
if MODE != 'kin':
    print('       KP=%.0f KD=%.2f' % (KP, KD) if MODE == 'pd' else '       open-loop recorded torque')
def set_kin(k):
    data.qpos[:3] = [x[k], y[k], z[k]]
    data.qpos[3:7] = quat[k]
    data.qpos[rb.q_ids] = q[k]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
def reset_dyn():
    data.qpos[:3] = [0, 0, 0.30]; data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[rb.q_ids] = q[0]; data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(int(0.6 / rt.DT)):            # settle on the first pose with a stiff PD
        qj = data.qpos[rb.q_ids]; dqj = data.qvel[rb.dq_ids]
        t_, _l = rt.tn_clamp(200.0 * (q[0] - qj) - 2.7 * dqj, dqj); rb.apply(t_)
def step_dyn(k):
    qj = data.qpos[rb.q_ids]; dqj = data.qvel[rb.dq_ids]
    if MODE == 'pd':
        tc = KP * (q[k] - qj) - KD * dqj
        t_, _l = rt.tn_clamp(tc, dqj)
    else:
        t_, _l = rt.tn_clamp(tau[k], dqj)        # recorded torque, re-clamped by the same envelope
    rb.apply(t_)
if GIF:
    import imageio
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 854)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 480)
    rend = mujoco.Renderer(model, height=480, width=854)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 2.6, -12, 125
    frames = []
    if MODE != 'kin': reset_dyn()
    every = max(1, int(round(FS / 50.0)))        # ~50 fps gif
    for k in range(W):
        if MODE == 'kin': set_kin(k)
        else: step_dyn(k)
        if k % every == 0:
            cam.lookat[:] = [data.qpos[0], 0.0, 0.30]
            rend.update_scene(data, cam) ; frames.append(rend.render())
    imageio.mimsave(GIF, frames, fps=int(50 / SLOW), loop=0)
    print('[play] wrote %s  (%d frames, %.1fx slow)' % (GIF, len(frames), SLOW))
    nz = z if MODE == 'kin' else None
    print('[play] base z range %.3f .. %.3f m' % (z.min(), z.max()))
else:
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model, data) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.6, -12, 125
        for _ in range(LOOP):
            if MODE != 'kin': reset_dyn()
            for k in range(W):
                if MODE == 'kin': set_kin(k)
                else: step_dyn(k)
                v.cam.lookat[:] = [data.qpos[0], 0.0, 0.30]
                v.sync(); time.sleep(dt * SLOW)
            time.sleep(0.4)
