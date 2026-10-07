# Replay the Isaac-extracted reference through deploy_mujoco/ref_traj.py's feasibility check,
# AS-IS (no clipping of the calf overshoot past the hard stop).
#   run A: native ref_traj.replay() on the push segment [-650 ms .. takeoff]
#   run B: same PD law / T-N clamp, full window incl. flight + landing, with a base trace
import os, sys, math, importlib.util
import numpy as np
HERE = '/home/qiwang/torque_jump2/SATA/deploy_mujoco'
PT = '/home/qiwang/torque_jump2/SATA/paper_tools'
spec = importlib.util.spec_from_file_location('ref_traj', os.path.join(HERE, 'ref_traj.py'))
rt = importlib.util.module_from_spec(spec); sys.modules['ref_traj'] = rt; spec.loader.exec_module(rt)
import mujoco
D = np.load(os.path.join(PT, 'ref_traj_14700.npz'))
PRE = int(D['pre'][0]); FS = float(D['fs'][0]); dt = 1.0 / FS
CMDS = os.environ.get('CMDS', '0.90,0.70,0.50').split(',')
# Isaac ground truth for the same conditions (nominal mass, noise-free, deterministic)
ISAAC_NET = {'0.30': 0.324, '0.50': 0.482, '0.70': 0.683, '0.80': 0.755, '0.90': 0.809}
print('ref_traj.py: KP=%.1f KD=%.1f DT=%.4f   our reference dt=%.4f  (%d frames, takeoff at idx %d)'
      % (rt.KP, rt.KD, rt.DT, dt, D['q_0.90'].shape[0], PRE))
mj_lo, mj_hi = -2.7227, -0.83776
def trace(q_ref, settle=0.6):
    rb = rt._load_robot(rt.DEFAULT_XML)
    model, data = rb.model, rb.data
    model.opt.timestep = rt.DT
    data.qpos[:3] = [0, 0, 0.30]; data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[rb.q_ids] = q_ref[0]; data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    def step_to(target):
        qj = data.qpos[rb.q_ids]; dqj = data.qvel[rb.dq_ids]
        tau_cmd = rt.KP * (target - qj) - rt.KD * dqj
        tau, env_lim = rt.tn_clamp(tau_cmd, dqj)
        rb.apply(tau)
        return tau_cmd, tau, env_lim
    for _ in range(int(settle / rt.DT)): step_to(q_ref[0])
    Z=[];X=[];VX=[];VZ=[];C=[];TE=[];SAT=np.zeros(12);DEM=np.zeros(12)
    for k in range(len(q_ref)):
        tau_cmd, tau, lim = step_to(q_ref[k])
        SAT += (np.abs(tau_cmd) > lim + 1e-6).astype(float); DEM = np.maximum(DEM, np.abs(tau_cmd))
        TE.append(data.qpos[rb.q_ids] - q_ref[k])
        Z.append(data.qpos[2]); X.append(data.qpos[0])
        VX.append(data.qvel[0]); VZ.append(data.qvel[2]); C.append(int((rb.foot_fz() > 1.0).sum()))
    return (np.array(Z), np.array(X), np.array(VX), np.array(VZ), np.array(C),
            np.array(TE), SAT / len(q_ref), DEM)
for cmd in CMDS:
    q = D['q_%s' % cmd]; base = D['base_%s' % cmd]
    print('\n' + '#' * 90)
    print('## cmd %s   Isaac reference: |v|=%.3f  angle=%.1f deg  peak z=%.3f  net flight=%.3f m'
          % (cmd, math.hypot(base[PRE,1], base[PRE,3]),
             math.degrees(math.atan2(base[PRE,3], base[PRE,1])), base[:,0].max(),
             ISAAC_NET.get(cmd, float('nan'))))
    out = q[:, 2::3]
    n_lo = int((out < mj_lo - 1e-6).sum()); n_hi = int((out > mj_hi + 1e-6).sum())
    print('## calf targets outside the MJCF range [%.4f, %.4f]: %d of %d samples, worst %+.4f rad past'
          % (mj_lo, mj_hi, n_lo + n_hi, out.size, min(out.min() - mj_lo, 0.0)))
    print('\n--- RUN A: ref_traj.replay() on the push segment, as-is ---')
    qp = q[:PRE + 1]
    phi = np.linspace(0.0, 1.0, len(qp))
    rA = rt.replay(rt.DEFAULT_XML, phi, qp, settle=0.6, verbose=True)
    print('--- RUN B: full window (push + flight + landing), same law ---')
    Z, X, VX, VZ, C, TE, SAT, DEM = trace(q)
    to = PRE
    fly = np.where((np.arange(len(C)) > to) & (C == 0))[0]
    tdi = None
    if len(fly):
        after = np.where((np.arange(len(C)) > fly[-1]) & (C > 0))[0]
        tdi = int(after[0]) if len(after) else None
    print('  takeoff  @idx %d: z=%.3f  vx=%+.3f  vz=%+.3f  |v|=%.3f  angle=%.1f deg'
          % (to, Z[to], VX[to], VZ[to], math.hypot(VX[to], VZ[to]),
             math.degrees(math.atan2(VZ[to], VX[to]))))
    print('  peak z   = %.3f   (Isaac %.3f)' % (Z.max(), base[:, 0].max()))
    if tdi is not None:
        print('  touchdown@idx %d (%.0f ms after takeoff, Isaac %.0f ms)  net flight = %.3f m  (Isaac %.3f)'
              % (tdi, 1000*(tdi-to)*dt, 1000*(np.argmax(D['grf_%s'%cmd][PRE:].sum(axis=1) > 20.0))*dt,
                 X[tdi] - X[to], ISAAC_NET.get(cmd, float('nan'))))
    else:
        print('  never came down inside the window; z_end=%.3f' % Z[-1])
    calf_te = np.abs(TE[:, 2::3])
    print('  tracking error: rms %.4f rad  max %.4f  | calf only: rms %.4f  max %.4f'
          % (np.sqrt((TE**2).mean()), np.abs(TE).max(), np.sqrt((calf_te**2).mean()), calf_te.max()))
    ov = [(i, SAT[i], DEM[i], rt.TAU_LIM[i]) for i in range(12) if SAT[i] > 0.01]
    jn = ['%s_%s' % (l, j) for l in rt.FEET for j in ('hip', 'thigh', 'calf')]
    if ov:
        print('  torque over envelope: ' + ', '.join('%s %.0f%% (peak %.0f/%.0f Nm)'
              % (jn[i], 100*s, d, l) for i, s, d, l in sorted(ov, key=lambda x: -x[1])[:6]))
