"""Build a COMMAND-CONDITIONED motion reference R(phi, c) from the extracted 14700 trajectories.

Storage:  anchors at the measured commands; the query interpolates linearly in c.
Time base: index 0 of the stored window is 650 ms before takeoff, takeoff at index `pre`.
           Online the env cannot know when takeoff will happen, so the reference clock is
           anchored on the UNLOAD event instead (GRF sum first drops below 50 N), whose
           offset to takeoff is stored per command. Measured spread of that offset over
           125-128 jumps: +-0.7..0.8 steps (+-4 ms), so a clock index is sufficient.
Clipping: calf joints are clipped to the URDF/MJCF hard range before storage, because the
          raw mean trajectory penetrates the lower stop by up to 0.082 rad.
"""
import os
import numpy as np
PT = os.path.dirname(os.path.abspath(__file__))
SP = os.environ.get('SP', '/tmp')
D = np.load(os.path.join(PT, 'ref_traj_14700.npz'))
R = np.load(os.path.join(SP, 'raw_base14700b_14700.npz'))
R2 = np.load(os.path.join(SP, 'raw_base14700_14700.npz'))
PRE = int(D['pre'][0]); FS = float(D['fs'][0]); dt = 1.0 / FS
W = D['q_0.90'].shape[0]
CAL_LO, CAL_HI = -2.7227, -0.83776
HIP_LO, HIP_HI = -1.0472, 1.0472
ANCH = [0.30, 0.50, 0.70, 0.80, 0.90]
def unload_offset(raw, c):
    cf, to, dn, lde, dxv = raw['cf'], raw['to'], raw['dn'], raw['lde'], raw['dx']
    S = cf.shape[0]; cols = np.where(np.abs(dxv - c) < 1e-6)[0]; out = []
    for n in cols:
        for k in np.where(to[:, n])[0]:
            i0, i1 = k - PRE, k + (W - PRE)
            if i0 < 0 or i1 > S or dn[i0:i1, n].any(): continue
            e = lde[k:i1, n]; e = e[e >= 0]
            if len(e) == 0 or e[0] > 0.12: continue
            g = cf[i0:k + 1, n].sum(axis=1); u = np.where(g < 50.0)[0]
            if len(u): out.append(PRE - int(u[0]))
    return (np.mean(out), np.std(out), len(out)) if out else (np.nan, np.nan, 0)
out = {'cmd': np.array(ANCH), 'fs': np.array([FS]), 'pre': np.array([PRE]),
       'calf_range': np.array([CAL_LO, CAL_HI])}
rows = []
for c in ANCH:
    k = '%.2f' % c
    q = D['q_' + k].copy()
    n_clip = int((q[:, 2::3] < CAL_LO).sum() + (q[:, 2::3] > CAL_HI).sum())
    worst = float(min(q[:, 2::3].min() - CAL_LO, 0.0))
    q[:, 2::3] = np.clip(q[:, 2::3], CAL_LO, CAL_HI)
    q[:, 0::3] = np.clip(q[:, 0::3], HIP_LO, HIP_HI)
    raw = R if c in (0.30, 0.80, 0.90) else R2
    mu, sd, n = unload_offset(raw, c)
    out['q_' + k] = q.astype(np.float32)
    out['dq_' + k] = D['dq_' + k].astype(np.float32)
    out['tau_' + k] = D['tau_' + k].astype(np.float32)
    out['base_' + k] = D['base_' + k].astype(np.float32)      # z vx vy vz pitch roll
    out['contact_' + k] = (D['grf_' + k] > 20.0).astype(np.uint8)
    out['q_std_' + k] = D['q_std_' + k].astype(np.float32)
    out['unload_' + k] = np.array([mu, sd, n])
    rows.append((c, int(D['n_' + k][0]), n_clip, worst, mu, sd, n))
np.savez_compressed(os.path.join(PT, 'commanded_ref_14700.npz'), **out)
print('%-6s %6s %10s %10s %14s %8s' % ('cmd', 'jumps', 'calf clip', 'worst[rad]', 'unload->TO[步]', '±'))
for c, nj, ncl, wo, mu, sd, n in rows:
    print('%-6.2f %6d %10d %10.4f %14.1f %8.2f' % (c, nj, ncl, wo, mu, sd))
print('\n[saved] commanded_ref_14700.npz   %.2f MB'
      % (os.path.getsize(os.path.join(PT, 'commanded_ref_14700.npz')) / 1048576))
