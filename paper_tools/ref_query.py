"""Query a command-conditioned motion reference R(phi, c) built by build_commanded_ref.py.

Indexing (the part that has to work ONLINE, where takeoff time is not yet known):
  * the clock is anchored on the UNLOAD event -- the step at which the foot force first
    collapses (GRF sum < 50 N). The env sees this directly; measured spread of
    unload->takeoff over 125-170 jumps per command is +-0.6..1.1 steps (+-3..6 ms).
  * BEFORE takeoff the index is the normalised progress phi = k / K_unload_to_takeoff(c),
    so takeoff lands exactly at phi = 1 for every command (K is itself interpolated in c:
    70.1 steps at cmd 0.30 -> 81.9 at cmd 0.90).
  * AFTER takeoff the index is absolute steps since takeoff, because flight duration is set
    by the launch state and genuinely differs per command.
Command axis: linear interpolation between the measured anchors, clamped at the ends.
    Leave-one-out check: predicting cmd 0.70 from 0.50/0.80 gives q RMS 0.027 rad, which is
    0.59x the jump-to-jump spread at that command -- i.e. below the natural noise floor.
"""
import os
import numpy as np
_F = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'commanded_ref_14700.npz')
class CommandedReference:
    KEYS = ('q', 'dq', 'tau', 'base', 'contact')
    def __init__(self, path=_F):
        d = np.load(path)
        self.c = d['cmd']; self.fs = float(d['fs'][0]); self.pre = int(d['pre'][0])
        self.dt = 1.0 / self.fs
        self.calf_range = d['calf_range']
        self.data = {k: [d['%s_%.2f' % (k, c)] for c in self.c] for k in self.KEYS}
        self.q_std = [d['q_std_%.2f' % c] for c in self.c]
        self.K = np.array([d['unload_%.2f' % c][0] for c in self.c])   # unload -> takeoff [steps]
        self.W = self.data['q'][0].shape[0]
    def _blend(self, c):
        c = float(np.clip(c, self.c[0], self.c[-1]))
        j = int(np.clip(np.searchsorted(self.c, c) - 1, 0, len(self.c) - 2))
        w = (c - self.c[j]) / (self.c[j + 1] - self.c[j])
        return j, w
    def takeoff_steps(self, c):
        """How many steps after the unload event takeoff is expected, for this command."""
        j, w = self._blend(c)
        return float((1 - w) * self.K[j] + w * self.K[j + 1])
    def _sample(self, arr, idx):
        i0 = int(np.floor(idx)); f = idx - i0
        i0 = int(np.clip(i0, 0, self.W - 1)); i1 = int(np.clip(i0 + 1, 0, self.W - 1))
        return (1 - f) * arr[i0] + f * arr[i1]
    def query(self, c, k, after_takeoff=False, keys=None):
        """k = steps since the unload event (after_takeoff=False)
                or steps since takeoff   (after_takeoff=True)."""
        j, w = self._blend(c)
        out = {}
        for key in (keys or self.KEYS):
            vals = []
            for m in (j, j + 1):
                if after_takeoff:
                    idx = self.pre + k
                else:
                    # phi-normalised so takeoff lands at phi=1 for both anchors and for the query.
                    # k past K spills into the post-takeoff branch in ABSOLUTE steps, matching the env's
                    # _ref_preview: a preview horizon may cross the reference's takeoff.
                    K = self.takeoff_steps(c)
                    if k > K:
                        idx = self.pre + (k - K)
                    else:
                        phi = k / max(K, 1e-9)
                        idx = self.pre - self.K[m] + phi * self.K[m]
                vals.append(self._sample(self.data[key][m].astype(np.float64), idx))
            out[key] = (1 - w) * vals[0] + w * vals[1]
        if 'contact' in out: out['contact'] = (out['contact'] > 0.5).astype(np.float32)
        return out
    def preview(self, c, k, horizons=(0, 10, 25), after_takeoff=False, key='q'):
        return np.concatenate([self.query(c, k + h, after_takeoff, keys=(key,))[key] for h in horizons])
if __name__ == '__main__':
    r = CommandedReference()
    print('anchors', r.c, ' fs', r.fs, ' window', r.W, ' takeoff idx', r.pre)
    print('unload->takeoff [steps]:', np.round(r.K, 1))
    print('\nround-trip at the anchors (query vs stored, q):')
    for c in r.c:
        K = r.takeoff_steps(c); m = list(r.c).index(c)
        err = max(np.abs(r.query(c, k)['q'] - r.data['q'][m][r.pre - int(round(K)) + k]).max()
                  for k in range(0, int(K), 5))
        print('  cmd %.2f  K=%.1f  max|err| %.4f rad' % (c, K, err))
    print('\nleave-one-out through the query path (drop 0.70, predict from 0.50/0.80):')
    import copy
    r2 = CommandedReference()
    keep = [0, 1, 3, 4]
    r2.c = r.c[keep]; r2.K = r.K[keep]
    for kname in r2.KEYS: r2.data[kname] = [r.data[kname][i] for i in keep]
    K = r.takeoff_steps(0.70)
    e = []
    for k in range(0, int(K)):
        e.append(r2.query(0.70, k)['q'] - r.query(0.70, k)['q'])
    e = np.array(e)
    print('  pre-takeoff  q RMS %.4f rad  max %.4f  (jump-to-jump 1sigma %.4f)'
          % (np.sqrt((e ** 2).mean()), np.abs(e).max(), r.q_std[2][:r.pre].mean()))
    e = np.array([r2.query(0.70, k, True)['q'] - r.query(0.70, k, True)['q'] for k in range(0, 170)])
    print('  post-takeoff q RMS %.4f rad  max %.4f  (jump-to-jump 1sigma %.4f)'
          % (np.sqrt((e ** 2).mean()), np.abs(e).max(), r.q_std[2][r.pre:].mean()))
    print('\nexample query, cmd 0.75 (never measured):')
    K = r.takeoff_steps(0.75); print('  expected unload->takeoff = %.1f steps (%.0f ms)' % (K, K * r.dt * 1000))
    for lab, k, af in (('unload', 0, False), ('mid-prep', int(K * 0.5), False),
                       ('takeoff', 0, True), ('apex', 51, True), ('touchdown', 89, True)):
        s = r.query(0.75, k, af)
        print('  %-9s z=%.3f vx=%+.2f vz=%+.2f pitch=%+.1f°  calf FL/RL %+.3f/%+.3f  contact %s'
              % (lab, s['base'][0], s['base'][1], s['base'][3], np.degrees(s['base'][4]),
                 s['q'][2], s['q'][8], s['contact'].astype(int)))
