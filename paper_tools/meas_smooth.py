# meas_land.py + OUTPUT SMOOTHNESS of the commanded torque.
#   The policy runs at current_freq (200 Hz at general_scale=1, decimation 1, dt=0.005)
#   action a -> tau_cmd = 60*a -> tanh/|tau_lim| -> first-order activation (alpha 0.6) -> T-N curve -> tau_applied
# Logged per policy step: tau_applied (what a real motor driver would receive) and a (raw policy output).
# Reported per joint group x phase: RMS |dtau| in Nm/step and Nm/s, jitter index, direction-reversal rate,
# torque sign-flip rate, lag-1 autocorrelation of dtau (-1 = pure Nyquist alternation, 0 = white, +1 = smooth ramp),
# plus a Welch PSD over reset-free windows (band powers + peak frequency) and the action_rate reward value.
import os, sys
sys.argv = ['m', '--task=go2_omnijump_landing_torque', '--headless',
            '--num_envs=' + os.environ.get('NENV', '256'),
            '--load_run=' + os.environ.get('RUN', ''),
            '--checkpoint=' + os.environ.get('CKPT', '14700')]
import isaacgym  # noqa
import legged_gym.envs  # noqa
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque import GO2OmniJumpLandingTorque as L
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play_landing import train_step_count_at_iter
import torch
import numpy as np
N = int(os.environ.get('NENV', '256')); STEPS = int(os.environ.get('STEPS', '900'))
HEIGHT = float(os.environ.get('HEIGHT', '0.45'))
BINS = [float(x) for x in os.environ.get('BINS', '0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0').split(',')]
TOL = float(os.environ.get('TOL', '0.10')); MINPEAK = float(os.environ.get('MINPEAK', '0.30'))
NOM = int(os.environ.get('NOM', '0'))       # 1 = play_landing.py / real-robot condition
NOISE = int(os.environ.get('NOISE', '1'))   # 0 = observation noise OFF (play_landing.py sets this)
args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = N
env_cfg.commands.jump_command_range = [1.0, 1.0]
env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
if not NOISE:
    env_cfg.noise.add_noise = False
if NOM:
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [0.1, 0.1]
train_cfg.runner.resume = True
_orig = L._resample_commands
def _p(self, env_ids):
    _orig(self, env_ids)
    if len(env_ids) > 0 and hasattr(self, '_fixed_dx'):
        self.commands[env_ids, 0] = self._fixed_dx[env_ids]; self.commands[env_ids, 1] = 0.0
    if len(env_ids) > 0:
        self.commands[env_ids, 3] = HEIGHT; self.commands[env_ids, 4] = 1.0
L._resample_commands = _p
log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
ck = int(os.path.basename(path).replace('model_', '').replace('.pt', ''))
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
dev = env.device
env._fixed_dx = torch.tensor([BINS[i % len(BINS)] for i in range(N)], device=dev)
runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
policy = runner.get_inference_policy(device=dev); ac = runner.alg.actor_critic
REPLAY = int(round(train_step_count_at_iter(ck, float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
    float(env.start_freq), float(env.max_freq), float(env.dt), int(train_cfg.runner.num_steps_per_env))))
env.reset()
env._fixed_dx = torch.tensor([BINS[i % len(BINS)] for i in range(N)], device=dev)
env._resample_commands(torch.arange(N, device=dev))
obs = env.get_observations(); spawn = env.env_origins[:, :2].clone()
R = {b: dict(n=0, hit=0, tot=[], err=[]) for b in BINS}
S = dict(settle=[], drift=[], slide=[]); njump = 0
td_xy = torch.zeros(N, 2, device=dev); maxd = torch.zeros(N, device=dev)
settle_t = torch.full((N,), -1.0, device=dev); calm_run = torch.zeros(N, device=dev); since = torch.zeros(N, device=dev)
# ---- smoothness buffers (T,N,12) / (T,N) ----
TQ = torch.zeros(STEPS, N, 12, device=dev)
AC = torch.zeros(STEPS, N, 12, device=dev)
PH = torch.zeros(STEPS, N, dtype=torch.uint8, device=dev)
DN = torch.zeros(STEPS, N, dtype=torch.bool, device=dev)
PHN = ['stand_pre', 'stand_post', 'loaded', 'ground', 'airborne', 'prelanding', 'landing', 'other']
_nsub = [0]
_ct = L._compute_torques
def _ct_count(self, actions):
    _nsub[0] += 1
    return _ct(self, actions)
L._compute_torques = _ct_count
for t in range(STEPS):
    env.step_count = REPLAY
    _nsub[0] = 0
    with torch.no_grad():
        c = ac.comp_forward(obs)
        if c is not None: env.comp_torque = c.detach()
        a = policy(obs.detach())
    obs, _, _, dones, _ = env.step(a)
    if t == 0:
        print('[freq] current_freq=%.1f Hz  dt=%.4f  substeps/step=%d  action_scale=%.1f  pd_prior_weight=%.3f'
              % (float(env.current_freq), float(env.dt), _nsub[0], float(env.cfg.control.action_scale),
                 float(env.cfg.control.pd_prior_weight)))
        print('[dr] friction=%s push=%s mass=%s %s | add_noise=%s' % (env_cfg.domain_rand.randomize_friction,
              env_cfg.domain_rand.push_robots, env_cfg.domain_rand.randomize_base_mass,
              env_cfg.domain_rand.added_mass_range, env_cfg.noise.add_noise))
    TQ[t] = env.torques.detach()
    AC[t] = a.detach()[:, :12]
    DN[t] = dones.bool() if dones is not None else torch.zeros(N, dtype=torch.bool, device=dev)
    js = env.jumping_state; hto = env.has_taken_off
    code = torch.full((N,), 7, dtype=torch.uint8, device=dev)
    code = torch.where(js & (~hto), torch.full_like(code, 3), code)
    code = torch.where(env.phase_loaded, torch.full_like(code, 2), code)
    code = torch.where(env.airborne & (~env.prelanding), torch.full_like(code, 4), code)
    code = torch.where(env.prelanding, torch.full_like(code, 5), code)
    code = torch.where(env.landing, torch.full_like(code, 6), code)
    stand = (~js)
    code = torch.where(stand & (env.jump_starts <= 0.0), torch.full_like(code, 0), code)
    code = torch.where(stand & (env.jump_starts > 0.0), torch.full_like(code, 1), code)
    PH[t] = code
    jl = getattr(env, 'just_landed', None)
    if jl is not None and jl.any():
        idx = jl.nonzero(as_tuple=False).flatten()
        since[idx] = 0.0; calm_run[idx] = 0.0; settle_t[idx] = -1.0
        td_xy[idx] = env.root_states[idx][:, :2].clone(); maxd[idx] = 0.0
        for i in idx.tolist():
            key = min(BINS, key=lambda x: abs(x - float(env._fixed_dx[i])))
            err = float(torch.norm(env.root_states[i, :2] - env.landing_target[i, :2]))
            pk = float(env.peak_base_height[i]); R[key]['n'] += 1; njump += 1
            R[key]['hit'] += int(err <= TOL and pk >= MINPEAK)
            R[key]['tot'].append(float(env.root_states[i, 0] - spawn[i, 0])); R[key]['err'].append(err)
    inl = env.landing if hasattr(env, 'landing') else torch.zeros(N, dtype=torch.bool, device=dev)
    if inl.any():
        since = torch.where(inl, since + 1, since)
        all4 = torch.all(env.contact_forces[:, env.feet_indices, 2] > 1.0, dim=1)
        calm = all4 & (env.root_states[:, 7:9].norm(dim=1) < 0.2) & (env.root_states[:, 10:13].norm(dim=1) < 0.5)
        calm_run = torch.where(inl & calm, calm_run + 1, torch.where(inl, torch.zeros_like(calm_run), calm_run))
        new = inl & (settle_t < 0) & (calm_run >= 10)
        dnow = (env.root_states[:, :2] - td_xy).norm(dim=1)
        maxd = torch.where(inl, torch.maximum(maxd, dnow), maxd)
        if new.any():
            settle_t = torch.where(new, since, settle_t)
            for i in new.nonzero(as_tuple=False).flatten().tolist():
                S['settle'].append(float(since[i]) * float(env.dt))
                S['drift'].append(float(dnow[i])); S['slide'].append(float(maxd[i]))
def m(v): return sum(v) / len(v) if v else float('nan')
print('RESULT %s ckpt=%d  (sanity: same protocol as meas_land)' % (os.environ.get('TAG', ''), ck))
A = dict(n=0, hit=0, tot=[], err=[])
for b in BINS:
    d = R[b]
    if d['n'] == 0: continue
    A['n'] += d['n']; A['hit'] += d['hit']; A['tot'] += d['tot']; A['err'] += d['err']
print('  ALL  %4d  hit %.2f  dist %.3f  err %.3f | settled %d/%d (%.0f%%) in %.2fs'
      % (A['n'], A['hit'] / max(A['n'], 1), m(A['tot']), m(A['err']), len(S['settle']), njump,
         100.0 * len(S['settle']) / max(njump, 1), m(S['settle'])))

# ================= SMOOTHNESS =================
FS = float(env.current_freq)                      # policy / torque-update rate [Hz]
tau = TQ.cpu().numpy(); act = AC.cpu().numpy()
ph = PH.cpu().numpy(); dn = DN.cpu().numpy()
lim = env.torque_limits.detach().cpu().numpy()    # (12,)
np.save(os.path.join(os.environ.get('OUTDIR', '/tmp'), 'smooth_%s_%d.npy' % (os.environ.get('TAG', 'x'), ck)),
        {'tau': tau.astype(np.float16), 'act': act.astype(np.float16), 'ph': ph, 'dn': dn,
         'fs': FS, 'lim': lim}, allow_pickle=True)
GRP = {'hip': [0, 3, 6, 9], 'thigh': [1, 4, 7, 10], 'calf': [2, 5, 8, 11]}
dtau = np.diff(tau, axis=0)                       # (T-1,N,12)
dact = np.diff(act, axis=0)
# a step pair (t,t+1) is valid only if no reset happened at either end
vp = ~(dn[:-1] | dn[1:])                          # (T-1,N)
d2 = dtau[:-1] * dtau[1:]                         # for lag-1 autocorr, needs 3 clean steps
vp2 = vp[:-1] & vp[1:]
print('\n[smooth] fs=%.0f Hz  valid step-pairs %d  (of %d)' % (FS, int(vp.sum()), vp.size))
print('  %-11s %-10s %8s %8s %8s %8s %8s %8s %8s' % ('phase', 'group', 'RMStau', 'RMSdtau', 'Nm/s', 'jitter',
                                                      'rev/s', 'flip/s', 'rho1'))
rows = []
for pidx in [0, 1, 2, 3, 4, 5, 6]:
    pm = (ph[:-1] == pidx) & vp                   # (T-1,N) mask on the dtau axis
    if pm.sum() < 200: continue
    pm2 = (ph[:-2] == pidx) & vp2
    for g, cols in GRP.items():
        x = dtau[:, :, cols][pm]                  # (K,4)
        tt = tau[:-1][:, :, cols][pm]
        rms_t = float(np.sqrt(np.mean(tt ** 2))); rms_d = float(np.sqrt(np.mean(x ** 2)))
        sg = np.sign(dtau[:, :, cols]); rev = (sg[1:] * sg[:-1] < 0)
        rev_r = float(rev[pm2].mean()) * FS if pm2.sum() > 200 else float('nan')
        st = np.sign(tau[:, :, cols]); flip = (st[1:] * st[:-1] < 0)
        flip_r = float(flip[pm].mean()) * FS
        num = d2[:, :, cols][pm2]; den = dtau[:-1][:, :, cols][pm2]
        rho1 = float(num.mean() / max(np.mean(den ** 2), 1e-12)) if pm2.sum() > 200 else float('nan')
        print('  %-11s %-10s %8.3f %8.3f %8.1f %8.3f %8.1f %8.1f %+8.2f'
              % (PHN[pidx], g, rms_t, rms_d, rms_d * FS, rms_d / max(rms_t, 1e-9), rev_r, flip_r, rho1))
        rows.append((PHN[pidx], g, rms_t, rms_d, rev_r, rho1))
# overall
for g, cols in GRP.items():
    x = dtau[:, :, cols][vp]; tt = tau[:-1][:, :, cols][vp]
    rms_t = float(np.sqrt(np.mean(tt ** 2))); rms_d = float(np.sqrt(np.mean(x ** 2)))
    sg = np.sign(dtau[:, :, cols]); rev_r = float((sg[1:] * sg[:-1] < 0)[vp2].mean()) * FS
    st = np.sign(tau[:, :, cols]); flip_r = float((st[1:] * st[:-1] < 0)[vp].mean()) * FS
    rho1 = float((d2[:, :, cols][vp2]).mean() / max(np.mean(dtau[:-1][:, :, cols][vp2] ** 2), 1e-12))
    print('  %-11s %-10s %8.3f %8.3f %8.1f %8.3f %8.1f %8.1f %+8.2f'
          % ('ALL', g, rms_t, rms_d, rms_d * FS, rms_d / max(rms_t, 1e-9), rev_r, flip_r, rho1))
# saturation duty + per-joint worst
sat = (np.abs(tau) / lim[None, None, :] > 0.95)
print('\n[sat] |tau|>0.95*limit duty: hip %.3f  thigh %.3f  calf %.3f'
      % (sat[:, :, GRP['hip']].mean(), sat[:, :, GRP['thigh']].mean(), sat[:, :, GRP['calf']].mean()))
# action_rate reward cross-check
sq = (dact ** 2).sum(axis=2)
print('[action_rate] mean sum_j (da)^2 per step = %.5f  -> reward/step = %.3e  (scale -0.03 x dt %.3f)'
      % (float(sq[vp].mean()), -0.03 * float(env.dt) * float(sq[vp].mean()), float(env.dt)))
print('[action_rate] implied RMS |da| per joint = %.4f  -> commanded torque step %.2f Nm (= %.0f Nm/s)'
      % (np.sqrt(sq[vp].mean() / 12), np.sqrt(sq[vp].mean() / 12) * float(env.cfg.control.action_scale),
         np.sqrt(sq[vp].mean() / 12) * float(env.cfg.control.action_scale) * FS))
# ---- Welch PSD over reset-free windows ----
def runs(mask, L):
    out = []
    for n in range(mask.shape[1]):
        col = mask[:, n]; i = 0
        while i < len(col):
            if col[i]:
                j = i
                while j < len(col) and col[j]: j += 1
                if j - i >= L: out.append((n, i, j))
                i = j
            else: i += 1
    return out
def psd(mask, L, cols, name):
    segs = runs(mask, L)
    if not segs:
        print('  %-12s no window >= %d steps' % (name, L)); return
    w = np.hanning(L); acc = None; cnt = 0
    for (n, i, j) in segs:
        for s in range(i, j - L + 1, L // 2):
            blk = tau[s:s + L, n][:, cols]
            blk = blk - blk.mean(axis=0, keepdims=True)
            F = np.fft.rfft(blk * w[:, None], axis=0)
            P = (np.abs(F) ** 2).mean(axis=1)
            acc = P if acc is None else acc + P; cnt += 1
    P = acc / cnt; f = np.fft.rfftfreq(L, 1.0 / FS)
    tot = P[1:].sum()
    def band(a, b): return 100.0 * P[1:][(f[1:] >= a) & (f[1:] < b)].sum() / tot
    pk = f[1:][np.argmax(P[1:])]
    print('  %-12s n=%4d win  peak %5.1f Hz | 0-5Hz %4.1f%%  5-20 %4.1f%%  20-50 %4.1f%%  50-100 %4.1f%%'
          % (name, cnt, pk, band(0, 5), band(5, 20), band(20, 50), band(50, 101)))
L = int(os.environ.get('NFFT', '128'))
print('\n[psd] applied torque, Hann %d (%.0f ms), reset-free windows' % (L, 1000.0 * L / FS))
nores = ~dn
for g, cols in GRP.items(): psd(nores, L, cols, 'all ' + g)
standm = nores & ((ph == 0) | (ph == 1))
for g, cols in GRP.items(): psd(standm, L, cols, 'stand ' + g)
