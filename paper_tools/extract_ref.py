# Extract a CLEAN reference trajectory from a trained policy.
#   perfect / noise-free conditions: deterministic mean action, add_noise=False, friction fixed,
#   no pushes, nominal trunk mass (+0.1 kg = play_landing.py), fixed jump height, PD replay pinned.
#   Many identical jumps per command -> aligned on the TAKEOFF step -> mean +- std across jumps,
#   so Isaac's residual non-determinism averages out and the spread is reported, not hidden.
# Saves ref_<tag>_<ckpt>.npz:  per command bin  q/dq/tau (W,12), base (W,6), contact (W,4), std, n
import os, sys
sys.argv = ['m', '--task=go2_omnijump_landing_torque', '--headless',
            '--num_envs=' + os.environ.get('NENV', '255'),
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
N = int(os.environ.get('NENV', '255')); STEPS = int(os.environ.get('STEPS', '1400'))
HEIGHT = float(os.environ.get('HEIGHT', '0.45'))
BINS = [float(x) for x in os.environ.get('BINS', '0.5,0.7,0.9').split(',')]
PRE = int(os.environ.get('PRE', '130'))     # steps kept before takeoff
POST = int(os.environ.get('POST', '170'))   # steps kept after takeoff
ETOL = float(os.environ.get('ETOL', '0.05'))  # only keep accurate jumps in the reference
MINPEAK = float(os.environ.get('MINPEAK', '0.30'))
OUTDIR = os.environ.get('OUTDIR', '/tmp')
TAG = os.environ.get('TAG', 'ref')
args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = N
env_cfg.commands.jump_command_range = [1.0, 1.0]
env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
# ---- perfect, noise-free, no-DR condition ----
env_cfg.noise.add_noise = False
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
obs = env.get_observations()
DP = torch.zeros(STEPS, N, 12, device=dev); DV = torch.zeros(STEPS, N, 12, device=dev)
TQ = torch.zeros(STEPS, N, 12, device=dev); CF = torch.zeros(STEPS, N, 4, device=dev)
BS = torch.zeros(STEPS, N, 6, device=dev)        # z vx vy vz pitch roll
TO = torch.zeros(STEPS, N, dtype=torch.bool, device=dev)   # takeoff rising edge
LDE = torch.full((STEPS, N), -1.0, device=dev)   # landing error at touchdown step
DN = torch.zeros(STEPS, N, dtype=torch.bool, device=dev)
prev_to = torch.zeros(N, dtype=torch.bool, device=dev)
for t in range(STEPS):
    env.step_count = REPLAY
    with torch.no_grad():
        c = ac.comp_forward(obs)
        if c is not None: env.comp_torque = c.detach()
        a = policy(obs.detach())
    obs, _, _, dones, _ = env.step(a)
    if t == 0:
        print('[cond] freq=%.1fHz dt=%.4f noise=%s fric=%s push=%s mass=%s | bins=%s h=%.2f etol=%.3f'
              % (float(env.current_freq), float(env.dt), env_cfg.noise.add_noise,
                 env_cfg.domain_rand.randomize_friction, env_cfg.domain_rand.push_robots,
                 env_cfg.domain_rand.added_mass_range, BINS, HEIGHT, ETOL))
    DP[t] = env.dof_pos.detach(); DV[t] = env.dof_vel.detach(); TQ[t] = env.torques.detach()
    CF[t] = env.contact_forces[:, env.feet_indices, 2].detach()
    rpy = env.base_lin_vel.detach()
    BS[t, :, 0] = env.root_states[:, 2]
    BS[t, :, 1:4] = env.root_states[:, 7:10]
    g = env.projected_gravity.detach()
    # project convention (legged_gym/scripts/pitch_sign_check.py): pitch = asin(projected_gravity[0]),
    # POSITIVE = NOSE-DOWN.  roll = asin(projected_gravity[1]).
    BS[t, :, 4] = torch.asin(g[:, 0].clamp(-1.0, 1.0))
    BS[t, :, 5] = torch.asin(g[:, 1].clamp(-1.0, 1.0))
    hto = env.has_taken_off
    TO[t] = hto & (~prev_to); prev_to = hto.clone()
    DN[t] = dones.bool() if dones is not None else torch.zeros(N, dtype=torch.bool, device=dev)
    jl = getattr(env, 'just_landed', None)
    if jl is not None and jl.any():
        e = (env.root_states[:, :2] - env.landing_target[:, :2]).norm(dim=1)
        ok = jl & (env.peak_base_height >= MINPEAK)
        LDE[t] = torch.where(ok, e, torch.full_like(e, -1.0))
np.savez(os.path.join(OUTDIR, 'raw_%s_%d.npz' % (TAG, ck)),
         dp=DP.cpu().numpy().astype(np.float32), dv=DV.cpu().numpy().astype(np.float32),
         tq=TQ.cpu().numpy().astype(np.float32), cf=CF.cpu().numpy().astype(np.float32),
         bs=BS.cpu().numpy().astype(np.float32), to=TO.cpu().numpy(), lde=LDE.cpu().numpy(),
         dn=DN.cpu().numpy(), dx=env._fixed_dx.cpu().numpy(), fs=float(env.current_freq),
         lim=env.torque_limits.detach().cpu().numpy(), dt=float(env.dt))
print('[saved] raw_%s_%d.npz' % (TAG, ck))
# =============== align on takeoff, average across jumps ===============
dp = DP.cpu().numpy(); dv = DV.cpu().numpy(); tq = TQ.cpu().numpy()
cf = CF.cpu().numpy(); bs = BS.cpu().numpy()
to = TO.cpu().numpy(); lde = LDE.cpu().numpy(); dn = DN.cpu().numpy()
dxv = env._fixed_dx.cpu().numpy(); FS = float(env.current_freq)
W = PRE + POST + 1
out = {}
print('\n[align] window %d steps (%.0f ms before takeoff, %.0f ms after), accept err<=%.3f'
      % (W, 1000.0 * PRE / FS, 1000.0 * POST / FS, ETOL))
for b in BINS:
    cols = np.where(np.abs(dxv - b) < 1e-6)[0]
    acc = []
    for n in cols:
        tos = np.where(to[:, n])[0]
        for k in tos:
            i0, i1 = k - PRE, k + POST + 1
            if i0 < 0 or i1 > STEPS: continue
            if dn[i0:i1, n].any(): continue                 # no reset inside the window
            e = lde[k:i1, n]; e = e[e >= 0.0]
            if len(e) == 0 or e[0] > ETOL: continue         # must land, and land accurately
            acc.append((n, i0, float(e[0])))
    if not acc:
        print('  cmd %.2f : no accepted jump' % b); continue
    Q = np.stack([dp[i0:i0 + W, n] for (n, i0, _) in acc])      # (J,W,12)
    QD = np.stack([dv[i0:i0 + W, n] for (n, i0, _) in acc])
    T_ = np.stack([tq[i0:i0 + W, n] for (n, i0, _) in acc])
    C_ = np.stack([cf[i0:i0 + W, n] for (n, i0, _) in acc])
    B_ = np.stack([bs[i0:i0 + W, n] for (n, i0, _) in acc])
    errs = np.array([e for (_, _, e) in acc])
    out['q_%.2f' % b] = Q.mean(0); out['q_std_%.2f' % b] = Q.std(0)
    out['dq_%.2f' % b] = QD.mean(0); out['dq_std_%.2f' % b] = QD.std(0)
    out['tau_%.2f' % b] = T_.mean(0); out['tau_std_%.2f' % b] = T_.std(0)
    out['grf_%.2f' % b] = C_.mean(0); out['base_%.2f' % b] = B_.mean(0)
    out['base_std_%.2f' % b] = B_.std(0); out['n_%.2f' % b] = np.array([len(acc)])
    print('  cmd %.2f : %4d jumps  land_err %.4f+-%.4f  | cross-jump std at takeoff: q %.4f rad  tau %.3f Nm  vx %.4f m/s'
          % (b, len(acc), errs.mean(), errs.std(), Q.std(0)[PRE].mean(), T_.std(0)[PRE].mean(),
             B_.std(0)[PRE, 1]))
out['fs'] = np.array([FS]); out['pre'] = np.array([PRE]); out['post'] = np.array([POST])
out['lim'] = env.torque_limits.detach().cpu().numpy()
np.savez(os.path.join(OUTDIR, 'ref_%s_%d.npz' % (TAG, ck)), **out)
print('[saved] ref_%s_%d.npz' % (TAG, ck))
# =============== smoothness of the MEAN reference vs a single jump ===============
GRP = {'hip': [0, 3, 6, 9], 'thigh': [1, 4, 7, 10], 'calf': [2, 5, 8, 11]}
def sm(x, name):
    d = np.diff(x, axis=0)
    r = []
    for g, c in GRP.items():
        rd = float(np.sqrt(np.mean(d[:, c] ** 2)))
        num = (d[:-1][:, c] * d[1:][:, c]).mean(); den = np.mean(d[:-1][:, c] ** 2)
        r.append((g, rd, float(num / max(den, 1e-12))))
    print('  %-26s ' % name + '  '.join('%s d=%.4f rho1=%+.2f' % t for t in r))
for b in BINS:
    k = 'tau_%.2f' % b
    if k not in out: continue
    cols = np.where(np.abs(dxv - b) < 1e-6)[0]
    print('\n[smooth] cmd %.2f  (n=%d jumps)' % (b, int(out['n_%.2f' % b][0])))
    sm(out[k], 'tau  MEAN reference')
    # one representative single jump for contrast
    n0 = cols[0]; tos = np.where(to[:, n0])[0]
    for kk in tos:
        if kk - PRE >= 0 and kk + POST + 1 <= STEPS and not dn[kk - PRE:kk + POST + 1, n0].any():
            sm(tq[kk - PRE:kk + POST + 1, n0], 'tau  single jump'); break
    sm(out['q_%.2f' % b] * 1000.0, 'q    MEAN ref (mrad)')
