import os, sys
sys.argv = ['m', '--task=go2_omnijump_landing_torque', '--headless',
            '--num_envs=' + os.environ.get('NENV', '512'),
            '--load_run=' + os.environ.get('RUN', ''),
            '--checkpoint=' + os.environ.get('CKPT', '12000')]
import isaacgym  # noqa
import legged_gym.envs  # noqa
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque import GO2OmniJumpLandingTorque as L
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play_landing import train_step_count_at_iter
import torch
N = int(os.environ.get('NENV', '512')); STEPS = int(os.environ.get('STEPS', '1700'))
HEIGHT = float(os.environ.get('HEIGHT', '0.45'))
BINS = [float(x) for x in os.environ.get('BINS', '0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0').split(',')]
TOL = float(os.environ.get('TOL', '0.10')); MINPEAK = float(os.environ.get('MINPEAK', '0.30'))
args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = N
env_cfg.commands.jump_command_range = [1.0, 1.0]
env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
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
for t in range(STEPS):
    env.step_count = REPLAY
    with torch.no_grad():
        c = ac.comp_forward(obs)
        if c is not None: env.comp_torque = c.detach()
        a = policy(obs.detach())
    obs, _, _, _, _ = env.step(a)
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
                S['drift'].append(float(dnow[i]))
                S['slide'].append(float(maxd[i]))
def m(v): return sum(v) / len(v) if v else float('nan')
print('RESULT %s ckpt=%d' % (os.environ.get('TAG', ''), ck))
print('  cmd   n   hit   dist    err')
A = dict(n=0, hit=0, tot=[], err=[])
for b in BINS:
    d = R[b]
    if d['n'] == 0: continue
    print('  %.2f %4d  %.2f  %.3f  %.3f' % (b, d['n'], d['hit'] / d['n'], m(d['tot']), m(d['err'])))
    A['n'] += d['n']; A['hit'] += d['hit']; A['tot'] += d['tot']; A['err'] += d['err']
TOLS=[0.08,0.10,0.12,0.15,0.20]
print('  tolerance sweep:  cmd ' + ''.join('  <=%.2f'%t for t in TOLS))
for b in BINS:
    e=R[b]['err']
    if not e: continue
    print('                   %.2f '%b + ''.join('  %5.2f'%(sum(1 for x in e if x<=t)/len(e)) for t in TOLS))
print('  ALL  %4d  %.2f  %.3f  %.3f | settled %d/%d (%.0f%%) in %.2fs' % (A['n'], A['hit']/A['n'], m(A['tot']), m(A['err']), len(S['settle']), njump, 100.0*len(S['settle'])/max(njump,1), m(S['settle'])))
print('  settle drift: mean %.3f m at settle, max excursion %.3f m' % (m(S['drift']), m(S['slide'])))
