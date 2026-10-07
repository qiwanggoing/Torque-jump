"""Per-bin creep / flight / total, so the command and the reported distance can be read together."""
import os, sys
sys.argv = ['m', '--task=go2_omnijump_landing_torque', '--headless',
            '--num_envs=' + os.environ.get('NENV', '512'),
            '--load_run=' + os.environ['RUN'], '--checkpoint=' + os.environ.get('CKPT', '13000')]
import isaacgym  # noqa
import legged_gym.envs  # noqa
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque import GO2OmniJumpLandingTorque as L
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play_landing import train_step_count_at_iter
import torch

N = int(os.environ.get('NENV', '512')); STEPS = int(os.environ.get('STEPS', '1700'))
HEIGHT = 0.45; TOL = 0.10; MINPEAK = 0.30
BINS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
args = get_args(); env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = N; env_cfg.commands.jump_command_range = [1.0, 1.0]
env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]; train_cfg.runner.resume = True
_o = L._resample_commands
def _p(self, e):
    _o(self, e)
    if len(e) > 0 and hasattr(self, '_fx'):
        self.commands[e, 0] = self._fx[e]; self.commands[e, 1] = 0.0
    if len(e) > 0:
        self.commands[e, 3] = HEIGHT; self.commands[e, 4] = 1.0
L._resample_commands = _p
lr = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
path = get_load_path(lr, load_run=args.load_run, checkpoint=args.checkpoint)
ck = int(os.path.basename(path).replace('model_', '').replace('.pt', ''))
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg); dev = env.device
env._fx = torch.tensor([BINS[i % len(BINS)] for i in range(N)], device=dev)
runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
policy = runner.get_inference_policy(device=dev); ac = runner.alg.actor_critic
RP = int(round(train_step_count_at_iter(ck, float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
        float(env.start_freq), float(env.max_freq), float(env.dt), int(train_cfg.runner.num_steps_per_env))))
env.reset(); env._fx = torch.tensor([BINS[i % len(BINS)] for i in range(N)], device=dev)
env._resample_commands(torch.arange(N, device=dev)); obs = env.get_observations()
spawn = env.env_origins[:, :2].clone()
xto = torch.zeros(N, device=dev); prev = torch.zeros(N, dtype=torch.bool, device=dev)
R = {b: dict(n=0, h=0, cr=[], fl=[], to=[], er=[]) for b in BINS}
for t in range(STEPS):
    env.step_count = RP
    with torch.no_grad():
        c = ac.comp_forward(obs)
        if c is not None: env.comp_torque = c.detach()
        a = policy(obs.detach())
    obs, _, _, _, _ = env.step(a)
    rise = env.has_taken_off & (~prev); prev = env.has_taken_off.clone()
    if rise.any(): xto = torch.where(rise, env.root_states[:, 0], xto)
    jl = getattr(env, 'just_landed', None)
    if jl is not None and jl.any():
        for i in jl.nonzero(as_tuple=False).flatten().tolist():
            k = min(BINS, key=lambda x: abs(x - float(env._fx[i])))
            er = float(torch.norm(env.root_states[i, :2] - env.landing_target[i, :2]))
            pk = float(env.peak_base_height[i]); d = R[k]
            d['n'] += 1; d['h'] += int(er <= TOL and pk >= MINPEAK)
            d['cr'].append(float(xto[i] - spawn[i, 0]))
            d['fl'].append(float(env.root_states[i, 0] - xto[i]))
            d['to'].append(float(env.root_states[i, 0] - spawn[i, 0]))
            d['er'].append(er)
def m(v): return sum(v) / len(v) if v else float('nan')
print('RESULT %s ckpt=%d' % (os.environ.get('TAG', ''), ck))
print('  cmd    n   hit   creep  flight  total   err')
A = dict(n=0, h=0, cr=[], fl=[], to=[], er=[])
for b in BINS:
    d = R[b]
    if not d['n']: continue
    print('  %.2f %4d  %.2f  %6.3f %6.3f %6.3f  %.3f'
          % (b, d['n'], d['h'] / d['n'], m(d['cr']), m(d['fl']), m(d['to']), m(d['er'])))
    for k in ('cr', 'fl', 'to', 'er'): A[k] += d[k]
    A['n'] += d['n']; A['h'] += d['h']
print('  ALL  %4d  %.2f  %6.3f %6.3f %6.3f  %.3f'
      % (A['n'], A['h'] / A['n'], m(A['cr']), m(A['fl']), m(A['to']), m(A['er'])))
