"""Attitude of the base while it stands, before the jump starts.

Not the feet and not the translation: roll / pitch / yaw and the body rates, sampled every
control step from the moment the episode settles until the load phase begins, reported as
peak-to-peak swing, end value and RMS rate.
"""
import os, sys
sys.argv = ['m', '--task=go2_omnijump_landing_torque', '--headless',
            '--num_envs=' + os.environ.get('NENV', '512'),
            '--load_run=' + os.environ.get('RUN', ''),
            '--checkpoint=' + os.environ.get('CKPT', '13000')]
import isaacgym  # noqa
import math
import legged_gym.envs  # noqa
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque import GO2OmniJumpLandingTorque as L
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play_landing import train_step_count_at_iter
from isaacgym.torch_utils import get_euler_xyz
import torch

N = int(os.environ.get('NENV', '512')); STEPS = int(os.environ.get('STEPS', '1700'))
HEIGHT = 0.45
BINS = [float(x) for x in os.environ.get('BINS', '0.85').split(',')]
SETTLE = int(os.environ.get('SETTLE', '20'))     # control steps to ignore after a reset
args = get_args(); env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = N; env_cfg.commands.jump_command_range = [1.0, 1.0]
env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]; train_cfg.runner.resume = True
if int(os.environ.get('DEADZONE', '0')):
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.noise.add_noise = False
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
STAND_Z = float(env.cfg.init_state.pos[2])

def wrap(a): return torch.atan2(torch.sin(a), torch.cos(a))

BIG = 1e9
zero = torch.zeros(N, device=dev)
lo = torch.full((N, 3), BIG, device=dev); hi = torch.full((N, 3), -BIG, device=dev)
first_rpy = torch.zeros(N, 3, device=dev); have_first = torch.zeros(N, dtype=torch.bool, device=dev)
z_ref = torch.zeros(N, device=dev)
w2 = torch.zeros(N, 3, device=dev); nsamp = zero.clone()
age = zero.clone()
done_load = torch.zeros(N, dtype=torch.bool, device=dev)
A = dict(pp=[], end=[], rms=[], nst=[])
for t in range(STEPS):
    env.step_count = RP
    with torch.no_grad():
        c = ac.comp_forward(obs)
        if c is not None: env.comp_torque = c.detach()
        a = policy(obs.detach())
    obs, _, _, _, _ = env.step(a)
    r, p, y = get_euler_xyz(env.base_quat)
    rpy = torch.stack([wrap(r), wrap(p), wrap(y)], dim=1)
    z = env.root_states[:, 2]
    age = age + 1
    standing = (~done_load) & (~env.has_taken_off) & (age > SETTLE)
    newf = standing & (~have_first)
    if newf.any():
        first_rpy = torch.where(newf.unsqueeze(1), rpy, first_rpy)
        z_ref = torch.where(newf, z, z_ref); have_first = have_first | newf
    rel = wrap(rpy - first_rpy)
    m3 = standing.unsqueeze(1)
    lo = torch.where(m3, torch.minimum(lo, rel), lo)
    hi = torch.where(m3, torch.maximum(hi, rel), hi)
    w2 = torch.where(m3, w2 + env.base_ang_vel[:, :3] ** 2, w2)
    nsamp = torch.where(standing, nsamp + 1, nsamp)
    begin = standing & have_first & (z < z_ref - 0.03) & (nsamp > 10)
    if begin.any():
        for i in begin.nonzero(as_tuple=False).flatten().tolist():
            A['pp'].append([math.degrees(float(hi[i, k] - lo[i, k])) for k in range(3)])
            A['end'].append([math.degrees(float(rel[i, k])) for k in range(3)])
            A['rms'].append([float((w2[i, k] / nsamp[i]) ** 0.5) for k in range(3)])
            A['nst'].append(float(nsamp[i]) * float(env.dt))
        done_load = done_load | begin
    rs = env.reset_buf
    if rs.any():
        done_load = done_load & (~rs); have_first = have_first & (~rs)
        z_ref = torch.where(rs, torch.zeros_like(z_ref), z_ref)
        lo = torch.where(rs.unsqueeze(1), torch.full_like(lo, BIG), lo)
        hi = torch.where(rs.unsqueeze(1), torch.full_like(hi, -BIG), hi)
        w2 = torch.where(rs.unsqueeze(1), torch.zeros_like(w2), w2)
        nsamp = torch.where(rs, torch.zeros_like(nsamp), nsamp)
        age = torch.where(rs, torch.zeros_like(age), age)

def col(key, k): return [v[k] for v in A[key]]
def m(v): return sum(v) / len(v) if v else float('nan')
def mx(v): return max(v) if v else float('nan')
print('RESULT stand-attitude ckpt=%d  n=%d  stand window %.2f s' % (ck, len(A['pp']), m(A['nst'])))
for k, nm in enumerate(('roll ', 'pitch', 'yaw  ')):
    print('  %s  swing(peak-to-peak) mean %5.2f deg  worst %5.2f deg |  end value mean %+5.2f deg  |w| rms %.3f rad/s'
          % (nm, m(col('pp', k)), mx(col('pp', k)), m(col('end', k)), m(col('rms', k))))
