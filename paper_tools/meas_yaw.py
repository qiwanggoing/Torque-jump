# Deterministic eval that ALSO measures the yaw hole:
#   takeoff omega_z (body-frame yaw rate at the takeoff instant)
#   heading change from takeoff to touchdown (the rotation actually accumulated in flight)
#   ground-phase heading change (command -> takeoff), to separate "spun on the ground" from "spun in the air"
# plus the standard hit / distance / error metrics so a regression is visible in the same run.
import os, sys, math
sys.argv = ['m', '--task=go2_omnijump_landing_torque', '--headless',
            '--num_envs=' + os.environ.get('NENV', '512'),
            '--load_run=' + os.environ.get('RUN', ''),
            '--checkpoint=' + os.environ.get('CKPT', '20000')]
import isaacgym  # noqa
import legged_gym.envs  # noqa
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque import GO2OmniJumpLandingTorque as L
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym.utils.math import wrap_to_pi
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play_landing import train_step_count_at_iter
from isaacgym.torch_utils import get_euler_xyz
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

def yaw_now():
    return get_euler_xyz(env.base_quat)[2]

R = {b: dict(n=0, hit=0, tot=[], err=[], dyaw=[], wz=[], gyaw=[]) for b in BINS}
prev_to = torch.zeros(N, dtype=torch.bool, device=dev)
prev_cmd = torch.zeros(N, dtype=torch.bool, device=dev)
yaw_cmd = yaw_now().clone(); yaw_to = yaw_now().clone(); wz_to = torch.zeros(N, device=dev)
njump = 0
for t in range(STEPS):
    env.step_count = REPLAY
    with torch.no_grad():
        c = ac.comp_forward(obs)
        if c is not None: env.comp_torque = c.detach()
        a = policy(obs.detach())
    obs, _, _, _, _ = env.step(a)
    y = yaw_now()
    # jump commanded rising edge -> remember heading at the start of the jump
    cmd = env.jumping_state
    newcmd = cmd & (~prev_cmd)
    if newcmd.any(): yaw_cmd = torch.where(newcmd, y, yaw_cmd)
    prev_cmd = cmd.clone()
    # takeoff rising edge -> remember heading and yaw RATE at the takeoff instant
    to = env.has_taken_off
    newto = to & (~prev_to)
    if newto.any():
        yaw_to = torch.where(newto, y, yaw_to)
        wz_to = torch.where(newto, env.base_ang_vel[:, 2], wz_to)
    prev_to = to.clone()
    jl = getattr(env, 'just_landed', None)
    if jl is not None and jl.any():
        for i in jl.nonzero(as_tuple=False).flatten().tolist():
            key = min(BINS, key=lambda x: abs(x - float(env._fixed_dx[i])))
            err = float(torch.norm(env.root_states[i, :2] - env.landing_target[i, :2]))
            pk = float(env.peak_base_height[i])
            R[key]['n'] += 1; njump += 1
            R[key]['hit'] += int(err <= TOL and pk >= MINPEAK)
            R[key]['tot'].append(float(env.root_states[i, 0] - spawn[i, 0]))
            R[key]['err'].append(err)
            R[key]['dyaw'].append(abs(float(wrap_to_pi(y[i] - yaw_to[i]))) * 180.0 / math.pi)
            R[key]['gyaw'].append(abs(float(wrap_to_pi(yaw_to[i] - yaw_cmd[i]))) * 180.0 / math.pi)
            R[key]['wz'].append(abs(float(wz_to[i])))
def m(v): return sum(v)/len(v) if v else float('nan')
def p95(v):
    if not v: return float('nan')
    s=sorted(v); return s[min(len(s)-1,int(0.95*len(s)))]
print('YAWRESULT %s ckpt=%d' % (os.environ.get('TAG',''), ck))
print('  cmd    n   hit   dist    err  | flightYaw p95  groundYaw | |wz|TO  p95')
A=dict(n=0,hit=0,tot=[],err=[],dyaw=[],wz=[],gyaw=[])
for b in BINS:
    d=R[b]
    if d['n']==0: continue
    print('  %.2f %4d  %.2f  %.3f  %.3f  |  %6.1f %6.1f   %6.1f    |  %.3f %.3f'
          % (b,d['n'],d['hit']/d['n'],m(d['tot']),m(d['err']),m(d['dyaw']),p95(d['dyaw']),m(d['gyaw']),m(d['wz']),p95(d['wz'])))
    for k in A:
        if k in ('n','hit'): A[k]+=d[k]
        else: A[k]+=d[k]
print('  ALL  %4d  %.2f  %.3f  %.3f  |  %6.1f %6.1f   %6.1f    |  %.3f %.3f'
      % (A['n'],A['hit']/max(A['n'],1),m(A['tot']),m(A['err']),m(A['dyaw']),p95(A['dyaw']),m(A['gyaw']),m(A['wz']),p95(A['wz'])))
print('  (flightYaw = |heading change takeoff->touchdown| deg; groundYaw = |cmd->takeoff| deg; |wz|TO = |body yaw rate| at takeoff rad/s)')
