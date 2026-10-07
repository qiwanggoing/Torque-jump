"""Where does the pre-take-off travel come from: standing drift, or the push itself?

Splits the spawn -> take-off displacement at the moment the load phase begins (the first
step where the base has dropped below the standing height by more than 3 cm), so the
"stand" part is what the robot gives away before it starts the jump at all.
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
BINS = [float(x) for x in os.environ.get('BINS', '0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0').split(',')]
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

spawn = env.env_origins[:, :2] + torch.tensor(
    [float(env.cfg.init_state.pos[0]), float(env.cfg.init_state.pos[1])], device=dev)
STAND_Z = float(env.cfg.init_state.pos[2])

zero = torch.zeros(N, device=dev)
load_x = zero.clone(); load_y = zero.clone(); load_yaw = zero.clone()   # state when the load phase starts
seen_load = torch.zeros(N, dtype=torch.bool, device=dev)
max_stand = zero.clone()                                               # worst excursion while standing
prev_to = torch.zeros(N, dtype=torch.bool, device=dev)
yaw0 = zero.clone()
first = True
A = dict(sx=[], sy=[], syaw=[], smax=[], px=[], py=[], tot=[],
         vx=[], vz=[], pitch=[], wy=[], fthigh=[], fcalf=[], rthigh=[], rcalf=[], hip=[], z=[])
for t in range(STEPS):
    env.step_count = RP
    with torch.no_grad():
        c = ac.comp_forward(obs)
        if c is not None: env.comp_torque = c.detach()
        a = policy(obs.detach())
    obs, _, _, _, _ = env.step(a)
    z = env.root_states[:, 2]
    yaw = get_euler_xyz(env.base_quat)[2]
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    if first:
        yaw0 = yaw.clone(); first = False
    standing = (~seen_load) & (~env.has_taken_off)
    d = (env.root_states[:, :2] - spawn).norm(dim=1)
    max_stand = torch.where(standing, torch.maximum(max_stand, d), max_stand)
    # load phase begins: base has dropped 3 cm below the standing height while still on the ground
    begin = standing & (z < STAND_Z - 0.03)
    if begin.any():
        load_x = torch.where(begin, env.root_states[:, 0] - spawn[:, 0], load_x)
        load_y = torch.where(begin, env.root_states[:, 1] - spawn[:, 1], load_y)
        load_yaw = torch.where(begin, yaw - yaw0, load_yaw)
        seen_load = seen_load | begin
    rise = env.has_taken_off & (~prev_to); prev_to = env.has_taken_off.clone()
    if rise.any():
        for i in rise.nonzero(as_tuple=False).flatten().tolist():
            if not bool(seen_load[i]):
                continue
            A['sx'].append(float(load_x[i])); A['sy'].append(float(load_y[i]))
            A['syaw'].append(math.degrees(float(load_yaw[i]))); A['smax'].append(float(max_stand[i]))
            A['px'].append(float(env.root_states[i, 0] - spawn[i, 0]) - float(load_x[i]))
            A['py'].append(float(env.root_states[i, 1] - spawn[i, 1]) - float(load_y[i]))
            A['tot'].append(float(env.root_states[i, 0] - spawn[i, 0]))
            q = env.dof_pos[i]
            A['vx'].append(float(env.root_states[i, 7])); A['vz'].append(float(env.root_states[i, 9]))
            _p = get_euler_xyz(env.base_quat)[1][i]
            _p = math.atan2(math.sin(float(_p)), math.cos(float(_p)))
            A['pitch'].append(_p); A['wy'].append(float(env.root_states[i, 11]))
            A['z'].append(float(env.root_states[i, 2]))
            A['hip'].append(float(q[[0,3,6,9]].abs().mean()))
            A['fthigh'].append(float(q[[1,4]].mean())); A['fcalf'].append(float(q[[2,5]].mean()))
            A['rthigh'].append(float(q[[7,10]].mean())); A['rcalf'].append(float(q[[8,11]].mean()))
        seen_load = torch.where(rise, torch.zeros_like(seen_load), seen_load)
        max_stand = torch.where(rise, torch.zeros_like(max_stand), max_stand)
    done = env.reset_buf
    if done.any():
        seen_load = seen_load & (~done); max_stand = torch.where(done, torch.zeros_like(max_stand), max_stand)

def m(v): return sum(v) / len(v) if v else float('nan')
def mx(v): return max(v) if v else float('nan')
def ab(v): return sum(abs(x) for x in v) / len(v) if v else float('nan')
print('RESULT stand-drift ckpt=%d  n=%d' % (ck, len(A['sx'])))
print('  while STANDING (spawn -> start of the load phase):')
print('    forward  %+.4f m   lateral |dy| %.4f m   yaw |dpsi| %.2f deg' % (m(A['sx']), ab(A['sy']), ab(A['syaw'])))
print('    worst excursion from spawn %.4f m (max over jumps %.4f m)' % (m(A['smax']), mx(A['smax'])))
print('  during LOAD+PUSH (start of the load phase -> take-off):')
print('    forward  %+.4f m   lateral |dy| %.4f m' % (m(A['px']), ab(A['py'])))
print('  spawn -> take-off total forward %.4f m  => standing share %.0f%%'
      % (m(A['tot']), 100.0 * m(A['sx']) / max(m(A['tot']), 1e-6)))
print('  TAKE-OFF state (for the MuJoCo comparison):')
print('    z=%.3f  vx=%.3f vz=%.3f  |v|=%.3f angle=%.1f deg  pitch=%+.3f wy=%+.2f'
      % (m(A['z']), m(A['vx']), m(A['vz']), (m(A['vx'])**2 + m(A['vz'])**2) ** 0.5,
         math.degrees(math.atan2(m(A['vz']), m(A['vx']))), m(A['pitch']), m(A['wy'])))
print('    |hip|=%.3f  F thigh %.2f / calf %.2f   R thigh %.2f / calf %.2f'
      % (m(A['hip']), m(A['fthigh']), m(A['fcalf']), m(A['rthigh']), m(A['rcalf'])))
