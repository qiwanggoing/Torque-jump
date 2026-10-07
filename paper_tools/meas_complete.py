# Non-terminal landing completion, with the DENOMINATOR the plan asks for.
#
# The hit rates elsewhere in this project use "recorded touchdown events" as the denominator,
# which silently drops every episode that never took off or terminated before landing. This
# script reports three counts per command bin instead:
#   n_ep        episodes that ran to completion inside the measurement window
#   n_td        of those, episodes with a VALID touchdown (peak >= MINPEAK)
#   n_complete  of those, episodes that then survived the post-touchdown window
#               (landing_buffer_steps) WITHOUT a terminal (non-timeout) reset
# and both rates:  n_complete/n_ep (unconditional) and n_complete/n_td (conditional).
# The strict settling check is kept alongside as a diagnostic, not as the success definition.
import os, sys
sys.argv = ['m', '--task=go2_omnijump_landing_torque', '--headless',
            '--num_envs=' + os.environ.get('NENV', '512'),
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
N = int(os.environ.get('NENV', '512')); STEPS = int(os.environ.get('STEPS', '2500'))
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
BUF = int(getattr(env.cfg.rewards, 'landing_buffer_steps', 150))
env.reset()
env._fixed_dx = torch.tensor([BINS[i % len(BINS)] for i in range(N)], device=dev)
env._resample_commands(torch.arange(N, device=dev))
obs = env.get_observations()
R = {b: dict(ep=0, td=0, comp=0, settle=0, err=[], dist=[]) for b in BINS}
spawn = env.env_origins[:, :2].clone()
# per-env episode bookkeeping
had_td   = torch.zeros(N, dtype=torch.bool, device=dev)
td_age   = torch.zeros(N, dtype=torch.long, device=dev)   # steps since touchdown
completed= torch.zeros(N, dtype=torch.bool, device=dev)
settled  = torch.zeros(N, dtype=torch.bool, device=dev)
calm_run = torch.zeros(N, device=dev)
ep_err   = torch.zeros(N, device=dev); ep_dist = torch.zeros(N, device=dev)
warm = 0   # ignore the first episode per env (starts mid-window)
first_done = torch.zeros(N, dtype=torch.bool, device=dev)
def flush(idx):
    for i in idx.tolist():
        if not first_done[i]:      # skip the partial first episode
            first_done[i] = True
            continue
        key = min(BINS, key=lambda x: abs(x - float(env._fixed_dx[i])))
        d = R[key]; d['ep'] += 1
        if had_td[i]:
            d['td'] += 1; d['err'].append(float(ep_err[i])); d['dist'].append(float(ep_dist[i]))
            if completed[i]: d['comp'] += 1
            if settled[i]:   d['settle'] += 1
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
        ok = idx[(env.peak_base_height[idx] >= MINPEAK)]
        if len(ok):
            had_td[ok] = True; td_age[ok] = 0; calm_run[ok] = 0.0
            ep_err[ok] = torch.norm(env.root_states[ok][:, :2] - env.landing_target[ok][:, :2], dim=1)
            ep_dist[ok] = env.root_states[ok][:, 0] - spawn[ok, 0]
    # age the post-touchdown window; completion = BUF steps survived
    td_age = torch.where(had_td & (~completed), td_age + 1, td_age)
    completed = completed | (had_td & (td_age >= BUF))
    # strict settling diagnostic (kept for comparison with the old criterion)
    all4 = torch.all(env.contact_forces[:, env.feet_indices, 2] > 1.0, dim=1)
    calm = all4 & (env.root_states[:, 7:9].norm(dim=1) < 0.2) & (env.root_states[:, 10:13].norm(dim=1) < 0.5)
    calm_run = torch.where(had_td & calm, calm_run + 1, torch.where(had_td, torch.zeros_like(calm_run), calm_run))
    settled = settled | (had_td & (calm_run >= 10))
    # episode boundary: reset_buf is set on BOTH termination and timeout
    rb = env.reset_buf.clone().bool()
    if rb.any():
        idx = rb.nonzero(as_tuple=False).flatten()
        flush(idx)
        had_td[idx] = False; completed[idx] = False; settled[idx] = False
        td_age[idx] = 0; calm_run[idx] = 0.0; ep_err[idx] = 0.0; ep_dist[idx] = 0.0
def m(v): return sum(v)/len(v) if v else float('nan')
print('COMPLETE %s ckpt=%d  buffer=%d steps' % (os.environ.get('TAG',''), ck, BUF))
print('  cmd   n_ep  n_td  n_comp | comp/ep  comp/td  td/ep | strictSettle/ep | err   dist')
T=dict(ep=0,td=0,comp=0,settle=0,err=[],dist=[])
for b in BINS:
    d=R[b]
    if d['ep']==0: continue
    print('  %.2f %5d %5d %6d |  %5.2f   %5.2f   %5.2f |     %5.2f       | %.3f %.3f'
      % (b,d['ep'],d['td'],d['comp'],d['comp']/d['ep'],d['comp']/max(d['td'],1),d['td']/d['ep'],
         d['settle']/d['ep'], m(d['err']), m(d['dist'])))
    for k in T:
        if isinstance(T[k],list): T[k]+=d[k]
        else: T[k]+=d[k]
print('  ALL  %5d %5d %6d |  %5.2f   %5.2f   %5.2f |     %5.2f       | %.3f %.3f'
  % (T['ep'],T['td'],T['comp'],T['comp']/max(T['ep'],1),T['comp']/max(T['td'],1),T['td']/max(T['ep'],1),
     T['settle']/max(T['ep'],1), m(T['err']), m(T['dist'])))
