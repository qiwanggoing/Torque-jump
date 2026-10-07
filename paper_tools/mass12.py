"""Payload robustness: every environment carries its own trunk payload, all at one command."""
import os, sys
sys.argv = ["m", "--task=go2_omnijump_landing_torque", "--headless",
            "--num_envs=" + os.environ.get("NENV", "512"),
            "--load_run=" + os.environ.get("RUN", ""), "--checkpoint=" + os.environ.get("CKPT", "13000")]
import isaacgym  # noqa
import numpy as np
import legged_gym.envs  # noqa
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque import GO2OmniJumpLandingTorque as L
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play_landing import train_step_count_at_iter
import torch
N = int(os.environ.get("NENV", "512")); STEPS = int(os.environ.get("STEPS", "1700"))
DX = float(os.environ.get("DX", "0.70")); HEIGHT = 0.45; TOL = 0.10
MINPEAK = float(os.environ.get("MINPEAK", "0.30"))
MASSES = [float(x) for x in os.environ.get("MASSES", "0,0.5,1,1.5,2,3,5").split(",")]
JITTER = bool(int(os.environ.get("JITTER", "1")))
args = get_args(); env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = N; env_cfg.commands.jump_command_range = [1.0, 1.0]
env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
env_cfg.domain_rand.randomize_base_mass = True; train_cfg.runner.resume = True
_plan = [MASSES[i % len(MASSES)] for i in range(N)]
_orig_props = L._process_rigid_body_props
def _props(self, props, env_id):
    m = _plan[env_id % N]
    self.cfg.domain_rand.added_mass_range = [m, m]
    props = _orig_props(self, props, env_id)
    for i in range(len(props)):
        props[i].mass -= m / 16.0
    if JITTER:
        for i in range(len(props)):
            props[i].mass += np.random.uniform(-1.0 / 16.0, 1.0 / 16.0)
    return props
L._process_rigid_body_props = _props
_orig_cmd = L._resample_commands
def _cmd(self, env_ids):
    _orig_cmd(self, env_ids)
    if len(env_ids) > 0:
        self.commands[env_ids, 0] = DX; self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 3] = HEIGHT; self.commands[env_ids, 4] = 1.0
L._resample_commands = _cmd
log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
ck = int(os.path.basename(path).replace('model_', '').replace('.pt', ''))
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg); dev = env.device
runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
policy = runner.get_inference_policy(device=dev); ac = runner.alg.actor_critic
REPLAY = int(round(train_step_count_at_iter(ck, float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
    float(env.start_freq), float(env.max_freq), float(env.dt), int(train_cfg.runner.num_steps_per_env))))
got = torch.tensor(_plan[:N], device=dev)
env.reset(); obs = env.get_observations(); spawn = env.env_origins[:, :2].clone()
R = {m: dict(n=0, hit=0, tot=[], err=[]) for m in MASSES}
for t in range(STEPS):
    env.step_count = REPLAY
    with torch.no_grad():
        comp = ac.comp_forward(obs)
        if comp is not None: env.comp_torque = comp.detach()
        a = policy(obs.detach())
    obs, _, _, _, _ = env.step(a)
    jl = getattr(env, "just_landed", None)
    if jl is not None and jl.any():
        for i in jl.nonzero(as_tuple=False).flatten().tolist():
            key = min(MASSES, key=lambda x: abs(x - float(got[i])))
            err = float(torch.norm(env.root_states[i, :2] - env.landing_target[i, :2]))
            pk = float(env.peak_base_height[i]); R[key]['n'] += 1
            R[key]['hit'] += int(err <= TOL and pk >= MINPEAK)
            R[key]['tot'].append(float(env.root_states[i, 0] - spawn[i, 0])); R[key]['err'].append(err)
def m_(v): return sum(v) / len(v) if v else float('nan')
print("\n===== payload sweep | ckpt=%d | dx=%.2f | link_jitter=%s =====" % (ck, DX, JITTER))
print("  payload(kg)   n   hit_rate   total_dist   land_err")
for m in MASSES:
    d = R[m]
    if d['n'] == 0: continue
    print("  %6.1f      %3d     %.2f       %.3f m      %.3f" % (m, d['n'], d['hit'] / d['n'], m_(d['tot']), m_(d['err'])))
