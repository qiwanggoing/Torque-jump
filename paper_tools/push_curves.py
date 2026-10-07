"""Per-joint torque and speed through the push, as time curves aligned on take-off."""
import os, sys
sys.argv = ["m", "--task=go2_omnijump_landing_torque", "--headless",
            "--num_envs=" + os.environ.get("NENV", "256"),
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
N = int(os.environ.get("NENV", "256")); STEPS = int(os.environ.get("STEPS", "1200"))
K = int(os.environ.get("K", "110")); DX = float(os.environ.get("DX", "0.70")); HEIGHT = 0.45
_orig = L._resample_commands
def _patched(self, env_ids):
    _orig(self, env_ids)
    if len(env_ids) > 0:
        self.commands[env_ids, 0] = DX; self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 3] = HEIGHT; self.commands[env_ids, 4] = 1.0
L._resample_commands = _patched
args = get_args(); env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = N; env_cfg.commands.jump_command_range = [1.0, 1.0]
env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]; train_cfg.runner.resume = True
log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
ck = int(os.path.basename(path).replace('model_', '').replace('.pt', ''))
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg); dev = env.device
runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
policy = runner.get_inference_policy(device=dev); ac = runner.alg.actor_critic
REPLAY = int(round(train_step_count_at_iter(ck, float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
    float(env.start_freq), float(env.max_freq), float(env.dt), int(train_cfg.runner.num_steps_per_env))))
env.reset(); obs = env.get_observations()
vlim, tlim = env.dof_vel_limits.clone(), env.torque_limits.clone()
ringV = torch.zeros(N, K, 12, device=dev); ringT = torch.zeros(N, K, 12, device=dev)
ringG = torch.zeros(N, K, device=dev); filled = torch.zeros(N, device=dev)
accV, accT, accG, nacc = [], [], [], 0
prev_to = torch.zeros(N, dtype=torch.bool, device=dev)
for t in range(STEPS):
    env.step_count = REPLAY
    with torch.no_grad():
        comp = ac.comp_forward(obs)
        if comp is not None: env.comp_torque = comp.detach()
        a = policy(obs.detach())
    obs, _, _, _, _ = env.step(a)
    ringV = torch.roll(ringV, -1, dims=1); ringT = torch.roll(ringT, -1, dims=1); ringG = torch.roll(ringG, -1, dims=1)
    ringV[:, -1] = env.dof_vel.abs() / vlim.unsqueeze(0)
    ringT[:, -1] = env.torques.abs() / tlim.unsqueeze(0)
    ringG[:, -1] = torch.clamp(env.contact_forces[:, env.feet_indices, 2], min=0.0).sum(dim=1)
    filled = torch.clamp(filled + 1, max=K)
    rise = env.has_taken_off & (~prev_to); prev_to = env.has_taken_off.clone()
    ok = rise & (filled >= K)
    if ok.any():
        idx = ok.nonzero(as_tuple=False).flatten()
        accV.append(ringV[idx].cpu().numpy()); accT.append(ringT[idx].cpu().numpy())
        accG.append(ringG[idx].cpu().numpy()); nacc += len(idx)
    done = env.reset_buf.nonzero(as_tuple=False).flatten()
    if len(done) > 0: filled[done] = 0
V = np.concatenate(accV, axis=0); T = np.concatenate(accT, axis=0); G = np.concatenate(accG, axis=0)
print("\n===== push curves | ckpt=%d | dx=%.2f | n_jumps=%d | K=%d steps (%.0f ms) =====" % (ck, DX, nacc, K, K * env.dt * 1000))
np.savez(os.environ.get("OUT", "/tmp/push_curves.npz"), V=V, T=T, G=G, dt=float(env.dt))
groups = {"front hip": [0, 3], "front thigh": [1, 4], "front calf": [2, 5],
          "rear hip": [6, 9], "rear thigh": [7, 10], "rear calf": [8, 11]}
ms = (np.arange(K) - (K - 1)) * float(env.dt) * 1000.0
g = G.mean(axis=0); air = (G < 1.0).mean(axis=0)
print("  peak foot force %.0f N at %.0f ms (%.1f x body weight)" % (g.max(), ms[g.argmax()], g.max() / 147.0))
un = [ms[i] for i in range(K - 3) if air[i] > 0.5]
if un: print("  feet off the ground: %.0f to %.0f ms (%.0f ms)" % (un[0], un[-1], un[-1] - un[0]))
for nm, ids in groups.items():
    v = V[:, :, ids].mean(axis=(0, 2)); q = T[:, :, ids].mean(axis=(0, 2))
    i_t = q.argmax(); i_v = v.argmax()
    print("  %-11s peak tau %.2f @ %4.0f ms | peak |w|/w0 %.2f @ %4.0f ms | above no-load %2.0f%%"
          % (nm, q.max(), ms[i_t], v.max(), ms[i_v], 100 * (v > 1.0).mean()))
