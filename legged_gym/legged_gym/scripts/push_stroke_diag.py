"""Is the PUSH cut short by early leg-tuck? Measures, per jump, how long the feet actually apply
propulsive force and whether the legs are still EXTENDING or already TUCKING at takeoff.

Per jump (latched at takeoff = has_taken_off rising edge):
  push_steps        : steps in the push window (jumping & ~taken_off & squat_qualified)
  contact_push_steps: of those, steps with total foot GRF > floor (a REAL propulsive push)
  calf_ext_used     : |calf_pos(takeoff) - calf_pos(push_start)|  (rad of knee extension traveled)
  calf_vel@takeoff  : signed knee speed at feet-off, projected onto the extension direction
                      (+ = still EXTENDING/pushing when it left the ground = used the wall;
                       - = already FLEXING/tucking before feet-off = push cut short by tuck)
  calf_v/vlim@takeoff: |calf speed|/vlim at feet-off (1.0 = at the velocity wall)
Reports population means + the fraction of jumps whose calves are already TUCKING at takeoff. Run:
  TQ_DX=0.8 python legged_gym/scripts/push_stroke_diag.py --task=go2_omnijump_landing_torque --load_run=RUN --checkpoint=N --headless
"""
import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

DX = float(os.environ.get("TQ_DX", "0.8"))
NUM_ENVS = 256
STEPS = 1500
CALF = [2, 5, 8, 11]


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.env.episode_length_s = 3
    env_cfg.commands.resampling_time = 20.0
    env_cfg.commands.landing_dx_curriculum = False
    env_cfg.commands.jump_command_range = [1.0, 1.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.rewards.landing_tilt_terminate = 0.0
    env_cfg.rewards.rsi_prob = 0.0
    train_cfg.runner.resume = True

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    _load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    _checkpoint = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    _resume_path = get_load_path(log_root, load_run=_load_run, checkpoint=_checkpoint)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic
    REPLAY_STEP = 200000
    floor = float(getattr(env.cfg.rewards, "four_leg_push_force_floor", 200.0))
    dt = float(env.dt)
    dev = env.device
    vlim_calf = env.dof_vel_limits[CALF].float()   # [4]

    env.command_ranges["lin_vel_x"] = [DX, DX]
    env.reset()
    obs = env.get_observations()

    prev_push = torch.zeros(NUM_ENVS, dtype=torch.bool, device=dev)
    prev_takeoff = torch.zeros(NUM_ENVS, dtype=torch.bool, device=dev)
    push_cnt = torch.zeros(NUM_ENVS, device=dev)
    contact_cnt = torch.zeros(NUM_ENVS, device=dev)
    vmax = torch.full((NUM_ENVS, 4), -1e9, device=dev)       # max raw calf_vel during push (per calf)
    vmin = torch.full((NUM_ENVS, 4), 1e9, device=dev)        # min raw calf_vel during push
    calf_at_start = torch.zeros(NUM_ENVS, 4, device=dev)      # calf pos latched at push start
    # extension direction per calf: sign of (pos moving away from squat). Determined online as sign of mean
    # calf velocity while pushing (whichever way the knee travels during push = "extension").
    ext_dir = torch.zeros(4, device=dev)
    ext_dir_n = 0

    N = 0
    s_push = s_contact = s_ext_used = s_vel_sign = s_vvlim = 0.0
    s_extpeak = s_tuckpeak = 0.0
    s_calf_squat = s_calf_takeoff = 0.0
    tuck_frac = 0
    lat_vmax = []; lat_vmin = []   # per-jump latched vmax/vmin (list of (k,4) tensors)
    for _ in range(STEPS):
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        with torch.no_grad():
            comp = actor_critic.comp_forward(obs)
            if comp is not None:
                env.comp_torque = comp.detach()
            actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

        try:
            squat = env._squat_deep_enough()
        except Exception:
            squat = torch.ones(NUM_ENVS, dtype=torch.bool, device=dev)
        push = env.jumping_state & (~env.has_taken_off) & squat
        fz = torch.clamp(env.contact_forces[:, env.feet_indices, 2], min=0.0)
        total = fz.sum(dim=1)
        calf_pos = env.dof_pos[:, CALF]
        calf_vel = env.dof_vel[:, CALF]

        # learn extension direction (sign of calf velocity during push, averaged)
        if torch.any(push):
            ext_dir += (torch.sign(calf_vel[push]).sum(dim=0))
            ext_dir_n += int(push.sum().item())

        start = push & (~prev_push)                      # push rising edge
        if torch.any(start):
            push_cnt[start] = 0; contact_cnt[start] = 0
            calf_at_start[start] = calf_pos[start]
            vmax[start] = -1e9; vmin[start] = 1e9
        push_cnt = torch.where(push, push_cnt + 1.0, push_cnt)
        contact_cnt = torch.where(push & (total > floor), contact_cnt + 1.0, contact_cnt)
        pm = push.unsqueeze(1)
        vmax = torch.where(pm, torch.maximum(vmax, calf_vel), vmax)
        vmin = torch.where(pm, torch.minimum(vmin, calf_vel), vmin)

        takeoff = env.has_taken_off & (~prev_takeoff)    # takeoff rising edge
        if torch.any(takeoff):
            ed = torch.sign(ext_dir) if ext_dir_n > 0 else torch.ones(4, device=dev)
            k = int(takeoff.sum().item()); N += k
            s_push += float(push_cnt[takeoff].sum().item())
            s_contact += float(contact_cnt[takeoff].sum().item())
            ext_used = (calf_pos[takeoff] - calf_at_start[takeoff]) * ed          # (k,4) signed extension traveled
            s_ext_used += float(ext_used.mean(dim=1).sum().item())
            velsign = (calf_vel[takeoff] * ed)                                    # (k,4) + = extending, - = tucking
            s_vel_sign += float(velsign.mean(dim=1).sum().item())
            s_vvlim += float((calf_vel[takeoff].abs() / vlim_calf).mean(dim=1).sum().item())
            tuck_frac += int((velsign.mean(dim=1) < 0).sum().item())             # jumps already tucking at takeoff
            lat_vmax.append(vmax[takeoff].clone()); lat_vmin.append(vmin[takeoff].clone())
            s_calf_squat += float(calf_at_start[takeoff].mean(dim=1).sum().item())
            s_calf_takeoff += float(calf_pos[takeoff].mean(dim=1).sum().item())

        prev_push = push.clone()
        prev_takeoff = env.has_taken_off.clone()

    print(f"\n[push_stroke] {os.path.basename(_resume_path)} | cmd dx={DX} | n_jumps={N} | dt={dt*1000:.1f}ms", flush=True)
    if N == 0:
        print("  no jumps", flush=True); return
    print(f"  push window          = {s_push/N:.1f} steps = {s_push/N*dt*1000:.0f} ms", flush=True)
    print(f"  CONTACT push (GRF>{floor:.0f}N) = {s_contact/N:.1f} steps = {s_contact/N*dt*1000:.0f} ms   <-- real force window", flush=True)
    print(f"  contact/push ratio   = {s_contact/max(1e-6,s_push):.2f}", flush=True)
    print(f"  calf @ squat-bottom  = {s_calf_squat/N:+.2f} rad  (start of push; more negative = deeper knee fold)", flush=True)
    print(f"  calf @ takeoff       = {s_calf_takeoff/N:+.2f} rad  (feet-off knee angle)", flush=True)
    print(f"  calf extension used  = {s_ext_used/N:+.2f} rad  (knee angle traveled during push)", flush=True)
    print(f"  calf vel @ takeoff   = {s_vel_sign/N:+.2f} rad/s (signed: + still EXTENDING, - already TUCKING)", flush=True)
    print(f"  calf |v|/vlim @ takeoff = {s_vvlim/N:.2f}  (1.0 = leaves ground AT the speed wall)", flush=True)
    print(f"  jumps TUCKING at takeoff = {tuck_frac/N:.2f}  (knee flexing before feet-off = push cut short)", flush=True)
    # peak EXTENSION vs peak TUCK velocity during the push, using the settled extension direction
    ed = torch.sign(ext_dir) if ext_dir_n > 0 else torch.ones(4, device=dev)
    VM = torch.cat(lat_vmax, dim=0); VN = torch.cat(lat_vmin, dim=0)   # (N,4)
    valid = (VM > -1e8).all(dim=1) & (VN < 1e8).all(dim=1)             # drop jumps whose push had no recorded step
    VM = VM[valid]; VN = VN[valid]
    ext_peak = torch.where(ed > 0, VM, -VN)     # fastest EXTENSION-direction speed reached in push
    tuck_peak = torch.where(ed > 0, -VN, VM)    # fastest TUCK-direction speed reached in push
    ext_peak_vv = (ext_peak / vlim_calf).mean(dim=1)
    tuck_peak_vv = (tuck_peak / vlim_calf).mean(dim=1)
    print(f"  --- peak EXTENSION |v|/vlim during push = {float(ext_peak_vv.mean()):.2f}  "
          f"(does the knee hit the wall while EXTENDING?)", flush=True)
    print(f"  --- peak TUCK      |v|/vlim during push = {float(tuck_peak_vv.mean()):.2f}  "
          f"(or only while TUCKING/retracting?)", flush=True)


if __name__ == '__main__':
    main()
