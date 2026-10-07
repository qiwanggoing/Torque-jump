"""Capture a phase filmstrip of ONE jump (side camera that follows the base).

Mirrors play_landing.py's env setup exactly (same command flow, same PD replay,
architecture read from the checkpoint), adds an Isaac Gym camera sensor and writes
one PNG per control step plus a CSV of the per-step state, so the six phase frames
can be picked afterwards from the recorded state rather than by eye.

Run (needs a display; graphics_device_id is -1 when headless):
  PD_FLOOR=0.2 STANCE_CURR=0 python phase_shots.py --task=go2_omnijump_landing_torque \
      --load_run=Sep22_04-02-15_inertiafix_s1 --checkpoint=13000
"""
import isaacgym
from isaacgym import gymapi
import torch
import os
import math
import csv

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR

OUT = os.environ.get("SHOT_DIR", "/tmp/frames")
DX = float(os.environ.get("PLAY_LANDING_DX", "0.80"))
DY = 0.0
HEIGHT = float(os.environ.get("PLAY_LANDING_HEIGHT", "0.50"))
CAM_W = int(os.environ.get("SHOT_W", "900"))
CAM_H = int(os.environ.get("SHOT_H", "600"))


def train_step_count_at_iter(target_iter, W, X, sf, mf, dt, nspe):
    r0 = nspe / (dt * sf)
    iw = W / r0
    if target_iter <= iw:
        return r0 * target_iter
    Delta = X - W
    a = (mf - sf) / (2.0 * Delta)
    b = sf
    iter_fade_end = iw + (dt / nspe) * (a * Delta * Delta + b * Delta)
    if target_iter <= iter_fade_end:
        c = (nspe / dt) * (target_iter - iw)
        u = (-b + math.sqrt(b * b + 4.0 * a * c)) / (2.0 * a)
        return W + u
    r_end = nspe / (dt * mf)
    return X + r_end * (target_iter - iter_fade_end)


def main():
    os.makedirs(OUT, exist_ok=True)
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 6
    env_cfg.commands.resampling_time = 20.0
    env_cfg.commands.landing_dx_curriculum = False
    env_cfg.commands.landing_disp_x_stage2 = [DX, DX]
    env_cfg.commands.landing_disp_y_stage2 = [DY, DY]
    env_cfg.commands.ranges.jump_height = [HEIGHT, HEIGHT]
    env_cfg.commands.ranges.jump_command = [1.0, 1.0]
    env_cfg.commands.jump_command_range = [1.0, 1.0]
    # CLEAN=1 strips noise/DR (play_landing's "isolate the policy" view). CLEAN=0 keeps the
    # config exactly as the validated batch eval (meas3col) runs it -- that is the condition the
    # reported tracking numbers come from, and the single-env jump only fires reliably there.
    if int(os.environ.get("SHOT_CLEAN", "0")):
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_base_mass = False
    else:
        env_cfg.domain_rand.push_robots = False   # a random shove mid-jump would be a bad figure
    env_cfg.rewards.landing_tilt_terminate = 0.0
    train_cfg.runner.resume = True

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    _load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    _checkpoint = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    _resume_path = get_load_path(log_root, load_run=_load_run, checkpoint=_checkpoint)
    ckpt_iter = int(os.path.basename(_resume_path).replace('model_', '').replace('.pt', ''))

    # architecture straight out of the checkpoint (same as play_landing)
    _sd = torch.load(_resume_path, map_location='cpu')['model_state_dict']

    def _dims(prefix):
        idx = sorted({int(k.split('.')[1]) for k in _sd if k.startswith(prefix + '.') and k.endswith('.weight')})
        return [int(_sd['%s.%d.weight' % (prefix, i)].shape[0]) for i in idx[:-1]], \
               int(_sd['%s.%d.weight' % (prefix, idx[0])].shape[1])
    _a_dims, _ = _dims('actor')
    _c_dims, _c_in = _dims('critic')
    train_cfg.policy.actor_hidden_dims = _a_dims
    train_cfg.policy.critic_hidden_dims = _c_dims
    _per = int(getattr(env_cfg.env, 'single_num_privileged_obs', 0))
    if _per > 0 and _c_in % _per == 0:
        env_cfg.env.c_frame_stack = _c_in // _per
        env_cfg.env.num_privileged_obs = env_cfg.env.c_frame_stack * _per

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic

    REPLAY_STEP = int(round(train_step_count_at_iter(
        ckpt_iter,
        float(env.cfg.growth.warmup_steps), float(env.cfg.growth.x0),
        float(env.start_freq), float(env.max_freq),
        float(env.dt), int(train_cfg.runner.num_steps_per_env))))
    _gs = min(1.0, max(0.0, (REPLAY_STEP - float(env.cfg.growth.warmup_steps))
                       / max(1.0, float(env.cfg.growth.x0) - float(env.cfg.growth.warmup_steps))))
    _floor = max(0.0, min(1.0, float(getattr(env.cfg.growth, "pd_fade_floor", 0.0))))
    _pd_a = float(env.cfg.control.pd_prior_weight) * max(_floor, 1.0 - _gs)
    print(f"[shots] {os.path.basename(_resume_path)} iter {ckpt_iter} pd_alpha={_pd_a:.3f} dx={DX} h={HEIGHT}", flush=True)

    env.reset()
    env.commands[:, 0] = DX
    env.commands[:, 1] = DY
    env.landing_target[:, 0] = env.root_states[:, 0] + DX
    env.landing_target[:, 1] = env.root_states[:, 1] + DY
    env.compute_observations()
    obs = env.get_observations()

    # side camera that follows the base
    cp = gymapi.CameraProperties()
    cp.width, cp.height = CAM_W, CAM_H
    cp.horizontal_fov = 42.0
    cam = env.gym.create_camera_sensor(env.envs[0], cp)

    def place_camera():
        x = float(env.root_states[0, 0])
        y = float(env.root_states[0, 1])
        env.gym.set_camera_location(
            cam, env.envs[0],
            gymapi.Vec3(x + 0.10, y - 2.30, 0.55),
            gymapi.Vec3(x + 0.10, y, 0.28))

    rows = []
    ep = 0
    ep_took_off = False
    ep_landed_at = None
    N_EP = int(os.environ.get("SHOT_EPISODES", "8"))
    z0 = float(env.root_states[0, 2])
    for k in range(20000):
        env.step_count = REPLAY_STEP
        env.common_step_counter = REPLAY_STEP
        with torch.no_grad():
            actions = policy(obs.detach())
            comp = actor_critic.comp_forward(obs.detach())
            if comp is not None:
                env.comp_torque = comp
        obs, _, _, dones, infos = env.step(actions.detach())

        t = float(env.gym.get_sim_time(env.sim))
        cs = env._get_contact_state()[0].int().tolist()
        nfeet = sum(cs)
        z = float(env.root_states[0, 2])
        x = float(env.root_states[0, 0])
        jumping = bool(env.jumping_state[0])
        hto = bool(env.has_taken_off[0])
        hld = bool(env.has_landed[0])
        if hto:
            ep_took_off = True
        if ep_took_off and hld and ep_landed_at is None:
            ep_landed_at = k

        place_camera()
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        fn = os.path.join(OUT, "f%05d.png" % k)
        env.gym.write_camera_image_to_file(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR, fn)
        rows.append(dict(k=k, ep=ep, t=t, x=x, z=z, nfeet=nfeet, jumping=int(jumping),
                         took_off=int(hto), landed=int(hld), done=int(bool(dones[0]))))

        if ep_landed_at is not None and k > ep_landed_at + 60:
            print("[shots] ep %d: clean jump, stopping" % ep, flush=True)
            break
        if bool(dones[0]):
            print("[shots] ep %d ended at step %d (took_off=%d landed=%s)"
                  % (ep, k, int(ep_took_off), ep_landed_at), flush=True)
            ep += 1
            ep_took_off = False
            ep_landed_at = None
            if ep >= N_EP:
                break

    with open(os.path.join(OUT, "state.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[shots] wrote", len(rows), "frames to", OUT, "| z0=%.3f" % z0, flush=True)


if __name__ == '__main__':
    main()
