# Pre-jump stance height: is the robot HOLDING a stance before the jump is commanded, or sinking?
# Behavioural test for STAND_ANCHOR (the x3 pre-jump default_pos penalty), which no config record survives for.
import os, sys
sys.argv = ['m', '--task=go2_omnijump_landing_torque', '--headless',
            '--num_envs=' + os.environ.get('NENV', '256'),
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
N=int(os.environ.get('NENV','256')); HEIGHT=float(os.environ.get('HEIGHT','0.45'))
DX=float(os.environ.get('DX','0.6'))
args=get_args(); env_cfg,train_cfg=task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs=N
env_cfg.commands.jump_command_range=[1.0,1.0]
env_cfg.commands.ranges.jump_height=[HEIGHT,HEIGHT]
train_cfg.runner.resume=True
_orig=L._resample_commands
def _p(self, ids):
    _orig(self, ids)
    if len(ids)>0:
        self.commands[ids,0]=DX; self.commands[ids,1]=0.0
        self.commands[ids,3]=HEIGHT; self.commands[ids,4]=1.0
L._resample_commands=_p
log_root=os.path.join(LEGGED_GYM_ROOT_DIR,'logs',train_cfg.runner.experiment_name)
path=get_load_path(log_root,load_run=args.load_run,checkpoint=args.checkpoint)
ck=int(os.path.basename(path).replace('model_','').replace('.pt',''))
env,_=task_registry.make_env(name=args.task,args=args,env_cfg=env_cfg); dev=env.device
runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train_cfg)
policy=runner.get_inference_policy(device=dev); ac=runner.alg.actor_critic
REPLAY=int(round(train_step_count_at_iter(ck,float(env.cfg.growth.warmup_steps),float(env.cfg.growth.x0),
    float(env.start_freq),float(env.max_freq),float(env.dt),int(train_cfg.runner.num_steps_per_env))))
env.reset(); obs=env.get_observations()
# sample base height ONLY while no jump is commanded yet (pre-jump stance window)
buckets={}   # step-since-episode-start -> [heights]
for t in range(400):
    env.step_count=REPLAY
    with torch.no_grad():
        c=ac.comp_forward(obs)
        if c is not None: env.comp_torque=c.detach()
        a=policy(obs.detach())
    obs,_,_,_,_=env.step(a)
    pre=(~env.jumping_state)
    if pre.any():
        for s in (10,30,50,80,110,140,170,200,250,300,350,395):
            if t==s:
                h=env.root_states[pre,2]-env.env_origins[pre,2]
                buckets[s]=(float(h.mean()), float(h.min()), float(h.max()), int(pre.sum()))
print('PREJUMP %s ckpt=%d  (base height above ground while NO jump commanded)'%(os.environ.get('TAG',''),ck))
print('  step   mean    min     max    n_env_still_prejump')
for s in sorted(buckets):
    m,lo,hi,n=buckets[s]
    print('  %4d  %.3f  %.3f  %.3f   %d'%(s,m,lo,hi,n))
