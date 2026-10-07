# Per-phase PD-scaffold contribution for a trained policy.
# Both command paths are stored on the env before activation/T-N limiting:
#     torques_action = residual_torques_action + pd_prior_torques
# so their magnitudes and their ALIGNMENT can be read directly, per jump phase.
#   share   = mean|tau_PD| / (mean|tau_PD| + mean|tau_RL|)
#   proj    = <tau_PD, tau_RL> / <tau_RL, tau_RL>   >0 PD reinforces RL, <0 PD opposes it
#   pdtrack = |q*(phase) - q|  the tracking error that drives the scaffold
import os, sys
sys.argv = ['m','--task=go2_omnijump_landing_torque','--headless',
            '--num_envs='+os.environ.get('NENV','256'),
            '--load_run='+os.environ.get('RUN',''),
            '--checkpoint='+os.environ.get('CKPT','14700')]
import isaacgym  # noqa
import legged_gym.envs  # noqa
from legged_gym.envs.go2.go2_omnijump_landing_torque.go2_omnijump_landing_torque import GO2OmniJumpLandingTorque as L
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play_landing import train_step_count_at_iter
import torch
N=int(os.environ.get('NENV','256')); STEPS=int(os.environ.get('STEPS','1700'))
HEIGHT=float(os.environ.get('HEIGHT','0.45')); DX=float(os.environ.get('DX','0.8'))
args=get_args(); env_cfg,train_cfg=task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs=N
env_cfg.commands.jump_command_range=[1.0,1.0]
env_cfg.commands.ranges.jump_height=[HEIGHT,HEIGHT]
train_cfg.runner.resume=True
_orig=L._resample_commands
def _p(self,ids):
    _orig(self,ids)
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
GRP={'hip':[0,3,6,9],'thigh':[1,4,7,10],'calf':[2,5,8,11]}
acc={}   # phase -> dict
def bump(ph, pd, rl, err, n):
    d=acc.setdefault(ph, dict(n=0.0, pd=0.0, rl=0.0, dot=0.0, rr=0.0, err=0.0,
                              g={k:[0.0,0.0] for k in GRP}))
    d['n']+=n
    d['pd']+=pd.abs().mean(dim=1).sum().item(); d['rl']+=rl.abs().mean(dim=1).sum().item()
    d['dot']+=(pd*rl).sum().item(); d['rr']+=(rl*rl).sum().item()
    d['err']+=err.abs().mean(dim=1).sum().item()
    for k,idx in GRP.items():
        d['g'][k][0]+=pd[:,idx].abs().mean(dim=1).sum().item()
        d['g'][k][1]+=rl[:,idx].abs().mean(dim=1).sum().item()
for t in range(STEPS):
    env.step_count=REPLAY
    with torch.no_grad():
        c=ac.comp_forward(obs)
        if c is not None: env.comp_torque=c.detach()
        a=policy(obs.detach())
    obs,_,_,_,_=env.step(a)
    pd=env.pd_prior_torques.detach(); rl=env.residual_torques_action.detach()
    err=(env.default_joint_pd_target-env.dof_pos).detach()
    jm=env.jumping_state; to=env.has_taken_off
    masks={'stand':(~jm),
           'load':(jm & (~to)),
           'flight':env.airborne,
           'prelanding':getattr(env,'prelanding',torch.zeros_like(jm)),
           'landing':env.landing}
    for ph,m in masks.items():
        if m.any(): bump(ph, pd[m], rl[m], err[m], int(m.sum()))
print('PDSHARE %s ckpt=%d  cmd dx=%.2f'%(os.environ.get('TAG',''),ck,DX))
print('  %-11s %7s %8s %8s %7s %7s | %s'%('phase','samples','|tau_PD|','|tau_RL|','PDshare','proj','PD share by group (hip/thigh/calf)'))
for ph in ['stand','load','flight','prelanding','landing']:
    d=acc.get(ph)
    if not d or d['n']==0: continue
    n=d['n']; pdm=d['pd']/n; rlm=d['rl']/n
    share=pdm/max(pdm+rlm,1e-9); proj=d['dot']/max(d['rr'],1e-9)
    gs=' '.join('%s %.2f'%(k, d['g'][k][0]/max(d['g'][k][0]+d['g'][k][1],1e-9)) for k in GRP)
    print('  %-11s %7d %8.3f %8.3f %7.3f %7.3f | %s'%(ph,int(n),pdm,rlm,share,proj,gs))
print('  (proj >0: PD reinforces the RL torque; <0: PD opposes it)')
