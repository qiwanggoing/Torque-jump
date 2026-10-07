# Decompose the reported "jump distance" into ground creep + flight, per command bin.
#   creep  = takeoff_x - spawn_x        ground shuffled before the feet leave
#   flight = touchdown_x - takeoff_x    the actual jump
#   total  = touchdown_x - spawn_x      what the thesis calls jump distance
import os, sys
sys.argv=['m','--task=go2_omnijump_landing_torque','--headless',
          '--num_envs='+os.environ.get('NENV','512'),
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
N=int(os.environ.get('NENV','512')); STEPS=int(os.environ.get('STEPS','1700'))
HEIGHT=float(os.environ.get('HEIGHT','0.45'))
BINS=[float(x) for x in os.environ.get('BINS','0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0').split(',')]
TOL=float(os.environ.get('TOL','0.10')); MINPEAK=float(os.environ.get('MINPEAK','0.30'))
args=get_args(); env_cfg,train_cfg=task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs=N
env_cfg.commands.jump_command_range=[1.0,1.0]
env_cfg.commands.ranges.jump_height=[HEIGHT,HEIGHT]
train_cfg.runner.resume=True
_orig=L._resample_commands
def _p(self,ids):
    _orig(self,ids)
    if len(ids)>0 and hasattr(self,'_fixed_dx'):
        self.commands[ids,0]=self._fixed_dx[ids]; self.commands[ids,1]=0.0
    if len(ids)>0:
        self.commands[ids,3]=HEIGHT; self.commands[ids,4]=1.0
L._resample_commands=_p
log_root=os.path.join(LEGGED_GYM_ROOT_DIR,'logs',train_cfg.runner.experiment_name)
path=get_load_path(log_root,load_run=args.load_run,checkpoint=args.checkpoint)
ck=int(os.path.basename(path).replace('model_','').replace('.pt',''))
env,_=task_registry.make_env(name=args.task,args=args,env_cfg=env_cfg); dev=env.device
env._fixed_dx=torch.tensor([BINS[i%len(BINS)] for i in range(N)],device=dev)
runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train_cfg)
policy=runner.get_inference_policy(device=dev); ac=runner.alg.actor_critic
REPLAY=int(round(train_step_count_at_iter(ck,float(env.cfg.growth.warmup_steps),float(env.cfg.growth.x0),
    float(env.start_freq),float(env.max_freq),float(env.dt),int(train_cfg.runner.num_steps_per_env))))
env.reset()
env._fixed_dx=torch.tensor([BINS[i%len(BINS)] for i in range(N)],device=dev)
env._resample_commands(torch.arange(N,device=dev))
obs=env.get_observations(); spawn=env.env_origins[:,:2].clone()
R={b:dict(n=0,hit=0,tot=[],fly=[],creep=[]) for b in BINS}
prev_to=torch.zeros(N,dtype=torch.bool,device=dev); to_x=torch.zeros(N,device=dev)
for t in range(STEPS):
    env.step_count=REPLAY
    with torch.no_grad():
        c=ac.comp_forward(obs)
        if c is not None: env.comp_torque=c.detach()
        a=policy(obs.detach())
    obs,_,_,_,_=env.step(a)
    to=env.has_taken_off; new=to&(~prev_to)
    if new.any(): to_x=torch.where(new, env.root_states[:,0], to_x)
    prev_to=to.clone()
    jl=getattr(env,'just_landed',None)
    if jl is not None and jl.any():
        for i in jl.nonzero(as_tuple=False).flatten().tolist():
            k=min(BINS,key=lambda x:abs(x-float(env._fixed_dx[i])))
            err=float(torch.norm(env.root_states[i,:2]-env.landing_target[i,:2])); pk=float(env.peak_base_height[i])
            td=float(env.root_states[i,0])
            R[k]['n']+=1; R[k]['hit']+=int(err<=TOL and pk>=MINPEAK)
            R[k]['tot'].append(td-float(spawn[i,0]))
            R[k]['fly'].append(td-float(to_x[i]))
            R[k]['creep'].append(float(to_x[i])-float(spawn[i,0]))
def m(v): return sum(v)/len(v) if v else float('nan')
def mx(v): return max(v) if v else float('nan')
def p95(v):
    if not v: return float('nan')
    q=sorted(v); return q[min(len(q)-1,int(0.95*len(q)))]
print('REACH %s ckpt=%d'%(os.environ.get('TAG',''),ck))
print('  cmd    n   hit | totMean  totMax totP95 | flyMean  flyMax | creep%%')
A=dict(n=0,hit=0,tot=[],fly=[],creep=[])
for b in BINS:
    d=R[b]
    if d['n']==0: continue
    tt,cc,ff=m(d['tot']),m(d['creep']),m(d['fly'])
    print('  %.2f %4d  %.2f | %7.3f %7.3f %6.3f | %7.3f %7.3f | %5.1f%%'%(
        b,d['n'],d['hit']/d['n'],tt,mx(d['tot']),p95(d['tot']),ff,mx(d['fly']),100*cc/max(tt,1e-6)))
    for k in A:
        if k in ('n','hit'): A[k]+=d[k]
        else: A[k]+=d[k]
tt,cc,ff=m(A['tot']),m(A['creep']),m(A['fly'])
print('  ALL  %4d  %.2f | %7.3f %7.3f %6.3f | %7.3f %7.3f | %5.1f%%'%(
    A['n'],A['hit']/max(A['n'],1),tt,mx(A['tot']),p95(A['tot']),ff,mx(A['fly']),100*cc/max(tt,1e-6)))
