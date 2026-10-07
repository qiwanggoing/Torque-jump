import os, sys, math, importlib.util
import numpy as np
HERE='/home/qiwang/torque_jump2/SATA/deploy_mujoco'; PT='/home/qiwang/torque_jump2/SATA/paper_tools'
spec=importlib.util.spec_from_file_location('ref_traj', os.path.join(HERE,'ref_traj.py'))
rt=importlib.util.module_from_spec(spec); sys.modules['ref_traj']=rt; spec.loader.exec_module(rt)
import mujoco
D=np.load(os.path.join(PT,'ref_traj_14700.npz')); PRE=int(D['pre'][0]); FS=float(D['fs'][0]); dt=1/FS
CMD=os.environ.get('CMD','0.90')
q=D['q_%s'%CMD]; tau_ref=D['tau_%s'%CMD]; base=D['base_%s'%CMD]
jn=['%s_%s'%(l,j) for l in rt.FEET for j in ('hip','thigh','calf')]
print('=== why a KP=40 PD must lag: the reference IS a torque trajectory ===')
pk=np.abs(tau_ref).max(axis=0)
print('  peak |tau_ref| per joint [Nm]      : '+' '.join('%5.1f'%v for v in pk))
print('  implied steady-state lag tau/KP [rad]: '+' '.join('%5.2f'%(v/40.0) for v in pk))
print('  -> a PD can only emit tau = KP*err, so tracking error tau/KP is structural, not a bug.')
print('     max over joints %.2f rad   (measured tracking max was 1.14 rad)' % (pk.max()/40.0))
def run(KP,KD,q_ref):
    rb=rt._load_robot(rt.DEFAULT_XML); model,data=rb.model,rb.data; model.opt.timestep=rt.DT
    data.qpos[:3]=[0,0,0.30]; data.qpos[3:7]=[1,0,0,0]; data.qpos[rb.q_ids]=q_ref[0]; data.qvel[:]=0
    mujoco.mj_forward(model,data)
    def step(target):
        qj=data.qpos[rb.q_ids]; dqj=data.qvel[rb.dq_ids]
        tc=KP*(target-qj)-KD*dqj; t,l=rt.tn_clamp(tc,dqj); rb.apply(t); return tc,l
    for _ in range(int(0.6/rt.DT)): step(q_ref[0])
    z0=data.qpos[2]
    Z=[];X=[];VX=[];VZ=[];C=[];TE=[];SAT=np.zeros(12)
    for k in range(len(q_ref)):
        tc,l=step(q_ref[k]); SAT+=(np.abs(tc)>l+1e-6)
        TE.append(data.qpos[rb.q_ids]-q_ref[k]); Z.append(data.qpos[2]); X.append(data.qpos[0])
        VX.append(data.qvel[0]); VZ.append(data.qvel[2]); C.append(int((rb.foot_fz()>1.0).sum()))
    Z=np.array(Z);X=np.array(X);VX=np.array(VX);VZ=np.array(VZ);C=np.array(C);TE=np.array(TE)
    to=PRE; fly=np.where((np.arange(len(C))>to)&(C==0))[0]; net=float('nan'); tms=float('nan')
    if len(fly):
        # FIRST touchdown after the airborne phase begins (matches the Isaac definition)
        af=np.where((np.arange(len(C))>fly[0])&(C>0))[0]
        if len(af): net=X[af[0]]-X[to]; tms=1000*(af[0]-to)*dt
    return dict(z_stand=z0, v=math.hypot(VX[to],VZ[to]), ang=math.degrees(math.atan2(VZ[to],VX[to])),
                peak=Z.max(), net=net, tms=tms, trms=float(np.sqrt((TE**2).mean())),
                tmax=float(np.abs(TE).max()), sat=float((SAT/len(q_ref)).max()))
print('\n=== KP sweep, cmd %s, reference AS-IS (no clipping) ==='%CMD)
print('  Isaac truth:  stand z 0.297   |v| %.2f   angle %.1f   peak %.3f   net %.3f   flight %.0f ms'
      %(math.hypot(base[PRE,1],base[PRE,3]), math.degrees(math.atan2(base[PRE,3],base[PRE,1])),
        base[:,0].max(), {'0.90':0.809,'0.70':0.683,'0.50':0.482}[CMD],
        1000*dt*int(np.argmax(D['grf_%s'%CMD][PRE:].sum(axis=1)>20.0))))
print('  %5s %5s | %7s %6s %7s %6s %7s %7s | %7s %7s %6s'
      %('KP','KD','stand z','|v|','angle','peak','net','flight','trk rms','trk max','sat'))
for KP in [40,100,200,400,800,1600]:
    KD=1.2*math.sqrt(KP/40.0)
    r=run(KP,KD,q)
    print('  %5d %5.1f | %7.3f %6.2f %6.1f° %6.3f %7.3f %6.0fms | %7.3f %7.3f %5.0f%%'
          %(KP,KD,r['z_stand'],r['v'],r['ang'],r['peak'],r['net'],r['tms'],r['trms'],r['tmax'],100*r['sat']))
