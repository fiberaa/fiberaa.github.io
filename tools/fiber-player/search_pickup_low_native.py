"""Batched real-policy, real-physics seed search from resting arms; no pose overrides."""
from pathlib import Path
import json,time,sys,numpy as np,torch,mujoco
from collections import deque
from lsvla import load_bfm
from lsvla.knot_planner_runtime import load_knot_planner_checkpoint,normalize_planner_state,bfm_backward_anchor,bfm_action_from_latent,RTCLatentScheduler
from lsvla.sim.mujoco_sim import BFMMuJoCoSim,KEYPOINT_NAMES
from lsvla.sim.pickup_mujoco_env import resize_depth_to_mm
root=Path.home()/'fiber-web-export';torch.set_num_threads(8);device='cuda';bfm=load_bfm(str(root/'models/model_best_emd.pt'),device=device,control_profile='bfm05_xl',verbose=False);bundle=load_knot_planner_checkpoint(root/'models/planner.pt',device=device);meta=json.loads((root/'browser/v27/planner_meta.json').read_text());am=json.loads((root/'browser/actor_meta.json').read_text());base=json.loads((root/'browser/scenes/pickup/layouts.json').read_text())['layouts']['support'];m=mujoco.MjModel.from_xml_path(str(root/'browser/scenes/pickup'/base['xml']));renderer=mujoco.Renderer(m,height=240,width=426);renderer.enable_depth_rendering();n=16
object_index=1;firstseed=int(sys.argv[2]) if len(sys.argv)>2 else 0;cfg=base.copy();object=cfg['boxes'][object_index];cfg.update(box_body=object['body'],box_qadr=object['qadr'],object_geom=object['geom']);qa=cfg['qadr'];va=cfg['vadr'];handgeoms=set(i for i in range(m.ngeom) if any(x in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '') for x in ['_hand_','_wrist_']));normal=np.zeros(6);started=time.time()
for startseed in range(firstseed,firstseed+16,n):
 states=[];histories=[];last=np.zeros((n,29));ht=np.zeros((n,14));generators=[];starts=[];holds=np.zeros(n,int);maxlift=np.zeros(n);windows=[deque(maxlen=51) for _ in range(n)];done=np.zeros(n,bool)
 for i in range(n):
  d=mujoco.MjData(m);d.qpos[:]=cfg['qpos'];d.mocap_pos[:]=np.array(cfg['mocap_pos']).reshape(-1,3);d.mocap_quat[:]=np.array(cfg['mocap_quat']).reshape(-1,4)
  shift=d.qpos[object['qadr']:object['qadr']+2]-d.qpos[cfg['boxes'][0]['qadr']:cfg['boxes'][0]['qadr']+2];d.qpos[:2]+=shift
  for j,other in enumerate(cfg['boxes']):
   if j==object_index:continue
   d.qpos[other['qadr']:other['qadr']+2]=[100.+j*5,100.];sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pickup_support_body'+('' if j==0 else '_'+str(j)));d.mocap_pos[m.body_mocapid[sid],:2]=[100.+j*5,100.]
  mujoco.mj_forward(m,d);states.append(d);starts.append(d.geom_xpos[cfg['object_geom'],2]);histories.append([{k:np.zeros(v) for k,v in [('actions',29),('ang_vel',3),('dof_pos',29),('dof_vel',29),('gravity',3)]} for _ in range(10)]);generators.append(torch.Generator(device=device).manual_seed(startseed+i))
 def frame(i):
  d=states[i];g=np.empty(3);mujoco.mju_rotVecQuat(g,np.array([0.,0.,-1.]),np.array([d.qpos[3],-d.qpos[4],-d.qpos[5],-d.qpos[6]]));return dict(actions=last[i].copy(),ang_vel=d.qvel[3:6]*.25,dof_pos=d.qpos[qa]-cfg['default_q'],dof_vel=d.qvel[va].copy(),gravity=g)
 for i in range(n):histories[i][0]=frame(i)
 def obs():
  obs=[]
  for i,d in enumerate(states):
   f=frame(i);shim=BFMMuJoCoSim.__new__(BFMMuJoCoSim);shim.model=m;shim.data=d;shim._mj=mujoco;shim._body_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name) for name in KEYPOINT_NAMES[:-1]];shim._body_ids_arr=np.array(shim._body_ids);hist=np.concatenate([np.concatenate([fr[k] for fr in histories[i][1:]]) for k in ['actions','ang_vel','dof_pos','dof_vel','gravity']]);obs.append(dict(state=np.r_[f['dof_pos'],f['dof_vel'],f['gravity'],f['ang_vel']],last_action=last[i],history_actor=hist,privileged_state=shim.get_privileged_state()))
  return {k:torch.tensor(np.stack([o[k] for o in obs]),device=device,dtype=torch.float32) for k in obs[0]}
 scheduler=RTCLatentScheduler(controls_per_knot=5,controls_per_plan=25,num_knots=10,rtc=True,ramp_controls=5,control_points=2)
 for pi in range(36):
  depths=[]
  for d in states:renderer.update_scene(d,camera=cfg['camera_id']);depths.append(resize_depth_to_mm(renderer.render(),144,256).float()/1000)
  depth=torch.stack(depths).to(device)[:,None];depth=torch.where((depth>=.12)&(depth<=3),depth,0);q=torch.tensor(np.stack([d.qpos[qa] for d in states]),device=device,dtype=torch.float32);v=torch.tensor(np.stack([d.qvel[va] for d in states]),device=device,dtype=torch.float32);state=normalize_planner_state(q,v,bundle)
  noises=[]
  for gen in generators:noises.append(torch.randn(1,10,512,generator=gen,device=device))
  endpoints=bundle.policy.sample(depth,state,steps=8,solver='euler',noise=torch.cat(noises),language=torch.tensor(meta['language']['pickup'],device=device)[None].expand(n,-1),command=torch.zeros(n,10,device=device),drop_command_condition=True,drop_state_condition=False);anchor=bfm_backward_anchor(bfm,obs()) if scheduler.needs_anchor() else None;controls=scheduler.plan(endpoints,anchor_latent=anchor,max_controls=25)
  for ci,z in enumerate(controls.unbind(1)):
   action=bfm_action_from_latent(bfm,obs(),z).cpu().numpy();last=np.clip(action*6,-6,6);targets=np.array(cfg['default_q'])+last*.25*np.array(am['action_rescale'])
   for i,d in enumerate(states):
    if done[i]:continue
    for _ in range(cfg['decimation']):
     mujoco.mj_step1(m,d);d.ctrl[cfg['actuator']]=np.clip(np.array(cfg['kp'])*(targets[i]-d.qpos[qa])-np.array(cfg['kd'])*d.qvel[va],-np.array(cfg['effort']),np.array(cfg['effort']));mujoco.mj_step2(m,d)
    histories[i].insert(0,frame(i));histories[i].pop();rise=d.geom_xpos[cfg['object_geom'],2]-starts[i];maxlift[i]=max(maxlift[i],rise);raised=rise>.1 and np.linalg.norm(d.geom_xpos[cfg['object_geom'],:2]-d.qpos[:2])<.8;holds[i]=holds[i]+1 if raised else 0;force=0
    for j,c in enumerate(d.contact):
     if (c.geom1==cfg['object_geom'] and c.geom2 in handgeoms) or (c.geom2==cfg['object_geom'] and c.geom1 in handgeoms):mujoco.mj_contactForce(m,d,j,normal);force+=abs(normal[0])
    windows[i].append(raised and force>.1)
    if holds[i]>=51 and sum(windows[i])>=45:
     done[i]=True;print(json.dumps(dict(object=object['label'],seed=startseed+i,success=True,lift=maxlift[i],simtime=pi*.5+(ci+1)*.02,contact=sum(windows[i]),seconds=time.time()-started),default=lambda x:x.item()),flush=True)
    if d.qpos[2]<.25:done[i]=True
  if any(holds>=51):break
 print(json.dumps(dict(object=object['label'],seeds=[startseed,startseed+n-1],max_lifts=maxlift.tolist(),holds=holds.tolist(),seconds=time.time()-started),default=lambda x:x.item()),flush=True)
 if any(holds>=51):break
renderer.close()
