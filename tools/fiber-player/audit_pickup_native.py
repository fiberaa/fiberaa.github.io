"""Audit actual box pickup defaults with FP32 learned models and physical contact gates."""
from pathlib import Path
import json,time,sys,argparse,numpy as np,torch,mujoco
from collections import deque
parser=argparse.ArgumentParser();parser.add_argument("--layout",choices=["mixed","ground","support"],default="mixed");parser.add_argument("--object",type=int,default=0);parser.add_argument("--seed",type=int,default=42);parser.add_argument("--all",action="store_true");parser.add_argument("--retain-state",action="store_true");parser.add_argument("--standoff",type=float,default=.45);parser.add_argument("--approach-gate",type=float,default=.18);parser.add_argument("--face-object",action="store_true");parser.add_argument("--problem-cases",action="store_true");parser.add_argument("--isolate",action="store_true");parser.add_argument("--ablate-state",action="store_true");parser.add_argument("--single-case",action="store_true");parser.add_argument("--config");args=parser.parse_args()
from lsvla import load_bfm
from lsvla.knot_planner_runtime import load_knot_planner_checkpoint,normalize_planner_state,bfm_backward_anchor,bfm_action_from_latent,RTCLatentScheduler
from lsvla.sim.mujoco_sim import BFMMuJoCoSim,KEYPOINT_NAMES
from lsvla.sim.pickup_mujoco_env import resize_depth_to_mm
root=Path.home()/'fiber-web-export';torch.set_num_threads(8)
bfm=load_bfm(str(root/'models/model_best_emd.pt'),device='cuda',control_profile='bfm05_xl',verbose=False);bundle=load_knot_planner_checkpoint(root/'models/planner.pt',device='cuda');meta=json.loads((root/'browser/v27/planner_meta.json').read_text());am=json.loads((root/'browser/actor_meta.json').read_text());device='cuda';results=[]
def run(task,seed,boxx=None,drop=False):
 cfg=json.loads((root/'browser/scenes/pickup/layouts.json').read_text())['layouts'][layout].copy()
 if args.single_case:
  manifest=json.loads((root/'browser/scenes/pickup/cases.json').read_text());filename=manifest['layouts'][layout][active]['config'];cfg=json.loads((root/'browser/scenes/pickup'/filename).read_text())
 if args.config:cfg=json.loads((root/'browser/scenes/pickup'/args.config).read_text())
 obj=cfg['boxes'][0 if args.single_case or args.config else active];cfg.update(box_body=obj['body'],box_qadr=obj['qadr'],object_geom=obj['geom']);m=mujoco.MjModel.from_xml_path(str(root/f"browser/scenes/{task}/{cfg.get('xml','scene.xml')}"));d=mujoco.MjData(m);d.qpos[:]=cfg['qpos'];d.qvel[:]=cfg['qvel'];d.mocap_pos[:]=np.array(cfg['mocap_pos']).reshape(-1,3);d.mocap_quat[:]=np.array(cfg['mocap_quat']).reshape(-1,4)
 if args.isolate:
  baseobj=cfg['boxes'][0];shift=d.qpos[obj['qadr']:obj['qadr']+2]-d.qpos[baseobj['qadr']:baseobj['qadr']+2];d.qpos[:2]+=shift
  for i,other in enumerate(cfg['boxes']):
   if i==active:continue
   d.qpos[other['qadr']:other['qadr']+2]=[100.+i*5,100.]
   sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pickup_support_body'+('' if i==0 else '_'+str(i)));d.mocap_pos[m.body_mocapid[sid],:2]=[100.+i*5,100.]
 if boxx is not None:d.qpos[cfg['box_qadr']]=boxx
 mujoco.mj_forward(m,d);renderer=mujoco.Renderer(m,height=240,width=426);renderer.enable_depth_rendering();qadr=cfg['qadr'];vadr=cfg['vadr'];last=np.zeros(29);zero={'actions':np.zeros(29),'ang_vel':np.zeros(3),'dof_pos':np.zeros(29),'dof_vel':np.zeros(29),'gravity':np.zeros(3)};history=[{k:v.copy() for k,v in zero.items()} for _ in range(10)]
 shim=BFMMuJoCoSim.__new__(BFMMuJoCoSim);shim.model=m;shim.data=d;shim._mj=mujoco;shim._body_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in KEYPOINT_NAMES[:-1]];shim._body_ids_arr=np.array(shim._body_ids)
 def frame():
  gravity=np.empty(3);mujoco.mju_rotVecQuat(gravity,np.array([0.,0.,-1.]),np.array([d.qpos[3],-d.qpos[4],-d.qpos[5],-d.qpos[6]]));return {'actions':last.copy(),'ang_vel':d.qvel[3:6]*.25,'dof_pos':d.qpos[qadr]-cfg['default_q'],'dof_vel':d.qvel[vadr].copy(),'gravity':gravity}
 history[0]=frame()
 def obs():
  f=frame();hist=np.concatenate([np.concatenate([a[k] for a in history[1:]]) for k in zero]);return {k:torch.tensor(v[None],dtype=torch.float32,device=device) for k,v in {'state':np.concatenate([f['dof_pos'],f['dof_vel'],f['gravity'],f['ang_vel']]),'last_action':last,'history_actor':hist,'privileged_state':shim.get_privileged_state()}.items()}
 scheduler=RTCLatentScheduler(controls_per_knot=5,controls_per_plan=25,num_knots=10,rtc=True,ramp_controls=5,control_points=2);torch.manual_seed(seed);start=float(d.xpos[cfg['box_body'],2]);startcenter=float(d.geom_xpos[cfg.get('object_geom',mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'pickup_object_box')),2]);maxlift=0;hold=0;besthold=0;hinge=0;minz=1;mode='pickup' if active==0 or args.isolate or args.single_case else 'walk';hand=cfg.get('hand');ht=d.qpos[hand['qadr']].copy() if hand else None;started=time.time()
 center=d.geom_xpos[cfg['object_geom']].copy();delta=d.qpos[:2]-center[:2];goal=center[:2]+delta/np.linalg.norm(delta)*args.standoff;goal_yaw=np.arctan2(-delta[1],-delta[0]);pending_drop=drop and mode=='pickup';drop_count=0;interaction_time=0. if mode=='pickup' else None;window=deque(maxlen=51);forcebuf=np.zeros(6);contactbest=0;handgeoms=set(g for g in range(m.ngeom) if any(t in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or "") for t in ["_hand_","_wrist_"]));success=False
 for pi in range(70):
  if mode=='walk':
   if args.face_object:goal_yaw=np.arctan2(center[1]-d.qpos[1],center[0]-d.qpos[0])
   w,x,y,z=d.qpos[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
   if np.linalg.norm(d.qpos[:2]-goal)<args.approach_gate and abs(np.arctan2(np.sin(goal_yaw-yaw),np.cos(goal_yaw-yaw)))<.35:mode='pickup';pending_drop=drop;interaction_time=float(d.time)
  if task=='door' and mode=='walk' and (np.hypot(d.qpos[0]-.58,d.qpos[1]+.33)<.25 or pi>=20):mode='open_door';scheduler=RTCLatentScheduler(controls_per_knot=5,controls_per_plan=25,num_knots=10,rtc=True,ramp_controls=5,control_points=2)
  renderer.update_scene(d,camera=cfg['camera_id']);depth=resize_depth_to_mm(renderer.render(),144,256).float().to(device)[None,None]/1000;depth=torch.where((depth>=.12)&(depth<=3),depth,0);q=torch.tensor(d.qpos[qadr][None],dtype=torch.float32,device=device);v=torch.tensor(d.qvel[vadr][None],dtype=torch.float32,device=device);state=normalize_planner_state(q,v,bundle)
  cmd=np.zeros(10,np.float32)
  if mode=='walk':
   if args.face_object:goal_yaw=np.arctan2(center[1]-d.qpos[1],center[0]-d.qpos[0])
   w,x,y,z=d.qpos[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));dx=goal[0]-d.qpos[0];dy=goal[1]-d.qpos[1];dist=np.hypot(dx,dy);err=np.arctan2(np.sin(np.arctan2(dy,dx)-yaw),np.cos(np.arctan2(dy,dx)-yaw));face=goal_yaw-yaw;face=np.arctan2(np.sin(face),np.cos(face))
   for i,h in enumerate([.5,1.]):mag=min(dist,.8*h);turn=np.clip(face,-.9*h,.9*h);cmd[i*5:i*5+5]=[mag*np.cos(err),mag*np.sin(err),np.sin(turn),np.cos(turn),.75]
  kwargs={'language':torch.tensor(meta['language'][mode],device=device)[None],'command':torch.tensor(cmd,device=device)[None],'drop_command_condition':mode!='walk','drop_state_condition':pending_drop}
  if task=='grasp':
   hs=(d.qpos[hand['qadr']]-meta['hand_state_normalization']['mean'])/np.maximum(meta['hand_state_normalization']['std'],1e-6);kwargs.update(normalized_hand_state=torch.tensor(hs[None],device=device,dtype=torch.float32),hand_mask=torch.tensor([True],device=device),return_hand_actions=True)
  drop_count+=int(pending_drop);pending_drop=False;sample=bundle.policy.sample(depth,state,steps=8,solver='euler',**kwargs);endpoints,hands=sample if task=='grasp' else (sample,None);anchor=bfm_backward_anchor(bfm,obs()) if scheduler.needs_anchor() else None;controls=scheduler.plan(endpoints,anchor_latent=anchor,max_controls=25)
  for ci,z in enumerate(controls.unbind(1)):
   action=bfm_action_from_latent(bfm,obs(),z)[0].cpu().numpy();last=np.clip(action*6,-6,6);target=np.array(cfg['default_q'])+last*.25*np.array(am['action_rescale'])
   if hands is not None:ht=np.array(meta['hand_action']['center'])+np.clip(hands[0,ci//5].cpu().numpy(),-1,1)*np.array(meta['hand_action']['scale'])
   for _ in range(cfg['decimation']):
    mujoco.mj_step1(m,d);d.ctrl[cfg['actuator']]=np.clip(np.array(cfg['kp'])*(target-d.qpos[qadr])-np.array(cfg['kd'])*d.qvel[vadr],-np.array(cfg['effort']),np.array(cfg['effort']))
    if ht is not None:d.ctrl[hand['actuator']]=np.clip(np.array(hand['kp'])*(ht-d.qpos[hand['qadr']])-np.array(hand['kd'])*d.qvel[hand['vadr']],-np.array(hand['effort']),np.array(hand['effort']))
    mujoco.mj_step2(m,d)
   # Match the browser arbiter: handoff is checked after every executed control.
   if mode=='walk':
    if args.face_object:goal_yaw=np.arctan2(center[1]-d.qpos[1],center[0]-d.qpos[0])
    w,x,y,z=d.qpos[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    if np.linalg.norm(d.qpos[:2]-goal)<args.approach_gate and abs(np.arctan2(np.sin(goal_yaw-yaw),np.cos(goal_yaw-yaw)))<.35:
     mode='pickup';pending_drop=drop;interaction_time=float(d.time)
   history.insert(0,frame());history.pop();gid=cfg.get('object_geom',mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'pickup_object_box'));rise=d.geom_xpos[gid,2]-startcenter;maxlift=max(maxlift,float(rise));raised=rise>.1 and np.linalg.norm(d.geom_xpos[gid,:2]-d.qpos[:2])<.8;hold=hold+1 if raised else 0;besthold=max(besthold,hold);minz=min(minz,float(d.qpos[2]));hinge=max(hinge,float(d.qpos[cfg['hinge_qadr']])) if cfg.get('hinge_qadr') is not None else 0
   force=0.
   for j,contact in enumerate(d.contact):
    if (contact.geom1==gid and contact.geom2 in handgeoms) or (contact.geom2==gid and contact.geom1 in handgeoms):mujoco.mj_contactForce(m,d,j,forcebuf);force+=abs(forcebuf[0])
   window.append(bool(raised and force>.1));contactbest=max(contactbest,sum(window))
   success=mode=='pickup' and hold>=51 and sum(window)>=45 and minz>.25
   if success:break
  if (task!='door' and success) or (task=='door' and hinge>.5 and d.qpos[0]>1.6):break
  if minz<.2:break
 w,x,y,z=d.qpos[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
 result={'single_case':args.single_case,'isolated_reset':args.isolate,'arbiter':'each executed control','precision':'native FP32 planner + actor','standoff':args.standoff,'approach_gate':args.approach_gate,'face_object':args.face_object,'layout':layout,'object':active,'success':bool(success),'mode':mode,'interaction_time':interaction_time,'drop_plans':drop_count,'min_root_z':minz,'contact_samples':sum(window),'best_contact_samples':contactbest,'goal_distance':float(np.linalg.norm(d.qpos[:2]-goal)),'goal_yaw_error':float(np.arctan2(np.sin(goal_yaw-yaw),np.cos(goal_yaw-yaw))),'task':task,'seed':seed,'boxx':boxx,'drop':drop,'lift':maxlift,'hold':besthold,'hinge':hinge,'root':d.qpos[:3].tolist(),'simtime':float(d.time),'seconds':time.time()-started};print(json.dumps(result),flush=True);results.append(result);renderer.close()
cases=[(l,i,not args.retain_state) for l in ['mixed','ground','support'] for i in range(3)] if args.all else [(args.layout,args.object,not args.retain_state)]
if args.ablate_state:cases += [(l,0,args.retain_state) for l in ['mixed','ground','support']]
if args.problem_cases:cases=[(l,i,not args.retain_state) for l,i in [('ground',1),('support',1),('support',2)]]
for layout,active,drop in cases:
 run('pickup',args.seed,None,drop)
