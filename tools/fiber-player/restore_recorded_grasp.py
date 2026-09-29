"""Restore recorded single-object support layouts on the export node.

Run repair_grasp_rest_pose.py afterward to apply collision-free resting arms
and rebuild anchors. The curated reset poses/noise seeds were verified in
Chromium and WebKit with the 3-step student and synchronous 1 Hz planning.
"""
from pathlib import Path
import json,numpy as np,torch,mujoco
from lsvla import load_bfm
from lsvla.knot_planner_runtime import bfm_backward_anchor
from lsvla.sim.mujoco_sim import BFMMuJoCoSim,KEYPOINT_NAMES
root=Path.home()/'fiber-web-export'; folder=root/'browser/scenes/grasp'; bank=torch.load(root/'grail/bank.pt',weights_only=False)
bfm=load_bfm(str(root/'models/model_best_emd.pt'),device='cpu',control_profile='bfm05_xl',verbose=False)
for key,clipkey in [('tangerine','pickup_table__tangerine_4__001'),('water-bottle','pickup_table__water_bottle_2__001'),('cup','pickup_table__coffee_cup_4__002')]:
 cfg=json.loads((folder/f'candidate-{key}.json').read_text()); c=next(c for c in bank['clips'] if c['clip_key']==clipkey); m=mujoco.MjModel.from_xml_path(str(folder/cfg['xml']));d=mujoco.MjData(m);d.qpos[:]=cfg['qpos']; d.qpos[:3]=c['root_pos_w'].numpy();q=c['root_quat_w_xyzw'].numpy();yaw=np.arctan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2));d.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)];d.qpos[2]=.8
 poses={'tangerine':(.25701123476,.63664693117,'grasp-student-5'),'water-bottle':(.21,.65,'grasp-student-7'),'cup':(.12,.82,'grasp-student-10')}
 x,y,noise=poses[key];d.qpos[:2]=[x,y];cfg['sampler_noise']=noise
 if key=='water-bottle':cfg['switch_sampler_noise']='grasp-student-3';cfg['browser_start_xy']={'default':[.21,.65],'webkit':[.21,.72]}
 if key in ('cup','water-bottle'):d.qpos[3:7]=[np.cos(-np.pi/4),0,0,np.sin(-np.pi/4)]
 cfg['provenance']['start_frame']=int(c['arrival_frame']);cfg['provenance']['grasp_onset_frame']=int(c['grasp_onset_frame'])
 cfg['provenance']['root_arrival_offset_xy_m']=(d.qpos[:2]-c['root_pos_w'][:2].numpy()).tolist()
 d.qpos[cfg['hand']['qadr']]=0; a=cfg['box_qadr'];d.qpos[a:a+3]=c['object_pos_w'].numpy();q=c['object_quat_w_xyzw'].numpy();d.qpos[a+3:a+7]=q[[3,0,1,2]];d.mocap_pos[0]=c['table_pos'].numpy();d.mocap_quat[0]=[1,0,0,0];mujoco.mj_forward(m,d)
 gravity=np.empty(3);mujoco.mju_rotVecQuat(gravity,np.array([0.,0.,-1.]),np.array([d.qpos[3],-d.qpos[4],-d.qpos[5],-d.qpos[6]]));obs={'state':torch.tensor(np.r_[d.qpos[cfg['qadr']]-cfg['default_q'],np.zeros(29),gravity,np.zeros(3)][None],dtype=torch.float32),'last_action':torch.zeros(1,29),'history_actor':torch.zeros(1,837)};shim=BFMMuJoCoSim.__new__(BFMMuJoCoSim);shim.model=m;shim.data=d;shim._mj=mujoco;shim._body_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in KEYPOINT_NAMES[:-1]];shim._body_ids_arr=np.array(shim._body_ids);obs['privileged_state']=torch.tensor(shim.get_privileged_state()[None]);cfg.update(qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),mocap_pos=d.mocap_pos.reshape(-1).tolist(),mocap_quat=d.mocap_quat.reshape(-1).tolist(),anchor=bfm_backward_anchor(bfm,obs)[0].tolist(),initial_obs={k:v[0].tolist() for k,v in obs.items()});cfg['provenance']['start']='recorded object and support placement; curated upright approach pose near the recorded arrival; resting arms and open fingers';cfg['provenance']['support_size_m']=c['table_size'].tolist();(folder/f'candidate-{key}.json').write_text(json.dumps(cfg,separators=(',',':')));print(key,d.qpos[:3],d.mocap_pos[0],flush=True)
