"""Run on the export node after restoring recorded grasp placements.

Use the G1 Dex3 training rest pose and recompute the BFM anchor.
Exclude only palm/thumb-base convex-hull overlaps within each hand;
all robot/object and other self contacts remain enabled.
"""
from pathlib import Path
import json,numpy as np,torch,mujoco
from lsvla import load_bfm
from lsvla.knot_planner_runtime import bfm_backward_anchor
from lsvla.sim.mujoco_sim import BFMMuJoCoSim,KEYPOINT_NAMES
root=Path.home()/'fiber-web-export'; folder=root/'browser/scenes/grasp'
bfm=load_bfm(str(root/'models/model_best_emd.pt'),device='cpu',control_profile='bfm05_xl',verbose=False)
import xml.etree.ElementTree as ET
for key in ['tangerine','water-bottle','cup']:
 cfg=json.loads((folder/f'candidate-{key}.json').read_text()); xp=folder/cfg['xml']; tree=ET.parse(xp); xml=tree.getroot(); contact=xml.find('contact')
 if contact is None: contact=ET.SubElement(xml,'contact')
 for side in ['left','right']:
  pair={'body1':f'{side}_hand_palm_link','body2':f'{side}_hand_thumb_1_link'}
  if not any(e.attrib==pair for e in contact): ET.SubElement(contact,'exclude',pair)
 if key in ('water-bottle','cup'):
  mesh=xml.find("./asset/mesh[@name='grasp_object_mesh']");mesh.set('inertia','convex')
  geom=xml.find(".//geom[@name='grasp_object']");geom.set('mass','1');tree.write(xp)
  probe=mujoco.MjModel.from_xml_path(str(xp));body=mujoco.mj_name2id(probe,mujoco.mjtObj.mjOBJ_BODY,'box')
  if body < 0: body=int(probe.geom_bodyid[mujoco.mj_name2id(probe,mujoco.mjtObj.mjOBJ_GEOM,'grasp_object')])
  inertia=probe.body_inertia[body];half_squared=1.5*(inertia.sum()-2*inertia)
  mass=float(np.clip(300*8*np.sqrt(np.prod(half_squared)),.02,1));geom.set('mass',str(mass));cfg['provenance']['mass_kg']=mass
  cfg['provenance']['mass_rule']='300 kg/m^3 times equivalent box volume from convex collision inertia, clamped to 0.02–1 kg'
 tree.write(xp)
 m=mujoco.MjModel.from_xml_path(str(xp));d=mujoco.MjData(m);d.qpos[:]=cfg['qpos'];d.mocap_pos[:]=np.array(cfg['mocap_pos']).reshape(-1,3);d.mocap_quat[:]=np.array(cfg['mocap_quat']).reshape(-1,4)
 for side,sign in [('left',1),('right',-1)]:
  for suffix,val in [('shoulder_pitch_joint',.2),('shoulder_roll_joint',sign*.2),('elbow_joint',.6)]:
   jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_{suffix}');d.qpos[m.jnt_qposadr[jid]]=val
 d.qpos[cfg['hand']['qadr']]=0;mujoco.mj_forward(m,d)
 gravity=np.empty(3);mujoco.mju_rotVecQuat(gravity,np.array([0.,0.,-1.]),np.array([d.qpos[3],-d.qpos[4],-d.qpos[5],-d.qpos[6]]));obs={'state':torch.tensor(np.r_[d.qpos[cfg['qadr']]-cfg['default_q'],np.zeros(29),gravity,np.zeros(3)][None],dtype=torch.float32),'last_action':torch.zeros(1,29),'history_actor':torch.zeros(1,837)};shim=BFMMuJoCoSim.__new__(BFMMuJoCoSim);shim.model=m;shim.data=d;shim._mj=mujoco;shim._body_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in KEYPOINT_NAMES[:-1]];shim._body_ids_arr=np.array(shim._body_ids);obs['privileged_state']=torch.tensor(shim.get_privileged_state()[None]);cfg.update(qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),mocap_pos=d.mocap_pos.reshape(-1).tolist(),mocap_quat=d.mocap_quat.reshape(-1).tolist(),anchor=bfm_backward_anchor(bfm,obs)[0].tolist(),initial_obs={k:v[0].tolist() for k,v in obs.items()});cfg['provenance']['rest_pose']='G1 Dex3 training rest: shoulder pitch .2, roll +/- .2, elbow .6, open fingers';(folder/f'candidate-{key}.json').write_text(json.dumps(cfg,separators=(',',':')));print(key,d.ncon,flush=True)
