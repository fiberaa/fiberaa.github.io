"""Build a MuJoCo tabletop demo with articulated URDF Dex3 hands.

Uses a held-out GRAIL arrival state; the fruit collider is a spherical proxy.
This is an interactive demo, not the GRAIL evaluation environment.
"""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as E
import numpy as np,torch,mujoco
from lsvla.dex_hand import DEX3_JOINT_ORDER
from lsvla import load_bfm
from export_actor_reference import JOINT_NAMES
root=Path.home()/'fiber-web-export';source=root/'browser/scenes/pickup';out=root/'browser/scenes/grasp';shutil.copytree(source,out,dirs_exist_ok=True)
meta=json.loads((source/'scene.json').read_text()); original=mujoco.MjModel.from_xml_path(str(source/'scene.xml'));body_names=[mujoco.mj_id2name(original,mujoco.mjtObj.mjOBJ_JOINT,next(i for i in range(original.njnt) if original.jnt_qposadr[i]==a)) for a in meta['qadr']];tree=E.parse(out/'scene.xml');r=tree.getroot();u=E.parse(root/'src/lsvla/assets/g1_dex3/g1_29dof_with_hand_rev_1_0.urdf').getroot();links={l.get('name'):l for l in u.findall('link')};children={}
for j in u.findall('joint'):
 if 'hand_' in j.get('name',''):children.setdefault(j.find('parent').get('link'),[]).append(j)
act=r.find('actuator')
def add(parent,name):
 for j in children.get(name,[]):
  child=j.find('child').get('link');link=links[child];origin=j.find('origin');assert origin.get('rpy','0 0 0')=='0 0 0'
  b=E.SubElement(parent,'body',name=child,pos=origin.get('xyz','0 0 0'));inert=link.find('inertial');o=inert.find('origin');I=inert.find('inertia');E.SubElement(b,'inertial',pos=o.get('xyz'),mass=inert.find('mass').get('value'),fullinertia=' '.join(I.get(k) for k in ['ixx','iyy','izz','ixy','ixz','iyz']))
  if j.get('type')=='revolute':
   limit=j.find('limit');E.SubElement(b,'joint',name=j.get('name'),type='hinge',axis=j.find('axis').get('xyz'),range=limit.get('lower')+' '+limit.get('upper'),armature='0.001',damping='0')
   effort='2.45' if 'thumb_0' in j.get('name') else '1.4';E.SubElement(act,'motor',name=j.get('name')+'_motor',joint=j.get('name'),ctrlrange='-'+effort+' '+effort,ctrllimited='true')
  E.SubElement(b,'geom',name=child+'_contact',type='mesh',mesh='d3_'+child,friction='1.8 0.01 0.0005',condim='4',rgba='0.22 0.22 0.26 1')
  add(b,child)
for side in ['left','right']:
 wrist=next(b for b in r.iter('body') if b.get('name')==side+'_wrist_yaw_link')
 for g in list(wrist.findall('geom')):
  if '_hand_collider_d3_' in g.get('name',''):wrist.remove(g)
 old=wrist.find('inertial');link=links[side+'_wrist_yaw_link'];li=link.find('inertial');I=li.find('inertia');old.attrib.clear();old.set('pos',li.find('origin').get('xyz'));old.set('mass',li.find('mass').get('value'));old.set('fullinertia',' '.join(I.get(k) for k in ['ixx','iyy','izz','ixy','ixz','iyz']))
 add(wrist,side+'_wrist_yaw_link')
bank=torch.load(root/'grail/bank.pt',weights_only=False);clip=next(c for c in bank['clips'] if c['clip_key']=='pickup_table__tangerine_4__002')
obj=next(b for b in r.iter('body') if b.get('name')=='pickup_object');obj.remove(obj.find('geom'));E.SubElement(obj,'geom',name='grasp_object',type='sphere',size='0.045',mass='0.12',rgba='1 .38 .035 1',friction='1.2 .01 .0005',condim='4',solref='.004 1')
support=next(b for b in r.iter('body') if b.find("geom[@name='pickup_support']") is not None);g=support.find('geom');g.set('size','1 .3 .02');g.set('rgba','.33 .38 .45 1')
r.find('keyframe') is not None and r.remove(r.find('keyframe'))
tree.write(out/'scene.xml',encoding='unicode');m=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));d=mujoco.MjData(m)
def jid(name):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
order=list(bank['joint_names']);by_name={name:i for i,name in enumerate(order)}
for name in body_names+list(DEX3_JOINT_ORDER):
 id=jid(name);d.qpos[m.jnt_qposadr[id]]=float(clip['joint_pos'][by_name[name]]);d.qvel[m.jnt_dofadr[id]]=float(clip['joint_vel'][by_name[name]])
d.qpos[:3]=clip['root_pos_w'].numpy();q=clip['root_quat_w_xyzw'].numpy();d.qpos[3:7]=q[[3,0,1,2]];d.qvel[:3]=clip['root_lin_vel_w'].numpy()
# MuJoCo free-joint angular velocity is in the body frame.
R=np.empty(9);mujoco.mju_quat2Mat(R,d.qpos[3:7]);d.qvel[3:6]=R.reshape(3,3).T@clip['root_ang_vel_w'].numpy()
objid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pickup_object');oq=int(m.jnt_qposadr[m.body_jntadr[objid]]);d.qpos[oq:oq+3]=clip['object_pos_w'].numpy();d.qpos[oq+2]=float(clip['pedestal_top_z'])+.045;d.qpos[oq+3:oq+7]=[1,0,0,0]
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,support.get('name'));d.mocap_pos[m.body_mocapid[sid]]=clip['table_pos'].numpy();d.mocap_quat[m.body_mocapid[sid]]=[1,0,0,0]
mujoco.mj_forward(m,d)
qadr=[int(m.jnt_qposadr[jid(n)]) for n in body_names];vadr=[int(m.jnt_dofadr[jid(n)]) for n in body_names]
# Preserve the BFM body contract and compute an anchor from this arrival state.
bfm=load_bfm(str(root/'models/model_best_emd.pt'),device='cpu',control_profile='bfm05_xl',verbose=False)
from lsvla.knot_planner_runtime import bfm_backward_anchor
qdev=d.qpos[qadr]-np.array(meta['default_q']);gravity=np.empty(3);mujoco.mju_rotVecQuat(gravity,np.array([0.,0.,-1.]),np.array([d.qpos[3],-d.qpos[4],-d.qpos[5],-d.qpos[6]]));angular=d.qvel[3:6]*.25
frame=np.concatenate([np.zeros(29),angular,qdev,d.qvel[vadr],gravity]);hist=np.concatenate([np.tile(np.zeros(29),9),np.tile(angular,9),np.tile(qdev,9),np.tile(d.qvel[vadr],9),np.tile(gravity,9)])
obs={'state':torch.tensor(np.concatenate([qdev,d.qvel[vadr],gravity,angular])[None],dtype=torch.float32),'last_action':torch.zeros(1,29),'history_actor':torch.tensor(hist[None],dtype=torch.float32)}
from lsvla.sim.mujoco_sim import BFMMuJoCoSim, KEYPOINT_NAMES
shim=BFMMuJoCoSim.__new__(BFMMuJoCoSim);shim.model=m;shim.data=d;shim._mj=mujoco;shim._body_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in KEYPOINT_NAMES[:-1]];shim._body_ids_arr=np.array(shim._body_ids)
obs['privileged_state']=torch.tensor(shim.get_privileged_state()[None])
anchor=bfm_backward_anchor(bfm,obs)[0].tolist()
visuals=[{'id':i,'type':int(m.geom_type[i]),'size':m.geom_size[i].tolist(),'rgba':m.geom_rgba[i].tolist(),'body':int(m.geom_bodyid[i]),'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''} for i in range(m.ngeom) if m.geom_group[i]<3]
hand={'names':list(DEX3_JOINT_ORDER),'qadr':[int(m.jnt_qposadr[jid(n)]) for n in DEX3_JOINT_ORDER],'vadr':[int(m.jnt_dofadr[jid(n)]) for n in DEX3_JOINT_ORDER],'actuator':[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n+'_motor') for n in DEX3_JOINT_ORDER],'kp':[2. if 'thumb_0' in n else .5 for n in DEX3_JOINT_ORDER],'kd':[.1]*14,'effort':[2.45 if 'thumb_0' in n else 1.4 for n in DEX3_JOINT_ORDER]}
meta.update(visuals=visuals,qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),mocap_pos=d.mocap_pos.reshape(-1).tolist(),mocap_quat=d.mocap_quat.reshape(-1).tolist(),qadr=qadr,vadr=vadr,actuator=list(range(29)),camera_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'depth_camera'),box_body=objid,box_qadr=oq,object_geom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'grasp_object'),hand=hand,anchor=anchor,initial_obs={k:v[0].tolist() for k,v in obs.items()},provenance={'arrival_clip':clip['clip_key'],'arrival_frame':clip['arrival_frame'],'object_geometry':'spherical proxy; not the GRAIL mesh'})
# Camera name is read from the original scene if the specific name differs.
if meta['camera_id']<0:meta['camera_id']=json.loads((source/'scene.json').read_text())['camera_id']
(out/'scene.json').write_text(json.dumps(meta,separators=(',',':')))
print('GRASP SCENE',m.nq,m.nv,m.nu,'hand joints',len(hand['qadr']),flush=True)
