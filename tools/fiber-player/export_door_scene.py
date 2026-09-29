"""Build a MuJoCo door demo with articulated URDF Dex3 hands.

Uses the existing latched-door scene with the Dex3 URDF hand joints.
This is an interactive demo, not the paper evaluation environment.
"""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as E
import numpy as np,torch,mujoco
from lsvla.dex_hand import DEX3_JOINT_ORDER
from lsvla import load_bfm
from export_actor_reference import JOINT_NAMES
root=Path.home()/'fiber-web-export';source=root/'browser/scenes/door';out=root/'browser/scenes/door_articulated';shutil.copytree(source,out,dirs_exist_ok=True)
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

if r.find('keyframe') is not None:r.remove(r.find('keyframe'))
r.find(".//geom[@name='pickup_object_box']").set('mass','0.3')
tree.write(out/'scene.xml',encoding='unicode');m=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));d=mujoco.MjData(m)
oldq=np.array(meta['qpos']);oldv=np.array(meta['qvel'])
for i in range(original.njnt):
 name=mujoco.mj_id2name(original,mujoco.mjtObj.mjOBJ_JOINT,i);new=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);kind=int(original.jnt_type[i]);nq={0:7,1:4,2:1,3:1}[kind];nv={0:6,1:3,2:1,3:1}[kind];q=int(original.jnt_qposadr[i]);v=int(original.jnt_dofadr[i]);a=int(m.jnt_qposadr[new]);b=int(m.jnt_dofadr[new]);d.qpos[a:a+nq]=oldq[q:q+nq];d.qvel[b:b+nv]=oldv[v:v+nv]
d.mocap_pos[:]=np.array(meta['mocap_pos']).reshape(-1,3);d.mocap_quat[:]=np.array(meta['mocap_quat']).reshape(-1,4);mujoco.mj_forward(m,d)
def jid(n):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)
hand={'names':list(DEX3_JOINT_ORDER),'qadr':[int(m.jnt_qposadr[jid(n)]) for n in DEX3_JOINT_ORDER],'vadr':[int(m.jnt_dofadr[jid(n)]) for n in DEX3_JOINT_ORDER],'actuator':[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n+'_motor') for n in DEX3_JOINT_ORDER],'kp':[2. if 'thumb_0' in n else .5 for n in DEX3_JOINT_ORDER],'kd':[.1]*14,'effort':[2.45 if 'thumb_0' in n else 1.4 for n in DEX3_JOINT_ORDER]}
objid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pickup_object');oq=int(m.jnt_qposadr[m.body_jntadr[objid]])
visuals=[{'id':i,'type':int(m.geom_type[i]),'size':m.geom_size[i].tolist(),'rgba':m.geom_rgba[i].tolist(),'body':int(m.geom_bodyid[i]),'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''} for i in range(m.ngeom) if m.geom_group[i]<3]
hinge_name=mujoco.mj_id2name(original,mujoco.mjtObj.mjOBJ_JOINT,next(i for i in range(original.njnt) if original.jnt_qposadr[i]==meta['hinge_qadr']))
meta.update(visuals=visuals,qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),mocap_pos=d.mocap_pos.reshape(-1).tolist(),mocap_quat=d.mocap_quat.reshape(-1).tolist(),qadr=[int(m.jnt_qposadr[jid(n)]) for n in body_names],vadr=[int(m.jnt_dofadr[jid(n)]) for n in body_names],box_body=objid,box_qadr=oq,hinge_qadr=int(m.jnt_qposadr[jid(hinge_name)]),hand=hand)
(out/'scene.json').write_text(json.dumps(meta,separators=(',',':')))
print('ARTICULATED DOOR',m.nq,m.nv,m.nu,flush=True)
