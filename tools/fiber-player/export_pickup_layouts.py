"""Build three real multi-box layouts: ground, support and mixed."""
from pathlib import Path
import copy,json,xml.etree.ElementTree as E
import numpy as np,torch,mujoco
from lsvla import load_bfm
from lsvla.knot_planner_runtime import bfm_backward_anchor
from lsvla.sim.mujoco_sim import BFMMuJoCoSim,KEYPOINT_NAMES
root=Path.home()/'fiber-web-export';folder=root/'browser/scenes/pickup';base=json.loads((folder/'scene.json').read_text());records=json.loads((root/'pickup-manifest.json').read_text())['records'];bfm=load_bfm(str(root/'models/model_best_emd.pt'),device='cpu',control_profile='bfm05_xl',verbose=False);layouts={}
for layout in ['ground','support','mixed']:
 tree=E.parse(folder/'scene.xml');world=tree.getroot().find('worldbody');ob=world.find("body[@name='pickup_object']");sb=world.find("body[@name='pickup_support_body']")
 for i in [1,2]:
  for original in [ob,sb]:
   node=copy.deepcopy(original)
   for el in node.iter():
    if el.get('name'):el.set('name',el.get('name')+'_'+str(i))
   world.append(node)
 for i,color in enumerate(['.72 .55 .35 1','.35 .63 .9 1','.66 .46 .82 1']):world.find(f".//geom[@name='pickup_object_box{'' if i==0 else '_'+str(i)}']").set('rgba',color)
 xml=f'scene-{layout}.xml';tree.write(folder/xml,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(folder/xml));d=mujoco.MjData(m);r=records[0 if layout=='ground' else 4];q=np.array(r['root_quat_xyzw']);d.qpos[:3]=r['root_pos_w'];d.qpos[3:7]=q[[3,0,1,2]];d.qpos[base['qadr']]=r['joint_pos'];yaw=np.arctan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2));R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]);origin=np.array(r['object_pos_w'][:2]);offsets=[np.zeros(2),R@np.array([1.4,.95]),R@np.array([2.1,-1.05])];boxes=[]
 for i in range(3):
  suffix='' if i==0 else '_'+str(i);bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pickup_object'+suffix);gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'pickup_object_box'+suffix);oq=int(m.jnt_qposadr[m.body_jntadr[bid]]);sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pickup_support_body'+suffix);mid=int(m.body_mocapid[sid]);supported=layout=='support' or (layout=='mixed' and i!=1);height=[.43958349,.25,.55][i] if supported else 0.;d.qpos[oq:oq+3]=[*list(origin+offsets[i]),height];q=np.array(r['object_quat_xyzw']);d.qpos[oq+3:oq+7]=q[[3,0,1,2]]
  pos=np.array(records[4]['support_pose_w'][:3]);pos[:2]+=origin-np.array(records[4]['object_pos_w'][:2])+offsets[i];pos[2]=height-float(m.geom_size[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'pickup_support'+suffix),2]) if supported else -5.;d.mocap_pos[mid]=pos;q=np.array(records[4]['support_pose_w'][3:]);d.mocap_quat[mid]=q[[3,0,1,2]];boxes.append({'label':f'Box {i+1} · '+('support' if supported else 'ground'),'body':bid,'geom':gid,'qadr':oq,'surface':'support' if supported else 'ground'})
 mujoco.mj_forward(m,d);qdev=d.qpos[base['qadr']]-base['default_q'];gravity=np.empty(3);mujoco.mju_rotVecQuat(gravity,np.array([0.,0.,-1.]),np.array([d.qpos[3],-d.qpos[4],-d.qpos[5],-d.qpos[6]]));obs={'state':torch.tensor(np.concatenate([qdev,np.zeros(29),gravity,np.zeros(3)])[None],dtype=torch.float32),'last_action':torch.zeros(1,29),'history_actor':torch.zeros(1,837)};shim=BFMMuJoCoSim.__new__(BFMMuJoCoSim);shim.model=m;shim.data=d;shim._mj=mujoco;shim._body_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in KEYPOINT_NAMES[:-1]];shim._body_ids_arr=np.array(shim._body_ids);obs['privileged_state']=torch.tensor(shim.get_privileged_state()[None]);cfg=copy.deepcopy(base);cfg.update(xml=xml,files=[xml]+[f for f in base['files'] if f.startswith('assets/')],boxes=boxes,object_geom=boxes[0]['geom'],box_body=boxes[0]['body'],box_qadr=boxes[0]['qadr'],qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),mocap_pos=d.mocap_pos.reshape(-1).tolist(),mocap_quat=d.mocap_quat.reshape(-1).tolist(),anchor=bfm_backward_anchor(bfm,obs)[0].tolist(),initial_obs={k:v[0].tolist() for k,v in obs.items()},visuals=[{'id':g,'type':int(m.geom_type[g]),'size':m.geom_size[g].tolist(),'rgba':m.geom_rgba[g].tolist(),'body':int(m.geom_bodyid[g]),'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''} for g in range(m.ngeom) if m.geom_group[g]<3],provenance={'layout':layout,'arrival':r['key'],'object_mass_kg':.5});layouts[layout]=cfg
(folder/'layouts.json').write_text(json.dumps({'layouts':layouts},separators=(',',':')));print('THREE MULTI-BOX LAYOUTS EXPORTED')
