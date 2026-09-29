"""Add independently simulated tabletop targets to the neutral grasp scene."""
from pathlib import Path
import json,copy,xml.etree.ElementTree as E
import numpy as np,mujoco
folder=Path.home()/'fiber-web-export/browser/scenes/grasp';cfg=json.loads((folder/'scene.json').read_text());tree=E.parse(folder/'scene.xml');w=tree.find('worldbody');original=w.find("body[@name='pickup_object']")
for i in [1,2]:
 old=w.find(f"body[@name='pickup_object_{i}']")
 if old is not None:w.remove(old)
 node=copy.deepcopy(original)
 for el in node.iter():
  if el.get('name'):el.set('name',el.get('name')+'_'+str(i))
 g=node.find('geom');g.attrib.pop('mesh',None);g.set('type','sphere' if i==1 else 'box');g.set('size','.045' if i==1 else '.035 .035 .035');g.set('mass','.12');g.set('rgba','.35 .68 .78 1' if i==1 else '.68 .55 .82 1');w.append(node)
tree.write(folder/'scene-objects.xml',encoding='unicode');m=mujoco.MjModel.from_xml_path(str(folder/'scene-objects.xml'));d=mujoco.MjData(m);d.qpos[:len(cfg['qpos'])]=cfg['qpos'];d.mocap_pos[:]=np.array(cfg['mocap_pos']).reshape(-1,3);d.mocap_quat[:]=np.array(cfg['mocap_quat']).reshape(-1,4);boxes=[]
for i,label in enumerate(['Tangerine','Ball','Cube']):
 suffix='' if i==0 else '_'+str(i);bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pickup_object'+suffix);gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'grasp_object'+suffix);a=int(m.jnt_qposadr[m.body_jntadr[bid]])
 if i:d.qpos[a:a+3]=[(-.38 if i==1 else .38),.02,float(d.mocap_pos[0,2])+.02+(.045 if i==1 else .035)];d.qpos[a+3:a+7]=[1,0,0,0]
 boxes.append(dict(label=label,body=bid,geom=gid,qadr=a,surface='table'))
mujoco.mj_forward(m,d);cfg.update(xml='scene-objects.xml',files=['scene-objects.xml']+[f for f in cfg['files'] if f.startswith('assets/')],boxes=boxes,qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),visuals=[dict(id=g,type=int(m.geom_type[g]),size=m.geom_size[g].tolist(),rgba=m.geom_rgba[g].tolist(),body=int(m.geom_bodyid[g]),name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '') for g in range(m.ngeom) if m.geom_group[g]<3]);(folder/'scene.json').write_text(json.dumps(cfg,separators=(',',':')));print('THREE GRASP OBJECTS EXPORTED')
