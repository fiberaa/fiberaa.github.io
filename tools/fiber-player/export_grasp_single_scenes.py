"""Export one-object grasp scenes; preserve the neutral robot reset for each."""
from pathlib import Path
import copy,json,xml.etree.ElementTree as E
import mujoco,numpy as np
folder=Path.home()/'fiber-web-export/browser/scenes/grasp'
source=json.loads((folder/'scene.json').read_text())
objects=[]
for i,(key,label,kind,size,color) in enumerate([
 ('tangerine','Tangerine','mesh',None,'1 .38 .035 1'),
 ('ball','Ball','sphere','.045','.35 .68 .78 1'),
 ('cube','Cube','box','.035 .035 .035','.68 .55 .82 1')]):
 tree=E.parse(folder/'scene-objects.xml');world=tree.find('worldbody')
 for body in list(world.findall('body')):
  if body.get('name','').startswith('pickup_object_'):world.remove(body)
 geom=world.find("body[@name='pickup_object']/geom")
 if kind!='mesh':
  geom.attrib.pop('mesh',None);geom.set('type',kind);geom.set('size',size);geom.set('mass','.12')
 geom.set('rgba',color);xml=f'scene-{key}.xml';tree.write(folder/xml,encoding='unicode')
 model=mujoco.MjModel.from_xml_path(str(folder/xml));data=mujoco.MjData(model)
 data.qpos[:]=source['qpos'][:model.nq];data.qvel[:]=0
 data.mocap_pos[:]=np.array(source['mocap_pos']).reshape(-1,3);data.mocap_quat[:]=np.array(source['mocap_quat']).reshape(-1,4)
 body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pickup_object');gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'grasp_object');adr=int(model.jnt_qposadr[model.body_jntadr[body]])
 if kind!='mesh':data.qpos[adr+2]=data.mocap_pos[0,2]+.02+(.045 if kind=='sphere' else .035);data.qpos[adr+3:adr+7]=[1,0,0,0]
 mujoco.mj_forward(model,data);cfg=copy.deepcopy(source)
 cfg.update(xml=xml,files=[xml]+[f for f in source['files'] if f.startswith('assets/')],boxes=[dict(label=label,body=body,geom=gid,qadr=adr,surface='table')],box_body=body,box_qadr=adr,object_geom=gid,qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),visuals=[dict(id=g,type=int(model.geom_type[g]),size=model.geom_size[g].tolist(),rgba=model.geom_rgba[g].tolist(),body=int(model.geom_bodyid[g]),name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '') for g in range(model.ngeom) if model.geom_group[g]<3])
 filename=f'scene-{key}.json';(folder/filename).write_text(json.dumps(cfg,separators=(',',':')));objects.append(dict(key=key,label=label,config=filename))
(folder/'objects.json').write_text(json.dumps(dict(objects=objects),separators=(',',':')))
print('Exported three separate one-object scenes')
