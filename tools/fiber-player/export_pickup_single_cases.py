"""Preserve all nine pickup placements as independent one-box trials."""
from pathlib import Path
import copy,json,xml.etree.ElementTree as E
import numpy as np,mujoco
folder=Path.home()/'fiber-web-export/browser/scenes/pickup';layouts=json.loads((folder/'layouts.json').read_text())['layouts'];manifest={'layouts':{}}
for layout,source in layouts.items():
 variants=[]
 for index,target in enumerate(source['boxes']):
  tree=E.parse(folder/source['xml']);world=tree.find('worldbody')
  for body in list(world.findall('body')):
   if body.get('name','') in ['pickup_object_1','pickup_object_2','pickup_support_body_1','pickup_support_body_2']:world.remove(body)
  color=next(g['rgba'] for g in source['visuals'] if g['id']==target['geom']);tree.find(".//geom[@name='pickup_object_box']").set('rgba',' '.join(map(str,color)))
  xml=f'case-{layout}-{index+1}.xml';tree.write(folder/xml,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(folder/xml));d=mujoco.MjData(m)
  d.qpos[:]=source['qpos'][:m.nq];d.qvel[:]=0
  body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pickup_object');geom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'pickup_object_box');adr=int(m.jnt_qposadr[m.body_jntadr[body]])
  selected=np.array(source['qpos'][target['qadr']:target['qadr']+7]);origin=np.array(source['qpos'][source['boxes'][0]['qadr']:source['boxes'][0]['qadr']+2]);shift=selected[:2]-origin;selected[:2]-=shift;d.qpos[adr:adr+7]=selected
  d.mocap_pos[0]=np.array(source['mocap_pos']).reshape(-1,3)[index];d.mocap_pos[0,:2]-=shift;d.mocap_quat[0]=np.array(source['mocap_quat']).reshape(-1,4)[index]
  mujoco.mj_forward(m,d);cfg=copy.deepcopy(source);one=dict(label=target['label'],body=body,geom=geom,qadr=adr,surface=target['surface'])
  cfg.update(xml=xml,files=[xml]+[f for f in source['files'] if f.startswith('assets/')],boxes=[one],box_body=body,box_qadr=adr,object_geom=geom,qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),mocap_pos=d.mocap_pos.reshape(-1).tolist(),mocap_quat=d.mocap_quat.reshape(-1).tolist(),sampler_noise='pickup',visuals=[dict(id=g,type=int(m.geom_type[g]),size=m.geom_size[g].tolist(),rgba=m.geom_rgba[g].tolist(),body=int(m.geom_bodyid[g]),name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '') for g in range(m.ngeom) if m.geom_group[g]<3])
  cfg['provenance'].update(single_object=True,variant=index,start='Original arrival pose relative to selected box; one-object trial',support_height=float(selected[2]))
  filename=f'case-{layout}-{index+1}.json';(folder/filename).write_text(json.dumps(cfg,separators=(',',':')));variants.append(dict(label=target['label'],surface=target['surface'],config=filename))
 manifest['layouts'][layout]=variants
(folder/'cases.json').write_text(json.dumps(manifest,separators=(',',':')));print('EXPORTED NINE ONE-BOX TRIALS')
