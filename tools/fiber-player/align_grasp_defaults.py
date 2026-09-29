"""Place authentic objects singly on the verified neutral-start tabletop setup."""
from pathlib import Path
import json,copy,xml.etree.ElementTree as E,numpy as np,mujoco
folder=Path.home()/'fiber-web-export/browser/scenes/grasp'
base=json.loads((folder/'scene-tangerine.json').read_text())
for key,label in [('tangerine','Tangerine'),('water-bottle','Water bottle'),('cup','Cup')]:
 cfg=json.loads((folder/f'candidate-{key}.json').read_text());tree=E.parse(folder/cfg['xml']);tree.find(".//geom[@name='grasp_object']").set('rgba',{'tangerine':'1 .38 .035 1','water-bottle':'.35 .68 .78 1','cup':'.68 .55 .82 1'}[key]);tree.write(folder/cfg['xml'],encoding='unicode');m=mujoco.MjModel.from_xml_path(str(folder/cfg['xml']));d=mujoco.MjData(m)
 d.qpos[:]=base['qpos'];d.qvel[:]=0;d.mocap_pos[:]=np.array(base['mocap_pos']).reshape(-1,3);d.mocap_quat[:]=np.array(base['mocap_quat']).reshape(-1,4)
 a=cfg['box_qadr'];gid=cfg['object_geom']
 if key!='tangerine':d.qpos[a+3:a+7]=cfg['qpos'][a+3:a+7]
 mujoco.mj_forward(m,d);mid=m.geom_dataid[gid];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]];world=v@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
 tablegeom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'pickup_support')
 if tablegeom<0:tablegeom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'pickup_support_geom')
 top=float(d.mocap_pos[0,2])+.02
 if key!='tangerine':d.qpos[a:a+3]+=np.array([.000637383-d.geom_xpos[gid,0],.06548656-d.geom_xpos[gid,1],top+.0005-world[:,2].min()])
 mujoco.mj_forward(m,d)
 cfg.update(qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),mocap_pos=d.mocap_pos.reshape(-1).tolist(),mocap_quat=d.mocap_quat.reshape(-1).tolist(),anchor=base['anchor'],initial_obs=base.get('initial_obs',{}))
 cfg['sampler_noise']='grasp-'+str({'tangerine':15,'water-bottle':4,'cup':52}[key])
 cfg['grasp_close_distance']=0
 cfg['grasp_palm_body']=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_hand_palm_link')
 for visual in cfg['visuals']:visual['rgba']=m.geom_rgba[visual['id']].tolist()
 cfg['provenance']['start']='neutral arms down, open fingers; one object on standardized tabletop'
 (folder/f'candidate-{key}.json').write_text(json.dumps(cfg,separators=(',',':')))
 print(label,'object center',d.geom_xpos[gid].tolist(),'bottom',top+.0005,flush=True)
