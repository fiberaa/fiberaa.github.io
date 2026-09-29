"""Export authentic low-support arrivals instead of a resized high-support case."""
from pathlib import Path
import json,copy,xml.etree.ElementTree as E
import numpy as np,mujoco,torch
from lsvla import load_bfm
from lsvla.knot_planner_runtime import bfm_backward_anchor
from lsvla.sim.pickup_mujoco_env import PickupMuJoCoEnv,BoxSpec,SupportSpec
root=Path.home()/'fiber-web-export';folder=root/'browser/scenes/pickup';base=json.loads((folder/'case-support-2.json').read_text());records=json.loads((root/'pickup-manifest.json').read_text())['records'];bfm=load_bfm(str(root/'models/model_best_emd.pt'),device='cpu',control_profile='bfm05_xl',verbose=False)
def quat(v):return [v[3],*v[:3]]
for index in [9,14,63]:
 r=records[index];env=PickupMuJoCoEnv(history_length=9,enable_rendering=False,action_rescale=bfm.cfg.action_rescale,hand_model='dex3',box_condim=4,box_torsional_friction=.2,box_solref=(.004,1.0));env.reset_to_default();env.reset_scene(root_position_w=r['root_pos_w'],root_quat_wxyz=quat(r['root_quat_xyzw']),joint_position_usd=r['joint_pos'],object_position_w=r['object_pos_w'],object_quat_wxyz=quat(r['object_quat_xyzw']),box=BoxSpec(scale_xyz=tuple(r['object_scale_xyz']),mass_kg=.5,friction=1.2),support=SupportSpec(position_w=tuple(r['support_pose_w'][:3]),quat_wxyz=tuple(quat(r['support_pose_w'][3:])),size_xyz=tuple(r['support_size_xyz'])))
 m,d=env.model,env.data;tree=E.parse(folder/'case-support-2.xml')
 for name in ['pickup_object_box','pickup_support']:
  gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name);g=tree.find(f".//geom[@name='{name}']");g.set('size',' '.join(map(str,m.geom_size[gid])));g.set('pos',' '.join(map(str,m.geom_pos[gid])));g.set('friction',' '.join(map(str,m.geom_friction[gid])));g.set('condim',str(int(m.geom_condim[gid])))
 tree.find(".//geom[@name='pickup_object_box']").set('mass','.5');xml=f'low-bank-{index}.xml';tree.write(folder/xml,encoding='unicode');cfg=copy.deepcopy(base);obs=env.get_obs_dict();cfg.update(xml=xml,files=[xml]+[f for f in base['files'] if f.startswith('assets/')],qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),mocap_pos=d.mocap_pos.reshape(-1).tolist(),mocap_quat=d.mocap_quat.reshape(-1).tolist(),anchor=bfm_backward_anchor(bfm,obs)[0].tolist(),initial_obs={k:v[0].tolist() for k,v in obs.items()},visuals=[dict(id=g,type=int(m.geom_type[g]),size=m.geom_size[g].tolist(),rgba=(base['visuals'][g]['rgba'] if g==base['object_geom'] else m.geom_rgba[g].tolist()),body=int(m.geom_bodyid[g]),name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '') for g in range(m.ngeom) if m.geom_group[g]<3]);cfg['provenance'].update(arrival=r['key'],support_height=r['support_height_m'],start='Original held-out low-support arrival and geometry');(folder/f'low-bank-{index}.json').write_text(json.dumps(cfg,separators=(',',':')));env.close();print('EXPORTED LOW',index,flush=True)

manifest=json.loads((folder/"cases.json").read_text());manifest["layouts"]["support"][1]["config"]="low-bank-14.json";(folder/"cases.json").write_text(json.dumps(manifest));cfg=json.loads((folder/"low-bank-14.json").read_text());cfg["provenance"].update(layout="support",variant=1,single_object=True);(folder/"low-bank-14.json").write_text(json.dumps(cfg))
