"""Export pickup from a held-out arrival pose, including runtime geometry changes."""
from pathlib import Path
import json,xml.etree.ElementTree as E
import numpy as np,torch,mujoco
from lsvla import load_bfm
from lsvla.knot_planner_runtime import bfm_backward_anchor
from lsvla.sim.pickup_mujoco_env import PickupMuJoCoEnv,BoxSpec,SupportSpec
root=Path.home()/'fiber-web-export';out=root/'browser/scenes/pickup';meta=json.loads((out/'scene.json').read_text());r=json.loads((root/'pickup-manifest.json').read_text())['records'][4]
def quat(v):return [v[3],*v[:3]]
bfm=load_bfm(str(root/'models/model_best_emd.pt'),device='cpu',control_profile='bfm05_xl',verbose=False)
env=PickupMuJoCoEnv(history_length=9,enable_rendering=False,action_rescale=bfm.cfg.action_rescale,hand_model='dex3',box_condim=4,box_torsional_friction=.2,box_solref=(.004,1.0));env.reset_to_default();env.reset_scene(root_position_w=r['root_pos_w'],root_quat_wxyz=quat(r['root_quat_xyzw']),joint_position_usd=r['joint_pos'],object_position_w=r['object_pos_w'],object_quat_wxyz=quat(r['object_quat_xyzw']),box=BoxSpec(scale_xyz=tuple(r['object_scale_xyz']),mass_kg=.5,friction=1.2),support=SupportSpec(position_w=tuple(r['support_pose_w'][:3]),quat_wxyz=tuple(quat(r['support_pose_w'][3:])),size_xyz=tuple(r['support_size_xyz'])))
m,d=env.model,env.data;tree=E.parse(out/'scene.xml')
for name in ['pickup_object_box','pickup_support']:
 gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name);g=tree.getroot().find(f".//geom[@name='{name}']");g.set('size',' '.join(map(str,m.geom_size[gid])));g.set('pos',' '.join(map(str,m.geom_pos[gid])));g.set('friction',' '.join(map(str,m.geom_friction[gid])));g.set('condim',str(int(m.geom_condim[gid])))
tree.getroot().find(".//geom[@name='pickup_object_box']").set('mass','.5');tree.write(out/'scene.xml',encoding='unicode')
visuals=[{'id':i,'type':int(m.geom_type[i]),'size':m.geom_size[i].tolist(),'rgba':m.geom_rgba[i].tolist(),'body':int(m.geom_bodyid[i]),'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''} for i in range(m.ngeom) if m.geom_group[i]<3]
obs=env.get_obs_dict();meta.update(visuals=visuals,qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),mocap_pos=d.mocap_pos.reshape(-1).tolist(),mocap_quat=d.mocap_quat.reshape(-1).tolist(),anchor=bfm_backward_anchor(bfm,obs)[0].tolist(),initial_obs={k:v[0].tolist() for k,v in obs.items()},provenance={'arrival':'held-out pickup bank ep_0004','object_mass_kg':.5,'scene':'box on support; natural arrival pose'})
(out/'scene.json').write_text(json.dumps(meta,separators=(',',':')));env.close();print('PICKUP SCENE EXPORTED',flush=True)
