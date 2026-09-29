"""Export inference-only FIBER assets on the GPU node; never copies checkpoints."""
from pathlib import Path
import json, math, shutil, hashlib, xml.etree.ElementTree as ET
import numpy as np
import torch, mujoco, onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from lsvla import load_bfm
from lsvla.knot_planner_runtime import load_knot_planner_checkpoint, normalize_planner_state, bfm_backward_anchor
from lsvla.knot_language import ModeLanguageEmbedding
from lsvla.sim.pickup_mujoco_env import PickupMuJoCoEnv, DoorSpec, BoxSpec
from export_actor_reference import ActorONNX, JOINT_NAMES

root=Path.home()/'fiber-web-export'; out=root/'browser'; out.mkdir(exist_ok=True)
torch.set_num_threads(8); torch.backends.mha.set_fastpath_enabled(False)
def dump(name,obj): (out/name).write_text(json.dumps(obj,separators=(',',':')))
def trace(model,inputs,name,names,output='output'):
    with torch.inference_mode(): ref=model(*inputs).numpy()
    torch.onnx.export(model.eval(),inputs,str(out/name),input_names=names,output_names=[output],opset_version=17,dynamo=False,external_data=False)
    sess=ort.InferenceSession(str(out/name),providers=['CPUExecutionProvider'])
    feed={n:x.numpy() for n,x in zip(names,inputs) if n in [v.name for v in sess.get_inputs()]}
    pred=sess.run(None,feed)[0]; error=float(np.max(np.abs(ref-pred)))
    print(name,'bytes',(out/name).stat().st_size,'parity',error,flush=True)
    assert error < .001, (name,error)
    return feed,ref

bfm=load_bfm(str(root/'models/model_best_emd.pt'),device='cpu',control_profile='bfm05_xl',verbose=False)
actor=ActorONNX(bfm.policy).eval(); H=bfm.cfg.obs_dims['history_actor']//93
trace(actor,(torch.zeros(1,64),torch.zeros(1,29),torch.zeros(1,H*93),bfm.sample_z(1)), 'actor.onnx',['state','last_action','history_actor','z'],'action')
meta={'z_dim':bfm.z_dim,'history_length':H,'joint_names':JOINT_NAMES,'action_rescale':list(bfm.cfg.action_rescale),'control_dt':.02}
scenes={}
for scene_name,door,box_pos,root_pos in [('pickup',None,[.85,0,0],[0,0,.8]),('door',DoorSpec(latch=True,hinge_side='left',handle_return_nm=.7,latch_release_rad=.25,lever_len_m=.15,handle_damping=.3,handle_height_m=.95,handle_kp=1,max_open_rad=3),[3.5,-1,0],[0,0,.8])]:
    env=PickupMuJoCoEnv(history_length=H,enable_rendering=False,action_rescale=bfm.cfg.action_rescale,door=door,hand_model='dex3',box_condim=4,box_torsional_friction=.2,box_solref=(.004,1.0))
    env.reset_to_default(); env.reset_scene(root_position_w=root_pos,root_quat_wxyz=[1,0,0,0],joint_position_usd=env._get_dof_pos_usd().copy(),object_position_w=box_pos,object_quat_wxyz=[1,0,0,0],box=BoxSpec(mass_kg=.3,friction=1.2),support=None)
    m,d=env.model,env.data
    folder=out/'scenes'/scene_name; folder.mkdir(parents=True,exist_ok=True)
    xml=folder/'scene.xml'; mujoco.mj_saveLastXML(str(xml),m)
    tree=ET.parse(xml); compiler=tree.getroot().find('compiler'); meshdir=compiler.get('meshdir','') if compiler is not None else ''
    if compiler is not None: compiler.attrib.pop('meshdir',None); compiler.attrib.pop('texturedir',None)
    assets=[]
    for elem in tree.getroot().iter():
        if 'file' in elem.attrib:
            old=Path(elem.get('file')); source=old if old.is_absolute() else Path(env.xml_path).parent/meshdir/old
            if not source.exists(): raise FileNotFoundError(source)
            target='assets/'+source.name; dest=folder/target; dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(source,dest); elem.set('file',target); assets.append(target)
    tree.getroot().find(".//geom[@name='pickup_object_box']").set('mass','0.3')
    tree.write(xml,encoding='unicode')
    # Default contact solver settings are retained in scene.xml.
    visuals=[]
    for g in range(m.ngeom):
        if m.geom_group[g]>=3: continue
        entry={'id':g,'type':int(m.geom_type[g]),'size':m.geom_size[g].tolist(),'rgba':m.geom_rgba[g].tolist(),'body':int(m.geom_bodyid[g]),'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''}
        if int(m.geom_type[g])==7:
            # Browser reads the compiled mesh vertices directly from MuJoCo.
            pass
        visuals.append(entry)
    obs=env.get_obs_dict(); anchor=bfm_backward_anchor(bfm,obs)[0].tolist()
    config={'files':['scene.xml']+sorted(set(assets)),'visuals':visuals,'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),'mocap_pos':d.mocap_pos.reshape(-1).tolist(),'mocap_quat':d.mocap_quat.reshape(-1).tolist(),'qadr':env._qpos_addrs_arr.tolist(),'vadr':env._qvel_addrs_arr.tolist(),'actuator':env._actuator_ids_arr.tolist(),'default_q':env.default_q.tolist(),'kp':env.kp.tolist(),'kd':env.kd.tolist(),'effort':env.effort_limit.tolist(),'camera_id':env._camera_id,'box_body':env._object_body_ids[0],'box_qadr':env._object_qpos_adrs[0],'hinge_qadr':env._door_hinge_qpos_adr,'anchor':anchor,'initial_obs':{k:v[0].tolist() for k,v in obs.items()},'sim_dt':float(m.opt.timestep),'decimation':env._decimation}
    (folder/'scene.json').write_text(json.dumps(config,separators=(',',':'))); scenes[scene_name]='scenes/'+scene_name+'/scene.json'; env.close()
dump('actor_meta.json',meta)
del actor,bfm

bundle=load_knot_planner_checkpoint(root/'models/planner.pt',device='cpu',expected_control_profile='bfm05_xl')
c=bundle.config; print('planner config',vars(c),flush=True)
class Planner(torch.nn.Module):
    def __init__(self,policy): super().__init__(); self.policy=policy
    def forward(self,depth,state,trajectory,flow_time,command,language,drop_command,drop_state):
        return self.policy(trajectory,flow_time,depth,state,command=command,language=language,drop_command_condition=drop_command,drop_state_condition=drop_state)
inputs=(torch.rand(1,1,c.depth_height,c.depth_width),torch.randn(1,c.state_dim),torch.randn(1,c.num_knots,c.z_dim),torch.tensor([.25]),torch.zeros(1,c.command_dim),torch.randn(1,c.language_dim),torch.tensor([False]),torch.tensor([False]))
names=['depth','state','trajectory','flow_time','command','language','drop_command','drop_state']
feed,ref=trace(Planner(bundle.policy),inputs,'planner.float32.onnx',names,'endpoint')
quantize_dynamic(str(out/'planner.float32.onnx'),str(out/'planner.int8.onnx'),weight_type=QuantType.QUInt8,op_types_to_quantize=['MatMul','Gemm'])
sess=ort.InferenceSession(str(out/'planner.int8.onnx'),providers=['CPUExecutionProvider']); quant=sess.run(None,feed)[0]
error=float(np.max(np.abs(ref-quant))); relative=float(np.linalg.norm(ref-quant)/max(np.linalg.norm(ref),1e-8)); print('int8 parity max',error,'relative',relative,'size',(out/'planner.int8.onnx').stat().st_size,flush=True)
# Preserve float16 alternative for WebGPU, with float32 IO.
from onnxruntime.transformers.float16 import convert_float_to_float16
model=onnx.load(str(out/'planner.float32.onnx')); model=convert_float_to_float16(model,keep_io_types=True); onnx.save(model,str(out/'planner.fp16.onnx'))
cache=ModeLanguageEmbedding.load(str(root/'models/language.pt'))
language={mode:cache.mode_vector(mode).tolist() for mode in ['walk','pickup','carry_locomotion','put_down_ground','open_door']}
dump('planner_meta.json',{'config':vars(c),'state_mean':bundle.state_mean.tolist(),'state_std':bundle.state_std.tolist(),'horizons':[float(v) for v in bundle.training_args['waypoint_horizons_s'].split(',')],'controls_per_knot':bundle.controls_per_knot,'step':bundle.step,'language':language,'scenes':scenes,'quantization':{'max_error':error,'relative_error':relative},'source_checkpoint':'v27-226000'})
print('EXPORT COMPLETE',flush=True)
