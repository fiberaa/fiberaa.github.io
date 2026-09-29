"""Remote-only v24 export, with its own Dex3 head and normalization contracts."""
from pathlib import Path
import json, numpy as np, torch, onnx, onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from lsvla.knot_planner_runtime import load_knot_planner_checkpoint
from lsvla.knot_language import ModeLanguageEmbedding
root=Path.home()/'fiber-web-export'; out=root/'browser/v24';out.mkdir(exist_ok=True)
torch.set_num_threads(8);torch.backends.mha.set_fastpath_enabled(False)
b=load_knot_planner_checkpoint(root/'models/planner-v24.pt',device='cpu',expected_control_profile='bfm05_xl');c=b.config
ckpt=torch.load(root/'models/planner-v24.pt',map_location='cpu',weights_only=False)
def serial(v):
 if isinstance(v,torch.Tensor):return v.tolist()
 if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
 return v
payload=torch.load(root/'models/language-v3.pt',weights_only=False)
cache=ModeLanguageEmbedding(payload['texts'],payload['modes'],payload['embeddings'],encoder=payload['encoder'])
meta=json.loads((root/'browser/planner_meta.json').read_text());meta.update(config=vars(c),state_mean=b.state_mean.tolist(),state_std=b.state_std.tolist(),step=b.step,source_checkpoint='v24dex-399500',source_sha256='f10a89af2119f8c3feeea775375327f80cfdcfb353b341b59ceff27faaf33934',hand_action=serial(ckpt['hand_action']),hand_state_normalization=serial(ckpt['hand_state_normalization']),language={m:cache.mode_vector(m).tolist() for m in ['walk','pickup','carry_locomotion','open_door']});del ckpt
class Planner(torch.nn.Module):
 def __init__(self,policy):super().__init__();self.policy=policy
 def forward(self,depth,state,trajectory,flow_time,command,language,drop_command,drop_state,hand_trajectory,hand_state,hand_mask):
  return self.policy(trajectory,flow_time,depth,state,command=command,language=language,drop_command_condition=drop_command,drop_state_condition=drop_state,noised_hand_actions=hand_trajectory,normalized_hand_state=hand_state,hand_mask=hand_mask,return_hand_actions=True)
model=Planner(b.policy).eval()
torch.manual_seed(17)
inputs=(torch.rand(1,1,c.depth_height,c.depth_width),torch.randn(1,c.state_dim),torch.randn(1,c.num_knots,c.z_dim),torch.tensor([.25]),torch.zeros(1,c.command_dim),torch.randn(1,c.language_dim),torch.tensor([False]),torch.tensor([False]),torch.randn(1,c.num_knots,c.hand_dim),torch.randn(1,c.hand_state_dim),torch.tensor([True]))
names=['depth','state','trajectory','flow_time','command','language','drop_command','drop_state','hand_trajectory','hand_state','hand_mask']
with torch.inference_mode():ref=[x.numpy() for x in model(*inputs)]
p=out/'planner.float32.onnx';torch.onnx.export(model,inputs,str(p),input_names=names,output_names=['endpoint','hand_endpoint'],opset_version=17,dynamo=False,external_data=False)
del model,b
m=onnx.load(p)
for n in m.graph.node:
 for a in n.attribute:
  if n.op_type=='Cast' and a.name=='to' and a.i==11:a.i=1
  if a.type==onnx.AttributeProto.TENSOR and a.t.data_type==11:a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32),a.t.name))
onnx.save(m,p);del m
opts=ort.SessionOptions();opts.intra_op_num_threads=8
s=ort.InferenceSession(str(p),opts);feed={n:x.numpy() for n,x in zip(names,inputs) if n in [v.name for v in s.get_inputs()]};pred=s.run(None,feed)
print('FP32 parity',[float(abs(x-y).max()) for x,y in zip(ref,pred)],flush=True);assert max(float(abs(x-y).max()) for x,y in zip(ref,pred))<1e-4;del s
q=MatMulNBitsQuantizer(str(p),bits=8,block_size=32,is_symmetric=True,accuracy_level=0);q.process();target=out/'planner.w8.onnx';q.model.save_model_to_file(str(target),use_external_data_format=False);del q
s=ort.InferenceSession(str(target),opts);pred=s.run(None,feed);errors=[{'max':float(abs(x-y).max()),'relative':float(np.linalg.norm(x-y)/np.linalg.norm(x))} for x,y in zip(ref,pred)];del s
meta['quantization']={'body':errors[0],'hand':errors[1]};(out/'planner_meta.json').write_text(json.dumps(meta,separators=(',',':')))
np.savez(out/'parity.npz',**feed,reference=pred[0],hand_reference=pred[1])
print('W8',target.stat().st_size,'errors',errors,flush=True)
