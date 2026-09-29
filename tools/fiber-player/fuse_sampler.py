"""Keep all Euler updates on GPU; read back only the final eight-step plan."""
from pathlib import Path
import os,numpy as np,onnx,onnxruntime as ort
from onnx import helper,TensorProto
p=Path.home()/'fiber-web-export/browser'/os.environ.get('FIBER_MODEL','v24');m=onnx.load(p/'planner.step.w8.onnx')
outputs={v.name:v for v in m.graph.output}
m.graph.input.append(helper.make_tensor_value_info('remaining_steps',TensorProto.FLOAT,[1]))
for source,current,target in [('endpoint','trajectory','next_trajectory'),('hand_endpoint','hand_trajectory','next_hand')]:
 m.graph.node.extend([helper.make_node('Sub',[source,current],[target+'_delta'],name=target+'_delta'),helper.make_node('Div',[target+'_delta','remaining_steps'],[target+'_increment'],name=target+'_increment'),helper.make_node('Add',[current,target+'_increment'],[target],name=target)])
 info=helper.make_tensor_value_info(target,TensorProto.FLOAT,[1,10,512 if source=='endpoint' else 14]);outputs[target]=info
 del m.graph.output[:]
m.graph.output.extend([outputs['next_trajectory'],outputs['next_hand']]);onnx.checker.check_model(m);onnx.save(m,p/'planner.step.gpu.w8.onnx')
print('FUSED SAMPLER', (p/'planner.step.gpu.w8.onnx').stat().st_size,flush=True)
opts=ort.SessionOptions();opts.intra_op_num_threads=8
sample=dict(np.load(p/'parity.npz'));condition=ort.InferenceSession(str(p/'planner.condition.w8.onnx'),opts);clock=ort.InferenceSession(str(p/'planner.time.w8.onnx'),opts);step=ort.InferenceSession(str(p/'planner.step.w8.onnx'),opts);fused=ort.InferenceSession(str(p/'planner.step.gpu.w8.onnx'),opts)
cache=dict(zip([v.name for v in condition.get_outputs()],condition.run(None,{v.name:sample[v.name] for v in condition.get_inputs()})))
a=sample['trajectory'].copy();h=sample['hand_trajectory'].copy();fa=a.copy();fh=h.copy()
for i in range(8):
 cache.update(dict(zip([v.name for v in clock.get_outputs()],clock.run(None,{'flow_time':np.array([i/8],np.float32)}))))
 feed={**sample,**cache,'trajectory':a,'hand_trajectory':h};pred=step.run(None,{v.name:feed[v.name] for v in step.get_inputs()});a=a+(pred[0]-a)/(8-i);h=h+(pred[1]-h)/(8-i)
 feed.update(trajectory=fa,hand_trajectory=fh,remaining_steps=np.array([8-i],np.float32));fa,fh=fused.run(None,{v.name:feed[v.name] for v in fused.get_inputs()})
errors=[float(abs(a-fa).max()),float(abs(h-fh).max())];assert max(errors)<2e-5,errors
print('Eight-step fused parity',errors,flush=True)
