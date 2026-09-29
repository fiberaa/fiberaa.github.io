from pathlib import Path
import os,onnx,onnxruntime as ort,numpy as np,json
from onnx import helper,numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
p=Path.home()/'fiber-web-export/browser'/os.environ.get('FIBER_MODEL','v24')/'planner.w8.onnx';m=onnx.load(p);initial={v.name:v for v in m.graph.initializer};nodes=[];count=0
for n in m.graph.node:
 if n.op_type!='Gemm':nodes.append(n);continue
 a={x.name:helper.get_attribute_value(x) for x in n.attribute};assert a.get('alpha',1)==1 and a.get('beta',1)==1 and a.get('transA',0)==0
 w=n.input[1]
 if a.get('transB',0):
  w+='_transposed';m.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(initial[n.input[1]]).T.copy(),w))
 mid=n.output[0]+'_mm';nodes.append(helper.make_node('MatMul',[n.input[0],w],[mid],name=n.name+'_mm'));nodes.append(helper.make_node('Add',[mid,n.input[2]],n.output,name=n.name+'_bias'));count+=1
if count:
 del m.graph.node[:];m.graph.node.extend(nodes);used={v for n in nodes for v in n.input};keep=[v for v in m.graph.initializer if v.name in used];del m.graph.initializer[:];m.graph.initializer.extend(keep)
 q=MatMulNBitsQuantizer(m,bits=8,block_size=32,is_symmetric=True,accuracy_level=0);q.process();q.model.save_model_to_file(str(p),use_external_data_format=False)
print('Additional compact linear layers',count,'bytes',p.stat().st_size,flush=True)

opts=ort.SessionOptions();opts.intra_op_num_threads=8
reference=ort.InferenceSession(str(p.parent/'planner.float32.onnx'),opts)
feed=dict(np.load(p.parent/'parity.npz'));feed={v.name:feed[v.name] for v in reference.get_inputs()};ref=reference.run(None,feed);del reference
session=ort.InferenceSession(str(p),opts);pred=session.run(None,feed)
errors=[{'max':float(abs(x-y).max()),'relative':float(np.linalg.norm(x-y)/np.linalg.norm(x))} for x,y in zip(ref,pred)]
meta_path=p.parent/'planner_meta.json';meta=json.loads(meta_path.read_text());meta['quantization']={'body':errors[0],'hand':errors[1]};meta_path.write_text(json.dumps(meta,separators=(',',':')))
print('Final W8 parity',errors,flush=True)
