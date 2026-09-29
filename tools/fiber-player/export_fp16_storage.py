from pathlib import Path
import json,onnx,numpy as np
from onnx import helper,numpy_helper,TensorProto
p=Path('/home/ubuntu/fiber-web-export/browser/v27');out=p.parent/'v27-f16-storage';out.mkdir(exist_ok=True)
source=onnx.load(p/'planner.float32.onnx');original={v.name:v for v in source.graph.initializer};count=0
for filename in ['planner.condition.w8.onnx','planner.time.w8.onnx','planner.step.gpu.w8.onnx']:
 m=onnx.load(p/filename);nodes=[];extra=[]; restored=0
 for n in m.graph.node:
  if n.op_type=='MatMulNBits':
   name=n.input[1].removesuffix('_Q8');key=name;transpose=False
   if key not in original and key.endswith('_transposed'):key=key.removesuffix('_transposed');transpose=True
   if key not in original:raise KeyError(key)
   a=numpy_helper.to_array(original[key]);a=a.T.copy() if transpose else a
   stored=name+'_f16';extra.append(numpy_helper.from_array(a.astype(np.float16),stored))
   nodes.append(helper.make_node('Cast',[stored],[name],name=name+'_restore',to=TensorProto.FLOAT))
   nodes.append(helper.make_node('MatMul',[n.input[0],name],list(n.output),name=n.name.removesuffix('_Q8')));restored+=1
  else:nodes.append(n)
 used={i for n in nodes for i in n.input};keep=[v for v in m.graph.initializer if v.name in used];keep+=extra
 del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.initializer[:];m.graph.initializer.extend(keep)
 filename=filename.replace('.w8.','.f16.');onnx.checker.check_model(m);onnx.save(m,out/filename);print(filename,(out/filename).stat().st_size,'restored',restored,flush=True)
meta=json.loads((p/'planner_meta.json').read_text());meta['weight_storage']='FP16 linear weights, FP32 operations';(out/'planner_meta.json').write_text(json.dumps(meta,separators=(',',':')))
