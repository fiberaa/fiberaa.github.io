from pathlib import Path
import onnx,numpy as np,onnxruntime as ort
from onnx import numpy_helper
p=Path.home()/'fiber-web-export/browser'; path=p/'planner.w8.onnx'
opts=ort.SessionOptions();opts.intra_op_num_threads=8
rng=np.random.default_rng(9)
s=ort.InferenceSession(str(path),opts);feed={v.name:(np.array([False]) if v.type=='tensor(bool)' else rng.standard_normal(v.shape).astype(np.float32)) for v in s.get_inputs()};feed['depth']=np.abs(feed['depth']);feed['flow_time'][:]=.25;ref=s.run(None,feed)[0];del s
m=onnx.load(str(path));count=0
for n in m.graph.node:
 for a in n.attribute:
  if n.op_type=='Cast' and a.name=='to' and a.i==11:a.i=1;count+=1
  if a.type==onnx.AttributeProto.TENSOR and a.t.data_type==11:a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32),a.t.name))
for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output):
 if v.type.tensor_type.elem_type==11:v.type.tensor_type.elem_type=1
onnx.save(m,str(path));del m
s=ort.InferenceSession(str(path),opts);pred=s.run(None,feed)[0];print('Removed float64 casts',count,'maximum output difference',float(abs(pred-ref).max()),flush=True)
assert np.max(abs(pred-ref))<1e-6
