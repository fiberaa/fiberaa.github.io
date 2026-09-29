"""Cache condition-only ONNX subgraphs, including every cross-attention K/V.

Run on the export node. This rewrites no model arithmetic or solver steps.
"""
from pathlib import Path
import os,copy,json,time
import numpy as np,onnx,onnxruntime as ort
from onnx import helper
p=Path.home()/'fiber-web-export/browser'/os.environ.get('FIBER_MODEL','v24')
m=onnx.load(p/'planner.w8.onnx');m=onnx.shape_inference.infer_shapes(m)
info={v.name:v for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)}
deps={v.name:{v.name} for v in m.graph.input}
for n in m.graph.node:
 d=set().union(*(deps.get(i,set()) for i in n.input))
 for o in n.output:deps[o]=d
vary={'trajectory','hand_trajectory'}
dynamic=[n for n in m.graph.node if set().union(*(deps.get(i,set()) for i in n.input))&vary]
cut=sorted({i for n in dynamic for i in n.input if deps.get(i) and not deps[i]&vary})
# A direct original input is already available to both graphs.
cut=[n for n in cut if n not in {v.name for v in m.graph.input}]
print('Cache tensors',len(cut),flush=True)
missing=[v for v in cut if v not in info or not info[v].type.tensor_type.HasField('shape')]
if missing:
 probe=copy.deepcopy(m)
 for name in missing:probe.graph.output.append(helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None))
 temp=p/'probe.onnx';onnx.save(probe,temp);del probe
 opts=ort.SessionOptions();opts.intra_op_num_threads=8
 session=ort.InferenceSession(str(temp),opts);sample=dict(np.load(p/'parity.npz'));sample={v.name:sample[v.name] for v in session.get_inputs()}
 values=session.run(missing,sample)
 for name,value in zip(missing,values):info[name]=helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,list(value.shape))
 del session;temp.unlink()
assert all(v in info for v in cut)
producer={o:n for n in m.graph.node for o in n.output}
def graph(inputs,outputs,filename):
 seen=set(inputs); selected=set();needed=set(outputs);stack=list(outputs)
 while stack:
  name=stack.pop()
  if name in seen:continue
  seen.add(name)
  if name in producer:
   node=producer[name];selected.add(node.name)
   for i in node.input:
    if i:needed.add(i);stack.append(i)
 nodes=[n for n in m.graph.node if n.name in selected]
 used={i for n in nodes for i in n.input}
 init=[v for v in m.graph.initializer if v.name in used]
 actual_inputs=[name for name in inputs if name in used or name in outputs]
 g=helper.make_graph(nodes,filename,[info[v] for v in actual_inputs],[info[v] for v in outputs],initializer=init,value_info=[v for name,v in info.items() if name in seen and name not in set(actual_inputs+outputs)])
 model=helper.make_model(g,opset_imports=m.opset_import,ir_version=m.ir_version);onnx.checker.check_model(model);onnx.save(model,p/filename)
 print(filename,len(nodes),(p/filename).stat().st_size,flush=True)
original=[v.name for v in m.graph.input]; outputs=[v.name for v in m.graph.output]
time_cut=[v for v in cut if deps[v]=={'flow_time'}]
condition_cut=[v for v in cut if v not in time_cut]
graph([v for v in original if v not in vary and v!='flow_time'],condition_cut,'planner.condition.w8.onnx')
graph(['flow_time'],time_cut,'planner.time.w8.onnx')
graph(cut+original,outputs,'planner.step.w8.onnx')
del m
opts=ort.SessionOptions();opts.intra_op_num_threads=8
s=ort.InferenceSession(str(p/'planner.w8.onnx'),opts);condition=ort.InferenceSession(str(p/'planner.condition.w8.onnx'),opts);step=ort.InferenceSession(str(p/'planner.step.w8.onnx'),opts); clock=ort.InferenceSession(str(p/'planner.time.w8.onnx'),opts)
feed=dict(np.load(p/'parity.npz'));feed={v.name:feed[v.name] for v in s.get_inputs()}
errors=[]; times=[]
for drop in [False,True]:
 feed['drop_command'][:]=drop
 started=time.perf_counter();cache=condition.run(None,{v.name:feed[v.name] for v in condition.get_inputs()});cache=dict(zip(condition_cut,cache));encode_ms=(time.perf_counter()-started)*1000
 for t in [0.,.25,.875]:
  feed['flow_time'][:]=t;ref=s.run(None,feed);cache.update(dict(zip(time_cut,clock.run(None,{'flow_time':feed['flow_time']}))))
  started=time.perf_counter();pred=step.run(None,{v.name:cache[v.name] if v.name in cache else feed[v.name] for v in step.get_inputs()});times.append((time.perf_counter()-started)*1000)
  errors.extend(float(abs(x-y).max()) for x,y in zip(ref,pred))
assert max(errors)<2e-5,errors
metrics={'cache_tensors':len(cut),'max_error':max(errors),'cpu_condition_ms':encode_ms,'cpu_step_ms':sum(times)/len(times)};(p/'split_metrics.json').write_text(json.dumps(metrics));print(metrics,flush=True)
