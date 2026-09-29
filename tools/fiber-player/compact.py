from pathlib import Path
import json,time
import numpy as np,onnxruntime as ort,onnx
from onnx import helper,numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
p=Path.home()/'fiber-web-export/browser'
opts=ort.SessionOptions();opts.intra_op_num_threads=8
rng=np.random.default_rng(17)
for name in ['planner','actor']:
    source=p/(name+'.float32.onnx' if name=='planner' else 'actor.onnx')
    # Normalize Gemm to MatMul + Add so all linear weights are compacted.
    normalized=p/f'{name}.matmul.onnx'
    model=onnx.load(str(source)); initial={v.name:v for v in model.graph.initializer}; nodes=[]
    for node in model.graph.node:
        if node.op_type!='Gemm': nodes.append(node); continue
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        assert attrs.get('alpha',1)==1 and attrs.get('beta',1)==1 and attrs.get('transA',0)==0
        weight=node.input[1]
        if attrs.get('transB',0):
            weight=weight+'_transposed'; value=numpy_helper.to_array(initial[node.input[1]]).T.copy(); model.graph.initializer.append(numpy_helper.from_array(value,weight))
        intermediate=node.output[0]+'_matmul'
        nodes.append(helper.make_node('MatMul',[node.input[0],weight],[intermediate],name=node.name+'_matmul'))
        nodes.append(helper.make_node('Add',[intermediate,node.input[2]],node.output,name=node.name+'_bias'))
    del model.graph.node[:];model.graph.node.extend(nodes)
    used={v for n in nodes for v in n.input}; keep=[v for v in model.graph.initializer if v.name in used];del model.graph.initializer[:];model.graph.initializer.extend(keep)
    onnx.save(model,str(normalized));del model
    reference=ort.InferenceSession(str(source),opts,providers=['CPUExecutionProvider'])
    feed={}
    for v in reference.get_inputs():
        shape=v.shape
        if v.type=='tensor(bool)':feed[v.name]=np.array([False])
        else:feed[v.name]=rng.standard_normal(shape).astype(np.float32)
    if name=='planner':feed['depth']=np.abs(feed['depth']);feed['flow_time'][:]=.25
    ref=reference.run(None,feed)[0]
    del reference
    for bits in ([8,4] if name=='planner' else [8,4]):
        target=p/f'{name}.w{bits}.onnx'; print('quantizing',target.name,flush=True)
        q=MatMulNBitsQuantizer(str(normalized),bits=bits,block_size=32,is_symmetric=True,accuracy_level=0)
        q.process();q.model.save_model_to_file(str(target),use_external_data_format=False);del q
        sess=ort.InferenceSession(str(target),opts,providers=['CPUExecutionProvider']);started=time.time();pred=sess.run(None,feed)[0];elapsed=time.time()-started
        metrics={'bytes':target.stat().st_size,'max_error':float(np.max(np.abs(pred-ref))),'relative_error':float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)),'seconds':elapsed}
        print(target.name,metrics,flush=True);(p/(target.name+'.json')).write_text(json.dumps(metrics));del sess
    np.savez(p/f'{name}_parity.npz',**feed,reference=ref)
print('COMPACTION COMPLETE',flush=True)
