"""Export reproducible Gaussian sampler inputs, never actions or planned motions."""
from pathlib import Path
import torch,numpy as np,json
root=Path.home()/'fiber-web-export/browser';out=root/'noise';out.mkdir(exist_ok=True);config={}
for task,seed in [('pickup',42),('door',42),('grasp',43),('grasp-14',14),('grasp-4',4),('grasp-13',13),('grasp-15',15),('grasp-9',9),('grasp-21',21),('grasp-37',37),('grasp-52',52)]:
 torch.manual_seed(seed);rows=[]
 for _ in range(100):
  body=torch.randn((1,10,512),device='cuda').cpu().numpy().reshape(-1)
  hand=torch.randn((1,10,14),device='cuda').cpu().numpy().reshape(-1) if task.startswith('grasp') else np.zeros(140,np.float32)
  rows.append(np.concatenate([body,hand]))
 np.asarray(rows,dtype='<f4').tofile(out/f'{task}.f32');config[task]={'seed':seed,'plans':100,'body_width':5120,'hand_width':140}
(out/'meta.json').write_text(json.dumps(config));print('SAMPLER NOISE EXPORTED')
