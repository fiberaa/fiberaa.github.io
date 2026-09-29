"""Resize the door opening and matching panel/frame collision geometry together."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--width',type=float,default=.85);a=p.parse_args()
if not .65<=a.width<=1.2:raise ValueError('Door width must be 0.65–1.2 m')
t=ET.parse(a.folder/'scene.xml');r=t.getroot();half=a.width/2;panel=a.width-.006
changes={
 'door_panel':{'size':f'.025 {panel/2} 1.015','pos':f'0 {-panel/2} 1.015'},
 'door_wall_left':{'pos':f'1.3 {half+1.5} 1.275'},
 'door_wall_right':{'pos':f'1.3 {-half-1.5} 1.275'},
 'door_lintel':{'size':f'.05 {half} .25'},
 # A recessed door needs a stop overlapping its silhouette. The stop ends
 # at x=1.364, 1 mm before the closed panel, and opens away from it (+x).
 'door_frame_l':{'size':'.082 .04 1.025','pos':f'1.282 {half+.025} 1.025'},
 'door_frame_r':{'size':'.082 .04 1.025','pos':f'1.282 {-half-.025} 1.025'},
 'door_frame_top':{'size':f'.082 {half+.08} .055','pos':'1.282 0 2.08'},
 'door_strike_front':{'pos':f'1.414 {-half-.01} 1.95'},
}
for name,attrs in changes.items():
 g=r.find(f".//geom[@name='{name}']")
 for key,value in attrs.items():g.set(key,value)
r.find(".//body[@name='door_panel_body']").set('pos',f'1.39 {half} 0')
r.find(".//body[@name='door_handle_body']").set('pos',f'0 {-panel+.1} .95')
r.find(".//body[@name='door_bolt_body']").set('pos',f'0 {-panel} 1.95')
t.write(a.folder/'scene.xml',encoding='unicode')
cpath=a.folder/'scene.json';c=json.loads(cpath.read_text())
for v in c['visuals']:
 attrs=changes.get(v.get('name'),{})
 if 'size' in attrs:v['size']=list(map(float,attrs['size'].split()))
c.setdefault('provenance',{})['door_opening_width_m']=a.width
c['provenance']['door_frame_stop_overlap_m']=.015
cpath.write_text(json.dumps(c,separators=(',',':')))
print(f'Door opening: {a.width:.2f} m; panel: {panel:.3f} m; matched frame/walls/latch')
