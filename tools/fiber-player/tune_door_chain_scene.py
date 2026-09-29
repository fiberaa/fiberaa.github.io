"""Match the chain box to the standalone demo and give the handle a left-hand approach.

Run from the website root. Only reset-time scene geometry is changed.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

folder = Path('playground/models/scenes/door')
path = folder / 'scene.json'
cfg = json.loads(path.read_text())
cfg['qpos'][:2] = [.15, -.6]
cfg['door_exit_x'] = 2.1
cfg['pickup_settle_seconds'] = 3
cfg['sampler_noise'] = 'door-student-1'
cfg['command_noise'] = {'pickup': 'pickup', 'carry_locomotion': 'pickup'}
address = cfg['box_qadr']
cfg['qpos'][address:address + 7] = [3.6, -.15, 0, 0.9975510002532796, 0, 0, 0.06994284733753277]
cfg['mocap_pos'][:3] = [0, 0, -5]
tree = ET.parse(folder / 'scene.xml')
tree.find(".//geom[@name='pickup_support']").set('size', '.25 .30 .025')
box = tree.find(".//geom[@name='pickup_object_box']")
box.set('size', '.12 .16 .17')
box.set('pos', '0 0 .17')
box.set('mass', '.5')
box.set('condim', '4')
box.set('friction', '1.2 .2 .0001')
for visual in cfg['visuals']:
    if visual['name'] == 'pickup_support':
        visual['size'] = [.25, .30, .025]
    if visual['name'] == 'pickup_object_box':
        visual['size'] = [.12, .16, .17]
cfg['provenance']['chain_scene'] = {
    'start_xy_m': [.15, -.6],
    'box_xy_m': [3.6, -.15],
    'box_dimensions_m': [.24, .32, .34],
    'box_mass_kg': .5,
    'support_dimensions_m': [.5, .6, .05],
    'support_top_m': None,
    'note': 'Handle is left of the approach; box is visible beyond the exit. No waypoint for either interaction.',
}
tree.write(folder / 'scene.xml', encoding='unicode')
path.write_text(json.dumps(cfg, separators=(',', ':')))
