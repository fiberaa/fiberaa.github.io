"""Restore native box contact dimensionality lost in the MJCF export.

Run from the website root. Preserve sliding friction and every other setting.
The higher-priority box must explicitly use condim=4 for torsional friction.
"""
from pathlib import Path
import re

for folder in ('pickup', 'door'):
    for path in Path('playground/models/scenes', folder).rglob('*.xml'):
        source = path.read_text()
        def repair(match):
            tag = match.group(0)
            if 'name="pickup_object_box"' not in tag:
                return tag
            if re.search(r'\bcondim="[^"]*"', tag):
                return re.sub(r'\bcondim="[^"]*"', 'condim="4"', tag)
            return tag.replace('<geom ', '<geom condim="4" ', 1)
        updated = re.sub(r'<geom\b[^>]*>', repair, source)
        if updated != source:
            path.write_text(updated)
            print(path)
