"""Reassemble the reviewed browser model set for GitHub Pages."""
import hashlib
import json
from pathlib import Path

source = Path('public-models')
destination = Path('_site/playground/models')
for entry in json.loads((source / 'manifest.json').read_text()):
    output = destination / entry['path']
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with output.open('wb') as stream:
        for part in entry['parts']:
            with (source / part).open('rb') as chunk:
                while block := chunk.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
                    stream.write(block)
    assert size == entry['size'] and digest.hexdigest() == entry['sha256'], entry['path']
print('Verified and assembled browser models.')
