"""Link the public arXiv record once the exact paper appears."""
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

TITLE = 'Action-Aligned Latent Planning for Versatile Humanoid Loco-Manipulation'
query = urllib.parse.urlencode({'search_query': 'ti:"' + TITLE + '"', 'max_results': 10})
request = urllib.request.Request('https://export.arxiv.org/api/query?' + query,
                                 headers={'User-Agent': 'FIBER-project-page/1.0'})
with urllib.request.urlopen(request, timeout=60) as response:
    feed = ET.fromstring(response.read())
ns = {'a': 'http://www.w3.org/2005/Atom'}
normalize = lambda text: re.sub(r'[^a-z0-9]', '', text.lower())
for entry in feed.findall('a:entry', ns):
    if normalize(entry.findtext('a:title', '', ns)) != normalize(TITLE):
        continue
    authors = [a.findtext('a:name', '', ns) for a in entry.findall('a:author', ns)]
    if not any('Yutang' in name and 'Lin' in name for name in authors):
        continue
    identifier = entry.findtext('a:id', '', ns).rsplit('/', 1)[-1]
    assert re.fullmatch(r'\d{4}\.\d{4,5}(v\d+)?', identifier), identifier
    url = 'https://arxiv.org/abs/' + identifier
    page = Path('index.html')
    content = page.read_text()
    content = content.replace('href="static/papers/FIBER.pdf" target="_blank" rel="noopener">Paper (PDF)</a>',
                              f'href="{url}" target="_blank" rel="noopener">Paper (arXiv)</a>')
    if 'eprint =' not in content:
        content = content.replace('  year   = {2026}', '  year   = {2026},\n  archivePrefix = {arXiv},\n  eprint = {' + re.sub(r'v\d+$', '', identifier) + '}')
    page.write_text(content)
    print('Linked', url)
    break
else:
    print('Paper not public on arXiv yet; keeping the local PDF link.')
