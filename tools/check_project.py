"""Check project references and aggregate identities using the standard library."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--allow-missing-legacy',action='store_true',help='For a text-only local preview; full-repository verification omits this option.')
args=parser.parse_args()
errors=[]
catalog=json.loads((ROOT/'data/catalog.json').read_text())
legacy=json.loads((ROOT/'data/legacy-manifest.json').read_text())['preserved_git_blobs']
ids=[x['id'] for x in catalog['entries']]
if len(ids)!=len(set(ids)) or len(ids)!=catalog['count']:errors.append('Catalog count or uniqueness mismatch')
for entry in catalog['entries']:
    p=ROOT/f'data/evidence/{entry["id"]}.json'
    d=json.loads(p.read_text())
    if d['id']!=entry['id'] or not d['source_fingerprints']:errors.append(f'Missing identity/provenance: {p.name}')
    for source in d['source_fingerprints']:
        if not re.fullmatch('[0-9a-f]{64}',source['sha256']):errors.append(f'Invalid source SHA256: {p.name}')
    if d['status']!=entry['status']:errors.append(f'Status mismatch: {p.name}')

skip_legacy=set(legacy)
documents=[p for p in ROOT.rglob('*.md') if '.git' not in p.parts and str(p.relative_to(ROOT)) not in skip_legacy and p.name!='legacy-dashboard.md']
for p in documents:
    text=p.read_text()
    # Ignore fenced code: placeholders in the template are not file references.
    text=re.sub(r'```.*?```','',text,flags=re.S)
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        parsed=urlsplit(link)
        if parsed.scheme or link.startswith('#'):continue
        target=(p.parent/unquote(parsed.path)).resolve()
        try:relative=str(target.relative_to(ROOT))
        except ValueError:errors.append(f'Escaping project link: {p.relative_to(ROOT)} -> {link}');continue
        known_legacy=relative in legacy or any(k.startswith(relative.rstrip('/')+'/') for k in legacy)
        if not target.exists() and not (args.allow_missing_legacy and known_legacy):
            errors.append(f'Missing link: {p.relative_to(ROOT)} -> {link}')
        if target.name=='evidence.md' and parsed.fragment and parsed.fragment.upper() not in ids:
            errors.append(f'Unknown evidence anchor: {link}')
    if re.search(r'\[(?:C\d{2}|S\d{2}|G-\d{2})(?:[、,，\s]+(?:C\d{2}|S\d{2}|G-\d{2}))*\](?!\()',text):
        errors.append(f'Unresolved evidence reference: {p.relative_to(ROOT)}')

for p in [*documents,*ROOT.glob('data/**/*.json'),*ROOT.glob('progress/*.json')]:
    text=p.read_text()
    if any(bad in text for bad in ['/localhome/','/raid/','BEGIN PRIVATE KEY','BEGIN OPENSSH PRIVATE KEY']):
        errors.append(f'Unexpected private path/key text: {p.relative_to(ROOT)}')

with (ROOT/'data/first-hit.csv').open() as f:rows=list(csv.DictReader(f))
if [int(r['attempt']) for r in rows]!=list(range(1,101)):errors.append('First-hit attempts are not exactly 1..100')
for key in ['base_coverage_fraction','step503_coverage_fraction']:
    vals=[float(r[key]) for r in rows]
    if any(not 0<=v<=1 for v in vals) or vals!=sorted(vals):errors.append(f'Invalid monotone coverage: {key}')
for e in json.loads((ROOT/'data/generated-figures.json').read_text()):
    if not (ROOT/e['figure']).is_file():errors.append(f'Missing figure: {e["figure"]}')
    for name,digest in e['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:errors.append(f'Figure source changed: {name}')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'PASS: {len(ids)} evidence units, {len(documents)} project documents, links, provenance, aggregate coverage, figure source hashes.')
if args.allow_missing_legacy:print('Local preview: legacy links checked against preserved Git-tree manifest; run again without this option in the full repository.')
