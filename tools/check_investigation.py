"""Validate published report data and reference-source identity (no model needed)."""
import csv
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/investigation'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(D/'manifest.json')
for x in manifest:
    assert sha(ROOT/x['file'])==x['sha256'],x['file']
    assert re.fullmatch('[a-f0-9]{64}',x['archived_source_sha256'])
index=read(D/'index.json');ids=[x['id']for x in index['claims']]
assert len(ids)==len(set(ids))==index['count']==52
for x in index['claims']:
    source=read(ROOT/x['file']);claims=source.get('claims',source.get('evidence'))
    assert sum(c['id']==x['id']for c in claims)==1,x['id']
for x in read(D/'figures.json'):
    assert (ROOT/x['figure']).is_file(),x['figure']
    for name,digest in x['source_sha256'].items():assert sha(ROOT/name)==digest,name
rows=list(csv.DictReader((D/'gap_gradient.csv').open()))
assert [int(r['step'])for r in rows]==list(range(651,851))
assert all(int(r['N'])==512 and r['finite_status']=='finite'for r in rows)
for r in rows:
    assert abs(float(r['exceed_count'])/float(r['D'])-float(r['exceed_fraction']))<1e-12
    norms=[float(r[f'grad_norm_B{i}'])for i in range(1,5)]
    assert abs(sum(norms)/4-float(r['grad_norm_round_mean']))<1e-8
    assert abs(max(norms)-float(r['grad_norm_inner_max']))<1e-8
short=read(D/'preventive20.json')
assert [x['step']for x in short['evaluations']]==[600,610,620]
for x in short['evaluations']:
    assert x['questions']==64 and x['responses']==512
    for k,v in x['counts'].items():assert abs(v/512-x['fractions'][k])<1e-12
assert short['auxiliary']['common_rank_updates']==80
assert short['auxiliary']['aux_nonzero_trajectory_uses']==303
assert short['auxiliary']['aux_unique_response_uids']==76
assert short['auxiliary']['aux_target_token_uses']==9032
for p in read(D/'learned_priority_endpoints.json')['priority_points']:
    assert p['responses']==512 and p['questions']==64
    assert p['correct_responses']/512==p['metrics']['accuracy']
    for k,v in p['response_counts'].items():assert v/512==p['metrics'][k]
ref=ROOT/'reference/antiloop'
for x in read(ref/'source_manifest.json')['files']:
    assert sha(ref/x['published_file'])==x['sha256'],x['published_file']
# Scan only this report release; legacy records are preserved separately.
release=[*D.rglob('*'),*ref.rglob('*'),*ROOT.glob('reports/*.md'),ROOT/'tools/render_investigation_figures.py']
for p in release:
    if not p.is_file()or p.suffix not in {'.py','.md','.json','.csv','.txt'}:continue
    t=p.read_text()
    assert not any(v in t for v in ['/localhome/','/raid/','/tmp/','BEGIN PRIVATE KEY','BEGIN OPENSSH PRIVATE KEY']),p
    assert not re.search(r'\b(?:10|192\.168)\.(?:\d{1,3}\.){1,2}\d{1,3}\b',t),p
print('PASS: 52 report claims, source manifests, 200-round identities, endpoint counts, 3 figure inputs, reference hashes and publication scan.')
