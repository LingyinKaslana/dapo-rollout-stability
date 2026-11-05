"""Rebuild the report figures from published aggregate results (no GPU)."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/investigation'
OUT=ROOT/'figures'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
manifest=[]
def save(fig,name,sources):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)
    for ext in ['png','svg']:
        manifest.append({'figure':'figures/'+name+'.'+ext,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}})

source=DATA/'gap_gradient.csv'
with source.open()as f: rows=list(csv.DictReader(f))
x=np.array([int(r['step'])for r in rows])
def col(k):return np.array([float(r[k])for r in rows])
f,ax=plt.subplots(3,1,figsize=(13,8),sharex=True)
ax[0].plot(x,col('grad_norm_round_mean'),color='#993a32',label='Mean of 4 optimizer updates')
ax[0].plot(x,col('grad_norm_inner_max'),color='#993a32',alpha=.35,lw=.8,label='Maximum update')
ax[0].set_yscale('log');ax[0].set_ylabel('Pre-clip gradient norm');ax[0].legend(loc='upper right')
ax[1].plot(x,col('conditional_mean_d_gt05'),color='#21669d');ax[1].set_ylabel('Mean d given d > 0.5\n(nats)')
ax[2].plot(x,col('exceed_fraction')*100,color='#227957');ax[2].set_ylabel('Tokens with d > 0.5 (%)');ax[2].set_xlabel('Training round: resumed full8 branch')
for a in ax:a.grid(alpha=.2)
f.suptitle('Training-token mismatch and gradient: 650-to-850 continuation\nd uses round-start p0; gradients use successive parameter states',fontsize=13)
f.tight_layout(rect=(0,0,1,.94));save(f,'investigation-gap-gradient',[source])

source=DATA/'learned_priority_endpoints.json'
d=json.loads(source.read_text());pts=d['priority_points']
labels=['Source 720','Baseline 740','UL keep 740','UL withdraw 740']
colors=['#7a818a','#3476a5','#bd5a4a','#9b7aa4']
f,ax=plt.subplots(1,3,figsize=(14,4.7))
for a,key,title,mult in zip(ax,['accuracy','normalized_text_cycle128_any','mean_length'],['Accuracy (%)','All-type cycle candidates (%)','Mean response tokens'],[100,100,1]):
 vals=[p['metrics'][key]*mult for p in pts]; bars=a.bar(range(4),vals,color=colors)
 a.bar_label(bars,fmt='%.1f',padding=3);a.set_xticks(range(4),labels,rotation=20,ha='right');a.set_title(title);a.set_ylim(0,max(vals)*1.22);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
f.suptitle('Fixed E-only UL: priority endpoints, same 64 questions x 8 samples\nOrdinary decoding; one training trajectory per branch',fontsize=13)
f.tight_layout(rect=(0,0,1,.87));save(f,'investigation-ul-endpoints',[source])

source=DATA/'preventive20.json';d=json.loads(source.read_text());pts=d['evaluations'];x=[r['step']for r in pts]
f,ax=plt.subplots(1,3,figsize=(13,4.5))
series=[([r['accuracy']*100 for r in pts],'Accuracy (%)','#3476a5'),([r['fractions']['normalized_text_cycle128_any']*100 for r in pts],'All-type cycle candidates (%)','#bd5a4a'),([r['mean_length']for r in pts],'Mean response tokens','#227957')]
for a,(vals,title,color)in zip(ax,series):
 a.plot(x,vals,'o-',color=color,lw=2);a.set_xticks(x);a.set_xlabel('Checkpoint');a.set_title(title);a.grid(alpha=.2)
 for xx,yy in zip(x,vals):a.annotate(f'{yy:.1f}',(xx,yy),xytext=(0,8),textcoords='offset points',ha='center')
 a.margins(x=.12,y=.3)
f.suptitle('600-to-620 online-bank UL pilot: same 64 questions x 8 samples\nSingle arm; no matched baseline; lines connect evaluation checkpoints only',fontsize=13)
f.tight_layout(rect=(0,0,1,.86));save(f,'investigation-preventive20',[source])
(ROOT/'data/investigation/figures.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Rebuilt 3 figures in PNG and SVG from published aggregates.')
