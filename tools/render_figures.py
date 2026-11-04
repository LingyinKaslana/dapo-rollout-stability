"""Render aggregate research figures; no model, raw responses, or GPU required."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':170})
COLORS=['#285f93','#cb6841','#3a846c','#8c68a6']
manifest=[]

def evidence(identifier):
    return json.loads((ROOT/f'data/evidence/{identifier}.json').read_text())['results']

def rows(filename):
    with (ROOT/'data'/filename).open() as f:return list(csv.DictReader(f))

def save(fig,name,sources):
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    manifest.append({'figure':f'figures/{name}.png','sources':sources,
                     'source_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}})

v=rows('eight-validation.csv');t=rows('eight-training.csv')
fig,axs=plt.subplots(2,2,figsize=(11.8,7),layout='constrained')
x=[int(r['step']) for r in v]
axs[0,0].plot(x,[float(r['validation_mean_length']) for r in v],'-o',color=COLORS[0],ms=4)
axs[0,0].set(title='Fixed DeepMath validation length',ylabel='Response tokens')
axs[0,1].plot(x,[float(r['Alright_fraction']) for r in v],'-o',label='Alright',color=COLORS[1],ms=4)
axs[0,1].plot(x,[float(r['To_fraction']) for r in v],'-o',label='To',color=COLORS[0],ms=4)
axs[0,1].yaxis.set_major_formatter(PercentFormatter(1));axs[0,1].set(title='Natural first-response-token share');axs[0,1].legend()
axs[1,0].semilogy([int(r['step']) for r in t],[float(r['actor/grad_norm']) for r in t],color=COLORS[1],lw=1)
axs[1,0].axhline(1,color='#555',ls='--',lw=1,label='Native clip threshold')
axs[1,0].set(title='Training pre-clip norm: mean of 4 updates',ylabel='Gradient norm (log scale)');axs[1,0].legend(fontsize=8)
axs[1,1].plot(x,[float(r['mechanical_repeat_fraction']) for r in v],'-o',color=COLORS[2],ms=4)
axs[1,1].yaxis.set_major_formatter(PercentFormatter(1));axs[1,1].set(title='Conservative mechanical repetition rule')
for ax in axs.flat:
    ax.set_xlabel('Original full8 training round');ax.grid(alpha=.18)
    ax.axvspan(350,400,color=COLORS[0],alpha=.08);ax.axvspan(600,650,color=COLORS[1],alpha=.08)
fig.suptitle('Qwen3-8B-Base APRIL full8: distinct early and late phases\nValidation: fixed 1,021 questions x 4, T=0.6; training batches differ',fontsize=13)
save(fig,'eight-course',['data/eight-validation.csv','data/eight-training.csv'])

c=evidence('C01')['heldout'];w=evidence('C02')
fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
names=['Input delta E','Cache delta K','Cache delta V','Weight delta Wk','Weight delta Wv']
values=[c[k]['median_error_r64'] for k in ['projection_input','k_cache','v_cache']]+[w[k]['median_r64_error'] for k in ['k_proj','v_proj']]
ax.bar(names,values,color=[COLORS[0]]*3+[COLORS[1]]*2)
for i,val in enumerate(values):ax.text(i,val+.018,f'{val:.1%}',ha='center')
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set(ylim=(0,.85),ylabel='Residual / true delta norm',title='Oracle rank-64 fit leaves substantial delta residual\nQwen3-1.7B, 3 small full-parameter updates; medians')
ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.15)
save(fig,'cache-rank64',['data/evidence/C01.json','data/evidence/C02.json'])

g=evidence('G-04');arms=['eight','nodiscard44','discard44']
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,key,title,ylabel in [(axs[0],'first500_selected_response_mean_tokens','Selected training-response length','Tokens'),(axs[1],'first500_eventwall_mean_s','Training event interval','Seconds')]:
    vals=[g[a][key] for a in arms];ax.bar(arms,vals,color=COLORS[:3])
    for i,val in enumerate(vals):ax.text(i,val*1.02,f'{val:.1f}',ha='center')
    ax.set(title=title,ylabel=ylabel,ylim=(0,max(vals)*1.18));ax.grid(axis='y',alpha=.15)
fig.suptitle('First 500 rounds: different online trajectories and workloads\nNot a fixed-weight scheduler-only benchmark',fontsize=12)
save(fig,'group-workload',['data/evidence/G-04.json'])

h=rows('first-hit.csv');fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for key,label,color in [('base_coverage_fraction','Base',COLORS[0]),('step503_coverage_fraction','RL step503',COLORS[1])]:
    ax.plot([int(r['attempt']) for r in h],[float(r[key]) for r in h],label=label,color=color,lw=2)
ax.set(xscale='log',xlim=(1,100),ylim=(.5,1),xlabel='Maximum attempts (log scale)',ylabel='Questions solved at least once',title='First-correct coverage on 1,621 existing validation questions\nT=0.6, cap 8,192; stop at first correct or attempt 100')
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();ax.grid(alpha=.2)
save(fig,'first-hit',['data/first-hit.csv'])

e=evidence('S02')['natural_lengths_four_arms'];names=['control','firsttokenzero','randomzero','NLLmatchedzero']
labels=['Original','First-token loss=0','Random token loss=0','NLL-matched loss=0']
fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for ax,key,title,ylabel in [(axs[0],'mean_length','Endpoint natural response length','Tokens'),(axs[1],'Alright','Endpoint Alright count','Responses out of 256')]:
    vals=[e[n][key] for n in names];ax.bar(range(4),vals,color=COLORS)
    ax.set_xticks(range(4),labels,rotation=18,ha='right',fontsize=8)
    for i,val in enumerate(vals):ax.text(i,val+max(vals)*.02,f'{val:.0f}',ha='center')
    ax.set(title=title,ylabel=ylabel,ylim=(0,max(vals)*1.18));ax.grid(axis='y',alpha=.15)
fig.suptitle('Early fixed-history replay: targeted first-token intervention\n64 fixed questions x 4 at evaluation; not online matched trajectories',fontsize=12)
save(fig,'early-first-token',['data/evidence/S02.json'])

g=evidence('S10')['fixed_history_501_520']['arms']
fig,ax=plt.subplots(figsize=(8.5,4.6),layout='constrained')
names=['Original PPO','Token TIS','Mean weight / update','Within-response shuffle']
vals=[r['max_preclip_norm'] for r in g]
ax.bar(names,vals,color=COLORS);ax.set_yscale('log')
for i,val in enumerate(vals):ax.text(i,val*1.18,f'{val:.2f}',ha='center')
ax.set(ylim=(1,5000),ylabel='Maximum pre-clip norm (log scale)',title='Same historical batches: 20 rounds / 80 optimizer updates\nSame optimizer start; each arm recomputes its own p0')
ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.15)
save(fig,'gradient-weight-controls',['data/evidence/S10.json'])

b=evidence('S06')['by_length'];fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
names=['<1,024','1,024–2,047','2,048–4,095','4,096–8,192']
vals=[r['confirmed_repetition_rate'] for r in b]
ax.bar(names,vals,color=COLORS[0])
for i,(val,row) in enumerate(zip(vals,b)):ax.text(i,val+.008,f'{val:.1%}\nn={row["n"]:,}',ha='center',fontsize=9)
ax.set(ylim=(0,.37),ylabel='Repetition label fraction',xlabel='Generated response length (tokens)',title='Base model: repetition labels increase across length bins\n10,000 questions, one draw each; labels are AI-assisted, not human gold')
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.15)
save(fig,'base-length-repetition',['data/evidence/S06.json'])
(ROOT/'data/generated-figures.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Rendered {len(manifest)} figures in PNG and SVG.')
