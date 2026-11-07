"""Redraw three panels from the public848-point CSV; no private inputs needed.

Requires matplotlib. First-word frequencies are observed accepted-answer
frequencies, not model probabilities. Primary accuracy is a separate population.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math

ARMS=('baseline','full_tis','negative_tis','abs_gap_mask','negative_down_mask','repeat_negative_down_mask','negative_mass','sync_fresh')
LABELS=dict(zip(ARMS,('Baseline','1 Full TIS','2 Negative TIS','3 Absolute-gap mask','4 Negative down-gap mask','5 Repeat + negative down-gap','6 Negative length weighting','7 Synchronous fresh G8')))
COLORS=dict(zip(ARMS,('#222222','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#7B3294')))
SUPPLEMENT=Path('data/final-stability-600-650/supplements')
POINTS=SUPPLEMENT/'eight_native_firstword_accuracy_points.csv'
COLUMNS=['arm','panel','round','value_fraction','numerator','denominator','population']
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_points(path,expected_sha):
    need(sha(path)==expected_sha,'Point CSV differs from reviewed snapshot')
    with path.open(newline='')as f:
        r=csv.DictReader(f);need(r.fieldnames==COLUMNS,'Unexpected point columns');rows=list(r)
    expected={(arm,panel,s)for arm in ARMS for panel in 'AB'for s in range(601,651)}
    expected|={(arm,'C',s)for arm in ARMS for s in (600,610,620,630,640,650)}
    found={};need(len(rows)==848,'All eight actual trajectories required')
    for row in rows:
        key=(row['arm'],row['panel'],int(row['round']));need(key in expected and key not in found,'Unexpected or duplicated point')
        value=float(row['value_fraction']);n=int(row['numerator']);denom=int(row['denominator'])
        need(denom==512 and 0<=n<=denom and math.isfinite(value) and math.isclose(value,n/denom,rel_tol=0,abs_tol=1e-12),'Count/frequency mismatch')
        want='primary_validation_64x8'if row['panel']=='C'else'accepted_training'
        need(row['population']==want,'Accepted training and primary evaluation must stay separate')
        found[key]={**row,'round':key[2],'value_fraction':value,'numerator':n,'denominator':denom}
    need(set(found)==expected,'Missing actual arm/round/panel')
    return list(found.values())
def render(repo_root,points_sha,output_directory=None):
    path=Path(repo_root)/POINTS;points=load_points(path,points_sha)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output_directory)if output_directory else Path(repo_root)/SUPPLEMENT
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for arm in ARMS:
        for i,panel in enumerate('ABC'):
            selected=sorted((v for v in points if v['arm']==arm and v['panel']==panel),key=lambda v:v['round'])
            axes[i].plot([v['round']for v in selected],[100*v['value_fraction']for v in selected],label=LABELS[arm],color=COLORS[arm],lw=1.4,alpha=.9,marker='o'if panel=='C'else None,markersize=3)
    titles=['A  Observed first-word Alright frequency — accepted training,512 responses/round',
            'B  Observed first-word To frequency — accepted training,512 responses/round',
            'C  Primary64×8 accuracy — separate fixed-question evaluation population']
    for ax,title in zip(axes,titles):
        ax.set_title(title,loc='left',fontsize=11);ax.set_ylabel('Responses (%)');ax.set_ylim(0,100);ax.set_xlim(599.5,650.5);ax.set_xticks([600,610,620,630,640,650]);ax.tick_params(labelbottom=True);ax.grid(True,alpha=.2)
    axes[-1].set_xlabel('Training round')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.952),ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Observed answer openings and primary accuracy: eight native trajectories',fontsize=14,y=.995)
    fig.text(.5,.032,'A/B:50 rounds per method; first-word frequencies are not model token probabilities. C:6 evaluations per method.',ha='center',fontsize=9)
    fig.text(.5,.012,'One trajectory per method; unsmoothed lines. Primary draws are not paired by per-request seeds; no secondary/candidate data.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.09,hspace=.37)
    outputs={}
    for ext in ('svg','png'):
        p=out/('eight_native_firstword_accuracy.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight');outputs[p.name]=sha(p)
    plt.close(fig);need(sha(path)==points_sha,'Point CSV changed while rendering')
    return {'points':848,'points_by_panel':{'A':400,'B':400,'C':48},'observed_frequency_not_model_probability':True,'populations_separate':True,'outputs_sha256':outputs}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--points-sha256',required=True);p.add_argument('--output-directory',type=Path);a=p.parse_args();print(json.dumps(render(a.repo_root,a.points_sha256,a.output_directory),indent=2))
