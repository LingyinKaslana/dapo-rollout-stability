"""Redraw the six-method native-training supplement from its public long CSV.

Requires Python and matplotlib. No private paths, raw responses, remote logs,
training framework or model files are used. The published CSV is an immutable
3,600-point snapshot; its hash, coordinates and denominator labels are checked.
Image byte identity can vary with matplotlib/fonts; numeric points do not.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS=('baseline','full_tis','negative_tis','abs_gap_mask','negative_down_mask','repeat_negative_down_mask')
LABELS=dict(zip(ARMS,('Baseline','1 Full TIS','2 Negative TIS','3 Absolute-gap mask','4 Negative down-gap mask','5 Repeat + negative down-gap')))
COLORS=dict(zip(ARMS,('#222222','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00')))
SUPPLEMENT=Path('data/final-stability-600-650/supplements')
POINTS=SUPPLEMENT/'six_native_plot_points.csv'
# The actual six-method CSV hash is supplied only after completed data exists.
COLUMNS=['arm','panel','round','inner','x','value','population','denominator_type','denominator']


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_points(path,expected_sha256):
    require(digest(path)==expected_sha256,'CSV differs from the published immutable snapshot')
    with path.open(newline='')as stream:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==COLUMNS,'Unexpected CSV columns')
        raw=list(reader)
    require(len(raw)==3600,'Expected exactly 3,600 plotted points')
    expected={(arm,panel,step,inner)for arm in ARMS for panel in 'AB'
              for step in range(601,651)for inner in range(1,5)}
    expected|={(arm,panel,step,None)for arm in ARMS for panel in 'CDEF'for step in range(601,651)}
    found={}
    for row in raw:
        arm,panel=row['arm'],row['panel']
        require(arm in ARMS and panel in 'ABCDEF'and len(panel)==1,'Unknown arm/panel')
        step=int(row['round']);inner=int(row['inner'])if row['inner']else None
        key=(arm,panel,step,inner)
        require(key in expected and key not in found,'Duplicate or unexpected round/update coordinate')
        require(row['population']=='accepted_training','Population must be accepted_training')
        x,value=float(row['x']),float(row['value'])
        require(math.isfinite(x)and math.isfinite(value)and value>=0,'Invalid plotting number')
        want_x=step+(inner-1)/4 if panel in 'AB'else step
        require(x==want_x,'Fractional update position differs')
        denominator=int(row['denominator'])if row['denominator']else None
        want_type={'A':'parameter_global_norm','B':'tail_tokens','C':'responses','D':'responses','E':'responses','F':'tokens'}[panel]
        require(row['denominator_type']==want_type,'Wrong denominator scope')
        if panel=='A':require(denominator is None,'Parameter norm has no response/token denominator')
        elif panel=='B':
            require(denominator is not None and denominator>=0,'Invalid tail count')
            require(value==0 if denominator==0 else value>.5,'Tail mean/empty-tail convention differs')
        elif panel in 'CDE':
            require(denominator==512,'Each round contains 512 accepted responses')
            if panel in 'CD':
                require(value<=100,'Rate outside0–100%')
                require(abs(value*512/100-round(value*512/100))<1e-7,'Rate does not represent an integer response count')
        else:require(denominator is not None and denominator>0,'Invalid token denominator')
        found[key]={**row,'round':step,'inner':inner,'x':x,'value':value,'denominator':denominator}
    require(set(found)==expected,'Missing arm/panel/round/update coordinates')
    for arm in ARMS:
        for step in range(601,651):
            e,f=found[(arm,'E',step,None)],found[(arm,'F',step,None)]
            require(math.isclose(e['value']*512,f['denominator'],rel_tol=1e-12,abs_tol=1e-7),'Length/token total mismatch')
            tail=sum(found[(arm,'B',step,b)]['denominator']for b in range(1,5))
            require(tail<=f['denominator'],'Tail count exceeds all valid tokens')
    return list(found.values())


def render(repo_root,expected_sha256,output_directory=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FuncFormatter
    path=repo_root/POINTS;points=load_points(path,expected_sha256)
    output=output_directory or repo_root/SUPPLEMENT
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(3,2,figsize=(15,10.3),sharex=True);axs=axs.ravel();peaks={}
    for arm in ARMS:
        for k,panel in enumerate('ABCDEF'):
            selected=sorted((r for r in points if r['arm']==arm and r['panel']==panel),key=lambda r:r['x'])
            xs=[r['x']for r in selected];ys=[r['value']for r in selected]
            axs[k].plot(xs,ys,color=COLORS[arm],label=LABELS[arm],lw=1.1 if k<2 else 1.45,
                        alpha=.9,zorder=3 if arm=='repeat_negative_down_mask' else 2)
            if panel=='A':peaks[arm]=max(selected,key=lambda r:r['value'])
    titles=['A  Pre-clip parameter gradient norm\nEach optimizer update (200 per method)',
            'B  Current mismatch: mean d among d > 0.5\nEach update; 0 if no token exceeds the threshold',
            'C  Repetition candidate rate (normalized-text cycle128)\nEach round; 512 accepted responses',
            'D  Garble candidate rate\nEach round; 512 accepted responses',
            'E  Mean accepted response length\nEach round; includes token-capped responses',
            'F  Historical rollout q: token-weighted mean NLL\nEach round; aggregate the four updates by token count']
    ylabels=['Global L2 norm (symlog)','Conditional mean d (nats)','Responses (%)','Responses (%)','Response tokens',r'$-\mathrm{mean}(\log q)$ (nats)']
    for k,ax in enumerate(axs):
        ax.set_title(titles[k],loc='left',fontsize=11,pad=9);ax.set_ylabel(ylabels[k]);ax.grid(True,alpha=.2,zorder=0)
        ax.set_xlim(600.7,650.95);ax.set_ylim(bottom=0);ax.set_xticks([601,610,620,630,640,650]);ax.tick_params(labelbottom=True)
        if k>=4:ax.set_xlabel('Training round (fractional positions show B1–B4 within a round)')
    axs[0].set_yscale('symlog',linthresh=1);axs[0].set_ylim(0,max(330,max(x['value']for x in peaks.values())*1.1))
    axs[0].yaxis.set_major_locator(FixedLocator([0,.5,1,10,100,300]));axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    peak_arm=max(peaks,key=lambda a:peaks[a]['value']);p=peaks[peak_arm];axs[0].annotate(f"{p['value']:.1f}",(p['x'],p['value']),xytext=(8,-15),textcoords='offset points',fontsize=9,color=COLORS[peak_arm])
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False,fontsize=10)
    fig.suptitle('Six completed native training trajectories: rounds 601–650',fontsize=16,y=.995)
    fig.text(.5,.025,r'Accepted training only. d = |log p(current actor) − log q(saved rollout)| includes masked tokens. No candidate-rollout or secondary-eval data.',ha='center',fontsize=9)
    fig.text(.5,.007,'One online trajectory per method; no multi-seed replication. Lines are unsmoothed; A/B have 200 points per method, C–F have 50.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.09,hspace=.43,wspace=.2)
    outputs={}
    for ext in('svg','png'):
        target=output/f'six_native_training.{ext}';fig.savefig(target,dpi=180,bbox_inches='tight');outputs[target.name]=digest(target)
    plt.close(fig)
    require(digest(path)==expected_sha256,'Input CSV changed during rendering')
    return {'points':3600,'unique_optimizer_updates':1200,'unique_method_rounds':300,
            'points_by_panel':{'A':1200,'B':1200,'C':300,'D':300,'E':300,'F':300},'outputs_sha256':outputs}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-directory',type=Path,help='Optional separate preview directory; default is the public supplements directory')
    parser.add_argument('--points-sha256',required=True,help='Reviewed completed six-method CSV SHA256')
    args=parser.parse_args()
    print(json.dumps(render(args.repo_root,args.points_sha256,args.output_directory),indent=2))


if __name__=='__main__':main()
