"""Stable FP32 family unlikelihood, exact packed-target alignment, grad budget.

Independent auxiliary branch only. Does not modify RL advantages, p0 or D.
"""
import math
import torch
import torch.nn.functional as F


def family_unlikelihood(logits,bad_token_ids):
    """-log(1-P_bad)=softplus(LSE_bad-LSE_good), no probability subtraction."""
    assert logits.ndim==2
    ids=sorted(set(map(int,bad_token_ids)))
    assert ids and min(ids)>=0 and max(ids)<logits.shape[-1] and len(ids)<logits.shape[-1]
    z=logits.float()
    idx=torch.tensor(ids,device=z.device,dtype=torch.long)
    bad_values=z.index_select(-1,idx)
    good=z.clone().index_fill(-1,idx,-float('inf'))
    # Center EACH partition before LSE so backward does not subtract a rounded
    # large (e.g.1000) log-normalizer from individual logits. Detached shifts
    # preserve the exact analytic derivative and avoid numerical gradient drift.
    bad_shift=bad_values.detach().amax(-1)
    good_shift=good.detach().amax(-1)
    assert torch.isfinite(bad_shift).all()and torch.isfinite(good_shift).all()
    bad_center=torch.logsumexp(bad_values-bad_shift[:,None],dim=-1)
    good_center=torch.logsumexp(good-good_shift[:,None],dim=-1)
    difference=(bad_shift-good_shift)+(bad_center-good_center)
    loss=F.softplus(difference)
    return loss,torch.sigmoid(difference).detach()


def packed_target_map(full_ids,offsets,responses,response_mask,ul_target_mask,ul_family_index,families):
    """Sparse query locations at offset+prompt_len+response_target_index-1."""
    assert responses.shape==response_mask.shape==ul_target_mask.shape==ul_family_index.shape
    assert ul_target_mask.dtype==torch.bool
    assert not (ul_target_mask&~response_mask.bool()).any()
    assert (ul_family_index[~ul_target_mask]==-1).all()
    assert responses.shape[0]+1==len(offsets)
    out=[]
    for row in range(responses.shape[0]):
        valid=int(response_mask[row].sum());lo,hi=int(offsets[row]),int(offsets[row+1]);P=hi-lo-valid
        assert P>0 and valid>0
        assert response_mask[row,:valid].all()and not response_mask[row,valid:].any()
        chosen=ul_target_mask[row].nonzero().flatten().tolist();assert len(chosen)<=64
        for t in chosen:
            fi=int(ul_family_index[row,t]);assert 0<=fi<len(families)
            target=int(responses[row,t]);query=lo+P+int(t)-1
            assert lo<=query<hi-1 and int(full_ids[query+1])==target
            assert target in families[fi], 'Observed repeated token must belong to declared bad family'
            out.append((query,fi,target,row,int(t)))
    return out


def sparse_family_ul(student_logits,data,families,*,padding_tokens=0):
    """Return full packed query-length vector with nonzero loss only at masks."""
    assert student_logits.ndim==3 and student_logits.shape[0]==1
    ids=data['input_ids'];assert ids.is_nested
    offsets=ids.offsets().detach().cpu().tolist()
    fields=data.select('responses','response_mask','ul_target_mask','ul_family_index').to_padded_tensor()
    mappings=packed_target_map(ids.values().detach().cpu(),offsets,fields['responses'].detach().cpu(),fields['response_mask'].detach().cpu(),
                              fields['ul_target_mask'].detach().cpu(),fields['ul_family_index'].detach().cpu(),families)
    assert offsets[-1]+int(padding_tokens)==student_logits.shape[1], 'Packed/SP alignment mismatch'
    z=student_logits.squeeze(0)
    # Connected zero path ensures ranks with empty masks still join FSDP backward.
    values=z[:,0].float()*0.
    masses=[]
    for fi in sorted({x[1]for x in mappings}):
        query=[x[0]for x in mappings if x[1]==fi]
        qi=torch.tensor(query,device=z.device,dtype=torch.long)
        loss,mass=family_unlikelihood(z.index_select(0,qi),families[fi])
        values=values.index_copy(0,qi,loss)
        masses.append(mass)
    sums={'mask_tokens':len(mappings),'sum_family_probability':float(torch.cat(masses).double().sum())if masses else 0.,
          'sum_UL':float(values.detach().double().sum()),'packed_tokens':int(z.shape[0])}
    return values.unsqueeze(0),sums


def gradient_scale(ul_norm,lambda_value,cap):
    assert math.isfinite(ul_norm)and ul_norm>=0 and math.isfinite(lambda_value)and lambda_value>=0
    assert math.isfinite(cap)and cap>=0
    if lambda_value==0 or ul_norm==0 or cap==0:return 0.
    return min(lambda_value,cap/ul_norm)


def gradient_budget(rl_norm,ul_norm,lambda_value,absolute_cap,relative_cap_fraction=None):
    """Detached scalar budget; legacy absolute mode unless fraction is explicit.

    Relative mode has no floor: cap=eta*||g_RL||, so zero RL implies zero UL.
    lambda_value remains an upper bound even when ||g_UL|| is very small.
    """
    assert math.isfinite(rl_norm)and rl_norm>=0
    assert math.isfinite(absolute_cap)and absolute_cap>=0
    fraction=None if relative_cap_fraction is None else float(relative_cap_fraction)
    if fraction is not None:assert math.isfinite(fraction)and 0<=fraction<1
    cap=absolute_cap if fraction is None else fraction*rl_norm
    scale=gradient_scale(ul_norm,lambda_value,cap)
    return {'scale':scale,'effective_cap':cap,'relative_fraction':fraction,
            'configured_absolute_cap':absolute_cap,
            'gradient_budget_mode':'absolute'if fraction is None else'relative_to_RL',
            'ideal_effective_UL_norm':scale*ul_norm}


def local_gradient_geometry(rl,ul):
    """Local flat-shard FP64 sums. Caller all-reduces once across eight ranks."""
    assert rl.keys()==ul.keys()and rl
    first=next(iter(rl.values()));s=torch.zeros(3,device=first.device,dtype=torch.float64)
    for key in rl:
        a,b=rl[key],ul[key];assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
        assert torch.isfinite(a).all()and torch.isfinite(b).all()
        # Chunk to avoid a whole additional FP64 copy of each model shard.
        for x,y in zip(a.flatten().split(1<<20),b.flatten().split(1<<20),strict=True):
            xd,yd=x.double(),y.double();s[0]+=xd.square().sum();s[1]+=yd.square().sum();s[2]+=(xd*yd).sum()
    return s
