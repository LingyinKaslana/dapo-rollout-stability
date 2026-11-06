"""CPU text-only repetition masks for frozen-loss diagnostics.

Primary: every valid continuing token AFTER its own prefix has confirmed a v2
cycle. All episodes, no64-position limit, no vocabulary/family construction.
Retrospective: all valid fully-contained tokens of a sufficiently long completed
cycle, including its unconfirmed beginning. Explicitly uses future information;
never substitute it for the primary prefix-confirmed mask in online conclusions.
"""
from __future__ import annotations
import numpy as np
from repeat_targets_v2 import (VERSION,MAX_PERIOD,MIN_CHARS,MIN_CYCLES,
    NONALPHABETIC_MIN_CHARS,NONALPHABETIC_MIN_VISIBLE_TOKENS,
    normalize,period_candidates,protection_reasons,decode)

MASK_VERSION='all-episodes-prefix-confirmed-v2-thresholds-v1'


def special_ids(tokenizer):
    values=set(map(int,getattr(tokenizer,'all_special_ids',[])))
    for name in ('eos_token_id','bos_token_id','pad_token_id','unk_token_id'):
        x=getattr(tokenizer,name,None)
        if x is not None:values.update(map(int,x if isinstance(x,(list,tuple))else[x]))
    return values


def build_repetition_masks(response_ids,tokenizer,prompt_text=''):
    """Return numpy bool masks in original response-target coordinates [0,N).

    Inputs must be original valid response IDs, excluding padding via the native
    response mask; never re-tokenize visible text to manufacture alignment.
    No p/q/advantage is read by the detector. Prompt protection uses the original
    user question and PRE-cycle context. The actual loss gate is separate.
    """
    ids=list(map(int,response_ids));n=len(ids);text=decode(tokenizer,ids)
    norm,candidates=period_candidates(text);special=special_ids(tokenizer)
    fragments={i:decode(tokenizer,[i])for i in set(ids)}
    pieces=[fragments[i]for i in ids];normalized=[normalize(p)for p in pieces]
    fast=''.join(pieces)==text
    offsets=np.r_[0,np.cumsum([len(p)for p in normalized])]
    cache={0:'',n:norm}
    def before(t):
        if fast:return norm[:int(offsets[t])]
        if t not in cache:cache[t]=normalize(decode(tokenizer,ids[:t]))
        return cache[t]
    if fast:assert offsets[-1]==len(norm)
    # Boundaries are actual prefix decodes. A byte-piece whose standalone decode
    # differs from its contextual addition is excluded, never blurred across IDs.
    boundary_checks=[];valid_target=[];visible=[]
    for t in range(n):
        if fast:
            # Exact concatenation + characterwise normalization proves stable
            # offsets without quadratic whole-prefix string copies.
            L,U=int(offsets[t]),int(offsets[t+1]);stable=True;addition=normalized[t]
        else:
            left,right=before(t),before(t+1);L=len(left);U=len(right)
            stable=left==norm[:L]and right==norm[:U]and right.startswith(left)
            addition=right[L:]if stable else''
        piece_exact=stable and normalized[t]==addition
        target_ok=piece_exact and bool(addition)and ids[t]not in special and '\ufffd'not in fragments[ids[t]]
        boundary_checks.append((stable,L,U,addition));valid_target.append(target_ok)
        visible.append(stable and U>L)
    confirmed=np.zeros(n,dtype=bool);retro=np.zeros(n,dtype=bool);episodes=[]
    for candidate in candidates:
        cycle=dict(candidate);lo=int(cycle['norm_start']);hi=int(cycle['norm_end_exclusive']);pattern=cycle['pattern'];p=int(cycle['period'])
        reasons=protection_reasons(prompt_text,text[:cycle['raw_char_start']],cycle['category'])
        cycle['protected_reasons']=reasons;cycle['confirmed_positions']=[];cycle['retrospective_positions']=[]
        cycle['retrospective_uses_future_confirmation']=True
        if reasons:
            cycle['mask_exclusion']='protected_mathematical_or_requested_output';episodes.append(cycle);continue
        if not cycle['eligible_by_text_threshold']:
            cycle['mask_exclusion']='below_v2_text_threshold';episodes.append(cycle);continue
        # Count only fully-contained, visible, stable original response tokens.
        fully_visible=[t for t,(stable,L,U,add)in enumerate(boundary_checks)
                       if stable and U>L and L>=lo and U<=hi]
        total_visible=len(fully_visible)
        cycle['total_visible_tokens_in_completed_episode']=total_visible
        min_chars=int(cycle['required_prefix_characters']);min_tokens=int(cycle['required_prefix_visible_tokens'])
        full_episode_qualifies=(hi-lo>=min_chars and total_visible>=min_tokens)
        seen_visible=0
        for t,(stable,L,U,addition)in enumerate(boundary_checks):
            if not stable or L<lo or U>hi:continue
            previous_visible=seen_visible
            if visible[t]:seen_visible+=1
            if not valid_target[t]:continue
            phase=(L-lo)%p
            if not all(c==pattern[(phase+j)%p]for j,c in enumerate(addition)):continue
            if full_episode_qualifies:
                retro[t]=True;cycle['retrospective_positions'].append(t)
            # Confirmation excludes the current target and every future token.
            if L-lo>=min_chars and previous_visible>=min_tokens:
                confirmed[t]=True;cycle['confirmed_positions'].append(t)
        cycle['eligible_for_primary_mask']=bool(cycle['confirmed_positions'])
        cycle['mask_exclusion']=None if cycle['confirmed_positions']else'insufficient_prior_confirmation_or_no_valid_continuation'
        episodes.append(cycle)
    assert not (confirmed&~retro).any()
    assert all(ids[t]not in special and normalized[t]and '\ufffd'not in fragments[ids[t]]for t in np.flatnonzero(confirmed|retro))
    return {'version':MASK_VERSION,'source_detector_version':VERSION,
        'confirmed_mask':confirmed,'retrospective_mask':retro,'episodes':episodes,
        'audit':{'response_tokens':n,'normalized_characters':len(norm),'episodes':len(episodes),
            'confirmed_tokens':int(confirmed.sum()),'retrospective_tokens':int(retro.sum()),
            'decoder_alignment':'exact_concatenated_token_decodes'if fast else'checked_original_prefix_decodes',
            'unstable_or_nonexact_visible_token_boundaries':sum(bool(normalized[t])and not valid_target[t]and ids[t]not in special for t in range(n)),
            'primary_confirmation_excludes_current_and_future_tokens':True,
            'retrospective_uses_future_confirmation':True,'retrospective_boundary_mapping':'fully-contained visible original tokens only; boundary-straddling tokens excluded',
            'all_episodes':True,'target_count_cap':None,'vocabulary_family_scan':False,
            'classification':'mechanical loop candidate positions, not semantic abnormality verdict',
            'thresholds':{'maximum_period':MAX_PERIOD,'alphabetic_characters':MIN_CHARS,'cycles':MIN_CYCLES,
                'numeric_symbol_characters':NONALPHABETIC_MIN_CHARS,'numeric_symbol_visible_tokens':NONALPHABETIC_MIN_VISIBLE_TOKENS}}}


def negative_down_gate(logp,logq,advantages,valid_mask,repeat_mask,threshold=.5):
    """Detached CPU gate; does not change native A, probabilities or denominator."""
    p,q,a,v,r=map(np.asarray,(logp,logq,advantages,valid_mask,repeat_mask))
    assert p.shape==q.shape==a.shape==v.shape==r.shape
    assert threshold==.5 and np.isfinite(p[v.astype(bool)]).all()and np.isfinite(q[v.astype(bool)]).all()
    return v.astype(bool)&r.astype(bool)&(a<0)&((p.astype(np.float64)-q.astype(np.float64))<-.5)
