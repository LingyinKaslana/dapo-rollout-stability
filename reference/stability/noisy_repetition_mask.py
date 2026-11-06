"""Exploratory prefix-only noisy repetition, distinct from frozen exact masks.

Fixed a priori: previous512 normalized chars; both256-char halves must have
1-unique4grams/253 >=.5 (primary) or .75 (sensitivity). At least half of the
current token's newly completed4grams must have occurred>=2 times in that OLD
512-char window. No probabilities, advantages, labels or gradients are inputs.
This is a mechanical low-diversity/reuse candidate, not a semantic verdict.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
import numpy as np
from repeat_targets_v2 import normalize,normalize_with_source_offsets,protection_reasons,decode
from repetition_mask import special_ids

VERSION='exploratory-noisy4gram-two256-prefix512-v1'
HISTORY=512
HALF=256
NGRAM=4
HALF_GRAMS=253
PRIMARY_THRESHOLD=.5
STRICT_THRESHOLD=.75
CURRENT_MATCH_THRESHOLD=.5


def intervals(mask):
    d=np.diff(np.r_[False,mask,False].astype(np.int8))
    return [[int(a),int(z)]for a,z in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))]


def rolling_unique256(norm):
    """At indexL: unique4grams wholly inside norm[L-256:L], zero below256."""
    grams=[norm[i:i+4]for i in range(max(0,len(norm)-3))]
    unique=np.zeros(len(norm)+1,dtype=np.int32);counts=Counter()
    for i,g in enumerate(grams):
        counts[g]+=1
        if i>=HALF_GRAMS:
            old=grams[i-HALF_GRAMS];counts[old]-=1
            if not counts[old]:del counts[old]
        if i+4>=HALF:unique[i+4]=len(counts)
    return grams,unique


def build_noisy_repetition_masks(response_ids,tokenizer,prompt_text=''):
    ids=list(map(int,response_ids));n=len(ids);text=decode(tokenizer,ids)
    norm,raw_offsets=normalize_with_source_offsets(text);special=special_ids(tokenizer)
    fragments={i:decode(tokenizer,[i])for i in set(ids)};pieces=[fragments[i]for i in ids]
    additions=[normalize(p)for p in pieces];fast=''.join(pieces)==text
    offsets=np.r_[0,np.cumsum([len(x)for x in additions])]
    if fast:assert offsets[-1]==len(norm)
    cache={0:'',n:norm}
    def before(t):
        if t not in cache:cache[t]=normalize(decode(tokenizer,ids[:t]))
        return cache[t]
    grams,unique=rolling_unique256(norm)
    numeric=np.r_[0,np.cumsum([c.isnumeric()for c in norm])]
    alphabetic=np.r_[0,np.cumsum([c.isalpha()for c in norm])]
    primary=np.zeros(n,dtype=bool);strict=np.zeros(n,dtype=bool)
    history=Counter();old_start=old_end=None
    def history_at(L):
        nonlocal old_start,old_end,history
        start,end=L-HISTORY,L-3 # gram end must be <=L: current token excluded
        if old_start is None or start<old_start or start>=old_end:
            history=Counter(grams[start:end])
        else:
            for j in range(old_end,end):history[grams[j]]+=1
            for j in range(old_start,start):
                g=grams[j];history[g]-=1
                if not history[g]:del history[g]
        old_start,old_end=start,end
        return history
    @lru_cache(maxsize=64)
    def protection(category,context):
        return tuple(protection_reasons(prompt_text,context,category))
    witnesses=[];protected=Counter();categories=Counter();checks=Counter()
    for t in range(n):
        if ids[t]in special or not additions[t]or '\ufffd'in fragments[ids[t]]:
            checks['special_empty_or_replacement_excluded']+=1;continue
        if fast:L=int(offsets[t]);U=int(offsets[t+1]);addition=additions[t]
        else:
            left,right=before(t),before(t+1);L=len(left);U=len(right)
            if left!=norm[:L]or right!=norm[:U]or not right.startswith(left):
                checks['unstable_prefix_boundary_excluded']+=1;continue
            addition=right[L:]
            if addition!=additions[t]:checks['nonexact_token_piece_excluded']+=1;continue
        if not addition or L<HISTORY:continue
        checks['eligible_history_tokens']+=1
        left_score=1-float(unique[L-HALF])/HALF_GRAMS
        right_score=1-float(unique[L])/HALF_GRAMS
        if min(left_score,right_score)<PRIMARY_THRESHOLD:continue
        checks['two_halves_pass_primary']+=1
        hist=history_at(L)
        introduced=norm[L-3:L]+addition
        target_grams=[introduced[j:j+4]for j in range(len(addition))]
        assert len(target_grams)==len(addition)and all(len(g)==4 for g in target_grams)
        hits=sum(hist[g]>=2 for g in target_grams);fraction=hits/len(target_grams)
        if fraction<CURRENT_MATCH_THRESHOLD:continue
        checks['current_reuse_pass']+=1
        category='numeric'if numeric[L]-numeric[L-HISTORY]==HISTORY else'alphabetic'if alphabetic[L]-alphabetic[L-HISTORY]>0 else'other'
        # Same v2 protections, applied to the category of the OLD window and its
        # preceding original text. Alphabetic protection never uses context.
        raw_start=raw_offsets[L-HISTORY]
        context=''if category=='alphabetic'else text[max(0,raw_start-800):raw_start]
        reasons=protection(category,context)
        if reasons:protected.update(reasons);checks['protected_targets']+=1;continue
        primary[t]=True;strict[t]=min(left_score,right_score)>=STRICT_THRESHOLD;categories[category]+=1
        if len(witnesses)<16:
            witnesses.append({'response_position':t,'prior_norm_interval':[L-HISTORY,L],
                'left_repeat_fraction':left_score,'right_repeat_fraction':right_score,
                'new_4grams':len(target_grams),'new_4grams_seen_twice':hits,'new_match_fraction':fraction,
                'category':category,'strict_075':bool(strict[t]),'token_piece':pieces[t][:120],
                'prior_tail':norm[max(L-96,L-HISTORY):L]})
    assert not(strict&~primary).any()
    return {'version':VERSION,'primary_mask':primary,'strict_mask':strict,
        'spans':{'primary':intervals(primary),'strict':intervals(strict)},'witnesses':witnesses,
        'audit':{'response_tokens':n,'normalized_characters':len(norm),
            'primary_tokens':int(primary.sum()),'strict_tokens':int(strict.sum()),
            'category_target_counts':dict(categories),'protected_reasons':dict(protected),'checks':dict(checks),
            'decoder_alignment':'exact_concatenated_token_decodes'if fast else'checked_original_prefix_decodes',
            'history_characters':HISTORY,'half_characters':HALF,'ngram':NGRAM,
            'primary_half_threshold':PRIMARY_THRESHOLD,'sensitivity_half_threshold':STRICT_THRESHOLD,
            'current_reused_gram_fraction_threshold':CURRENT_MATCH_THRESHOLD,
            'new_gram_definition':'Every4gram ending in current normalized token addition, including cross-prefix boundary.',
            'prior_occurrence_minimum':2,'uses_probabilities_advantages_labels_or_gradients':False,
            'uses_future_confirmation':False,'math_protections':'Unchanged v2 protection_reasons, OLD-window category and pre-window context.',
            'numeric_visible_token_floor':'None added: this exploratory rule has512prior characters, not the exact-cycle256visible-token requirement.',
            'classification':'Exploratory noisy-reuse candidate; mathematical false positives require separate audit.'}}
