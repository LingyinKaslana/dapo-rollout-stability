"""Conservative causal targets for general sustained-loop UL (CPU only).

Alphabetic cycles retain v1's128-character/16-cycle threshold. Numeric and
symbol cycles require512 normalized characters,16 cycles AND256 visible
response tokens already in the loop. Only subsequent continuation targets are
penalized. Explicit requested mathematical expansions/repetitions are protected.
Protections are conservative heuristics, not a proof of semantic correctness.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
import numpy as np

VERSION='causal-text-cycle-family-v2'
MIN_CHARS=128
MIN_CYCLES=16
MAX_PERIOD=16
NONALPHABETIC_MIN_CHARS=512
NONALPHABETIC_MIN_VISIBLE_TOKENS=256
MAX_TARGETS=64
GLOBAL_AUX_ROWS=8
D_REF=GLOBAL_AUX_ROWS*MAX_TARGETS


def normalize(text):
    # Per-codepoint normalization is deliberately prefix-stable; full-string NFC
    # can alter a preceding combining-character boundary after a new token.
    return ''.join(k for c in text for k in unicodedata.normalize('NFKC',c)if not k.isspace())


def normalize_with_source_offsets(text):
    chars=[];offset=[]
    for i,c in enumerate(text):
        for k in unicodedata.normalize('NFKC',c):
            if not k.isspace():chars.append(k);offset.append(i)
    return ''.join(chars),offset


def period_candidates(text):
    norm,offset=normalize_with_source_offsets(text);a=np.asarray([ord(c)for c in norm],dtype=np.int64);found=set()
    for p in range(1,min(MAX_PERIOD,len(a)-1)+1):
        eq=a[p:]==a[:-p];d=np.diff(np.r_[False,eq,False].astype(np.int8))
        for lo,hi0 in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)):
            lo,hi=int(lo),int(hi0+p)
            if hi-lo<max(MIN_CHARS,MIN_CYCLES*p):continue
            segment=a[lo:hi];mp=next(j for j in range(1,p+1)if np.all(segment[j:]==segment[:-j]));found.add((lo,hi,mp))
    candidates=[]
    for lo,hi,p in sorted(found,key=lambda z:(z[0]+max(MIN_CHARS,MIN_CYCLES*z[2]),z[2],z[0],-z[1])):
        pattern=norm[lo:lo+p];numeric=all(c.isnumeric()for c in pattern)
        alphabetic=any(c.isalpha()for c in pattern)
        category='alphabetic'if alphabetic and not numeric else'numeric'if numeric else'other'
        required_chars=max(MIN_CHARS if category=='alphabetic'else NONALPHABETIC_MIN_CHARS,MIN_CYCLES*p)
        text_qualified=hi-lo>=required_chars
        candidates.append({'norm_start':lo,'norm_end_exclusive':hi,'period':p,'pattern':pattern,
            'confirm_norm_position':lo+required_chars,
            'raw_char_start':offset[lo],'raw_char_end_exclusive':offset[hi-1]+1,
            'category':category,'required_prefix_characters':required_chars,
            'required_prefix_visible_tokens':0 if category=='alphabetic'else NONALPHABETIC_MIN_VISIBLE_TOKENS,
            'eligible_by_text_threshold':text_qualified,'eligible_for_UL':False,'pure_numeric':numeric,
            'exclusion':None if text_qualified else'below_stricter_text_threshold',
            'eligibility_note':'Final eligibility requires token-prefix confirmation and prompt/context checks in select_targets.'})
    return norm,candidates


def requests_short_numeric_answer(prompt_text):
    """A large input expression does not imply a requested long output."""
    words={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'twelve':12,'twenty':20,'hundred':100}
    pattern=r'(?i)\b(?:last|first|leading|trailing)\s+(\d+|'+ '|'.join(words)+r')\s+(?:decimal\s+)?digits?\b'
    for match in re.finditer(pattern,str(prompt_text)):
        word=match.group(1).lower();n=int(word)if word.isdigit()else words[word]
        if n<128:return True
    return bool(re.search(r'(?i)\b(?:units|ones|tens|hundreds)\s+digit\b|\b(?:number|sum)\s+of\s+(?:the\s+)?digits\b|末[一二三四五六七八九十两\d]{1,3}位|首[一二三四五六七八九十两\d]{1,3}位|数字和|位数',str(prompt_text)))


def protection_reasons(prompt_text, context_before_cycle, category):
    """Use prompt and PRE-loop context only; no rewards or future tokens."""
    prompt=str(prompt_text);context=str(context_before_cycle)[-800:];reasons=[]
    requested=[]
    for match in re.finditer(r'(?i)\b(\d{3,})\s*(?:times|characters?|symbols?|letters?|rows?|terms?|copies|repetitions?)\b',prompt):
        if int(match.group(1))>=128:requested.append(match.group())
    for match in re.finditer(r'(\d{3,})\s*(?:次|个字符|行|个字母|个符号)',prompt):
        if int(match.group(1))>=128:requested.append(match.group())
    if requested:reasons.append('prompt_explicit_long_repeated_or_formatted_output')
    if re.search(r'(?i)\b(?:matrix|table)\b|矩阵|表格',prompt):
        for match in re.finditer(r'(\d+)\s*(?:[x×]|\\times|by)\s*(\d+)',prompt,re.I):
            if int(match.group(1))*int(match.group(2))>=128:
                reasons.append('prompt_explicit_large_matrix_or_table');break
    if re.search(r'(?i)\bexpand\b|without\s+simplifying|展开',prompt)and re.search(r'(?:[A-Za-z)]\s*\^\s*\{?\s*\d{3,})',prompt):
        reasons.append('prompt_explicit_long_algebraic_expansion')
    if category=='alphabetic':return reasons
    short_numeric=requests_short_numeric_answer(prompt)
    numeric_markers=r'(?i)(?:decimal|binary|hexadecimal|base[- ]?\d+)\s+(?:expansion|representation)|(?:repeating|recurring|periodic)\s+decimal|\brepetend\b|\brepdigits?\b|\brepunits?\b|小数展开|十进制展开|二进制展开|循环小数'
    if not short_numeric and re.search(numeric_markers,prompt):reasons.append('prompt_requests_numeric_expansion_or_representation')
    for match in re.finditer(r'(?i)\b(\d{3,})\s*(?:digits?|decimal\s+places?)\b|(\d{3,})\s*位',prompt):
        if int(match.group(1)or match.group(2))>=128:
            reasons.append('prompt_requests_many_digits');break
    #10^1000-1 is a legitimate long9 answer even without saying "print digits".
    if not short_numeric and re.search(r'\d+\s*(?:\^\s*\{?\s*|\*\*\s*)(\d{3,})',prompt):reasons.append('prompt_contains_large_explicit_integer_power')
    if not short_numeric and re.search(r'\b\d{3,}\s*!',prompt):reasons.append('prompt_contains_large_factorial')
    if not short_numeric and re.search(r'\\sum.{0,100}?\^\s*\{?\s*\d{3,}',prompt,re.S):reasons.append('prompt_contains_large_explicit_sum_bound')
    if not short_numeric and re.search(r'\d{128,}',normalize(prompt)):reasons.append('prompt_contains_long_numeric_literal')
    if not short_numeric and(re.search(numeric_markers,context)or re.search(r'(?i)\b(?:exact|full|complete)\s+(?:integer|digits?|decimal)|\ball\s+(?:the\s+)?digits\b|\bdigits\s+after\s+the\s+decimal\s+point\b',context)):
        reasons.append('preloop_context_explicit_numeric_expansion')
    if category=='other':
        symbol_markers=r'(?i)ascii\s+art|\b(?:matrix|table|separator|horizontal\s+line|divider)\b|矩阵|表格|分隔线|字符画'
        if re.search(symbol_markers,prompt):reasons.append('prompt_requests_symbolic_layout')
        if re.search(symbol_markers,context):reasons.append('preloop_context_symbolic_layout')
    return sorted(set(reasons))


def decode(tokenizer,ids,skip_special_tokens=False):
    try:return tokenizer.decode(list(ids),skip_special_tokens=skip_special_tokens,clean_up_tokenization_spaces=False)
    except TypeError:return tokenizer.decode(list(ids),skip_special_tokens=skip_special_tokens)


@dataclass
class FamilyCatalog:
    tokenizer:object
    vocab_limit:int|None=None

    def __post_init__(self):
        self.special=set(map(int,getattr(self.tokenizer,'all_special_ids',[])))
        for name in ['eos_token_id','bos_token_id','pad_token_id','unk_token_id']:
            value=getattr(self.tokenizer,name,None)
            if value is not None:self.special.update(map(int,value if isinstance(value,(list,tuple))else[value]))
        self.ids=sorted(set(map(int,self.tokenizer.get_vocab().values())))
        if self.vocab_limit is not None:self.ids=[i for i in self.ids if 0<=i<self.vocab_limit]
        self.fragments={};self.family_by_key={};self.families=[];self.keys=[]

    def fragment(self,token):
        token=int(token)
        if token not in self.fragments:
            piece=decode(self.tokenizer,[token]);self.fragments[token]=(piece,normalize(piece))
        return self.fragments[token]

    def family(self,pattern,phase):
        assert pattern and 0<=phase<len(pattern)
        key=(pattern,int(phase))
        if key not in self.family_by_key:
            bad=[]
            for token in self.ids:
                if token in self.special:continue
                raw,normalized=self.fragment(token)
                if not normalized or '\ufffd'in raw:continue
                if all(c==pattern[(phase+i)%len(pattern)]for i,c in enumerate(normalized)):bad.append(token)
            assert bad and not(self.special&set(bad))
            index=len(self.families);self.family_by_key[key]=index;self.families.append(bad);self.keys.append({'pattern':pattern,'phase':phase})
        return self.family_by_key[key]

    def receipt(self):
        return {'version':VERSION,'family_keys':self.keys,'families':self.families,
            'special_ids_excluded':sorted(self.special),'vocab_limit':self.vocab_limit,
            'empty_and_whitespace_only_tokens_excluded':True,'replacement_character_pieces_excluded':True,
            'catalog_sha256':hashlib.sha256(json.dumps({'keys':self.keys,'families':self.families},sort_keys=True).encode()).hexdigest()}


def select_targets(response_ids,catalog,prompt_text=''):
    ids=list(map(int,response_ids));text=decode(catalog.tokenizer,ids);norm,candidates=period_candidates(text)
    base={'version':VERSION,'response_tokens':len(ids),'decoded_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
          'normalized_characters':len(norm),'cycle_candidates':candidates,'numeric_cycle_candidates':sum(c['pure_numeric']for c in candidates),
          'max_targets':MAX_TARGETS,'D_ref':D_REF,'targets':[],'selected_episode':None,'decoder_alignment':'none','skipped_unstable_boundaries':0,
          'prompt_sha256':hashlib.sha256(str(prompt_text).encode()).hexdigest(),'monitor_only_candidates':[],
          'protected_reason':None,'target_config':{'alphabetic_min_characters':MIN_CHARS,'min_cycles':MIN_CYCLES,
          'nonalphabetic_min_characters':NONALPHABETIC_MIN_CHARS,'nonalphabetic_min_visible_tokens':NONALPHABETIC_MIN_VISIBLE_TOKENS,
          'max_period':MAX_PERIOD,'max_targets':MAX_TARGETS,'normalization':'characterwise_NFKC_then_remove_whitespace',
          'visible_token_count':'Nonempty normalized contribution, fully inside the confirmed cycle; excludes target token and whitespace-only tokens.'}}
    if not candidates:return base
    # Fast exact token-decode concatenation is the common ASCII/digit path.
    pieces=[catalog.fragment(i)[0]for i in ids]
    fast=''.join(pieces)==text
    if fast:
        offsets=np.r_[0,np.cumsum([len(catalog.fragment(i)[1])for i in ids])]
        assert int(offsets[-1])==len(norm)
        before=lambda t:norm[:int(offsets[t])]
        visible=np.r_[0,np.cumsum(np.diff(offsets)>0)]
        base['decoder_alignment']='exact_concatenated_token_decodes'
    else:
        cache={0:'',len(ids):norm}
        def before(t):
            if t not in cache:cache[t]=normalize(decode(catalog.tokenizer,ids[:t]))
            return cache[t]
        base['decoder_alignment']='verified_original_prefix_decode_boundaries'
    def boundary_at(position):
        if fast:return int(np.searchsorted(offsets,position,side='left'))
        lo,hi=0,len(ids)
        while lo<hi:
            mid=(lo+hi)//2
            if len(before(mid))>=position:hi=mid
            else:lo=mid+1
        return lo
    for cycle in candidates:
        cycle['protected_reasons']=protection_reasons(prompt_text,text[:cycle['raw_char_start']],cycle['category'])
        cycle['protected_reason']=';'.join(cycle['protected_reasons'])or None
        if cycle['protected_reasons']:
            cycle['exclusion']='protected_mathematical_or_requested_output';continue
        if not cycle['eligible_by_text_threshold']:continue
        start=boundary_at(cycle['confirm_norm_position']);loop_start=boundary_at(cycle['norm_start'])
        # Count only visible, fully-contained tokens in the PREVIOUS loop.
        seen_visible=0
        if not fast:
            for u in range(loop_start,start):
                left,right=before(u),before(u+1)
                if left==norm[:len(left)]and right==norm[:len(right)]and right.startswith(left)and len(right)>len(left)and len(left)>=cycle['norm_start']:
                    seen_visible+=1
        pattern=cycle['pattern'];p=cycle['period'];targets=[]
        for t in range(start,len(ids)):
            if len(targets)>=MAX_TARGETS:break
            left,right=before(t),before(t+1);L=len(left)
            if L>=cycle['norm_end_exclusive']:break
            if left!=norm[:L]or not right.startswith(left)or right!=norm[:len(right)]:
                base['skipped_unstable_boundaries']+=1;continue
            addition=right[L:]
            if fast:seen_visible=int(visible[t]-visible[loop_start])
            previous_visible=seen_visible
            if not fast and addition:seen_visible+=1
            if L<cycle['confirm_norm_position']or previous_visible<cycle['required_prefix_visible_tokens']:continue
            if not addition:continue
            phase=(L-cycle['norm_start'])%p
            if not all(c==pattern[(phase+j)%p]for j,c in enumerate(addition)):break
            raw_piece,normalized_piece=catalog.fragment(ids[t])
            if ids[t]in catalog.special or '\ufffd'in raw_piece or normalized_piece!=addition:
                base['skipped_unstable_boundaries']+=1;continue
            family=catalog.family(pattern,phase)
            assert ids[t]in catalog.families[family]
            assert L-cycle['norm_start']>=cycle['required_prefix_characters']
            targets.append({'response_position':t,'observed_token_id':ids[t],'family_index':family,
                'normalized_piece':addition,'confirmed_prefix_characters':L-cycle['norm_start'],
                'confirmed_prefix_visible_tokens':previous_visible,'pattern':pattern,'phase':phase,'category':cycle['category']})
        if targets:
            cycle.update(eligible_for_UL=True,exclusion=None,
                first_target_response_position=targets[0]['response_position'],
                confirmed_prefix_characters=targets[0]['confirmed_prefix_characters'],
                confirmed_prefix_visible_tokens=targets[0]['confirmed_prefix_visible_tokens'])
            base['targets']=targets;base['selected_episode']=cycle;break
        cycle['exclusion']='insufficient_confirmed_token_history_or_no_eligible_continuation'
    # Preserve candidates/protections even if an earlier episode supplies targets.
    for cycle in candidates:
        if 'protected_reasons'not in cycle:
            cycle['protected_reasons']=protection_reasons(prompt_text,text[:cycle['raw_char_start']],cycle['category'])
            cycle['protected_reason']=';'.join(cycle['protected_reasons'])or None
            cycle['exclusion']='protected_mathematical_or_requested_output'if cycle['protected_reasons']else'not_selected_later_episode'
    base['monitor_only_candidates']=[c for c in candidates if not c['eligible_for_UL']]
    protected=sorted({reason for c in candidates for reason in c['protected_reasons']})
    base['protected_reasons_observed']=protected
    base['protected_reason']=(';'.join(protected)or None)if not base['targets']else None
    assert len(base['targets'])<=MAX_TARGETS and len({x['response_position']for x in base['targets']})==len(base['targets'])
    return base
