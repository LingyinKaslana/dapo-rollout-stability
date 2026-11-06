"""Synthetic contracts only; no scientific model evidence."""
import unittest
from collections import Counter
import numpy as np
from noisy_repetition_mask import build_noisy_repetition_masks,rolling_unique256

class Tokenizer:
    all_special_ids=[0];eos_token_id=0;pad_token_id=0
    def decode(self,ids,**kwargs):return ''.join('<EOS>'if i==0 else chr(i)for i in ids)

def run(text,prompt='Solve the problem.'):
    return build_noisy_repetition_masks(list(map(ord,text)),Tokenizer(),prompt)

class Tests(unittest.TestCase):
    def test_rolling_matches_naive(self):
        text=('es_ez乎eX'*140)
        _,u=rolling_unique256(text)
        for L in [256,257,511,512,700,len(text)]:
            w=text[L-256:L];self.assertEqual(u[L],len({w[i:i+4]for i in range(253)}))
    def test_prefix_threshold_and_current_match(self):
        x=run('E'*700+'XYZ')
        self.assertEqual(np.flatnonzero(x['primary_mask']).tolist(),list(range(512,700)))
        self.assertTrue(np.array_equal(x['primary_mask'],x['strict_mask']))
        for L in [511,512,513,600,700,701,702,703]:
            y=run(('E'*700+'XYZ')[:L]);self.assertEqual(bool(y['primary_mask'][-1]),bool(x['primary_mask'][L-1]))
    def test_two_halves_required(self):
        diverse=''.join(chr(0x500+i)for i in range(256))
        x=run(diverse+'E'*257)
        self.assertFalse(x['primary_mask'][-1])
    def test_normal_math_protections_and_specials(self):
        self.assertFalse(run('9'*800,'Compute 10^1000-1.')['primary_mask'].any())
        self.assertTrue(run('9'*800,'Find the last two digits of 2013^2013.')['primary_mask'].any())
        self.assertFalse(run('E'*800,'Print E 800 times.')['primary_mask'].any())
        self.assertFalse(run(' '*800)['primary_mask'].any())
        ids=list(map(ord,'E'*700))+[0,0xfffd]
        x=build_noisy_repetition_masks(ids,Tokenizer());self.assertFalse(x['primary_mask'][-2:].any())
    def test_noisy_pattern_and_strict_subset(self):
        text=''.join('es_ez乎'+chr(65+i%23)for i in range(180))
        x=run(text);self.assertTrue(x['primary_mask'].any())
        self.assertFalse((x['strict_mask']&~x['primary_mask']).any())
        for L in [512,550,700,900,len(text)]:
            y=run(text[:L]);self.assertEqual(bool(y['primary_mask'][-1]),bool(x['primary_mask'][L-1]))

if __name__=='__main__':unittest.main()
