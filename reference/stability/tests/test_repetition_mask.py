"""Synthetic CPU contracts, not experimental training evidence."""
import unittest
import numpy as np
from repetition_mask import build_repetition_masks,negative_down_gate

class Tokenizer:
    pieces={0:'<EOS>',1:'E',2:'EE',3:' E',4:'Ｅ',5:'9',6:'9999',7:' ',8:'@',9:'X',10:'\ufffd',11:'\ufffd'}
    all_special_ids=[0];eos_token_id=0;pad_token_id=0
    def decode(self,ids,**kwargs):
        if ids[:2]==[10,11]:return 'é'+''.join(self.pieces[i]for i in ids[2:])
        return ''.join(self.pieces[i]for i in ids)

class Tests(unittest.TestCase):
    def masks(self,ids,prompt='Solve the problem.'):
        return build_repetition_masks(ids,Tokenizer(),prompt)
    def test_all_positions_no64_cap_and_censored_loop(self):
        x=self.masks([1]*300+[0]);self.assertEqual(np.flatnonzero(x['confirmed_mask']).tolist(),list(range(128,300)))
        self.assertEqual(int(x['retrospective_mask'].sum()),300)
        self.assertFalse(x['confirmed_mask'][-1])
        y=self.masks([1]*300);self.assertEqual(int(y['confirmed_mask'].sum()),172)
    def test_all_episodes_and_break_excluded(self):
        x=self.masks([1]*200+[9]+[1]*220)
        expected=list(range(128,200))+list(range(329,421))
        self.assertEqual(np.flatnonzero(x['confirmed_mask']).tolist(),expected)
        self.assertFalse(x['retrospective_mask'][200])
    def test_numeric_guard_protection_and_short_answer(self):
        self.assertFalse(self.masks([6]*200)['confirmed_mask'].any())
        self.assertFalse(self.masks([5]*700,'Compute 10^1000-1.')['confirmed_mask'].any())
        x=self.masks([6]*320,'Find the last two digits of 2013^2013.')
        self.assertEqual(np.flatnonzero(x['confirmed_mask']).tolist(),list(range(256,320)))
        x=self.masks([6]*128+[7]*200+[6]*200)
        self.assertEqual(np.flatnonzero(x['confirmed_mask'])[0],456)
    def test_normal_requested_math_and_whitespace(self):
        self.assertFalse(self.masks([1]*300,'Print E 300 times.')['confirmed_mask'].any())
        self.assertFalse(self.masks([1]*300,'Write a 20 by 20 matrix of E.')['confirmed_mask'].any())
        self.assertFalse(self.masks([1,9]*5)['confirmed_mask'].any())
        self.assertFalse(self.masks([7]*500)['confirmed_mask'].any())
    def test_aliases_exact_prefix_fallback_and_replacement_exclusion(self):
        x=self.masks([1,2,3,4]*120);self.assertEqual(x['audit']['decoder_alignment'],'exact_concatenated_token_decodes')
        self.assertGreater(x['audit']['confirmed_tokens'],64)
        y=self.masks([10,11]+[1]*300);self.assertEqual(y['audit']['decoder_alignment'],'checked_original_prefix_decodes')
        self.assertEqual(np.flatnonzero(y['confirmed_mask']).tolist(),list(range(130,302)))
        self.assertFalse(self.masks([10]*700)['confirmed_mask'].any())
    def test_no_future_confirmation_and_primary_prefix_consistency(self):
        ids=[1]*200+[9]+[1,2,3,4]*90
        full=self.masks(ids)['confirmed_mask']
        for n in [1,63,127,128,129,150,200,201,202,320,400,len(ids)]:
            x=self.masks(ids[:n]);self.assertEqual(bool(x['confirmed_mask'][-1]),bool(full[n-1]))
        x=self.masks([1]*128);self.assertFalse(x['confirmed_mask'].any());self.assertEqual(int(x['retrospective_mask'].sum()),128)
    def test_signed_negative_gate_strict_threshold(self):
        p=np.array([-.6,-.5,-.7,-.8,-.9]);q=np.zeros(5);A=np.array([-1,-1,1,-1,-1]);valid=np.array([1,1,1,0,1]);repeat=np.array([1,1,1,1,0])
        self.assertEqual(negative_down_gate(p,q,A,valid,repeat).tolist(),[True,False,False,False,False])

if __name__=='__main__':unittest.main()
