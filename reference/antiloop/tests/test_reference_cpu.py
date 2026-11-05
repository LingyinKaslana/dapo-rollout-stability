"""Synthetic CPU unit tests, never scientific measurements of a trained model."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repeat_targets_v2 import FamilyCatalog,select_targets
from ul_loss import family_unlikelihood,sparse_family_ul,gradient_budget


class ToyTokenizer:
    pieces={0:'<eos>',1:'E',2:'EE',3:' E',4:'Ｅ',5:'9',6:'9999',7:' ',8:'@',9:'９'}
    all_special_ids=[0]
    eos_token_id=0
    pad_token_id=0
    def get_vocab(self):return {str(i):i for i in self.pieces}
    def decode(self,ids,**kwargs):return ''.join(self.pieces[int(i)]for i in ids)


class Fields(dict):
    def to_padded_tensor(self):return self


class Batch(dict):
    def select(self,*keys):return Fields({k:self[k]for k in keys})


class ReferenceTests(unittest.TestCase):
    def detect(self,ids,prompt='Solve this problem.'):
        catalog=FamilyCatalog(ToyTokenizer())
        return select_targets(ids,catalog,prompt_text=prompt),catalog

    def test_confirmed_prefix_and_alias_family(self):
        info,catalog=self.detect([1]*220)
        self.assertEqual(len(info['targets']),64)
        self.assertEqual(info['targets'][0]['response_position'],128)
        self.assertEqual(set(catalog.families[0]),{1,2,3,4})
        self.assertNotIn(0,catalog.families[0])
        self.assertFalse(self.detect([1]*128)[0]['targets'])
        self.assertFalse(self.detect([1]*300,'Print E 300 times.')[0]['targets'])

    def test_numeric_guard_and_short_answer_exception(self):
        self.assertFalse(self.detect([6]*200)[0]['targets']) # many chars, <256 visible tokens
        info,_=self.detect([6]*320,'Find the last two digits of 2013^2013.')
        self.assertEqual(info['targets'][0]['response_position'],256)
        self.assertFalse(self.detect([5]*700,'Compute 10^1000-1.')[0]['targets'])
        info,_=self.detect([6]*128+[7]*200+[6]*200)
        self.assertEqual(info['targets'][0]['response_position'],456) # whitespace does not count

    def test_stable_family_gradient_near_unit_probability(self):
        z=torch.tensor([[1000.,999.7,0.,-.2]],requires_grad=True)
        loss,mass=family_unlikelihood(z,[0,1])
        g=torch.autograd.grad(loss.sum(),z)[0]
        self.assertTrue(torch.isfinite(loss).all()and torch.isfinite(g).all())
        self.assertEqual(float(mass[0]),1.)
        self.assertGreater(float(g[0,:2].sum()),.999)
        self.assertLess(abs(float(g.sum())),2e-6)

    def test_empty_family_and_zero_mask_has_exact_zero_gradient(self):
        ids=torch.tensor([2,3,1]);offsets=torch.tensor([0,3])
        batch=Batch(input_ids=torch.nested.nested_tensor_from_jagged(ids,offsets),
            responses=torch.tensor([[1]]),response_mask=torch.tensor([[1]]),
            ul_target_mask=torch.tensor([[False]]),ul_family_index=torch.tensor([[-1]]))
        z=torch.ones(1,3,10,requires_grad=True)
        values,stats=sparse_family_ul(z,batch,[])
        g=torch.autograd.grad(values.sum(),z)[0]
        self.assertEqual(stats['mask_tokens'],0)
        self.assertTrue(torch.equal(g,torch.zeros_like(g)))

    def test_five_percent_gradient_budget_and_no_floor(self):
        for nr in [0.,.001,1.,1000.]:
            for nu in [0.,.001,10.,1e6]:
                b=gradient_budget(nr,nu,.01,.05,relative_cap_fraction=.05)
                self.assertLessEqual(b['scale'],.01)
                self.assertLessEqual(b['ideal_effective_UL_norm'],.05*nr+1e-12)
                if nr==0 or nu==0:self.assertEqual(b['scale'],0.)
                for sign in [-1.,1.]:
                    combined=abs(nr+sign*b['scale']*nu)
                    self.assertGreaterEqual(combined,.95*nr-1e-10)
                    self.assertLessEqual(combined,1.05*nr+1e-10)


if __name__=='__main__':
    assert not torch.cuda.is_initialized()
    unittest.main()
