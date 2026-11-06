"""Native reduction edge contracts; synthetic CPU data only."""
import math
import unittest
import torch

from native_masked_reference import masked_sum, masked_mean
from native_ppo_reference import agg_loss, compute_policy_loss_vanilla


class Config(dict):
    def __getattr__(self,key):return self[key]


def config(tokens):
    return Config(clip_ratio=.2,clip_ratio_low=.2,clip_ratio_high=.28,clip_ratio_c=3.,
        global_batch_info={'dp_size':8,'batch_num_tokens':tokens})


class NativeMaskedReferenceTests(unittest.TestCase):
    def test_numeric_mask_axis_and_padding_nan(self):
        values=torch.tensor([[1.,float('nan'),3.],[float('nan')]*3],dtype=torch.float64)
        mask=torch.tensor([[1.,0.,.5],[0.,0.,0.]],dtype=torch.float64)
        self.assertTrue(torch.equal(masked_sum(values,mask,axis=1),torch.tensor([2.5,0.],dtype=torch.float64)))
        self.assertTrue(torch.equal(masked_mean(values,mask,axis=1),torch.tensor([2.5/(1.5+1e-8),0.],dtype=torch.float64)))
        self.assertEqual(masked_sum(values,mask,axis=(0,1)).item(),2.5)

    def test_zero_mask_mean_and_sum_are_zero(self):
        values=torch.full((2,3),float('nan'));mask=torch.zeros_like(values)
        self.assertEqual(masked_sum(values,mask).item(),0.)
        self.assertEqual(masked_mean(values,mask).item(),0.)

    def test_unmasked_nan_is_not_silently_dropped(self):
        values=torch.tensor([float('nan'),1.]);mask=torch.ones_like(values)
        self.assertTrue(torch.isnan(masked_sum(values,mask)))
        self.assertTrue(torch.isnan(masked_mean(values,mask)))

    def test_padded_rank_zero_forward_with_nonzero_global_denominator(self):
        values=torch.full((1,3),float('nan'));mask=torch.zeros_like(values)
        loss=agg_loss(values,mask,'token-mean',dp_size=8,batch_num_tokens=128)
        self.assertEqual(loss.item(),0.)
        loss,metrics=compute_policy_loss_vanilla(values,values,values,mask,config=config(128))
        self.assertEqual(loss.item(),0.)
        self.assertTrue(all(v==0. for v in metrics.values()))
        # The native aggregator does not invent a positive global denominator.
        self.assertTrue(torch.isnan(agg_loss(values,mask,'token-mean')))

    def test_native_policy_forward_ignores_nan_padding(self):
        values=torch.tensor([[-3.,float('nan')]])
        advantages=torch.tensor([[1.,float('nan')]])
        mask=torch.tensor([[1.,0.]])
        loss,metrics=compute_policy_loss_vanilla(values,values,advantages,mask,config=config(1))
        self.assertEqual(loss.item(),-8.)
        self.assertTrue(all(math.isfinite(v)for v in metrics.values()))


if __name__=='__main__':unittest.main()
