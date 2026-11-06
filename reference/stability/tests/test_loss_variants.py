"""Synthetic CPU contracts only; no experimental outcomes or private data."""
import math
from pathlib import Path
import subprocess
import sys
import unittest
import torch
import loss_variants as lv
from native_ppo_reference import compute_policy_loss_vanilla

def native_vanilla():
    return compute_policy_loss_vanilla

class Config(dict):

    def __getattr__(self, key):
        return self[key]

def config(mask):
    return Config(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=3.0, entropy_coeff=0.0, use_kl_loss=False, global_batch_info={'dp_size': 8, 'batch_num_tokens': int(mask.sum()), 'global_batch_size': 128, 'loss_scale_factor': None})

def fixtures():
    old = torch.full((3, 8), -4.0, dtype=torch.float32)
    s = torch.tensor([[-1.0, -0.5, -0.499, 0, 0.499, 0.5, 0.501, 1.0]]).repeat(3, 1)
    q = old - s
    A = torch.tensor([-1.0, 1.0, 0.0])[:, None].expand_as(old).clone()
    mask = torch.ones_like(old, dtype=torch.bool)
    mask[:, -1] = False
    repeat = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0]], dtype=torch.bool).repeat(3, 1)
    return (old, q, A, mask, repeat)

class LossVariantsTest(unittest.TestCase):

    def test_loss_module_import_does_not_eagerly_import_torch(self):
        result = subprocess.run([sys.executable, '-c', 'import sys; import loss_variants; assert "torch" not in sys.modules'], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_strict_thresholds_and_negative_repeat_intersection(self):
        old, q, A, mask, repeat = fixtures()
        full = lv.make_loss_weights('abs_gap_mask', old, q, A, mask, repeat)
        down = lv.make_loss_weights('negative_down_mask', old, q, A, mask, repeat)
        repeated = lv.make_loss_weights('repeat_negative_down_mask', old, q, A, mask, repeat)
        self.assertEqual(int((full[mask] == 0).sum()), 6)
        self.assertEqual(int((down[mask] == 0).sum()), 1)
        self.assertEqual(int((repeated[mask] == 0).sum()), 1)
        self.assertTrue(torch.all(full[:, 1] == 1))
        self.assertTrue(torch.all(full[:, 5] == 1))
        repeat[0, 0] = False
        repeated = lv.make_loss_weights('repeat_negative_down_mask', old, q, A, mask, repeat)
        self.assertTrue(torch.all(repeated == 1))

    def test_tis_cap_lower_guard_and_stopgrad(self):
        old, q, A, mask, repeat = fixtures()
        old.requires_grad_()
        q = q.clone().requires_grad_()
        with torch.no_grad():
            q[0, 0] = old[0, 0] + 100
            q[0, 1] = old[0, 1] - 100
        full = lv.make_loss_weights('full_tis', old, q, A, mask, repeat)
        negative = lv.make_loss_weights('negative_tis', old, q, A, mask, repeat)
        self.assertFalse(full.requires_grad)
        self.assertFalse(negative.requires_grad)
        self.assertEqual(float(full[0, 0]), float(torch.exp(torch.tensor(-20.0))))
        self.assertEqual(float(full[0, 1]), 2.0)
        self.assertTrue(torch.equal(negative[mask & (A >= 0)], torch.ones_like(negative[mask & (A >= 0)])))
        self.assertTrue(torch.all(full[~mask] == 1))

    def test_hard_gate_compares_fp64_operand_difference_at_rounding_edge(self):
        below_half = torch.nextafter(torch.tensor(0.5, dtype=torch.float32), torch.tensor(0.0))
        old = torch.tensor([[-1.0, -float(below_half)]], dtype=torch.float32)
        q = old.flip(-1).clone()
        A, mask, repeat = (-torch.ones_like(old), torch.ones_like(old, dtype=torch.bool), torch.ones_like(old, dtype=torch.bool))
        self.assertTrue(torch.equal(old - q, torch.tensor([[-0.5, 0.5]])))
        self.assertTrue(torch.all((old.double() - q.double()).abs() > 0.5))
        self.assertTrue(torch.equal(lv.make_loss_weights('abs_gap_mask', old, q, A, mask, repeat), torch.zeros_like(old)))
        for arm in ('negative_down_mask', 'repeat_negative_down_mask'):
            self.assertTrue(torch.equal(lv.make_loss_weights(arm, old, q, A, mask, repeat), torch.tensor([[0.0, 1.0]])))
        tis = lv.make_loss_weights('full_tis', old, q, A, mask, repeat)
        self.assertTrue(torch.equal(tis, torch.exp(old - q).clamp(max=2.0)))

    def test_invalid_padding_does_not_change_valid_weights(self):
        old, q, A, mask, repeat = fixtures()
        old[~mask] = float('nan')
        q[~mask] = float('inf')
        for arm in lv.ARMS:
            supplied = torch.ones_like(old) if arm == 'negative_mass' else None
            weights = lv.make_loss_weights(arm, old, q, A, mask, repeat, supplied)
            self.assertTrue(torch.isfinite(weights).all())
            self.assertTrue(torch.all(weights[~mask] == 1))

    def test_global128_partition_matches_native_rank_layout(self):
        groups = lv.original_global128_groups()
        self.assertEqual(len(groups), 4)
        self.assertTrue(all((len(group) == 128 for group in groups)))
        self.assertEqual(sorted(sum(groups, [])), list(range(512)))
        self.assertEqual(groups[1][:16], list(range(16, 32)))
        self.assertEqual(groups[1][16:32], list(range(80, 96)))

    def test_negative_mass_matches_each_global_substep(self):
        lengths = [1, 2, 4, 8, 1, 8]
        mask = torch.arange(8)[None, :] < torch.tensor(lengths)[:, None]
        A = torch.tensor([-1.0, -2.0, 3.0, -1.0, -2.0, 0.0])[:, None].expand(6, 8).clone()
        original_A = A.clone()
        groups = [[0, 1, 2], [3, 4, 5]]
        weights, records = lv.negative_mass_weights(A, mask, groups, length_cap=2)
        self.assertTrue(torch.equal(A, original_A))
        self.assertTrue(torch.all(weights[(A >= 0) & mask] == 1))
        self.assertEqual(records[0]['lambda'], 1.0)
        self.assertEqual(records[1]['lambda'], 2.5)
        self.assertEqual(float(weights[3, 0]), 0.625)
        self.assertEqual(float(weights[4, 0]), 2.5)
        for record in records:
            self.assertLess(record['relative_mass_error'], 1e-06)
        global_weights, _ = lv.negative_mass_weights(A, mask, [list(range(6))], length_cap=2)
        self.assertFalse(torch.equal(weights, global_weights))
        self.assertFalse(weights.requires_grad)

    def test_mass_empty_negative_and_invalid_partition(self):
        A, mask = (torch.ones(4, 3), torch.ones(4, 3, dtype=torch.bool))
        weights, records = lv.negative_mass_weights(A, mask, [[0, 1], [2, 3]])
        self.assertTrue(torch.all(weights == 1))
        self.assertTrue(all((r['lambda'] == 1 for r in records)))
        with self.assertRaises(ValueError):
            lv.negative_mass_weights(A, mask, [[0, 1], [1, 2, 3]])
        A[0, 1] = -1
        with self.assertRaises(ValueError):
            lv.negative_mass_weights(A, mask, [[0, 1], [2, 3]])

    def test_additive_tail_rms_and_directions(self):
        old, q, A, mask, repeat = fixtures()
        whole = lv.discrepancy_sums(old, q, A, mask)
        parts = {}
        for i in range(old.shape[0]):
            lv.merge_sums(parts, lv.discrepancy_sums(old[i:i + 1], q[i:i + 1], A[i:i + 1], mask[i:i + 1]))
        self.assertEqual(whole, parts)
        self.assertEqual(whole['all']['tokens'], 21)
        self.assertEqual(whole['all']['abs_gt_0_5_tokens'], 6)
        self.assertEqual(whole['all']['down_lt_minus_0_5_tokens'], 3)
        self.assertEqual(whole['all']['up_gt_0_5_tokens'], 3)
        expected = (old.double() - q.double())[mask]
        self.assertAlmostEqual(whole['all']['sum_squared'], float(expected.square().sum()), places=12)

    def test_repeat_cross_sign_statistics_partition_and_tis_floor(self):
        old, q, A, mask, repeat = fixtures()
        gap = lv.discrepancy_sums(old, q, A, mask, repeat)
        likelihood = lv.likelihood_sums(old, A, mask, repeat)
        for stats in (gap, likelihood):
            self.assertEqual(len(stats), 12)
            for key in stats['all']:
                self.assertAlmostEqual(stats['all'][key], stats['repeat'][key] + stats['nonrepeat'][key], places=10)
                for sign in ('positive', 'negative', 'zero'):
                    self.assertAlmostEqual(stats[sign][key], stats[sign + '_repeat'][key] + stats[sign + '_nonrepeat'][key], places=10)
        q[0, 0] = old[0, 0] + 30
        q[1, 0] = old[1, 0] + 30
        full = lv.make_loss_weights('full_tis', old, q, A, mask, repeat)
        negative = lv.make_loss_weights('negative_tis', old, q, A, mask, repeat)
        self.assertEqual(lv.multiplier_sums(full, A, mask, repeat)['all']['tis_floor_weight_tokens'], 2)
        self.assertEqual(lv.multiplier_sums(negative, A, mask, repeat)['all']['tis_floor_weight_tokens'], 1)

    def test_native_ppo_multiplier_preserves_clipping_and_direct_gradient(self):
        native = native_vanilla()
        old, q, A, mask, repeat = fixtures()
        delta = torch.tensor([[-2.0, -0.1, 0.0, 0.1, 0.4, 0.8, 2.0, 0.0]]).repeat(3, 1)
        reference_logp = (old + delta).clone().requires_grad_()
        reference_loss, reference_metrics = native(old_log_prob=old, log_prob=reference_logp, advantages=A, response_mask=mask, loss_agg_mode='token-mean', config=config(mask), rollout_is_weights=None)
        reference_loss.backward()
        for arm in lv.ARMS:
            supplied = torch.where(A < 0, 0.7, 1.0) if arm == 'negative_mass' else None
            weights = lv.make_loss_weights(arm, old, q, A, mask, repeat, supplied)
            logp = (old + delta).clone().requires_grad_()
            loss, metrics = native(old_log_prob=old, log_prob=logp, advantages=A, response_mask=mask, loss_agg_mode='token-mean', config=config(mask), rollout_is_weights=weights)
            loss.backward()
            self.assertEqual(metrics, reference_metrics)
            torch.testing.assert_close(logp.grad, reference_logp.grad * weights, rtol=2e-06, atol=1e-08)
            if arm in ('baseline', 'sync_fresh'):
                self.assertTrue(torch.equal(logp.grad, reference_logp.grad))
                self.assertTrue(torch.equal(loss, reference_loss))

if __name__=="__main__":unittest.main()
