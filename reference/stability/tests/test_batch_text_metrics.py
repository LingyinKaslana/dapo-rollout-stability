"""CPU contract tests only. These synthetic cases are not experimental evidence."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import unittest
import numpy as np
import batch_text_metrics as btm
from repetition_mask import build_repetition_masks
from noisy_repetition_mask import build_noisy_repetition_masks


class CharacterTokenizer:
    all_special_ids = [0]
    eos_token_id = 0
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        return ''.join('<EOS>' if i == 0 else chr(i) for i in ids)


def initialize_test_cpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    btm._TOKENIZER = CharacterTokenizer()


def prompt(question='Solve this problem.'):
    return list(map(ord, '<|im_start|>user\n' + question + '<|im_end|>\n<|im_start|>assistant\n'))


def task(text, question='Solve this problem.', **kw):
    return {'response_ids': list(map(ord, text)), 'prompt_ids': prompt(question), **kw}


class Tests(unittest.TestCase):
    def test_frozen_algorithm_union_and_distinct_monitoring(self):
        tok = CharacterTokenizer()
        texts = ['E'*800, '9'*800, ''.join('es_ez乎'+chr(65+i%23) for i in range(180)),
                 'We integrate x squared and obtain x cubed divided by three.']
        for text in texts:
            record = task(text)
            metrics, mask = btm._process_one(record, tok)
            exact = build_repetition_masks(record['response_ids'], tok, 'Solve this problem.')
            noisy = build_noisy_repetition_masks(record['response_ids'], tok, 'Solve this problem.')
            self.assertTrue(np.array_equal(mask, exact['confirmed_mask'] | noisy['strict_mask']))
            self.assertEqual(metrics['repeat_strict_union_tokens'], int(mask.sum()))
        metrics, mask = btm._process_one(task('9'*200), tok)
        self.assertTrue(metrics['normalized_text_cycle128_numeric'])
        self.assertFalse(mask.any())
        metrics, mask = btm._process_one(task('9'*800, 'Compute 10^1000-1.'), tok)
        self.assertTrue(metrics['normalized_text_cycle128_numeric'])
        self.assertFalse(mask.any())

    def test_raw_ids_prompt_text_and_expected_sha_fail_closed(self):
        tok = CharacterTokenizer()
        record = task('To solve this, take x=2.')
        metrics, _ = btm._process_one(record, tok)
        verified = dict(record, expected_response_text_sha256=metrics['response_text_sha256'],
                        expected_prompt_token_ids_sha256=metrics['prompt_token_ids_sha256'],
                        expected_user_text='Solve this problem.')
        btm._process_one(verified, tok)
        with self.assertRaises(ValueError):
            btm._process_one(dict(record, expected_response_text_sha256='0'*64), tok)
        with self.assertRaises(ValueError):
            btm._process_one(dict(record, expected_user_text='Another question.'), tok)
        with self.assertRaises(ValueError):
            btm._process_one(dict(record, prompt_ids=list(map(ord, 'untemplated text'))), tok)
        self.assertEqual(btm.sha_ids([1, 2]), btm.sha_ids(np.array([1, 2], dtype='>i8')))

    def test_first_word_is_not_model_first_token_probability(self):
        tok = CharacterTokenizer()
        a, _ = btm._process_one(task(' Alright, let us begin.'), tok)
        t, _ = btm._process_one(task('<think>To solve it.'), tok)
        self.assertTrue(a['first_word_alright'])
        self.assertFalse(t['first_word_to'])
        self.assertTrue(t['has_think_tag_prefix'])
        self.assertFalse(btm.aggregate_rows([a,t])['probability_of_first_token_measured'])

    def test_spawn_pool_order_original_padding_and_special_exclusion(self):
        # A deterministic local character tokenizer replaces only the external
        # dependency. This exercises the actual spawn pool and batch interface.
        pool = btm.TextMetricsPool.__new__(btm.TextMetricsPool)
        pool.workers = 2
        pool.chunksize = 1
        pool.creator_pid = os.getpid()
        pool.pool = ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize_test_cpu)
        ids = [list(map(ord, 'E'*700))+[0], list(map(ord, 'To solve x=2.'))]
        prompts = [prompt(), prompt('Find x.')]
        r = np.zeros((2, 720), dtype=np.int64)
        v = np.zeros_like(r, dtype=bool)
        p = np.zeros((2, max(map(len,prompts))+4), dtype=np.int64)
        pm = np.zeros_like(p, dtype=bool)
        for i in range(2):
            r[i,:len(ids[i])] = ids[i]
            v[i,:len(ids[i])] = True
            p[i,-len(prompts[i]):] = prompts[i]
            pm[i,-len(prompts[i]):] = True
        try:
            result = pool.process_batch(r,v,p,pm,[{'uid':'a'},{'uid':'b'}])
            self.assertEqual([x['uid'] for x in result['rows']], ['a','b'])
            mask = result['repeat_mask_strict_union']
            self.assertEqual(mask.dtype,np.bool_)
            self.assertEqual(mask.shape,r.shape)
            self.assertEqual(int(mask.sum()),572)
            self.assertFalse(mask[0,700:].any())
            self.assertFalse(mask[1].any())
            self.assertEqual(result['rows'][0]['response_token_ids_sha256'], btm.sha_ids(ids[0]))
            bad = v.copy();bad[1,0] = False
            with self.assertRaises(ValueError):
                pool.process_batch(r,bad,p,pm)
            with self.assertRaises(ValueError):
                pool.process_batch(r,v,p,pm,[{'response_ids':[]},{}])
        finally:
            pool.close()

    def test_signed_probability_metrics_threshold_and_empty_mean(self):
        # Signed gaps [-1, +.6, +.5, 0]; the exact boundary+.5 is not in the tail.
        q = np.array([-1., -1., -1., -1.])
        p = np.array([-2., -.4, -.5, -1.])
        result = btm.probability_summary(p,q,np.ones(4,bool))
        self.assertEqual(result['absolute_gap_threshold_count'],2)
        self.assertAlmostEqual(result['absolute_gap_threshold_conditional_mean_or_zero'],.8)
        self.assertAlmostEqual(result['absolute_gap_RMS'],np.sqrt(1.61/4))
        self.assertAlmostEqual(result['absolute_gap_L2'],np.sqrt(1.61))
        self.assertAlmostEqual(result['log_PPL_p'],.975)
        zero = btm.probability_summary(q,q,np.ones(4,bool))
        self.assertEqual(zero['absolute_gap_threshold_conditional_mean_or_zero'],0.)
        with self.assertRaises(ValueError):
            btm.probability_summary([np.nan],[0.],[True])


if __name__ == '__main__':
    unittest.main()
