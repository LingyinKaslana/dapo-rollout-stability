"""Redraw final-study secondary figures from the six published aggregate CSVs.

Usage from the published repository root:
  python plot_secondary_aggregates.py \
    --data-dir data/final-stability-600-650/secondary \
    --output figures/final-stability-600-650/secondary-redraw

Requires Python 3.10+ and Matplotlib. No model, raw response, tokenizer, remote
service, or original machine path is used. Stored confidence intervals are
plotted, not recomputed. Pixel/font identity with the original plots is not
promised. --allow-partial is for explicitly labelled internal previews only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import sys

ENDPOINTS = ('source600', 'baseline_650', 'full_tis_650', 'negative_tis_650',
             'abs_gap_mask_650', 'negative_down_mask_650',
             'repeat_negative_down_mask_650', 'negative_mass_650', 'sync_fresh_650')
LABELS = ('Source 600', 'Baseline', '1 Full TIS', '2 Negative TIS',
          '3 Absolute-gap mask', '4 Negative-down mask', '5 Repeat-gated mask',
          '6 Negative length weighting', '7 Fresh synchronous')
METRICS = ('accuracy', 'unknown', 'strict_repetition', 'normalized_text_cycle128_any',
           'normalized_text_cycle128_alphabetic', 'normalized_text_cycle128_numeric',
           'normalized_text_cycle128_other', 'repeat_strict_union_any',
           'noisy_repeat_primary_any', 'noisy_repeat_strict_any', 'gibberish_candidate',
           'response_hit_token_cap', 'first_word_alright', 'first_word_to',
           'tokens', 'log_PPL_q', 'q_ge_0999_fraction')
TARGETS = ('Alright', 'To', 'Let')
FILES = ('endpoint_metrics.csv', 'prompt_only_probabilities.csv', 'per_question.csv',
         'paired_natural_contrasts.csv', 'paired_prompt_probability_contrasts.csv',
         'all_cohort_descriptive.csv')
COHORTS = {'aime24': 3, 'aime25': 1, 'amc23': 4, 'math500': 42,
           'deepmath_D5_6': 6, 'deepmath_D6_7': 15, 'deepmath_D7_8': 26,
           'deepmath_D8_9': 29, 'deepmath_D9_10': 2}
FAMILIES = {'DeepMath': 78, 'aime24': 3, 'aime25': 1, 'amc23': 4, 'math500': 42}
CORE = (('accuracy', 'Accuracy (%)', 'Accuracy difference (pp)', 100),
        ('response_hit_token_cap', 'Response cap (%)', 'Cap difference (pp)', 100),
        ('normalized_text_cycle128_any', 'Cycle candidates (%)', 'Cycle difference (pp)', 100),
        ('tokens', 'Mean response tokens', 'Length difference (tokens)', 1))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    result = float(value)
    require(math.isfinite(result), 'Nonfinite aggregate value')
    return result


def integer(value):
    result = number(value)
    require(result == int(result), 'Nonintegral count')
    return int(result)


def close(left, right, message):
    require(math.isclose(number(left), number(right), rel_tol=1e-10, abs_tol=1e-10), message)


def indexed(rows, columns):
    result = {}
    for row in rows:
        key = tuple(row[c] for c in columns)
        require(key not in result, 'Duplicate aggregate coordinate: ' + repr(key))
        result[key] = row
    return result


def final_mode(available, allow_partial):
    require(set(available) <= set(ENDPOINTS), 'Unrecognized endpoint')
    require(bool(available), 'No observed endpoints')
    complete = set(available) == set(ENDPOINTS)
    require(complete or allow_partial,
            f'Final redraw requires all nine endpoints; found {len(set(available))}/9. '
            'Use --allow-partial only for an internal preview.')
    return complete


def load_and_validate(data_dir, allow_partial=False):
    tables, input_hashes = {}, {}
    for name in FILES:
        raw = (data_dir / name).read_bytes()
        input_hashes[name] = hashlib.sha256(raw).hexdigest()
        # csv.reader handles quoted newlines and Unicode separators correctly;
        # never split serialized records with str.splitlines().
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode('utf-8'), newline='')))
    natural = indexed(tables[FILES[0]], ('endpoint',))
    available = [e for e in ENDPOINTS if (e,) in natural]
    require(len(available) == len(natural), 'Unrecognized endpoint in natural metrics')
    complete = final_mode(available, allow_partial)
    probability = indexed(tables[FILES[1]], ('endpoint',))
    require(set(probability) == set(natural), 'Natural/HF endpoint sets differ')
    questions = indexed(tables[FILES[2]], ('endpoint', 'qid'))
    require({e for e, _ in questions} == set(available), 'Per-question endpoint sets differ')
    common_qids = None
    for endpoint in available:
        n, p = natural[(endpoint,)], probability[(endpoint,)]
        expected_step = 600 if endpoint == 'source600' else 650
        require(integer(n['step']) == integer(p['step']) == expected_step, 'Wrong endpoint step')
        require(integer(n['questions']) == integer(p['questions']) == 128, 'Expected 128 questions')
        require(integer(n['responses']) == 512 and integer(n['samples_per_question']) == 4,
                'Expected 128 x 4 responses')
        require(p['backend'] == 'HF BF16 flash_attention_2', 'HF probe backend changed')
        qids = {q for e, q in questions if e == endpoint}
        require(len(qids) == 128, 'Incomplete per-question table')
        if common_qids is None:
            common_qids = qids
        require(qids == common_qids, 'Question sets differ across endpoints')
        rows = [questions[(endpoint, q)] for q in sorted(qids)]
        require(all(integer(row['response_samples']) == 4 for row in rows), 'Unequal samples per question')
        for metric in METRICS:
            value = number(n[metric])
            require(value >= 0, 'Negative natural metric')
            if metric not in ('tokens', 'log_PPL_q'):
                require(value <= 1, 'Fraction outside [0,1]')
            close(value, sum(number(row[metric]) for row in rows) / 128,
                  'Endpoint/per-question mean mismatch: ' + metric)
        for target in TARGETS:
            vals = [number(row['HF_P_' + target]) for row in rows]
            require(all(0 <= x <= 1 for x in vals), 'HF probability outside [0,1]')
            close(p['P_' + target], sum(vals) / 128, 'HF prompt mean mismatch')
        require(all(sum(number(row['HF_P_' + k]) for k in TARGETS) <= 1.000001 for row in rows),
                'HF target probabilities sum above one')
    pairs = {(ref, e) for ref in ('source600', 'baseline_650') if ref in available
             for e in available if e != ref and not (ref == 'baseline_650' and e == 'source600')}
    natural_ci = indexed(tables[FILES[3]], ('reference', 'endpoint', 'metric'))
    probability_ci = indexed(tables[FILES[4]], ('reference', 'endpoint', 'target_token'))
    require(set(natural_ci) == {(r, e, m) for r, e in pairs for m in METRICS},
            'Natural contrast grid is incomplete or contains extra rows')
    require(set(probability_ci) == {(r, e, t) for r, e in pairs for t in TARGETS},
            'HF contrast grid is incomplete or contains extra rows')
    for (ref, endpoint, metric), row in natural_ci.items():
        require(integer(row['questions']) == 128 and integer(row['samples_per_question']) == 4
                and integer(row['bootstrap_draws']) == 10000, 'Natural CI sampling scope changed')
        require(row['multiplicity_adjusted'].lower() == 'false', 'Expected unadjusted intervals')
        require(number(row['ci95_low']) <= number(row['ci95_high']), 'Reversed confidence interval')
        close(row['mean_difference'], number(natural[(endpoint,)][metric]) - number(natural[(ref,)][metric]),
              'Natural contrast point estimate mismatch')
    for (ref, endpoint, target), row in probability_ci.items():
        require(integer(row['questions']) == 128 and row['backend'] == 'HF BF16 flash_attention_2',
                'HF contrast scope changed')
        require(row['multiplicity_adjusted'].lower() == 'false', 'Expected unadjusted intervals')
        require(number(row['ci95_low']) <= number(row['ci95_high']), 'Reversed HF interval')
        close(row['mean_probability_difference'], number(probability[(endpoint,)]['P_' + target])
              - number(probability[(ref,)]['P_' + target]), 'HF contrast point estimate mismatch')
    cohorts = indexed(tables[FILES[5]], ('endpoint', 'level', 'cohort'))
    group_spec = {('ALL', 'ALL'): 128, **{('family', k): v for k, v in FAMILIES.items()},
                  **{('original_cohort', k): v for k, v in COHORTS.items()}}
    require(set(cohorts) == {(e, level, c) for e in available for level, c in group_spec},
            'Cohort table is incomplete or selectively filtered')
    for endpoint in available:
        for (level, cohort), count in group_spec.items():
            row = cohorts[(endpoint, level, cohort)]
            require(integer(row['questions']) == count and integer(row['responses']) == count * 4,
                    'Wrong frozen cohort count')
            close(row['accuracy'], integer(row['correct_responses']) / (4 * count), 'Cohort accuracy mismatch')
            close(row['unknown'], integer(row['unknown_responses']) / (4 * count), 'Cohort unknown count mismatch')
            require(row['small_group_warning'].lower() == str(count < 10).lower(), 'Small cohort flag mismatch')
        for metric in METRICS:
            all_value = cohorts[(endpoint, 'ALL', 'ALL')][metric]
            close(all_value, natural[(endpoint,)][metric], 'Cohort ALL differs from endpoint')
            close(sum(number(cohorts[(endpoint, 'original_cohort', c)][metric]) * n
                      for c, n in COHORTS.items()) / 128, all_value, 'Cohort decomposition mismatch')
    return {'complete': complete, 'available': available, 'natural': natural, 'probability': probability,
            'natural_ci': natural_ci, 'input_hashes': input_hashes,
            'table_rows': {name: len(rows) for name, rows in tables.items()}}


def redraw(data_dir, output, allow_partial=False):
    data = load_and_validate(data_dir, allow_partial)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    available = data['available']
    labels = dict(zip(ENDPOINTS, LABELS))
    xs = list(range(len(available)))
    names = [labels[e] for e in available]
    prefix = 'Final secondary 128 x 4' if data['complete'] else f'PARTIAL PREVIEW: {len(available)}/9 endpoints'
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    outputs = []

    def save(fig, stem):
        for suffix in ('png', 'svg'):
            path = output / f'{stem}.{suffix}'
            fig.savefig(path, dpi=170)
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    colors = ['#969696' if e == 'source600' else '#222222' if e == 'baseline_650' else '#4079a6'
              for e in available]
    for ax, (metric, label, _, scale) in zip(axes.flat, CORE):
        values = [number(data['natural'][(e,)][metric]) * scale for e in available]
        ax.bar(xs, values, color=colors)
        ax.set_xticks(xs, names, rotation=35, ha='right')
        ax.set_ylabel(label)
        ax.set_ylim(0, 100 if metric == 'accuracy' else max(1, max(values) * 1.18))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(prefix + '; all unknown grades retained')
    save(fig, 'endpoint_metrics')

    treatments = [e for e in available if e not in ('source600', 'baseline_650')]
    fig, axes = plt.subplots(1, 4, figsize=(17, max(3.4, 1.5 + .6 * len(treatments))),
                             sharey=True, layout='constrained')
    for ax, (metric, _, label, scale) in zip(axes, CORE):
        if 'baseline_650' in available and treatments:
            for y, endpoint in enumerate(treatments):
                row = data['natural_ci'][('baseline_650', endpoint, metric)]
                mean, low, high = [number(row[k]) * scale for k in ('mean_difference', 'ci95_low', 'ci95_high')]
                ax.plot([low, high], [y, y], color='#4079a6')
                ax.plot(mean, y, 'o', color='#1c4e75')
            ax.set_yticks(list(range(len(treatments))), [labels[e] for e in treatments])
        else:
            ax.text(.5, .5, 'No observed baseline contrast', transform=ax.transAxes, ha='center')
            ax.set_yticks([])
        ax.axvline(0, color='grey', lw=1)
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle(prefix + '\nPaired-question 95% intervals: 10,000 resamples; unadjusted; one training trajectory/arm')
    save(fig, 'paired_vs_baseline')

    fig, ax = plt.subplots(figsize=(14, 5.5), layout='constrained')
    for offset, target, color in ((-.26, 'Alright', '#a75745'), (0, 'To', '#4079a6'), (.26, 'Let', '#75965b')):
        ax.bar([x + offset for x in xs], [number(data['probability'][(e,)]['P_' + target]) * 100 for e in available],
               width=.25, label=target, color=color)
    ax.set_xticks(xs, names, rotation=30, ha='right')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Mean prompt-only next-token probability (%)')
    ax.set_title(prefix + '\nSeparate HF BF16/FA2 probe; exact token IDs, 128 original prompts')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    save(fig, 'prompt_only_probabilities')
    for name, digest in data['input_hashes'].items():
        require(hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == digest, 'Input changed during redraw')
    receipt = {'complete': data['complete'], 'final_result': data['complete'],
               'render_mode': 'final' if data['complete'] else 'internal_partial_preview',
               'available_endpoints': available, 'missing_endpoints': [e for e in ENDPOINTS if e not in available],
               'input_sha256': data['input_hashes'], 'input_table_rows': data['table_rows'],
               'outputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'runtime': {'python': sys.version.split()[0], 'matplotlib': importlib.metadata.version('matplotlib')},
               'figure_meanings': {'endpoint_metrics': 'Accuracy, response-cap rate, mechanical cycle candidates, and mean length; percentages or token counts as labelled.',
                                   'paired_vs_baseline': 'Provided paired-qid percentile95% intervals, 10000 resamples; seven final comparisons unadjusted and conditional on one trajectory per arm.',
                                   'prompt_only_probabilities': 'HF BF16/FA2 prompt-only P(Alright), P(To), P(Let); distinct from sampled first visible word frequency.'},
               'scope': ['Only six published aggregate CSVs are read; no raw answers, model, tokenizer, API, or GPU.',
                         'Unknown grades and all original cohorts retained; all four samples per question have equal question weight.',
                         'Stored intervals are drawn without recomputation. Training-seed uncertainty is excluded.',
                         'Cycle and garble flags are mechanical candidates, not semantic ground truth.',
                         'Secondary response-mean logPPL_q is not global-token-weighted training PPL or p-q discrepancy.',
                         'Rendering is scientifically equivalent; pixel/font/hash identity to original figures is not promised.']}
    (output / 'redraw_manifest.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/final-stability-600-650/secondary'))
    parser.add_argument('--output', type=Path, default=Path('figures/final-stability-600-650/secondary-redraw'))
    parser.add_argument('--allow-partial', action='store_true', help='Internal preview with visible PARTIAL labels; never marks final.')
    args = parser.parse_args()
    try:
        result = redraw(args.data_dir, args.output, args.allow_partial)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, 'Redraw refused: ' + str(error) + '\n')
    print(json.dumps({k: result[k] for k in ('complete', 'render_mode', 'available_endpoints', 'missing_endpoints')}))
