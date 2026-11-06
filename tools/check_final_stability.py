"""Validate the completed eight-arm release using only published aggregates.

Expected layout: data/final-stability-600-650/{training,secondary}/ and a release
manifest at data/final-stability-600-650/manifest.json. Figures and portable code
may live elsewhere in the repository; their root-relative paths are manifested.
This checks the release, not the unavailable raw model or response artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ARMS = (
    'baseline', 'full_tis', 'negative_tis', 'abs_gap_mask',
    'negative_down_mask', 'repeat_negative_down_mask', 'negative_mass', 'sync_fresh',
)
STEPS = list(range(601, 651))
EVAL_STEPS = [600, 610, 620, 630, 640, 650]
ENDPOINTS = ['source600', *[a + '_650' for a in ARMS]]
SHA = re.compile(r'[0-9a-f]{64}')
SIGNS = ('all', 'positive', 'negative', 'zero')
STRATA = (*SIGNS, 'repeat', 'nonrepeat', 'positive_repeat', 'positive_nonrepeat',
          'negative_repeat', 'negative_nonrepeat', 'zero_repeat', 'zero_nonrepeat')
GAP_REFERENCES = ('historical_q', 'own_round_p0', 'anchor_p0_minus_historical_q')
PROBABILITY_SOURCES = ('actor_p', 'historical_q', 'own_round_p0')
PRIMARY_CONTRAST_METRICS = ('accuracy', 'mean_length', 'normalized_text_cycle128_any',
                          'response_hit_token_cap', 'gibberish_candidate', 'first_word_alright')
SECONDARY_METRICS = ('accuracy', 'unknown', 'strict_repetition', 'normalized_text_cycle128_any',
    'normalized_text_cycle128_alphabetic', 'normalized_text_cycle128_numeric', 'normalized_text_cycle128_other',
    'repeat_strict_union_any', 'noisy_repeat_primary_any', 'noisy_repeat_strict_any', 'gibberish_candidate',
    'response_hit_token_cap', 'first_word_alright', 'first_word_to', 'tokens', 'log_PPL_q', 'q_ge_0999_fraction')
TARGET_TOKENS = ('Alright', 'To', 'Let')
TRAINING_FIGURE_STEMS = ('gradient_by_inner_all_arms', 'gradient_baseline_vs_seven_methods',
    'gradient_and_gap_all_arms', 'gap_by_sign_and_repeat', 'learning_signal_and_adam_displacement',
    'accepted_training_metrics', 'all_completed_rollout_metrics', 'primary_validation_64x8_metrics',
    'native_performance_and_collection')
COHORT_COUNTS = {('ALL', 'ALL'): 128, ('family', 'DeepMath'): 78,
    ('family', 'aime24'): 3, ('family', 'aime25'): 1, ('family', 'amc23'): 4, ('family', 'math500'): 42,
    ('original_cohort', 'aime24'): 3, ('original_cohort', 'aime25'): 1,
    ('original_cohort', 'amc23'): 4, ('original_cohort', 'deepmath_D5_6'): 6,
    ('original_cohort', 'deepmath_D6_7'): 15, ('original_cohort', 'deepmath_D7_8'): 26,
    ('original_cohort', 'deepmath_D8_9'): 29, ('original_cohort', 'deepmath_D9_10'): 2,
    ('original_cohort', 'math500'): 42}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts, f'Nonportable path: {name}')
    out = root / p
    require(out.is_file() and not out.is_symlink(), f'Missing/nonregular artifact: {name}')
    require(out.resolve().is_relative_to(root.resolve()), f'Outside release: {name}')
    return out


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def coordinates(rows, columns):
    return [tuple(r[c] for c in columns) for r in rows]


def finite(row, key, nullable=False):
    value = row[key]
    if value is None or value == '':
        require(nullable, f'Missing numeric value: {key}')
        return None
    value = float(value)
    require(math.isfinite(value), f'Nonfinite value: {key}')
    return value


def count(row, key):
    value = finite(row, key)
    require(value >= 0 and value == int(value), f'Invalid count: {key}')
    return int(value)


def flag(row, key):
    require(row[key] in (True, False, 'True', 'False'), f'Invalid boolean: {key}')
    return row[key] is True or row[key] == 'True'


def near(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-9),
            f'Aggregate identity failed: {label}: {actual} vs {expected}')


def indexed(rows, fields, expected, label):
    keys = coordinates(rows, fields)
    require(len(keys) == len(set(keys)) and set(keys) == set(expected),
            f'Missing/duplicate/unexpected coordinates: {label}')
    return dict(zip(keys, rows))


def fraction(row, key, numerator, denominator):
    actual = finite(row, key, nullable=True)
    if denominator:
        require(actual is not None, f'Missing nonempty fraction: {key}')
        near(actual, numerator / denominator, key)
    else:
        require(actual is None, f'Unavailable fraction was zero-filled: {key}')


def moments(row, table):
    """Recover additive moments from public scalars, preserving empty strata."""
    n = count(row, 'tokens')
    out = {'tokens': n}

    def mean_sum(key, output, squared=False):
        x = finite(row, key, nullable=True)
        require((x is None) == (n == 0), f'Wrong empty-stratum semantics: {key}')
        if squared and x is not None:
            require(x >= 0, f'Negative RMS: {key}')
        out[output] = 0.0 if x is None else n * (x * x if squared else x)

    if table == 'gaps':
        mean_sum('signed_mean', 'sum_signed')
        mean_sum('absolute_mean', 'sum_abs')
        mean_sum('RMS', 'sum_squared', True)
        l2 = finite(row, 'L2')
        require(l2 >= 0, 'Negative discrepancy L2')
        near(l2 * l2, out['sum_squared'], 'L2 versus token-normalized RMS')
        require(out['sum_abs'] >= 0, 'Negative absolute gap sum')
        for direction in ('tail', 'down', 'up'):
            c = count(row, direction + '_tokens')
            require(c <= n, 'Tail exceeds valid token population')
            fraction(row, direction + '_fraction', c, n)
            key = 'tail_mean_or_zero' if direction == 'tail' else direction + '_tail_mean_or_zero'
            value = finite(row, key)
            require(value > .5 if c else value == 0, f'Invalid strict tail mean: {key}')
            out[direction + '_tokens'] = c
            out['sum_' + direction] = value * c
        require(out['tail_tokens'] == out['down_tokens'] + out['up_tokens'], 'Directional tail counts differ')
        near(out['sum_tail'], out['sum_down'] + out['sum_up'], 'Directional tail sums')
        near(finite(row, 'tail_excess_mean_or_zero'),
             max(0., finite(row, 'tail_mean_or_zero') - .5) if out['tail_tokens'] else 0., 'Tail excess')
    elif table == 'likelihood':
        mean_sum('log_PPL_nats', 'sum_nll')
        mean_sum('token_NLL_RMS', 'sum_nll_squared', True)
        nll, ppl = finite(row, 'log_PPL_nats', True), finite(row, 'PPL', True)
        overflow = nll is not None and nll > math.log(float.fromhex('0x1.fffffffffffffp+1023'))
        require(flag(row, 'PPL_overflow') == overflow, 'Wrong PPL overflow flag')
        if nll is None or overflow:
            require(ppl is None, 'Unavailable/overflow PPL was filled')
        else:
            require(ppl is not None, 'Missing finite PPL')
            near(ppl, math.exp(nll), 'PPL must exponentiate token-weighted NLL')
        for threshold in ('0_9', '0_99', '0_999', '0_9999'):
            key = 'probability_ge_' + threshold + '_fraction'
            v = finite(row, key, True)
            require((v is None) == (n == 0), 'Wrong empty probability fraction')
            require(v is None or 0 <= v <= 1, 'Probability fraction outside [0,1]')
            out[key] = 0. if v is None else v * n
    elif table == 'multipliers':
        mean_sum('weight_mean', 'weight_sum')
        mean_sum('weight_RMS', 'weight_squared_sum', True)
        for k in ('anchor_abs_A_mass_unweighted', 'anchor_abs_A_mass_weighted'):
            out[k] = finite(row, k)
            require(out[k] >= 0, 'Negative anchor mass')
        fraction(row, 'anchor_mass_retained_ratio', out['anchor_abs_A_mass_weighted'], out['anchor_abs_A_mass_unweighted'])
        for stem in ('changed', 'zero_weight', 'increased', 'reduced', 'cap2', 'tis_floor_weight', 'repeat'):
            c = count(row, stem + '_tokens')
            require(c <= n, 'Multiplier count exceeds population')
            fraction(row, stem + '_fraction', c, n)
            out[stem + '_tokens'] = c
        require(out['changed_tokens'] == out['increased_tokens'] + out['reduced_tokens'], 'Changed weight partition')
        require(out['zero_weight_tokens'] <= out['reduced_tokens'], 'Zero weight is not a reduced weight')
    elif table == 'coefficients':
        active = count(row, 'active_tokens')
        require(active <= n, 'Active direct coefficient count exceeds population')
        fraction(row, 'active_fraction', active, n)
        fraction(row, 'zero_coefficient_fraction', n - active, n)
        out.update(active_tokens=active, sum_signed=finite(row, 'direct_coefficient_signed_sum'),
                   sum_abs=finite(row, 'direct_coefficient_abs_mass'),
                   sum_squared=finite(row, 'direct_coefficient_L2') ** 2,
                   nonzero_at_zero_weight_tokens=count(row, 'nonzero_at_zero_weight_tokens'))
        require(out['sum_abs'] >= 0 and finite(row, 'direct_coefficient_L2') >= 0
                and out['nonzero_at_zero_weight_tokens'] == 0,
                'Invalid coefficient mass or direct gradient at a zero weight')
    else:
        raise ValueError(table)
    if n == 0:
        require(all(value == 0 for value in out.values()), 'Empty stratum has nonzero additive mass')
    return out


def same_sums(total, parts, label):
    for key, value in total.items():
        near(value, sum(part[key] for part in parts), label + ':' + key)


def validate_training_details(tables, arms=ARMS, steps=STEPS):
    """Full coordinate/denominator/moment checks; parameters support CPU audit subsets."""
    grid = {(a, str(s)) for a in arms for s in steps}
    updates = indexed(tables['optimizer_updates'], ('arm', 'round', 'inner'),
                      {(a, s, str(i)) for a, s in grid for i in range(1, 5)}, 'optimizer updates')
    rounds = indexed(tables['training_rounds'], ('arm', 'round'), grid, 'training rounds')
    identities = {(a, s, 'update', str(i)) for a, s in grid for i in range(1, 5)} | {
                  (a, s, 'round', '') for a, s in grid}
    physical = {key: count(updates[(key[0], key[1], key[3])] if key[2] == 'update'
                           else rounds[key[:2]], 'valid_tokens') for key in identities}
    indexes, sums = {}, {}
    for table, field, references, strata in (
        ('gaps', 'reference', GAP_REFERENCES, STRATA),
        ('likelihood', 'probability_source', PROBABILITY_SOURCES, STRATA),
        ('multipliers', None, ('',), SIGNS), ('coefficients', None, ('',), SIGNS)):
        fields = ('arm', 'round', 'level', 'inner') + ((field,) if field else ()) + ('stratum',)
        expected = {base + ((ref,) if field else ()) + (sign,)
                    for base in identities for ref in references for sign in strata}
        idx = indexed(tables[table], fields, expected, table)
        indexes[table], sums[table] = idx, {}
        for key, row in idx.items():
            require(row['population'] == 'accepted_training', 'Mixed training-detail population')
            sums[table][key] = moments(row, table)
        for base in identities:
            for ref in references:
                prefix = base + ((ref,) if field else ())
                group = {sign: sums[table][prefix + (sign,)] for sign in strata}
                require(group['all']['tokens'] == physical[base], 'Study weights changed the physical metric denominator')
                same_sums(group['all'], [group[s] for s in SIGNS[1:]], table + ' sign partition')
                if strata == STRATA:
                    same_sums(group['all'], [group['repeat'], group['nonrepeat']], table + ' repeat partition')
                    for sign in SIGNS[1:]:
                        same_sums(group[sign], [group[sign + '_repeat'], group[sign + '_nonrepeat']], table + ' sign/repeat partition')
            for ref in references:
                for sign in strata:
                    key = base + ((ref,) if field else ()) + (sign,)
                    canonical = base + ('historical_q', sign)
                    require(sums[table][key]['tokens'] == sums['gaps'][canonical]['tokens'],
                            'Different token populations across gap/PPL/weight/coefficient tables')
        for a, s in grid:
            for ref in references:
                for sign in strata:
                    suffix = ((ref,) if field else ()) + (sign,)
                    total = sums[table][(a, s, 'round', '') + suffix]
                    parts = [sums[table][(a, s, 'update', str(i)) + suffix] for i in range(1, 5)]
                    same_sums(total, parts, table + ' round versus four disjoint updates')
    for base in identities:
        flat = updates[(base[0], base[1], base[3])] if base[2] == 'update' else rounds[base[:2]]
        require(flat['population'] == 'accepted_training', 'Mixed flattened training population')
        for table, field, references in (('gaps', 'reference', GAP_REFERENCES),
                ('likelihood', 'probability_source', PROBABILITY_SOURCES), ('multipliers', None, ('',))):
            for ref in references:
                row = indexes[table][base + ((ref,) if field else ()) + ('all',)]
                prefix = ref + '_' if field else 'multiplier_'
                for key, value in row.items():
                    if key in ('arm', 'round', 'level', 'inner', 'population', 'stratum', field):
                        continue
                    actual = flat[prefix + key]
                    if value in ('', None):
                        require(actual in ('', None), 'Empty scalar was filled in flattened table')
                    elif key == 'PPL_overflow':
                        require(flag(flat, prefix + key) == flag(row, key), 'Flattened overflow differs')
                    else:
                        near(finite(flat, prefix + key), finite(row, key), 'Flattened ' + prefix + key)
        for sign in ('positive', 'negative'):
            coefficient = indexes['coefficients'][base + (sign,)]
            multiplier = indexes['multipliers'][base + (sign,)]
            for key, source, source_key in (
                (sign + '_direct_coefficient_abs_mass', coefficient, 'direct_coefficient_abs_mass'),
                (sign + '_active_token_fraction', coefficient, 'active_fraction'),
                (sign + '_anchor_mass_retained_ratio', multiplier, 'anchor_mass_retained_ratio')):
                a, b = finite(flat, key, True), finite(source, source_key, True)
                require((a is None) == (b is None), 'Flattened sign empty-stratum mismatch')
                if a is not None:
                    near(a, b, 'Flattened direct coefficient/anchor mass')
    for coord, row in rounds.items():
        children = [updates[coord + (str(i),)] for i in range(1, 5)]
        require(count(row, 'updates') == 4 and count(row, 'valid_tokens') == sum(count(c, 'valid_tokens') for c in children),
                'Round token denominator does not sum four disjoint updates')
        norms, deltas = [finite(c, 'preclip_norm') for c in children], [finite(c, 'adam_delta_l2') for c in children]
        for i, value in enumerate(norms, 1):
            near(finite(row, f'B{i}_preclip_norm'), value, 'Inner norm')
        near(finite(row, 'preclip_norm_mean'), sum(norms) / 4, 'Round mean norm')
        near(finite(row, 'preclip_norm_max'), max(norms), 'Round maximum norm')
        near(finite(row, 'preclip_norm_median'), sum(sorted(norms)[1:3]) / 2, 'Round median norm')
        peak = count(row, 'peak_inner'); require(1 <= peak <= 4, 'Invalid peak inner')
        near(norms[peak - 1], max(norms), 'Peak inner norm')
        near(finite(row, 'adam_path_l2_sum'), sum(deltas), 'Adam path is sum of update norms')
        near(finite(row, 'adam_update_l2_max'), max(deltas), 'Maximum Adam update')
    return indexes


def validate_accepted_signs(tables, detail_indexes, arms=ARMS, steps=STEPS):
    accepted = indexed(tables['accepted_training'], ('arm', 'round'),
                       {(a, str(s)) for a in arms for s in steps}, 'accepted training')
    keys = coordinates(tables['accepted_training_by_sign'], ('arm', 'round', 'stratum'))
    expected = set()
    for a, s in accepted:
        for sign in SIGNS[1:]:
            n = count(detail_indexes['gaps'][(a, s, 'round', '', 'historical_q', sign)], 'tokens')
            if n:
                expected.add((a, s, sign))
        require((a, s, 'positive') in expected and (a, s, 'negative') in expected,
                'Mixed accepted accuracy groups must have both advantage signs')
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            'Accepted sign rows must match actual nonempty strata; do not invent zero rows')
    idx = dict(zip(keys, tables['accepted_training_by_sign']))
    for (a, s), row in accepted.items():
        parts = [idx[(a, s, sign)] for sign in SIGNS[1:] if (a, s, sign) in idx]
        require(sum(count(x, 'responses') for x in parts) == count(row, 'responses') == 512,
                'Accepted sign response partition is incomplete')
        n = count(row, 'tokens')
        require(sum(count(x, 'tokens') for x in parts) == n and
                n == count(detail_indexes['gaps'][(a, s, 'round', '', 'historical_q', 'all')], 'tokens'),
                'Accepted token count differs from original training-mask denominator')
        near(finite(row, 'mean_length'), n / 512, 'Accepted mean length')
        for x in parts:
            require(x['population'] == 'accepted_training' and count(x, 'responses') > 0, 'Invalid sign population')
            require(count(x, 'tokens') == count(detail_indexes['gaps'][(a, s, 'round', '', 'historical_q', x['stratum'])], 'tokens'),
                    'Sign token denominator changed')
            near(finite(x, 'mean_length'), count(x, 'tokens') / count(x, 'responses'), 'Sign mean length')
        for key in ('accuracy', 'strict_repetition', 'normalized_text_cycle128_any', 'gibberish_candidate',
                    'response_hit_token_cap', 'first_word_alright', 'first_word_to'):
            near(finite(row, key), sum(finite(x, key) * count(x, 'responses') for x in parts) / 512,
                 'Response-weighted accepted sign fractions: ' + key)
        for key in ('exact_confirmed_repeat_tokens_token_fraction', 'noisy_repeat_primary_tokens_token_fraction',
                    'noisy_repeat_strict_tokens_token_fraction', 'repeat_strict_union_tokens_token_fraction'):
            near(finite(row, key), sum(finite(x, key) * count(x, 'tokens') for x in parts) / n,
                 'Token-weighted accepted repeat fractions: ' + key)


def validate_secondary_tables(tables, endpoints=ENDPOINTS):
    """Question-weighted natural NLL and HF probabilities are separate populations."""
    endpoint_index = indexed(tables['endpoint_metrics.csv'], ('endpoint',), {(e,) for e in endpoints}, 'secondary endpoint')
    probability_index = indexed(tables['prompt_only_probabilities.csv'], ('endpoint',), {(e,) for e in endpoints}, 'HF endpoint')
    qrows = tables['per_question.csv']
    keys = coordinates(qrows, ('endpoint', 'qid'))
    require(len(keys) == len(set(keys)), 'Duplicate secondary question')
    by = {e: {} for e in endpoints}
    for r in qrows:
        require(r['endpoint'] in by and count(r, 'response_samples') == 4, 'Wrong secondary question population')
        by[r['endpoint']][r['qid']] = r
    qids = set(by[endpoints[0]])
    require(len(qids) == 128 and all(set(v) == qids for v in by.values()), 'Secondary question grid changed')
    for endpoint in endpoints:
        natural, probe = endpoint_index[(endpoint,)], probability_index[(endpoint,)]
        step = 600 if endpoint == 'source600' else 650
        require(count(natural, 'step') == count(probe, 'step') == step, 'Source600/650 endpoint step mislabeled')
        require(count(natural, 'questions') == count(probe, 'questions') == 128 and
                count(natural, 'responses') == 512 and count(natural, 'samples_per_question') == 4,
                'Wrong secondary sampling counts')
        require(probe['backend'] == 'HF BF16 flash_attention_2', 'Prompt-probability backend changed')
        for metric in SECONDARY_METRICS:
            values = [finite(r, metric) for r in by[endpoint].values()]
            require(all(0 <= v <= 1 for v in values) if metric not in ('tokens', 'log_PPL_q')
                    else all(v >= 0 for v in values), 'Invalid secondary metric: ' + metric)
            near(finite(natural, metric), sum(values) / 128, 'Equal-question secondary mean: ' + metric)
        for token in TARGET_TOKENS:
            values = [finite(r, 'HF_P_' + token) for r in by[endpoint].values()]
            require(all(0 <= v <= 1 for v in values), 'Invalid HF probability')
            near(finite(probe, 'P_' + token), sum(values) / 128, 'HF equal-question probability')
    pairs = {(reference, endpoint) for reference in ('source600', 'baseline_650') if reference in endpoints
             for endpoint in endpoints if endpoint != reference and not (reference == 'baseline_650' and endpoint == 'source600')}
    for table, field, metrics, value_key in (
        ('paired_natural_contrasts.csv', 'metric', SECONDARY_METRICS, 'mean_difference'),
        ('paired_prompt_probability_contrasts.csv', 'target_token', TARGET_TOKENS, 'mean_probability_difference')):
        idx = indexed(tables[table], ('reference', 'endpoint', field),
                      {(a, b, m) for a, b in pairs for m in metrics}, table)
        for (reference, endpoint, metric), row in idx.items():
            require(count(row, 'questions') == 128 and not flag(row, 'multiplicity_adjusted'), 'Contrast scope changed')
            require(finite(row, 'ci95_low') <= finite(row, 'ci95_high'), 'Inverted contrast interval')
            if field == 'metric':
                require(count(row, 'samples_per_question') == 4 and count(row, 'bootstrap_draws') == 10000, 'Natural bootstrap protocol changed')
                key = metric
            else:
                require(row['backend'] == 'HF BF16 flash_attention_2', 'HF contrast backend changed')
                key = 'HF_P_' + metric
            delta = sum(finite(by[endpoint][q], key) - finite(by[reference][q], key) for q in qids) / 128
            near(finite(row, value_key), delta, 'Paired-question contrast mean')
    cohorts = indexed(tables['all_cohort_descriptive.csv'], ('endpoint', 'level', 'cohort'),
                      {(e, level, cohort) for e in endpoints for level, cohort in COHORT_COUNTS}, 'cohort grid')
    for endpoint in endpoints:
        for (level, cohort), n in COHORT_COUNTS.items():
            row = cohorts[(endpoint, level, cohort)]
            require(count(row, 'questions') == n and count(row, 'responses') == 4 * n, 'Cohort sample count changed')
            correct, unknown = count(row, 'correct_responses'), count(row, 'unknown_responses')
            require(correct <= 4 * n and unknown <= 4 * n, 'Cohort grade count exceeds responses')
            near(finite(row, 'accuracy'), correct / (4 * n), 'Cohort accuracy')
            near(finite(row, 'unknown'), unknown / (4 * n), 'Cohort unknown fraction')
        all_row = cohorts[(endpoint, 'ALL', 'ALL')]
        for metric in SECONDARY_METRICS:
            near(finite(all_row, metric), finite(endpoint_index[(endpoint,)], metric), 'ALL cohort matches endpoint')
            for level in ('family', 'original_cohort'):
                rows = [cohorts[(endpoint, lv, c)] for lv, c in COHORT_COUNTS if lv == level]
                near(finite(all_row, metric), sum(finite(r, metric) * count(r, 'questions') for r in rows) / 128,
                     'Question-weighted cohort partition: ' + metric)
    return qids


def validate_auxiliary_training_tables(tables, arms=ARMS):
    grid = {(a, str(s)) for a in arms for s in STEPS}
    updates = {a: [r for r in tables['optimizer_updates'] if r['arm'] == a] for a in arms}
    stability = indexed(tables['stability_summary'], ('arm',), {(a,) for a in arms}, 'stability summary')
    for arm, rows in updates.items():
        row = stability[(arm,)]
        require(count(row, 'observed_updates') == count(row, 'expected_updates') == 200
                and flag(row, 'complete_200_updates'), 'Partial stability summary')
        values = sorted(finite(r, 'preclip_norm') for r in rows)
        near(finite(row, 'maximum'), values[-1], 'Stability maximum')
        near(finite(row, 'mean'), sum(values) / 200, 'Stability mean')
        for field, q in (('median', .5), ('p95', .95), ('p99', .99)):
            position = 199 * q
            lower, upper = math.floor(position), math.ceil(position)
            near(finite(row, field), values[lower] + (values[upper] - values[lower]) * (position - lower), 'Stability quantile')
        for threshold in (1, 10, 50, 100):
            require(count(row, f'count_gt_{threshold}') == sum(v > threshold for v in values), 'Stability threshold count')
        peak = [r for r in rows if r['round'] == row['peak_round'] and r['inner'] == row['peak_inner']]
        require(len(peak) == 1, 'Invalid stability peak coordinate')
        near(finite(peak[0], 'preclip_norm'), values[-1], 'Stability peak coordinate/value')
        window = (count(row, 'peak_round') - 601) // 10
        require(row['peak_window'] == f'{601 + 10 * window}-{610 + 10 * window}', 'Wrong stability peak window')
    matching = indexed(tables['initial_batch_and_validation_matching'], ('arm',),
                       {(a,) for a in arms if a != 'baseline'}, 'initial matching audit')
    for (arm,), row in matching.items():
        require(row['reference'] == 'baseline' and flag(row, 'accepted601_available')
                and flag(row, 'validation600_available') and flag(row, 'validation600_same_panel_sha'),
                'Incomplete initial batch/primary600 comparison')
        require(flag(row, 'collector_intentionally_changed') == (arm == 'sync_fresh'), 'Collector audit mislabeled')
    indexed(tables['native_round_performance'], ('arm', 'round'), grid, 'native round performance')
    require(all(r['population'] == 'native_logged_round_metrics' for r in tables['native_round_performance']),
            'Wrong native performance population')
    native = tables['native_logged_metrics_long']
    keys = coordinates(native, ('arm', 'round', 'metric'))
    require(len(keys) == len(set(keys)) and {k[:2] for k in keys} == grid, 'Incomplete/duplicate native logged coordinates')
    for row in native:
        require(row['aggregation_scope'] == 'native_logged_value_no_reinterpretation', 'Native metric scope changed')
        # Native optional/derived nonfinite values are deliberately left blank by
        # the parser; unavailable metrics must not become fabricated zeros.
        if row['value'] not in ('', None, 'True', 'False'):
            finite(row, 'value')
    contrast = indexed(tables['endpoint_paired_differences'], ('arm', 'reference', 'round', 'metric'),
                       {(a, 'baseline', '650', m) for a in arms if a != 'baseline' for m in PRIMARY_CONTRAST_METRICS},
                       'primary endpoint contrasts')
    by_arm = {a: {r['qid']: r for r in tables['primary_validation_questions'] if r['arm'] == a and r['round'] == '650'}
              for a in arms}
    for (arm, _, _, metric), row in contrast.items():
        require(row['population'] == 'primary_validation_64x8' and count(row, 'paired_questions') == 64
                and count(row, 'bootstrap_draws') == 10000 and count(row, 'seed') == 20261106
                and not flag(row, 'multiple_comparison_adjusted'), 'Primary contrast protocol changed')
        require(finite(row, 'CI95_low') <= finite(row, 'CI95_high'), 'Inverted primary interval')
        delta = sum(finite(by_arm[arm][q], metric) - finite(by_arm['baseline'][q], metric) for q in by_arm['baseline']) / 64
        near(finite(row, 'mean_difference'), delta, 'Primary paired-question contrast')


def validate(root):
    data = root / 'data/final-stability-600-650'
    manifest = read(data / 'manifest.json')
    require(manifest['complete'] is True, 'Release is not complete')
    require(manifest['scientific_artifact'] is True and manifest['fixture_only'] is False,
            'Non-scientific release')
    files = manifest['files']
    names = [r['file'] for r in files]
    require(len(names) == len(set(names)), 'Duplicate release artifact')
    for row in files:
        path = safe_path(root, row['file'])
        require(SHA.fullmatch(row['published_sha256']) is not None, 'Invalid published SHA')
        require(digest(path) == row['published_sha256'], f'Hash mismatch: {row["file"]}')
        original = row.get('original_sha256')
        require(original is None or SHA.fullmatch(original) is not None, 'Invalid source SHA')
        if original is not None and original != row['published_sha256']:
            require(bool(row.get('transformation')), f'Unexplained publication transformation: {row["file"]}')
        if path.suffix in {'.json', '.csv', '.md', '.py', '.txt', '.svg'}:
            text = path.read_text()
            private_prefixes = [f'/{name}/' for name in ('localhome', 'raid', 'tmp')]
            key_delimiters = ['BEGIN ' + kind for kind in ('PRIVATE KEY', 'OPENSSH PRIVATE KEY')]
            require(not any(x in text for x in private_prefixes + key_delimiters),
                    f'Private runtime material in {row["file"]}')
            require(re.search(r'\b(?:10|192\.168)\.(?:\d{1,3}\.){1,2}\d{1,3}\b', text) is None,
                    f'Private address in {row["file"]}')

    tr = data / 'training'
    summary = read(tr / 'summary.json')
    require(summary['complete'] is True and summary['training_complete'] is True,
            'Training analysis is incomplete')
    require(summary['scientific_artifact'] is True and summary['fixture_only'] is False,
            'Training parser fixtures cannot be published as results')
    require(set(summary['arms']) == set(ARMS), 'Wrong training arms')
    for arm in ARMS:
        s = summary['arms'][arm]
        require(s['state'] == 'complete' and s['verified_updates'] == 200
                and s['complete_rounds'] == 50 and not s['pending'], f'Incomplete arm: {arm}')
        require(s['accepted_assessed_steps'] == STEPS and s['completed_assessed_steps'] == STEPS,
                f'Missing accepted/all-completed population: {arm}')
        require(s['validation_steps'] == EVAL_STEPS, f'Wrong primary evaluations: {arm}')
    tables = {}
    for name, row in summary['tables'].items():
        path = safe_path(tr, row['file'])
        require(digest(path) == row['sha256'], f'Training table hash mismatch: {name}')
        tables[name] = csv_rows(path)
        require(len(tables[name]) == row['rows'], f'Training table count mismatch: {name}')
    training_figures = []
    expected_training_figures = {f'figures/{name}.{suffix}' for name in TRAINING_FIGURE_STEMS for suffix in ('png', 'svg')}
    require(len(summary['figures']) == 18 and {row['file'] for row in summary['figures']} == expected_training_figures,
            'Final training figures must contain all nine plots in both PNG and SVG; --no-plots is incomplete')
    for row in summary['figures']:
        path = safe_path(tr, row['file'])
        require(digest(path) == row['sha256'], f'Training figure hash mismatch: {row["file"]}')
        training_figures.append(path)
    require(len(training_figures) == len(set(training_figures)), 'Duplicate training figure')
    expected_tables = {'optimizer_updates', 'training_rounds', 'accepted_training', 'accepted_training_by_sign',
        'all_completed_rollout', 'primary_validation', 'primary_validation_questions', 'endpoint_paired_differences',
        'stability_summary', 'initial_batch_and_validation_matching', 'native_round_performance',
        'native_logged_metrics_long', 'gaps', 'likelihood', 'multipliers', 'coefficients'}
    require(set(tables) == expected_tables, 'Missing or unexpected training result table')
    updates = tables['optimizer_updates']
    expected = {(a, str(s), str(i)) for a in ARMS for s in STEPS for i in range(1, 5)}
    coords = coordinates(updates, ['arm', 'round', 'inner'])
    require(len(coords) == len(set(coords)) == 1600 and set(coords) == expected,
            'Optimizer update coordinates do not form the complete eight-arm campaign')
    for r in updates:
        require(int(r['adam_step']) == 4 * (int(r['round']) - 1) + int(r['inner']), 'Wrong Adam step')
        require(int(r['ranks_verified']) == 8 and int(r['valid_tokens']) > 0, 'Incomplete rank reduction')
        for key in ('preclip_norm', 'adam_delta_l2'):
            v = float(r[key])
            require(math.isfinite(v) and v >= 0, f'Invalid actual update metric {key}')
    details = validate_training_details(tables)
    validate_accepted_signs(tables, details)
    for name, population in (('accepted_training', 'accepted_training'),
                             ('all_completed_rollout', 'all_completed_rollout')):
        rows = tables[name]
        coords = coordinates(rows, ['arm', 'round'])
        require(len(coords) == len(set(coords)) == 400 and
                set(coords) == {(a, str(s)) for a in ARMS for s in STEPS}, f'Wrong population rows: {name}')
        require(all(r['population'] == population for r in rows), f'Mixed population: {name}')
        if name == 'accepted_training':
            require(all(int(r['responses']) == 512 for r in rows), 'Wrong accepted batch size')
        for row in rows:
            n, responses = count(row, 'tokens'), count(row, 'responses')
            require(n >= responses and (responses > 0 or n == 0), 'Invalid response/token population size')
            if responses:
                near(finite(row, 'mean_length'), n / responses, 'Population mean length')
            else:
                require(finite(row, 'mean_length', True) is None, 'Empty completed population was zero-filled')
            if name == 'all_completed_rollout' and row.get('generation_q_tokens') not in (None, ''):
                q = {key.removeprefix('generation_q_'): value for key, value in row.items() if key.startswith('generation_q_')}
                qm = moments(q, 'likelihood')
                require(qm['tokens'] <= n, 'Known generated-q population exceeds all completed tokens')
                if count(row, 'q_missing_responses') == 0:
                    require(qm['tokens'] == n, 'Complete generated-q population uses a different token denominator')
    validation = tables['primary_validation']
    coords = coordinates(validation, ['arm', 'round'])
    require(len(coords) == len(set(coords)) == 48 and
            set(coords) == {(a, str(s)) for a in ARMS for s in EVAL_STEPS}, 'Wrong primary evaluation grid')
    require(all(int(r['responses']) == 512 for r in validation), 'Wrong primary response counts')
    question_rows = tables['primary_validation_questions']
    qcoords = coordinates(question_rows, ['arm', 'round', 'qid'])
    require(len(qcoords) == len(set(qcoords)) == 3072, 'Wrong primary question coordinates')
    primary_qids = None
    for point in validation:
        qrows = [r for r in question_rows if (r['arm'], r['round']) == (point['arm'], point['round'])]
        qids = {r['qid'] for r in qrows}
        require(len(qrows) == len(qids) == 64, 'Wrong primary question count')
        if primary_qids is None:
            primary_qids = qids
        require(qids == primary_qids, 'Primary question identities changed')
        for key in ('accuracy', 'mean_length', 'strict_repetition', 'normalized_text_cycle128_any',
                    'normalized_text_cycle128_numeric', 'normalized_text_cycle128_alphabetic',
                    'normalized_text_cycle128_other', 'response_hit_token_cap', 'first_word_alright',
                    'first_word_to', 'has_think_tag_prefix', 'gibberish_candidate'):
            mean = sum(float(r[key]) for r in qrows) / 64
            require(math.isclose(mean, float(point[key]), rel_tol=1e-10, abs_tol=1e-10),
                    f'Primary question/aggregate identity failed: {key}')
    validate_auxiliary_training_tables(tables)

    sec = data / 'secondary'
    endpoint = read(sec / 'summary.json')
    require(endpoint['complete'] is True and endpoint['available_endpoints'] == ENDPOINTS
            and not endpoint['missing_endpoints'], 'Secondary endpoints incomplete or reordered')
    require(endpoint['baseline_treatment_comparisons'] == 7, 'Wrong baseline comparison count')
    require(endpoint['available_cohort_endpoints'] == ENDPOINTS and not endpoint['missing_cohort_endpoints'],
            'Incomplete descriptive cohort breakdown')
    require(endpoint['bootstrap'] == {'unit': 'qid', 'questions': 128, 'equal_question_weights': True,
            'samples_per_question': 4, 'draws': 10000, 'seed': 20261106}, 'Bootstrap protocol changed')
    require(endpoint['unknown_grades_retained'] is True and endpoint['multiplicity_adjusted'] is False
            and endpoint['conditional_single_training_trajectory'] is True, 'Statistical scope changed')
    require(endpoint['panel_cases_sha256'] == '408b1d118eae44e5084f4d6b29fb64e77381af9e25f32c69ae6de7cf93ab58d3',
            'Secondary panel changed')
    require(endpoint['jobs_sha256'] == 'e4ba8f4ea2e157d1b12e3d2bffab2e664bb1940bf0215c630a2e6d3e196dce6c',
            'Secondary seeded jobs changed')
    sm = read(sec / 'ARTIFACT_MANIFEST.json')
    require(sm['complete'] is True, 'Secondary artifact manifest incomplete')
    for name, expected_sha in sm['outputs_sha256'].items():
        require(digest(safe_path(sec, name)) == expected_sha, f'Secondary hash mismatch: {name}')
    counts = {'endpoint_metrics.csv': 9, 'prompt_only_probabilities.csv': 9,
              'per_question.csv': 1152, 'paired_natural_contrasts.csv': 255,
              'paired_prompt_probability_contrasts.csv': 45, 'all_cohort_descriptive.csv': 135}
    for name, expected_count in counts.items():
        require(len(csv_rows(sec / name)) == expected_count, f'Wrong secondary count: {name}')
    require(set(counts).issubset(sm['outputs_sha256']), 'Secondary CSV missing from artifact manifest')
    validate_secondary_tables({name: csv_rows(sec / name) for name in counts})
    secondary_questions = csv_rows(sec / 'per_question.csv')
    coords = coordinates(secondary_questions, ['endpoint', 'qid'])
    require(len(coords) == len(set(coords)) == 1152, 'Duplicate secondary question')
    secondary_qids = None
    for name in ENDPOINTS:
        qids = {r['qid'] for r in secondary_questions if r['endpoint'] == name}
        require(len(qids) == 128, f'Wrong secondary question count: {name}')
        if secondary_qids is None:
            secondary_qids = qids
        require(qids == secondary_qids, 'Secondary question identities changed')
    require(not (primary_qids & secondary_qids), 'Primary and secondary panels overlap')

    ref = root / 'reference/stability'
    reference_files = read(ref / 'source_manifest.json')['files']
    for row in reference_files:
        require(digest(safe_path(ref, row['file'])) == row['published_sha256'],
                f'Portable reference changed: {row["file"]}')
    critical = [tr / 'summary.json', sec / 'summary.json', sec / 'ARTIFACT_MANIFEST.json',
                ref / 'source_manifest.json']
    critical += [tr / row['file'] for row in summary['tables'].values()]
    critical += training_figures
    critical += [sec / name for name in sm['outputs_sha256']]
    critical += [ref / row['file'] for row in reference_files]
    require({str(p.relative_to(root)) for p in critical}.issubset(set(names)),
            'Critical result or reference file is not covered by the release manifest')
    return {'complete_arms': 8, 'training_rounds': 400, 'Adam_updates': 1600,
            'primary_evaluations': 48, 'secondary_endpoints': 9,
            'release_files_verified': len(files)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print('PASS:', json.dumps(validate(args.root), sort_keys=True))
