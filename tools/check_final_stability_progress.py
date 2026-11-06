"""Validate a clearly incomplete, arm-by-arm scientific progress release.

Completed arms must still contain all 200 updates and all primary evaluations.
Incomplete all-completed text assessment is reported explicitly. This validator
does not waive the separate final eight-arm release gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import check_final_stability as final


def validate(root):
    data = root / 'data/final-stability-600-650'
    manifest = final.read(data / 'manifest.json')
    final.require(manifest['scientific_artifact'] is True and manifest['fixture_only'] is False,
                  'Progress data must be genuine scientific observations')
    if manifest['complete']:
        return final.validate(root)
    final.require(manifest['release_kind'] == 'incremental_scientific_progress', 'Wrong progress-release scope')
    arms = tuple(manifest['completed_training_arms'])
    final.require(arms and arms == tuple(a for a in final.ARMS if a in arms), 'Wrong completed arm identities/order')
    names = [r['file'] for r in manifest['files']]
    final.require(len(names) == len(set(names)), 'Duplicate released artifact')
    for row in manifest['files']:
        path = final.safe_path(root, row['file'])
        final.require(final.digest(path) == row['published_sha256'], 'Release hash mismatch: ' + row['file'])
        if row.get('original_sha256') and row['original_sha256'] != row['published_sha256']:
            final.require(bool(row.get('transformation')), 'Unexplained publication transformation')
        if path.suffix in ('.json', '.csv', '.md', '.py', '.txt', '.svg'):
            text = path.read_text()
            forbidden = [f'/{name}/' for name in ('localhome', 'raid', 'tmp')]
            forbidden += ['BEGIN ' + kind for kind in ('PRIVATE KEY', 'OPENSSH PRIVATE KEY')]
            final.require(not any(x in text for x in forbidden), 'Private runtime material: ' + row['file'])
            final.require(final.re.search(r'\b(?:10|192\.168)\.(?:\d{1,3}\.){1,2}\d{1,3}\b', text) is None,
                          'Private address: ' + row['file'])

    training = data / 'training'
    summary = final.read(training / 'summary.json')
    final.require(summary['complete'] is False and summary['scientific_artifact'] is True
                  and summary['fixture_only'] is False, 'Incomplete campaign was promoted to final')
    final.require(set(summary['arms']) == set(final.ARMS), 'Planned arms omitted')
    for arm, state in summary['arms'].items():
        if arm in arms:
            final.require(state['state'] == 'complete' and state['verified_updates'] == 200
                          and state['complete_rounds'] == 50 and not state['pending'], 'Incomplete training arm')
            final.require(state['accepted_assessed_steps'] == final.STEPS
                          and state['validation_steps'] == final.EVAL_STEPS, 'Missing completed-arm observations')
        else:
            final.require(state['verified_updates'] == 0, 'An unfinished training arm entered the completed-arm release')
    tables, critical = {}, [training / 'summary.json']
    for name, entry in summary['tables'].items():
        path = final.safe_path(training, entry['file'])
        final.require(final.digest(path) == entry['sha256'], 'Training CSV hash mismatch')
        tables[name] = final.csv_rows(path)
        final.require(len(tables[name]) == entry['rows'], 'CSV row count mismatch')
        critical.append(path)
    details = final.validate_training_details(tables, arms=arms)
    final.validate_accepted_signs(tables, details, arms=arms)
    # The producer keeps explicit "not available" rows for all seven planned
    # initial-batch comparisons, even before their training archives arrive.
    matching = final.indexed(tables['initial_batch_and_validation_matching'], ('arm',),
                             {(a,) for a in final.ARMS if a != 'baseline'}, 'Planned matching grid')
    for (arm,), row in matching.items():
        if arm not in arms:
            final.require(not final.flag(row, 'accepted601_available')
                          and not final.flag(row, 'validation600_available'),
                          'An unreported arm already has initial-matching observations')
    completed_tables = dict(tables)
    completed_tables['initial_batch_and_validation_matching'] = [
        row for row in tables['initial_batch_and_validation_matching'] if row['arm'] in arms]
    final.validate_auxiliary_training_tables(completed_tables, arms=arms)
    for row in tables['optimizer_updates']:
        final.require(final.count(row, 'ranks_verified') == 8
                      and final.count(row, 'adam_step') == 4 * (int(row['round']) - 1) + int(row['inner']),
                      'Wrong actual rank/update counters')
        final.require(final.finite(row, 'preclip_norm') >= 0 and final.finite(row, 'adam_delta_l2') >= 0,
                      'Invalid actual optimizer metric')

    primary = final.indexed(tables['primary_validation'], ('arm', 'round'),
                            {(a, str(s)) for a in arms for s in final.EVAL_STEPS}, 'Primary progress grid')
    questions = tables['primary_validation_questions']
    final.require(len(final.coordinates(questions, ('arm', 'round', 'qid'))) ==
                  len(set(final.coordinates(questions, ('arm', 'round', 'qid')))) == len(arms) * 6 * 64,
                  'Wrong primary question coordinates')
    primary_qids = None
    for coord, row in primary.items():
        qrows = [r for r in questions if (r['arm'], r['round']) == coord]
        qids = {r['qid'] for r in qrows}
        final.require(len(qids) == 64 and final.count(row, 'responses') == 512, 'Incomplete primary panel')
        if primary_qids is None:
            primary_qids = qids
        final.require(qids == primary_qids, 'Primary panel changed')
        for metric in final.PRIMARY_CONTRAST_METRICS:
            final.near(final.finite(row, metric), sum(final.finite(q, metric) for q in qrows) / 64,
                       'Primary question-weighted mean')

    completed = tables['all_completed_rollout']
    coords = final.coordinates(completed, ('arm', 'round'))
    expected = {(arm, str(step)) for arm in arms for step in summary['arms'][arm]['completed_assessed_steps']}
    final.require(len(coords) == len(set(coords)) and set(coords) == expected, 'Partial rollout coverage mismatch')
    for row in completed:
        final.require(row['population'] == 'all_completed_rollout' and int(row['round']) in final.STEPS,
                      'Wrong all-completed population')
    pending = {arm: sorted(set(final.STEPS) - set(summary['arms'][arm]['completed_assessed_steps'])) for arm in arms}
    final.require(manifest['pending_all_completed_rounds'] == pending, 'Pending text assessment was hidden')
    closure_path = data / 'full_tis_candidate_closure.json'
    if closure_path.is_file():
        closure = final.read(closure_path)
        crows = [r for r in completed if r['arm'] == 'full_tis']
        final.require(closure['complete'] is True and closure['arm'] == 'full_tis'
                      and closure['not_heldout_or_fixed_checkpoint_accuracy'] is True,
                      'Wrong closure population/scope')
        final.require({int(r['round']) for r in crows} == set(final.STEPS), 'Partial closure promoted to full')
        n = sum(final.count(r, 'responses') for r in crows)
        tokens = sum(final.count(r, 'tokens') for r in crows)
        groups = sum(final.count(r, 'groups') for r in crows)
        hist = [sum(final.count(r, f'groups_correct_{k}_of8') for r in crows) for k in range(9)]
        correct = sum(k * hist[k] for k in range(9))
        final.require((n, tokens, groups, correct) == (closure['candidate_responses'],
                      closure['candidate_tokens'], closure['candidate_groups'], closure['candidate_correct']),
                      'Closure totals disagree with immutable CSV')
        final.require(closure['group_class_counts'] == {'all_wrong': hist[0], 'all_correct': hist[8],
                      'mixed': sum(hist[1:8])}, 'Closure group classes disagree')
        final.near(closure['candidate_accuracy'], correct / n, 'Candidate accuracy')
        final.near(closure['length_mean'], tokens / n, 'Candidate length')
        for key, count in closure['candidate_counts'].items():
            total = sum(round(final.finite(r, key) * final.count(r, 'responses'))
                        for r in crows if final.count(r, 'responses'))
            final.require(total == count, 'Closure mechanical candidate mismatch: ' + key)
            final.near(closure['candidate_fractions'][key], count / n, 'Candidate fraction')
        q = closure['rollout_q']
        qrows = [r for r in crows if final.count(r, 'responses')]
        qn = sum(final.count(r, 'generation_q_tokens') for r in qrows)
        final.require(qn == q['token_count'] == tokens and q['actor_p_measured'] is False
                      and q['actor_minus_rollout_d_measured'] is False, 'Wrong candidate q scope')
        qmean = sum(final.count(r, 'generation_q_tokens') * final.finite(r, 'generation_q_log_PPL_nats')
                    for r in qrows if final.count(r, 'generation_q_tokens')) / qn
        final.near(q['mean_NLL_nats_log_PPL'], qmean, 'Pooled candidate q NLL')
        rms = (sum(final.count(r, 'generation_q_tokens') * final.finite(r, 'generation_q_token_NLL_RMS') ** 2
                   for r in qrows if final.count(r, 'generation_q_tokens')) / qn) ** .5
        final.near(q['NLL_RMS_nats'], rms, 'Pooled candidate q RMS')
        account = closure['accepted_population_accounting']; inherited = account['inherited_ready_at600']
        final.require(inherited['unmatched_exactly_source600_ready_multiset'] is True
                      and inherited['direct_original_ID_proof'] is True
                      and inherited['source_ready_groups'] * 8 == inherited['source_ready_responses']
                      and sum(inherited['source_ready_consumed_in_training_steps'].values()) == inherited['source_ready_responses']
                      and account['matched_new_completions'] + inherited['source_ready_responses'] == account['accepted_uses']
                      and account['matched_new_completions'] + sum(account['newly_completed_not_trained_in601_650'].values()) == n,
                      'Candidate/accepted accounting does not close')
        final.require(account['matched_all42_pure_fields_equal'] is True and account['mismatch_identities'] == 0,
                      'Candidate row-statistics mismatch')
        critical.append(closure_path)
    figure_names = {f'figures/{stem}.{suffix}' for stem in final.TRAINING_FIGURE_STEMS for suffix in ('png', 'svg')}
    final.require({r['file'] for r in summary['figures']} == figure_names and len(summary['figures']) == 18,
                  'Expected all 18 clearly partial training figures')
    for row in summary['figures']:
        path = final.safe_path(training, row['file'])
        final.require(final.digest(path) == row['sha256'], 'Training figure SHA mismatch')
        critical.append(path)

    secondary = data / 'secondary'
    secondary_summary = final.read(secondary / 'summary.json')
    endpoints = ['source600', *[a + '_650' for a in arms]]
    final.require(secondary_summary['available_endpoints'] == endpoints
                  and secondary_summary['complete'] == (len(endpoints) == 9), 'Wrong secondary progress')
    final.require(manifest['secondary_endpoints'] == endpoints, 'Secondary release state mismatch')
    artifact_manifest = final.read(secondary / 'ARTIFACT_MANIFEST.json')
    final.require(artifact_manifest['complete'] == (len(endpoints) == 9), 'Secondary manifest promoted to final')
    secondary_tables = {}
    for name, sha in artifact_manifest['outputs_sha256'].items():
        path = final.safe_path(secondary, name)
        final.require(final.digest(path) == sha, 'Secondary artifact SHA mismatch')
        if path.suffix == '.csv':
            secondary_tables[name] = final.csv_rows(path)
        critical.append(path)
    secondary_qids = final.validate_secondary_tables(secondary_tables, endpoints=endpoints)
    final.require(not primary_qids.intersection(secondary_qids), 'Primary and secondary panels overlap')
    critical.append(secondary / 'ARTIFACT_MANIFEST.json')
    reference = root / 'reference/stability'
    for row in final.read(reference / 'source_manifest.json')['files']:
        path = final.safe_path(reference, row['file'])
        final.require(final.digest(path) == row['published_sha256'], 'Portable reference changed')
        critical.append(path)
    critical.append(reference / 'source_manifest.json')
    final.require({str(p.relative_to(root)) for p in critical}.issubset(names), 'Unmanifested progress artifact')
    return {'campaign_complete': False, 'completed_training_arms': list(arms),
            'actual_updates': 200 * len(arms), 'secondary_endpoints': len(endpoints),
            'pending_all_completed_rounds': {a: len(s) for a, s in pending.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print('PASS:', json.dumps(validate(args.root), sort_keys=True))
