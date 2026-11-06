"""Redraw the eight-arm training figures from published aggregate CSVs only.

No raw responses, token IDs, model files or training framework are required.
This redraw preserves the plotted statistics, not original image byte identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS = (
    'baseline', 'full_tis', 'negative_tis', 'abs_gap_mask',
    'negative_down_mask', 'repeat_negative_down_mask', 'negative_mass', 'sync_fresh',
)
LABELS = dict(zip(ARMS, (
    'Baseline', '1 Full TIS', '2 Negative TIS', '3 Absolute-gap mask',
    '4 Negative down-gap mask', '5 Repeat + negative down-gap',
    '6 Negative length weighting', '7 Fresh synchronous G8',
)))
COLORS = dict(zip(ARMS, (
    '#222222', '#0072B2', '#D55E00', '#009E73',
    '#CC79A7', '#E69F00', '#56B4E9', '#7B3294',
)))
TABLES = (
    'optimizer_updates', 'training_rounds', 'gaps', 'accepted_training',
    'all_completed_rollout', 'primary_validation', 'native_round_performance',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    if value in ('', None):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite plotting value')
    return value


def load_tables(source, allow_partial):
    summary = json.loads((source / 'summary.json').read_text())
    if not summary.get('scientific_artifact') or summary.get('fixture_only') is not False:
        raise ValueError('Only scientific aggregates may be plotted')
    if not summary.get('complete') and not allow_partial:
        raise ValueError('Incomplete analysis; --allow-partial requires an explicitly marked partial snapshot')
    result, source_hashes = {}, {}
    for name in TABLES:
        entry = summary['tables'][name]
        relative = Path(entry['file'])
        path = source / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink():
            raise ValueError('Unsafe aggregate path')
        if digest(path) != entry['sha256']:
            raise ValueError(f'Aggregate hash mismatch: {name}')
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != entry['rows']:
            raise ValueError(f'Aggregate row count mismatch: {name}')
        result[name] = rows
        source_hashes[str(relative)] = entry['sha256']
    source_hashes['summary.json'] = digest(source / 'summary.json')
    return result, source_hashes, not summary.get('complete', False)


def render(source, output, allow_partial=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tables, sources, partial = load_tables(source, allow_partial)
    if output.resolve() == source.resolve() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Use a separate output directory to preserve the published artifacts')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    outputs = {}

    def save(fig, name):
        note = ('PARTIAL STUDY SNAPSHOT — missing arms or observations; '
                if partial else '')
        note += 'One online trajectory per method; populations are kept separate.'
        fig.text(.5, .005, note, ha='center', fontsize=9,
                 color='crimson' if partial else '#555555')
        fig.tight_layout(rect=(0, .035, 1, .97))
        for extension in ('png', 'svg'):
            path = output / f'{name}.{extension}'
            fig.savefig(path, dpi=160, bbox_inches='tight')
            outputs[path.name] = digest(path)
        plt.close(fig)

    def lines(ax, rows, key, title, log=False):
        values = []
        for arm in ARMS:
            selected = [r for r in rows if r['arm'] == arm]
            selected.sort(key=lambda r: int(r['round']))
            # Keep missing observations as NaN gaps; never join across them.
            if not any(number(r.get(key)) is not None for r in selected):
                continue
            y = [number(r.get(key)) for r in selected]
            values.extend(v for v in y if v is not None)
            ax.plot([int(r['round']) for r in selected],
                    [math.nan if v is None else v for v in y],
                    label=LABELS[arm], color=COLORS[arm],
                    lw=1.8 if arm == 'baseline' else 1.2,
                    marker='o', markersize=3 if len(selected) <= 6 else 2)
        if log:
            ax.set_yscale('symlog', linthresh=1.)
        if values and min(values) >= 0:
            ax.set_ylim(bottom=0)
        ax.set(xlim=(600, 650), title=title, xlabel='Completed training round')
        ax.grid(alpha=.2)

    updates, rounds = tables['optimizer_updates'], tables['training_rounds']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for inner, ax in enumerate(axes.flat, 1):
        lines(ax, [r for r in updates if int(r['inner']) == inner],
              'preclip_norm', f'B{inner}: pre-clip global gradient norm', log=True)
    axes[0, 0].legend(fontsize=8)
    save(fig, 'gradient_by_inner_all_arms')

    fig, axes = plt.subplots(4, 2, figsize=(13, 15), sharex=True)
    observed_arms = {row['arm'] for row in rounds}
    for arm, ax in zip(ARMS[1:], axes.flat):
        if arm not in observed_arms:
            ax.axis('off')
            ax.text(.5, .5, LABELS[arm] + '\nPending', ha='center', va='center',
                    transform=ax.transAxes, color='#777777')
            continue
        lines(ax, [r for r in rounds if r['arm'] in ('baseline', arm)],
              'preclip_norm_max', LABELS[arm] + ' vs baseline', log=True)
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    save(fig, 'gradient_baseline_vs_seven_methods')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, key, title, log in zip(axes.flat,
            ('preclip_norm_max', 'historical_q_tail_mean_or_zero',
             'historical_q_tail_fraction', 'historical_q_RMS'),
            ('Maximum pre-clip gradient per round',
             'Mean |log p - log q| above 0.5 (zero if none)',
             'Fraction with |log p - log q| > 0.5', 'RMS |log p - log q|'),
            (True, False, False, False)):
        lines(ax, rounds, key, title, log)
    axes[0, 0].legend(fontsize=8)
    save(fig, 'gradient_and_gap_all_arms')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, stratum in zip(axes.flat, ('positive_repeat', 'negative_repeat',
                                     'positive_nonrepeat', 'negative_nonrepeat')):
        selected = [r for r in tables['gaps'] if r['level'] == 'round'
                    and r['reference'] == 'historical_q' and r['stratum'] == stratum]
        lines(ax, selected, 'tail_mean_or_zero', stratum.replace('_', ' ') + ': d tail mean')
    axes[0, 0].legend(fontsize=8)
    save(fig, 'gap_by_sign_and_repeat')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, key, title in zip(axes.flat,
            ('negative_anchor_mass_retained_ratio', 'negative_direct_coefficient_abs_mass',
             'multiplier_zero_weight_fraction', 'adam_path_l2_sum'),
            ('Retained anchor negative |A| mass',
             'Direct negative coefficient mass\n(not parameter gradient)',
             'Fraction of tokens with zero intervention weight',
             'Sum of four actual Adam displacement L2 norms')):
        lines(ax, rounds, key, title)
    axes[0, 0].legend(fontsize=8)
    save(fig, 'learning_signal_and_adam_displacement')

    for table, population in (('accepted_training', 'Accepted training'),
                              ('all_completed_rollout', 'New all-completed rollout'),
                              ('primary_validation', 'Primary validation: 64 questions x 8')):
        fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
        for ax, key, title in zip(axes.flat,
                ('accuracy', 'mean_length', 'normalized_text_cycle128_any',
                 'response_hit_token_cap', 'first_word_alright', 'gibberish_candidate'),
                ('Accuracy', 'Mean response length', 'Mechanical cycle candidate fraction',
                 '8192-token cap fraction', 'First visible word Alright (frequency)',
                 'Mechanical garble candidate fraction')):
            lines(ax, tables[table], key, title)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(population)
        stem = 'primary_validation_64x8' if table == 'primary_validation' else table
        save(fig, stem + '_metrics')

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, key, title in zip(axes.flat,
            ('step_wall_seconds_native', 'CPU_text_assessment_seconds',
             'gen_wall_seconds_native', 'throughput_native',
             'selected_cross_version_response_fraction_native', 'delta_continuation_attempts'),
            ('Native step wall time (s)', 'CPU text assessment (inside actor wall time)',
             'Native generation wall time (s)', 'Native logged throughput',
             'Native selected multi-version response fraction',
             'Increment in cumulative continuation attempts')):
        lines(ax, tables['native_round_performance'], key, title)
    axes[0, 0].legend(fontsize=8)
    save(fig, 'native_performance_and_collection')
    receipt = {'complete': not partial, 'partial_preview': partial, 'sources_sha256': sources,
               'outputs_sha256': outputs, 'script_sha256': digest(Path(__file__)),
               'scope': 'Redraw only; no regrading or model execution. Image bytes depend on plotting environment.'}
    (output / 'REDRAW_MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=root / 'data/final-stability-600-650/training')
    parser.add_argument('--output', type=Path, default=root / 'figures/final-stability-600-650-redraw/training')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = render(args.data, args.output, args.allow_partial)
    print(json.dumps({'complete': result['complete'], 'figures': len(result['outputs_sha256'])}))
