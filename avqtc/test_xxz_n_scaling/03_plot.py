"""Original maxiter=100 results: selected depths and fixed-depth HSF at s=20."""
import argparse
import csv
import os
from pathlib import Path
import numpy as np
from scaling import settings, library_path, validate_library
from xxz_core import ROOT, read_json, write_json, file_hash
from figure_style import apply_aps_style, plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter


def export_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(data, output):
    cfg, _ = settings(ROOT/'config.json')
    assert cfg['training']['iterations'] == 100
    sizes = [6, 8, 10, 12]
    depth_rows, fidelity_rows, provenance = [], [], []
    for n in sizes:
        manifest = library_path(data, n)
        run = read_json(manifest)
        validate_library(run, cfg, n)
        assert run['status'] == 'completed' and len(run['library']) == 20
        for s in range(1, 21):
            block = run['library'][str(s)]
            assert block['threshold_reached']
            depth_rows.append(dict(n=n, s=s, selected_pqc_depth=block['pqc_steps'],
                                   selected_hsf=block['exact_power_validation_hs_fidelity']))
        shared_n8 = ROOT.parent/'test_xxz_r1/data/training/full_direct/seed_202506/manifest.json'
        source = data/'n08' if manifest.resolve() == shared_n8.resolve() else manifest.parent
        provenance.append(dict(n=n, manifest=Path(os.path.relpath(manifest, output)).as_posix(),
                               manifest_sha256=file_hash(manifest)))
        for depth in [9, 10]:
            files = sorted((source/'attempts').glob(f'k20_d{depth:02d}_*.json'))
            if not files:
                assert (n, depth) == (6, 10), (n, depth)
                fidelity_rows.append(dict(n=n, s=20, pqc_depth=depth, best_hsf='',
                    status='not_tested', selected_attempt='', optimizer_iterations='', attempts=0))
                continue
            attempts = [(p, read_json(p)) for p in files]
            assert all(a['status'] == 'completed' and a['max_iterations'] == 100 for _, a in attempts)
            chosen_path, chosen = max(attempts, key=lambda pair: pair[1]['best_validation_fidelity'])
            candidate = next(c for c in run['library']['20']['depth_candidates'] if c['depth'] == depth)
            assert abs(candidate['fidelity']-chosen['best_validation_fidelity']) < 1e-12
            fidelity_rows.append(dict(n=n, s=20, pqc_depth=depth, best_hsf=chosen['best_validation_fidelity'],
                status='completed', selected_attempt=Path(os.path.relpath(chosen_path, output)).as_posix(),
                optimizer_iterations=chosen['optimizer_iterations'], attempts=len(attempts)))
            provenance.append(dict(n=n, depth=depth, attempt=Path(os.path.relpath(chosen_path, output)).as_posix(),
                                   attempt_sha256=file_hash(chosen_path)))

    apply_aps_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8))
    fig.subplots_adjust(left=.14, right=.985, bottom=.215, top=.91, wspace=.54)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for n, color, marker in zip(sizes, colors, ['o', 's', '^', 'D']):
        points = [r for r in depth_rows if r['n'] == n]
        axes[0].plot([r['s'] for r in points], [r['selected_pqc_depth'] for r in points],
            color=color, marker=marker, markersize=2.1, markeredgewidth=.4,
            linewidth=.65, label=rf'$n={n}$')
    axes[0].set(xlabel=r'Repetitions, $s$', ylabel='Selected PQC depth',
                xlim=(.5, 20.5), ylim=(.5, 11.7), xticks=[1, 5, 10, 15, 20], yticks=[1, 3, 5, 7, 9, 11])
    axes[0].legend(loc='upper left', fontsize=5.2, labelspacing=.15, handlelength=1.2)
    for depth, color, marker in [(9, colors[0], 'o'), (10, colors[1], 's')]:
        # Missing n=6, depth=10 is NaN: no extrapolated/imputed point is plotted.
        values = [next(r['best_hsf'] for r in fidelity_rows if r['n'] == n and r['pqc_depth'] == depth)
                  for n in sizes]
        values = [float(v) if v != '' else np.nan for v in values]
        axes[1].plot(sizes, values, color=color, marker=marker, markersize=2.5,
            markeredgewidth=.4, linewidth=.7, label=rf'PQC depth $={depth}$')
    axes[1].axhline(.99, color='.5', linestyle=':', linewidth=.6, zorder=0)
    axes[1].set(xlabel=r'Number of qubits, $n$', ylabel=r'HSF to $(U_0)^s$',
                xlim=(5.7, 12.3), ylim=(.9778, .9953), xticks=sizes, yticks=[.980, .985, .990, .995])
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[1].legend(loc='lower left', fontsize=5.2, labelspacing=.2, handlelength=1.4)
    axes[1].text(.96, .96, r'$s=20$', transform=axes[1].transAxes, ha='right', va='top', fontsize=6.5)
    for i, ax in enumerate(axes):
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.text(-.29, 1.04, f'({chr(97+i)})', transform=ax.transAxes, fontweight='bold', fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    stem = output/'figure_xxz_size_depth_comparison'
    for ext in ['pdf']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=400)
    plt.close(fig)
    export_csv(output/'figure_xxz_size_depth_comparison_a.csv', depth_rows)
    export_csv(output/'figure_xxz_size_depth_comparison_b.csv', fidelity_rows)
    write_json(output/'figure_xxz_size_depth_comparison_sources.json', dict(
        training='Full-space direct training, maxiter=100, seed=202506',
        panel_a='Selected depth versus s for n=6,8,10,12',
        panel_b='Best recorded HSF across tested initializations at fixed s=20 and depths 9,10',
        missing='n=6, depth=10 was not tested and is left blank',
        interpretation='Finite-budget achieved fidelities; not proven globally optimal fidelities',
        sources=provenance))
    print(f'Saved {stem}.pdf and panel data/source records.', flush=True)
    for row in fidelity_rows:
        print(f'n={row["n"]}, depth={row["pqc_depth"]}, HSF={row["best_hsf"]}, status={row["status"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data', help='Training data directory')
    args = parser.parse_args()
    plot(args.output.resolve(), args.output.resolve()/'figures')
