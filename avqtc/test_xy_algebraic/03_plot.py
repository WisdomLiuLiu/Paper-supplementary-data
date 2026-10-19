"""APS-style full-width comparison: Kraus errors and single/two-qubit counts."""
import csv
import numpy as np
from common import ROOT, read_json, logger, DATA, arguments
from figure_style import apply_aps_style, plt, fit_error_limits
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


def plot(output=DATA):
    summary = read_json(output/'summary.json')
    assert summary['status'] == 'completed' and summary['trajectories'] == 12800
    curves = np.load(output/'curves.npz')
    resources = np.genfromtxt(output/'resources.csv', delimiter=',', names=True)
    times = curves['time']
    apply_aps_style()
    plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5,
                         'ytick.labelsize': 7.5, 'legend.fontsize': 6.5})
    fig, axes = plt.subplots(1, 3, figsize=(7., 2.15))
    fig.subplots_adjust(left=.075, right=.985, bottom=.21, top=.91, wspace=.35)
    styles = [('original', 'Original trajectory', 'black', '-'),
              ('compressed', 'Algebraically compressed', '#0072B2', '--')]
    columns = dict(time=times)
    for prefix, label, color, ls in styles:
        error = curves[prefix+'_error_to_kraus']
        axes[0].plot(times, error, color=color, ls=ls, lw=.8, label=label,
            marker='o' if prefix == 'compressed' else None, markevery=10,
            ms=2.6, markerfacecolor='white', markeredgewidth=.65)
        columns[prefix+'_error_to_kraus'] = error
        for ax, kind in zip(axes[1:], ['single_qubit', 'cnot']):
            means, stds = [0.], [0.]
            for index in range(1, len(times)):
                values = resources[prefix+'_'+kind][resources['time_index'] == index]
                assert len(values) == 128
                means.append(float(values.mean()))
                stds.append(float(values.std(ddof=0)))
            means, stds = np.array(means), np.array(stds)
            ax.plot(times, means, color=color, ls=ls, lw=.85)
            ax.fill_between(times, means-stds, means+stds, color=color, alpha=.10, lw=0)
            columns[prefix+'_'+kind+'_mean'] = means
            columns[prefix+'_'+kind+'_std'] = stds
    axes[0].axhline(0, color='.6', ls=':', lw=.5, zorder=0)
    axes[0].set_ylabel(r'Error to Kraus, $\Delta I$')
    axes[0].legend(loc='upper left', labelspacing=.25, handlelength=2.0)
    axes[1].set_ylabel('Single-qubit gate count')
    axes[2].set_ylabel('Two-qubit gate count')
    for i, ax in enumerate(axes):
        ax.set(xlabel=r'Time, $t$', xlim=(0, 10), xticks=[0, 2.5, 5, 7.5, 10])
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.text(-.22, 1.04, f'({chr(97+i)})', transform=ax.transAxes, fontweight='bold', fontsize=10)
        if i:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    fig.canvas.draw()
    fit_error_limits(axes[0], fig.canvas.get_renderer())
    dest = output/'figures'
    dest.mkdir(exist_ok=True)
    stem = dest/'figure_xy_algebraic_compression'
    for ext in ['pdf']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=400)
    plt.close(fig)
    with stem.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for j in range(len(times)):
            writer.writerow({key: float(value[j]) for key, value in columns.items()})
    logger(f'FIGURE COMPLETE: {stem}.pdf/csv')


if __name__ == '__main__':
    plot(arguments(__doc__).output)
