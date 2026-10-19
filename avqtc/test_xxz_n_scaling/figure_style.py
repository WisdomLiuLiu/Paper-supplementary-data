"""APS-like house style from OQS 202506/test_xy_paper/figure/plot_xy_figures.py."""
import shutil
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FormatStrFormatter


def apply_aps_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use('science')
    except ImportError:
        plt.style.use('default')
    tex = bool(shutil.which('latex') and shutil.which('dvipng'))
    plt.rcParams.update({
        'text.usetex': tex, 'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
        'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 5.4,
        'lines.linewidth': .7, 'lines.markersize': 2.5, 'axes.linewidth': .5,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'xtick.major.width': .5, 'ytick.major.width': .5,
        'xtick.minor.width': .4, 'ytick.minor.width': .4,
        'xtick.major.size': 2.7, 'ytick.major.size': 2.7,
        'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
        'axes.grid': False, 'axes.labelpad': 2.,
        'legend.frameon': False, 'legend.handlelength': 1.6,
        'legend.handletextpad': .4, 'legend.borderaxespad': .3,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': None,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}',
    })
    return 'LaTeX/newtx' if tex else 'serif/STIX fallback'


def fit_error_limits(ax, renderer):
    """Fit the plotted errors, leaving only the local clearance needed by the legend."""
    curves = [(np.asarray(line.get_xdata(), dtype=float),
               np.asarray(line.get_ydata(), dtype=float)) for line in ax.lines]
    values = np.concatenate([y[np.isfinite(y)] for _, y in curves])
    low, high = min(0., values.min()), max(0., values.max())
    span = max(high-low, 1e-6)
    low -= .04*span
    high += .04*span
    legend = ax.get_legend()
    if legend is not None:
        box = legend.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        xmin, xmax = ax.get_xlim()
        left = xmin + max(0., box.x0-.02)*(xmax-xmin)
        right = xmin + min(1., box.x1+.02)*(xmax-xmin)
        local = []
        for x, y in curves:
            # axhline uses axes coordinates; its zero is already included below.
            if len(x) <= 2:
                continue
            local.extend(y[(x >= left) & (x <= right) & np.isfinite(y)])
            local.extend(np.interp([left, right], x, y))
        local = np.asarray(local if local else [0.])
        if legend._loc == 2:  # upper left
            high = max(high, low+(local.max()-low)/max(box.y0-.025, .1))
        elif legend._loc == 3:  # lower left
            low = min(low, high-(high-local.min())/max(1-box.y1-.025, .1))
        else:
            raise ValueError('Error-panel legend must be upper left or lower left')
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))


def finish(fig, axes, path, compact=True, error_axes=()):
    flat = list(axes.flat)
    for i, ax in enumerate(flat):
        label = rf'\textbf{{({chr(97+i)})}}' if plt.rcParams['text.usetex'] else f'({chr(97+i)})'
        ax.text(-.28 if i % 2 == 0 else -.40, 1.02, label,
                transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, fontweight='bold')
        if ax.get_xscale() == 'linear':
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        if ax.get_yscale() == 'linear':
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(False)
    if compact:
        # Compact margins follow test_xy_paper; preserve room for tick labels.
        fig.subplots_adjust(left=.14, right=.975, bottom=.108 if len(flat)>2 else .20,
                            top=.945 if len(flat)>2 else .92, wspace=.43, hspace=.40)
    else:
        fig.tight_layout(pad=.8, h_pad=1.5, w_pad=1.2)
    if error_axes:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in error_axes:
            fit_error_limits(ax, renderer)
    for suffix in ['pdf', 'png']:
        fig.savefig(path.with_suffix('.'+suffix), bbox_inches=None)
    plt.close(fig)


def time_axis(ax, time):
    ax.set_xlim(time[0], time[-1])
    ax.set_xticks([time[-1]*i/4 for i in range(5)])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_xlabel(r'Time, $t$')


def compact_y(ax, integer=False):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=integer))
