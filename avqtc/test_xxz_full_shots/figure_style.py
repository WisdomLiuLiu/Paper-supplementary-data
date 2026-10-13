"""APS-like house style from OQS 202506/test_xy_paper/figure/plot_xy_figures.py."""
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


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


def finish(fig, axes, path, compact=True):
    flat = list(axes.flat)
    for i, ax in enumerate(flat):
        label = rf'\textbf{{({chr(97+i)})}}' if plt.rcParams['text.usetex'] else f'({chr(97+i)})'
        ax.text(-.31 if i % 2 == 0 else -.36, 1.025, label,
                transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, fontweight='bold')
        if ax.get_xscale() == 'linear':
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        if ax.get_yscale() == 'linear':
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(False)
    if compact:
        fig.subplots_adjust(left=.15, right=.975, bottom=.115 if len(flat)>2 else .23,
                            top=.94 if len(flat)>2 else .90, wspace=.56, hspace=.55)
    else:
        fig.tight_layout(pad=.8, h_pad=1.5, w_pad=1.2)
    for suffix in ['pdf', 'png']:
        fig.savefig(path.with_suffix('.'+suffix), bbox_inches=None)
    plt.close(fig)


def time_axis(ax, time):
    ax.set_xlim(time[0], time[-1])
    ax.set_xticks([time[-1]*i/4 for i in range(5)])
    ax.set_xlabel(r'Time, $t$')


def compact_y(ax, integer=False):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=integer))
