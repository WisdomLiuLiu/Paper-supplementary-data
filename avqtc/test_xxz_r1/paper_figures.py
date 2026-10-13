"""Five APS-sized figures from the existing, validated revision data."""
import numpy as np

from figure_style import apply_aps_style, compact_y, finish, time_axis


def figures(cfg, dest, ref, baseline, manifests, results, seed, log):
    renderer = apply_aps_style()
    import matplotlib.pyplot as plt
    time = ref['time']
    out = dest/f'figures_seed_{seed}'
    out.mkdir(exist_ok=True)
    xxz = 'circuit' in cfg
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    styles = ['--', '-.', (0, (2., 1.2)), (0, (1., 1.))]
    log(f'APS style: 3.4-inch width, {renderer}, seed={seed}')

    def done(fig, axes, stem, error_axes=()):
        finish(fig, axes, out/stem, error_axes=error_axes)
        for panel, ax in zip(['a', 'd'], error_axes):
            log(f'{stem} ({panel}) error-axis limits: {ax.get_ylim()}')
        log(f'Figure saved: {stem}, APS style, training seed={seed}')

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.62))
    axes[0].plot(time, ref['exact'], 'k-', lw=.85, label='Exact')
    axes[0].plot(time, ref['kraus'], color='#D55E00', ls='--', label='Kraus')
    counts = cfg['sampling']['m_values']
    for count, color in zip(counts, plt.get_cmap('viridis')(np.linspace(.08, .88, len(counts)))):
        axes[0].plot(time, ref[f'sample_M{count}'].mean(0), color=color, lw=.48, label=rf'$M={count}$')
    axes[0].set_ylabel(r'Imbalance, $\langle I(t)\rangle$')
    time_axis(axes[0], time)
    axes[0].legend(loc='lower right', ncol=2, columnspacing=.65, labelspacing=.18, fontsize=4.8)
    for point, color, marker in [(time[len(time)//2], colors[0], 'o'), (time[-1], colors[1], 's')]:
        j = int(np.argmin(abs(time-point)))
        axes[1].errorbar(counts, [ref[f'sample_M{m}'].mean(0)[j] for m in counts],
                         yerr=[ref[f'sample_M{m}'].std(0, ddof=1)[j] for m in counts],
                         color=color, marker=marker, ls='--', lw=.65, capsize=1.6,
                         capthick=.5, elinewidth=.5, label=rf'Sampling, $t={point:g}$')
        axes[1].axhline(ref['kraus'][j], color=color, lw=.65, label=rf'Kraus, $t={point:g}$')
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks(counts, labels=[str(c) for c in counts])
    axes[1].set_xlabel(r'Sampling size, $M$')
    axes[1].set_ylabel(r'Imbalance, $\langle I(t_\star)\rangle$')
    axes[1].margins(y=.25)
    compact_y(axes[1])
    axes[1].legend(loc='best', fontsize=4.6, labelspacing=.15)
    done(fig, axes, 'figure_01_sampling_imbalance')

    # Error panels are fitted individually, after measuring their legends.
    all_f = []
    for (channel, run_seed), run in manifests.items():
        if run_seed == seed:
            all_f += [b['exact_power_validation_hs_fidelity'] for b in run['library'].values()]
    f_lower = np.floor((min(all_f)-.001)*100)/100

    for mode, train_number, gate_number in [('b32', 2, 4), ('full', 3, 5)]:
        specs = [(f'{mode}_direct', cfg['training']['max_k'], 'Direct')]
        specs += [(f'{mode}_iterative', k, rf'Iterative, $s_{{\max}}={k}$') for k in cfg['sampling']['iterative_cutoffs']]
        fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.85))
        for label, a, color, ls in [
            (f'{cfg["sampling"]["repeats"]}-run mean', ref[f'sample_M{cfg["sampling"]["trajectory_m"]}'].mean(0), '.5', (0, (3, 1, 1, 1))),
            ('Same-seed exact MC', ref['same_seed_exact'], '#009E73', '--'),
            ('Same-seed circuit', baseline, 'black', '-')]:
            axes[0, 0].plot(time, ref['kraus']-a, color=color, ls=ls, lw=.6, label=label)
        for route, color, marker in [('direct', colors[0], 'o'), ('iterative', colors[1], '^')]:
            lib = manifests[f'{mode}_{route}', seed]['library']
            ks = sorted(map(int, lib))
            for ax, key in [(axes[0, 1], 'pqc_steps'), (axes[1, 0], 'exact_power_validation_hs_fidelity')]:
                ax.plot(ks, [lib[str(k)][key] for k in ks], color=color, marker=marker,
                        ls='--', lw=.6, ms=2.5, markeredgewidth=.5, label=route.capitalize())
                ax.set_xlim(.5, cfg['training']['max_k']+.5)
                ax.set_xticks([1, 5, 10, 15, 20] if cfg['training']['max_k']==20 else ks)
                ax.set_xlabel(r'Repetitions, $s$')
        for (name, cutoff, label), color, ls in zip(specs, colors, styles):
            data = np.load(results[name, seed, cutoff]/'curves.npz')
            axes[1, 1].plot(time, data['error_to_kraus'], color=color, ls=ls, lw=.65, label=label)
        for ax in [axes[0, 0], axes[1, 1]]:
            time_axis(ax, time)
            ax.axhline(0, color='.6', lw=.4, ls=':')
            ax.set_ylabel(r'Error to Kraus, $\Delta I$')
            compact_y(ax)
            ax.legend(loc='upper left', fontsize=4.5, labelspacing=.12)
        axes[1, 1].set_ylabel(r'Replace error, $\Delta I$')
        if xxz:
            axes[1, 1].legend(loc='lower left', fontsize=4.5, labelspacing=.12)
        axes[0, 1].set_ylabel('PQC depth')
        compact_y(axes[0, 1], integer=True)
        axes[0, 1].legend(loc='lower right', fontsize=5.)
        axes[1, 0].set_ylabel(r'HSF to $(U_0)^s$')
        axes[1, 0].set_ylim(f_lower, 1.0015)
        f_step = .02 if 1-f_lower > .041 else .01
        axes[1, 0].set_yticks(np.arange(1., f_lower-.0001, -f_step)[::-1])
        axes[1, 0].legend(loc='lower left', fontsize=5.)
        done(fig, axes, f'figure_{train_number:02d}_{mode}_training_replacement',
             error_axes=[axes[0, 0], axes[1, 1]])

        datasets = [(label, np.genfromtxt(results[name, seed, cutoff]/'resources.csv', delimiter=',', names=True))
                    for name, cutoff, label in specs]
        base = datasets[0][1]
        for _, a in datasets[1:]:
            for field in ['time_index', 'trajectory_index', 'trajectory_seed', 'original_cnot', 'original_single_qubit']:
                if not np.array_equal(base[field], a[field]):
                    raise ValueError('Resource figures require identical paired trajectories')
        fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.85))
        final = len(time)-1
        series = [('Original', base, 'original_', 'black', '-')]
        series += [(label, a, 'replaced_', color, ls) for (label, a), color, ls in zip(datasets, colors, styles)]
        for row, (kind, ylabel) in enumerate([('single_qubit', 'Single-qubit gate count'), ('cnot', 'Two-qubit gate count')]):
            order = np.argsort(base['original_'+kind][base['time_index']==final], kind='stable')
            for label, a, prefix, color, ls in series:
                means, stds = [0], [0]
                for index in range(1, len(time)):
                    values = a[prefix+kind][a['time_index']==index]
                    means.append(values.mean()); stds.append(values.std())
                axes[row, 0].plot(time, means, color=color, ls=ls, lw=.58, label=label)
                axes[row, 0].fill_between(time, np.array(means)-stds, np.array(means)+stds,
                                         color=color, alpha=.09, lw=0)
                axes[row, 1].plot(np.arange(1, len(order)+1),
                                  a[prefix+kind][a['time_index']==final][order], color=color, ls=ls, lw=.52)
            time_axis(axes[row, 0], time)
            axes[row, 0].set_ylabel(ylabel)
            axes[row, 0].legend(loc='upper left', fontsize=4.4, labelspacing=.12, handlelength=1.45)
            axes[row, 0].set_title('Gate count', pad=2)
            axes[row, 1].set(xlabel='Trajectory index', ylabel=ylabel, xlim=(1, len(order)))
            axes[row, 1].set_xticks([1, 32, 64, 96, 128] if len(order)==128 else np.arange(1, len(order)+1))
            axes[row, 1].set_title(rf'$t={time[-1]:g}$', pad=2)
            for ax in axes[row]:
                compact_y(ax, integer=True)
        done(fig, axes, f'figure_{gate_number:02d}_{mode}_gate_resources')
