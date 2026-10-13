"""Reference generation, paired trajectories, and replacement resource accounting."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from training import Logger
from xy_core import (ROOT, Engine, channel, digest, file_hash, gate_resources, load_config,
                     read_json, segments, stamp, trajectory, write_json)


def times(cfg):
    m = cfg['model']
    return np.arange(round(m['total_time']/m['dt'])+1)*m['dt']


def reference_spec(cfg):
    return {'model': cfg['model'], 'sampling': cfg['sampling']}


def ensure_reference(cfg, output):
    meta = read_json(output / 'reference' / 'metadata.json')
    if meta['specification'] != reference_spec(cfg):
        raise ValueError('Reference config differs. Use a separate --output directory for each experiment.')
    return np.load(output / 'reference' / 'reference.npz'), meta


def exact_sampling(engine, u0, probs, cfg, count, stream_seed, log, label):
    rng = np.random.default_rng(stream_seed)
    result = [-1.0]
    grid = times(cfg)
    start = time.perf_counter()
    for index in range(1, len(grid)):
        branches = rng.choice(engine.n+1, size=(count, index), p=probs)
        states = engine.xp.tile(engine.initial[None, :], (count, 1))
        for j in range(index):
            zero = np.flatnonzero(branches[:, j] == 0)
            if zero.size:
                rows = engine.xp.asarray(zero)
                states[rows] = states[rows] @ u0.T
            for q in range(engine.n):
                chosen = np.flatnonzero(branches[:, j] == q+1)
                if chosen.size:
                    rows = engine.xp.asarray(chosen)
                    states[rows] *= engine.xp.asarray(engine.signs[:, q])[None, :]
        result.append(float(engine.xp.mean(engine.xp.sum(engine.xp.abs(states)**2 * engine.imbalance, axis=1)).item()))
        if index == 1 or index % 20 == 0 or index == len(grid)-1:
            log(f'{label} time={grid[index]:g}/{grid[-1]:g} step={index}/{len(grid)-1} elapsed={time.perf_counter()-start:.1f}s')
    return np.asarray(result)


def prepare_reference(cfg, output, source=None, recompute=False, backend=None):
    output = Path(output)
    dest = output / 'reference'
    dest.mkdir(parents=True, exist_ok=True)
    log = Logger(output / 'logs' / '01_reference.log')
    metadata = dest / 'metadata.json'
    if metadata.exists():
        ensure_reference(cfg, output)
        log('Reference is already complete; reuse existing data')
        return
    m, s = cfg['model'], cfg['sampling']
    data = {'time': times(cfg)}
    meta = {'specification': reference_spec(cfg), 'created_at': stamp(),
            'gamma_convention': 'L_i=sqrt(gamma)*Z_i; H=sum_j(XX+YY), open boundaries',
            'initial_state': '|1>^(n/2)|0>^(n/2)',
            'observable': '(sum_left Z - sum_right Z)/n',
            'trajectory_policy': 'independent ensemble at each time; time-major, trajectory-minor RNG stream',
            'source_files': {}}
    if not recompute:
        expected = {'n': 10, 'dt': 0.1, 'gamma': 0.06, 'total_time': 10.0}
        if m != expected or s['repeats'] != 100 or s['master_seed'] != 202506:
            raise ValueError('Old references support only the original model, 100 repeats, seed 202506. Use --recompute.')
        source = Path(source or ROOT / 'reference_inputs')
        log(f'Importing unchanged physical reference data from {source}')
        names = {'exact': 'exact_dt0.1_gamma0.06.txt', 'kraus': 'kraus_dt0.1_gamma0.06.txt'}
        for count in sorted(set(s['m_values'] + [s['trajectory_m']])):
            names[f'sample_M{count}'] = f'sample_dt0.1_gamma0.06_M{count}.txt'
            names[f'seeds_M{count}'] = f'sample_seed_dt0.1_gamma0.06_M{count}.txt'
        for key, filename in names.items():
            path = source / filename
            data[key] = np.loadtxt(path, dtype=np.uint32 if key.startswith('seeds') else float)
            sha = file_hash(path)
            if (source / 'provenance.json').exists():
                if read_json(source / 'provenance.json')['files'][filename]['sha256'] != sha:
                    raise ValueError(f'Bundled reference checksum mismatch: {filename}')
            meta['source_files'][key] = {'file': filename, 'sha256': sha}
            log(f'Imported {filename}: shape={data[key].shape}')
        matches = np.flatnonzero(data[f'seeds_M{s["trajectory_m"]}'] == s['stream_seed'])
        if len(matches) != 1:
            raise ValueError('stream_seed not present in source repeats. Use --recompute or restore original stream_seed.')
        data['same_seed_exact'] = data[f'sample_M{s["trajectory_m"]}'][matches[0]]
        meta['selected_original_row_zero_based'] = int(matches[0])
        meta['origin'] = 'imported original reference; no seed reselection'
    else:
        engine = Engine(m['n'], backend or cfg['training']['backend'])
        log(f'Computing reference on {engine.backend}; n={m["n"]}')
        u0, h = engine.u0(m['dt'], m['gamma'])
        step, probs = channel(m['n'], m['dt'], m['gamma'])
        import qutip as qt
        dims = [[2]*m['n'], [2]*m['n']]
        # CSR is essential: dense Liouvillian construction at n=10 would need 16 TiB.
        H = qt.Qobj(h, dims=dims).to('csr')
        ket = qt.Qobj(engine.cpu(engine.initial), dims=[[2]*m['n'], [1]*m['n']])
        jumps = [np.sqrt(m['gamma'])*qt.Qobj(np.diag(engine.signs[:, j]), dims=dims).to('csr') for j in range(m['n'])]
        obs = qt.Qobj(np.diag(engine.cpu(engine.imbalance)), dims=dims).to('csr')
        log('Lindblad solver starting (QuTiP progress follows)')
        solved = qt.mesolve(H, ket, data['time'], jumps, e_ops=[obs],
                           options={'progress_bar': 'text', 'rtol': 1e-9, 'atol': 1e-11})
        data['exact'] = np.real(solved.expect[0])
        rho = engine.xp.outer(engine.initial, engine.initial.conj())
        sign_factor = engine.xp.asarray(engine.signs @ engine.signs.T)
        kraus = [-1.0]
        for index in range(1, len(data['time'])):
            rho = probs[0]*(u0 @ rho @ u0.conj().T) + probs[1]*sign_factor*rho
            kraus.append(float(engine.xp.real(engine.xp.sum(engine.xp.diag(rho)*engine.imbalance)).item()))
            log(f'Kraus step {index}/{len(data["time"])-1}, imbalance={kraus[-1]:.8f}')
        data['kraus'] = np.array(kraus)
        for count in sorted(set(s['m_values'] + [s['trajectory_m']])):
            seeds = np.random.default_rng(s['master_seed']).integers(0, 2**32-1, size=s['repeats'], dtype=np.uint32)
            data[f'seeds_M{count}'] = seeds
            rows = []
            for repeat, seed in enumerate(seeds):
                rows.append(exact_sampling(engine, u0, probs, cfg, count, int(seed), log,
                                           f'M={count} repeat={repeat+1}/{len(seeds)}'))
                # Recovery material for a long reference run, independently of final NPZ.
                np.save(dest / f'sample_M{count}_partial.npy', np.asarray(rows))
            data[f'sample_M{count}'] = np.array(rows)
        seeds = data[f'seeds_M{s["trajectory_m"]}']
        matches = np.flatnonzero(seeds == s['stream_seed'])
        data['same_seed_exact'] = (data[f'sample_M{s["trajectory_m"]}'][matches[0]] if len(matches)
                                   else exact_sampling(engine, u0, probs, cfg, s['trajectory_m'], s['stream_seed'], log, 'paired stream'))
        meta['origin'] = 'recomputed'
        meta['environment'] = engine.environment()
        meta['qutip_version'] = qt.__version__
    for key in ('exact', 'kraus', 'same_seed_exact'):
        if data[key].shape != data['time'].shape or not np.isfinite(data[key]).all():
            raise ValueError(f'{key}: invalid reference data')
    for count in s['m_values']:
        if data[f'sample_M{count}'].shape != (s['repeats'], len(data['time'])):
            raise ValueError(f'M={count}: invalid repeat matrix')
    np.savez_compressed(dest / 'reference.npz', **data)
    meta['npz_sha256'] = file_hash(dest / 'reference.npz')
    write_json(metadata, meta)
    log('REFERENCE COMPLETE')


def paired_branch_table(cfg, output):
    dest = Path(output) / 'baseline'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'branches.npz'
    signature = digest({'model': cfg['model'], 'M': cfg['sampling']['trajectory_m'],
                        'stream_seed': cfg['sampling']['stream_seed']})
    if path.exists():
        old = np.load(path)
        if str(old['signature']) != signature:
            raise ValueError('Stored branch stream uses a different model/M/seed')
        return old['branches'], signature
    grid = times(cfg)
    n, s = cfg['model']['n'], cfg['sampling']
    _, probs = channel(n, cfg['model']['dt'], cfg['model']['gamma'])
    rng = np.random.default_rng(s['stream_seed'])
    out = np.full((len(grid)-1, s['trajectory_m'], len(grid)-1), -1, np.int16)
    for index in range(1, len(grid)):
        out[index-1, :, :index] = rng.choice(n+1, size=(s['trajectory_m'], index), p=probs)
    np.savez_compressed(path, branches=out, signature=signature)
    return out, signature


def run_baseline(cfg, output, backend=None, recompute=False):
    output = Path(output)
    ensure_reference(cfg, output)
    branches, signature = paired_branch_table(cfg, output)
    dest = output / 'baseline'
    meta_path = dest / 'metadata.json'
    log = Logger(output / 'logs' / '02_baseline.log')
    if meta_path.exists():
        if read_json(meta_path)['signature'] != signature:
            raise ValueError('Baseline mismatch')
        log('Baseline already complete')
        return
    original_setting = (cfg['model'] == {'n': 10, 'dt': .1, 'gamma': .06, 'total_time': 10.0}
                        and cfg['sampling']['trajectory_m'] == 128
                        and cfg['sampling']['stream_seed'] == 1225499654)
    if original_setting and not recompute:
        bundled = ROOT / 'reference_inputs'
        curve_name = 'circuit_sample_dt0.1_gamma0.06_M128.txt'
        counts_name = 'original_paired_resources.csv'
        provenance = read_json(bundled / 'provenance.json')
        for filename in [curve_name, counts_name]:
            if file_hash(bundled / filename) != provenance['files'][filename]['sha256']:
                raise ValueError(f'Bundled baseline checksum mismatch: {filename}')
        values = np.loadtxt(bundled / curve_name)
        if values.shape != times(cfg).shape:
            raise ValueError('Bundled baseline has wrong length')
        with (bundled / counts_name).open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != branches.shape[0]*branches.shape[1]:
            raise ValueError('Bundled resource table has wrong row count')
        log('Importing paired original circuit curve and gate counts; verifying every branch sequence')
        for number, row in enumerate(rows):
            index, j = number//branches.shape[1]+1, number % branches.shape[1]
            if (int(row['time_index']) != index or int(row['trajectory_index']) != j or
                    int(row['trajectory_seed']) != cfg['sampling']['stream_seed'] or
                    not np.isclose(float(row['time']), times(cfg)[index])):
                raise ValueError('Bundled trajectory ordering/seed does not match')
            cnot = single = depth = 0
            for branch, length in segments(branches[index-1, j, :index]):
                if branch:
                    single += 1; depth += 1
                else:
                    r = gate_resources(cfg['model']['n'], length)
                    cnot += r['cnot']; single += r['single_qubit']; depth += r['primitive_depth_upper_bound']
            if cnot != int(row['original_cnot']) or single != int(row['original_single_qubit']):
                raise ValueError('Bundled original gate counts do not match sampled branches')
            row['original_depth_upper_bound'] = depth
        with (dest / 'resources.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(dest / 'baseline.npz', time=times(cfg), imbalance=values)
        write_json(meta_path, {'signature': signature, 'model': cfg['model'], 'created_at': stamp(),
                              'origin': 'bundled same-seed compiled ensemble curve + extracted paired original counts',
                              'source_files': {name: provenance['files'][name] for name in [curve_name, counts_name]},
                              'per_trajectory_imbalance': 'not present in imported source; only ensemble curve is available; use --recompute in a new output directory to generate individual values',
                              'gate_count_scope': 'evolution only; initial state preparation/readout excluded',
                              'verified_paired_trajectories': len(rows)})
        log(f'BASELINE COMPLETE: imported curve and verified {len(rows)} paired resource records')
        return
    engine = Engine(cfg['model']['n'], backend or cfg['training']['backend'])
    step, _ = channel(engine.n, cfg['model']['dt'], cfg['model']['gamma'])
    grid, averages = times(cfg), [-1.0]
    fields = ['time_index', 'time', 'trajectory_index', 'trajectory_seed',
              'original_cnot', 'original_single_qubit', 'original_depth_upper_bound', 'imbalance']
    start = time.perf_counter()
    with (dest / 'resources.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index in range(1, len(grid)):
            values = []
            for j in range(branches.shape[1]):
                value, resource = trajectory(engine, branches[index-1, j, :index], step)
                values.append(value)
                writer.writerow({'time_index': index, 'time': grid[index], 'trajectory_index': j,
                                 'trajectory_seed': cfg['sampling']['stream_seed'], 'imbalance': value,
                                 **{key: resource[key] for key in fields if key.startswith('original_')}})
            averages.append(float(np.mean(values)))
            f.flush()
            log(f'BASELINE step={index}/{len(grid)-1} t={grid[index]:g} M={branches.shape[1]} '
                f'I={averages[-1]:.8f} elapsed={time.perf_counter()-start:.1f}s')
    np.savez_compressed(dest / 'baseline.npz', time=grid, imbalance=averages)
    write_json(meta_path, {'signature': signature, 'model': cfg['model'], 'created_at': stamp(),
                          'wall_seconds': time.perf_counter()-start,
                          'convention': 'alternate odd/even Trotter order in each contiguous U0 run, merge adjacent equal parity layers',
                          'gate_count_scope': 'evolution only; initial state preparation/readout excluded as in original paper',
                          'environment': engine.environment()})
    log('BASELINE COMPLETE')


def accuracy(a, b):
    delta = np.asarray(a) - np.asarray(b)
    return {'rmse': float(np.sqrt(np.mean(delta**2))), 'mae': float(np.mean(abs(delta))),
            'max_abs_error': float(np.max(abs(delta)))}


def replace(cfg, output, backend=None, seeds=None, channels=None):
    output = Path(output)
    reference, _ = ensure_reference(cfg, output)
    branches, signature = paired_branch_table(cfg, output)
    base_meta = read_json(output / 'baseline' / 'metadata.json')
    if base_meta['signature'] != signature:
        raise ValueError('Run 02 for the same paired stream first')
    baseline = np.load(output / 'baseline' / 'baseline.npz')['imbalance']
    channels = channels or ['full_direct', 'full_iterative', 'b32_direct', 'b32_iterative']
    seeds = seeds or cfg['training']['seeds']
    # Validate ALL dependencies before spending hours on partially available channels.
    jobs = []
    for name in channels:
        route = name.split('_')[1]
        for seed in seeds:
            manifest_path = output / 'training' / name / f'seed_{seed}' / 'manifest.json'
            manifest = read_json(manifest_path)
            if manifest['status'] != 'completed' or manifest['specification']['model'] != cfg['model']:
                raise ValueError(f'{manifest_path}: missing/incompatible completed library')
            if manifest['specification']['training'] != cfg['training']:
                raise ValueError(f'{manifest_path}: training config changed')
            cutoffs = [cfg['training']['max_k']] if route == 'direct' else cfg['sampling']['iterative_cutoffs']
            for cutoff in cutoffs:
                if any(str(k) not in manifest['library'] for k in range(1, cutoff+1)):
                    raise ValueError(f'{name} seed={seed}: library does not cover K={cutoff}')
                jobs.append((name, seed, cutoff, manifest, file_hash(manifest_path)))
    engine = Engine(cfg['model']['n'], backend or cfg['training']['backend'])
    step, _ = channel(engine.n, cfg['model']['dt'], cfg['model']['gamma'])
    grid = times(cfg)
    log = Logger(output / 'logs' / '07_replace.log')
    for job_index, (name, seed, cutoff, manifest, manifest_hash) in enumerate(jobs, 1):
        dest = output / 'replacement' / name / f'seed_{seed}' / f'K{cutoff}'
        dest.mkdir(parents=True, exist_ok=True)
        meta_path = dest / 'summary.json'
        if meta_path.exists():
            old = read_json(meta_path)
            if old['branch_signature'] != signature or old['training_manifest_sha256'] != manifest_hash:
                raise ValueError(f'{dest}: completed replacement uses different input data')
            log(f'RESUME job={job_index}/{len(jobs)} {name} seed={seed} K={cutoff}: complete')
            continue
        library = {k: np.asarray(manifest['library'][str(k)]['params']) for k in range(1, cutoff+1)}
        averages, all_counts, usage = [-1.0], [], {str(k): 0 for k in library}
        fields = ['time_index', 'time', 'trajectory_index', 'trajectory_seed', 'u0_count', 'z_count',
                  'original_cnot', 'replaced_cnot', 'original_single_qubit', 'replaced_single_qubit',
                  'original_depth_upper_bound', 'replaced_depth_upper_bound', 'pqc_block_count',
                  'largest_pqc_k', 'imbalance']
        start = time.perf_counter()
        log(f'START job={job_index}/{len(jobs)} {name} seed={seed} K={cutoff}')
        with (dest / 'resources.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for index in range(1, len(grid)):
                values = []
                for j in range(branches.shape[1]):
                    seq = branches[index-1, j, :index]
                    value, resource = trajectory(engine, seq, step, library, cutoff)
                    for k in resource.pop('blocks'):
                        usage[str(k)] += 1
                    values.append(value)
                    row = {'time_index': index, 'time': grid[index], 'trajectory_index': j,
                           'trajectory_seed': cfg['sampling']['stream_seed'], 'u0_count': int((seq==0).sum()),
                           'z_count': int((seq>0).sum()), 'imbalance': value, **resource}
                    writer.writerow(row)
                    all_counts.append(row)
                averages.append(float(np.mean(values)))
                f.flush()
                elapsed = time.perf_counter()-start
                log(f'{name} seed={seed} K={cutoff} step={index}/{len(grid)-1} '
                    f't={grid[index]:g} I={averages[-1]:.8f} elapsed={elapsed:.1f}s '
                    f'ETA={(len(grid)-1-index)*elapsed/index:.1f}s')
        result = np.asarray(averages)
        np.savez_compressed(dest / 'curves.npz', time=grid, imbalance=result,
                            error_to_kraus=reference['kraus']-result,
                            error_to_exact=reference['exact']-result,
                            error_to_baseline=baseline-result)
        summary = {'channel': name, 'training_seed': seed, 'cutoff': cutoff,
                   'branch_signature': signature, 'training_manifest_sha256': manifest_hash,
                   'M': branches.shape[1], 'wall_seconds': time.perf_counter()-start,
                   'error_definition': 'reference minus approximation (signed)',
                   'accuracy_to_kraus': accuracy(result, reference['kraus']),
                   'accuracy_to_exact': accuracy(result, reference['exact']),
                   'accuracy_to_baseline': accuracy(result, baseline),
                   'accuracy_to_same_seed_exact': accuracy(result, reference['same_seed_exact']),
                   'block_usage': usage, 'gate_statistics': {}}
        for which, rows in [('all_times', all_counts), ('final_time', [a for a in all_counts if a['time_index']==len(grid)-1])]:
            summary['gate_statistics'][which] = {}
            for kind in ['cnot', 'single_qubit', 'depth_upper_bound']:
                old = np.array([a['original_'+kind] for a in rows], float)
                new = np.array([a['replaced_'+kind] for a in rows], float)
                summary['gate_statistics'][which][kind] = {
                    'original_mean': float(old.mean()), 'replaced_mean': float(new.mean()),
                    'original_std_ddof0': float(old.std()), 'replaced_std_ddof0': float(new.std()),
                    'original_total': float(old.sum()), 'replaced_total': float(new.sum()),
                    'saving_fraction_ratio_of_totals': float(1-new.sum()/old.sum()) if old.sum() else None}
        write_json(meta_path, summary)
        log(f'COMPLETE {name} seed={seed} K={cutoff}; Kraus RMSE={summary["accuracy_to_kraus"]["rmse"]:.8f}')


def cli(stage):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT / 'config.json')
    p.add_argument('--output', type=Path, default=ROOT / 'data')
    p.add_argument('--backend', choices=['auto', 'cpu', 'gpu'])
    if stage == 'reference':
        p.add_argument('--source', type=Path)
        p.add_argument('--recompute', action='store_true')
    if stage == 'baseline':
        p.add_argument('--recompute', action='store_true')
    if stage == 'replace':
        p.add_argument('--seeds', type=int, nargs='+')
        p.add_argument('--channels', nargs='+', choices=['full_direct', 'full_iterative', 'b32_direct', 'b32_iterative'])
    args = p.parse_args()
    cfg = load_config(args.config)
    if stage == 'reference':
        prepare_reference(cfg, args.output, args.source, args.recompute, args.backend)
    elif stage == 'baseline':
        run_baseline(cfg, args.output, args.backend, args.recompute)
    else:
        replace(cfg, args.output, args.backend, args.seeds, args.channels)
