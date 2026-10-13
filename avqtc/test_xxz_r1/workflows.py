"""XXZ reference generation and paired original/replaced trajectory workflows."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from checkpoint_io import replace_with_retry
from circuit_export import export_circuit
from training import Logger
from xxz_core import (ROOT, Engine, analytic_params, channel, digest, file_hash, load_config,
                      model_record, read_json, save_model_record, source_hashes,
                      stamp, trajectory, write_json)


def times(cfg):
    m = cfg['model']
    return np.arange(round(m['total_time']/m['dt'])+1)*m['dt']


def reference_spec(cfg):
    return {'model': cfg['model'], 'sampling': cfg['sampling'],
            'resolved_model': model_record(cfg)}


def generator_hashes():
    return {name: file_hash(ROOT/name) for name in ['xxz_core.py', 'workflows.py', 'checkpoint_io.py', 'circuit_export.py']}


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    replace_with_retry(temporary, path)


def ensure_reference(cfg, output):
    output = Path(output)
    meta = read_json(output/'reference/metadata.json')
    if meta['specification'] != reference_spec(cfg):
        raise ValueError('Reference model/disorder/config differs; use a separate output directory')
    if meta['generator_source_hashes'] != generator_hashes():
        raise ValueError('Reference generator changed after this run; preserve original code or use new output')
    if meta['npz_sha256'] != file_hash(output/'reference/reference.npz'):
        raise ValueError('Reference NPZ checksum mismatch')
    return np.load(output/'reference/reference.npz'), meta


def exact_sampling(engine, u0, probs, cfg, count, stream_seed, log, label):
    rng = np.random.default_rng(stream_seed)
    result = [-1.0]
    grid = times(cfg)
    started = time.perf_counter()
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
                    states[rows] *= engine.z_signs[:, q][None, :]
        result.append(float(engine.xp.mean(engine.xp.sum(
            engine.xp.abs(states)**2 * engine.imbalance, axis=1)).item()))
        if index == 1 or index % 10 == 0 or index == len(grid)-1:
            log(f'{label} t={grid[index]:g}/{grid[-1]:g}, step={index}/{len(grid)-1}, '
                f'elapsed={time.perf_counter()-started:.1f}s')
    return np.asarray(result)


def prepare_reference(cfg, output, source=None, recompute=False, backend=None):
    if source is not None:
        raise ValueError('XXZ references must be generated for the configured disorder; XY imports are unsupported')
    output = Path(output)
    record = save_model_record(cfg, output)
    dest = output/'reference'
    dest.mkdir(parents=True, exist_ok=True)
    log = Logger(output/'logs/01_reference.log')
    metadata = dest/'metadata.json'
    if metadata.exists():
        ensure_reference(cfg, output)
        log('REFERENCE RESUME: complete reference verified')
        return
    m, s = cfg['model'], cfg['sampling']
    partial_spec = {'specification': reference_spec(cfg), 'generator_source_hashes': generator_hashes()}
    partial_path = dest/'partial_manifest.json'
    if partial_path.exists() and read_json(partial_path) != partial_spec:
        raise ValueError('Partial reference belongs to a different model/config/source version')
    write_json(partial_path, partial_spec)
    engine = Engine(m['n'], backend or cfg['training']['backend'], m, cfg['circuit'])
    log(f'XXZ reference: n={m["n"]}, backend={engine.backend}, fixed h_i={record["fields"]}')
    started = time.perf_counter()
    u0, h = engine.u0(m['dt'], m['gamma'])
    _, probs = channel(m['n'], m['dt'], m['gamma'])
    data = {'time': times(cfg)}
    import qutip as qt
    deterministic_path = dest/'deterministic_curves.npz'
    if deterministic_path.exists():
        with np.load(deterministic_path) as saved:
            data['exact'], data['kraus'] = saved['exact'], saved['kraus']
        log('REFERENCE RESUME: Lindblad and density-channel curves reused')
    else:
        dims = [[2]*m['n'], [2]*m['n']]
        H = qt.Qobj(h, dims=dims).to('csr')
        ket = qt.Qobj(engine.cpu(engine.initial), dims=[[2]*m['n'], [1]*m['n']])
        jumps = [np.sqrt(m['gamma'])*qt.Qobj(np.diag(engine.signs[:, j]), dims=dims).to('csr')
                 for j in range(m['n'])]
        obs = qt.Qobj(np.diag(engine.cpu(engine.imbalance)), dims=dims).to('csr')
        log('Lindblad sparse solver starting; QuTiP progress follows')
        solved = qt.mesolve(H, ket, data['time'], jumps, e_ops=[obs],
                           options={'progress_bar': 'text', 'rtol': 1e-9, 'atol': 1e-11})
        data['exact'] = np.real(solved.expect[0])
        rho = engine.xp.outer(engine.initial, engine.initial.conj())
        signs = engine.xp.asarray(engine.signs @ engine.signs.T)
        kraus = [-1.0]
        for index in range(1, len(data['time'])):
            rho = probs[0]*(u0 @ rho @ u0.conj().T) + probs[1]*signs*rho
            kraus.append(float(engine.xp.real(engine.xp.sum(engine.xp.diag(rho)*engine.imbalance)).item()))
            log(f'Kraus step={index}/{len(data["time"])-1} I={kraus[-1]:.8f}')
        data['kraus'] = np.asarray(kraus)
        atomic_npz(deterministic_path, exact=data['exact'], kraus=data['kraus'])
    for count in sorted(set(s['m_values']+[s['trajectory_m']])):
        seeds = np.random.default_rng(s['master_seed']).integers(0, 2**32-1, size=s['repeats'], dtype=np.uint32)
        data[f'seeds_M{count}'] = seeds
        partial = dest/f'sample_M{count}_partial.npz'
        rows = []
        if partial.exists():
            with np.load(partial) as previous:
                if not np.array_equal(previous['seeds'], seeds):
                    raise ValueError('Reference checkpoint seed mismatch')
                values = previous['values']
                if values.ndim != 2 or values.shape[1] != len(data['time']) or len(values)>len(seeds) or not np.isfinite(values).all():
                    raise ValueError('Invalid reference checkpoint')
                rows = list(values)
            log(f'REFERENCE RESUME: M={count}, completed repeats={len(rows)}/{len(seeds)}')
        for repeat in range(len(rows), len(seeds)):
            rows.append(exact_sampling(engine, u0, probs, cfg, count, int(seeds[repeat]), log,
                                       f'M={count}, repeat={repeat+1}/{len(seeds)}'))
            atomic_npz(partial, values=np.asarray(rows), seeds=seeds)
        data[f'sample_M{count}'] = np.asarray(rows)
    matches = np.flatnonzero(data[f'seeds_M{s["trajectory_m"]}'] == s['stream_seed'])
    data['same_seed_exact'] = (data[f'sample_M{s["trajectory_m"]}'][matches[0]] if len(matches)
                              else exact_sampling(engine, u0, probs, cfg, s['trajectory_m'],
                                                  s['stream_seed'], log, 'paired exact MC'))
    for key in ['exact', 'kraus', 'same_seed_exact']:
        if data[key].shape != data['time'].shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid reference curve: {key}')
    atomic_npz(dest/'reference.npz', **data)
    meta = {**partial_spec, 'created_at': stamp(), 'wall_seconds_this_session': time.perf_counter()-started,
            'npz_sha256': file_hash(dest/'reference.npz'), 'origin': 'generated for this XXZ disorder',
            'observable': 'domain wall: (sum_left Z-sum_right Z)/n; Neel: sum((-1)^j Z_j)/n, zero-based j',
            'initial_imbalance': -1.0, 'environment': engine.environment(), 'qutip_version': qt.__version__,
            'solver_rtol': 1e-9, 'solver_atol': 1e-11,
            'trajectory_policy': 'independent ensemble at each time; time-major, trajectory-minor RNG'}
    write_json(metadata, meta)
    log('REFERENCE COMPLETE')


def paired_branch_table(cfg, output):
    dest = Path(output)/'baseline'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/'branches.npz'
    signature = digest({'resolved_model': model_record(cfg), 'dt': cfg['model']['dt'],
                        'total_time': cfg['model']['total_time'], 'M': cfg['sampling']['trajectory_m'],
                        'stream_seed': cfg['sampling']['stream_seed']})
    if path.exists():
        with np.load(path) as old:
            if str(old['signature']) != signature:
                raise ValueError('Stored branches use a different model/circuit/M/seed')
            return old['branches'], signature
    grid, s = times(cfg), cfg['sampling']
    n = cfg['model']['n']
    _, probs = channel(n, cfg['model']['dt'], cfg['model']['gamma'])
    rng = np.random.default_rng(s['stream_seed'])
    out = np.full((len(grid)-1, s['trajectory_m'], len(grid)-1), -1, np.int16)
    for index in range(1, len(grid)):
        out[index-1, :, :index] = rng.choice(n+1, size=(s['trajectory_m'], index), p=probs)
    atomic_npz(path, branches=out, signature=signature)
    return out, signature


def run_baseline(cfg, output, backend=None, recompute=False):
    output = Path(output)
    save_model_record(cfg, output)
    ensure_reference(cfg, output)
    branches, signature = paired_branch_table(cfg, output)
    dest = output/'baseline'
    meta_path = dest/'metadata.json'
    log = Logger(output/'logs/02_baseline.log')
    if meta_path.exists():
        old = read_json(meta_path)
        if old['signature'] != signature or old['generator_source_hashes'] != generator_hashes():
            raise ValueError('Baseline config/code mismatch')
        for filename, sha in old['files_sha256'].items():
            if file_hash(dest/filename) != sha:
                raise ValueError(f'Baseline checksum mismatch: {filename}')
        log('BASELINE RESUME: complete baseline verified')
        return
    m = cfg['model']
    engine = Engine(m['n'], backend or cfg['training']['backend'], m, cfg['circuit'])
    step, _ = channel(engine.n, m['dt'], m['gamma'])
    grid, averages = times(cfg), [-1.0]
    fields = ['time_index', 'time', 'trajectory_index', 'trajectory_seed',
              'original_cnot', 'original_single_qubit', 'original_depth_upper_bound', 'imbalance']
    started = time.perf_counter()
    log(f'BASELINE START: n={m["n"]}, substeps={engine.baseline_substeps}; second-order XXZ circuit')
    export_circuit(dest/'circuits/original_one_step',
                   analytic_params(m['n'], 1, engine.baseline_substeps, step, m), m['n'],
                   {'role': 'original second-order approximation to U0', 'resolved_model': model_record(cfg)})
    with (dest/'resources.csv').open('w', newline='', encoding='utf-8') as f:
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
            elapsed = time.perf_counter()-started
            log(f'BASELINE step={index}/{len(grid)-1}, t={grid[index]:g}, I={averages[-1]:.8f}, '
                f'elapsed={elapsed:.1f}s, ETA={(len(grid)-1-index)*elapsed/index:.1f}s')
    atomic_npz(dest/'baseline.npz', time=grid, imbalance=averages)
    write_json(meta_path, {'signature': signature, 'resolved_model': model_record(cfg), 'created_at': stamp(),
                          'wall_seconds': time.perf_counter()-started, 'generator_source_hashes': generator_hashes(),
                          'convention': 'symmetric D/2 E/2 O E/2 D/2; merge consecutive diagonal halves inside no-jump run',
                          'gate_count_scope': 'evolution only; initial preparation/readout excluded; no angle pruning',
                          'files_sha256': {name: file_hash(dest/name) for name in ['baseline.npz','branches.npz','resources.csv']},
                          'environment': engine.environment()})
    log('BASELINE COMPLETE')


def accuracy(a, b):
    delta = np.asarray(a) - np.asarray(b)
    return {'rmse': float(np.sqrt(np.mean(delta**2))), 'mae': float(np.mean(abs(delta))),
            'max_abs_error': float(np.max(abs(delta)))}


def replace(cfg, output, backend=None, seeds=None, channels=None):
    output = Path(output)
    save_model_record(cfg, output)
    reference, _ = ensure_reference(cfg, output)
    branches, signature = paired_branch_table(cfg, output)
    base_meta = read_json(output / 'baseline' / 'metadata.json')
    if base_meta['signature'] != signature or base_meta['generator_source_hashes'] != generator_hashes():
        raise ValueError('Run 02 for the same paired stream first')
    for filename, sha in base_meta['files_sha256'].items():
        if file_hash(output/'baseline'/filename) != sha:
            raise ValueError(f'Baseline file changed: {filename}')
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
            if (manifest['specification']['circuit'] != cfg['circuit']
                    or manifest['specification']['resolved_model'] != model_record(cfg)
                    or manifest['specification']['source_hashes'] != source_hashes()):
                raise ValueError(f'{manifest_path}: circuit/disorder/training code changed')
            cutoffs = [cfg['training']['max_k']] if route == 'direct' else cfg['sampling']['iterative_cutoffs']
            for cutoff in cutoffs:
                if any(str(k) not in manifest['library'] for k in range(1, cutoff+1)):
                    raise ValueError(f'{name} seed={seed}: library does not cover K={cutoff}')
                jobs.append((name, seed, cutoff, manifest, file_hash(manifest_path)))
    engine = Engine(cfg['model']['n'], backend or cfg['training']['backend'], cfg['model'], cfg['circuit'])
    step, _ = channel(engine.n, cfg['model']['dt'], cfg['model']['gamma'])
    grid = times(cfg)
    log = Logger(output / 'logs' / '07_replace.log')
    for job_index, (name, seed, cutoff, manifest, manifest_hash) in enumerate(jobs, 1):
        dest = output / 'replacement' / name / f'seed_{seed}' / f'K{cutoff}'
        dest.mkdir(parents=True, exist_ok=True)
        meta_path = dest / 'summary.json'
        if meta_path.exists():
            old = read_json(meta_path)
            if (old['branch_signature'] != signature or old['training_manifest_sha256'] != manifest_hash
                    or old['generator_source_hashes'] != generator_hashes()):
                raise ValueError(f'{dest}: completed replacement uses different input data')
            for filename, sha in old['files_sha256'].items():
                if file_hash(dest/filename) != sha:
                    raise ValueError(f'Replacement file checksum mismatch: {dest/filename}')
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
        atomic_npz(dest / 'curves.npz', time=grid, imbalance=result,
                            error_to_kraus=reference['kraus']-result,
                            error_to_exact=reference['exact']-result,
                            error_to_baseline=baseline-result,
                            error_to_same_seed_exact=reference['same_seed_exact']-result)
        summary = {'channel': name, 'training_seed': seed, 'cutoff': cutoff,
                   'branch_signature': signature, 'training_manifest_sha256': manifest_hash,
                   'generator_source_hashes': generator_hashes(), 'resolved_model': model_record(cfg),
                   'files_sha256': {filename: file_hash(dest/filename) for filename in ['curves.npz','resources.csv']},
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
    if stage == 'replace' and args.seeds and not set(args.seeds).issubset(cfg['training']['seeds']):
        p.error('--seeds must be a subset of the configured training seeds')
    if stage == 'reference':
        prepare_reference(cfg, args.output, args.source, args.recompute, args.backend)
    elif stage == 'baseline':
        run_baseline(cfg, args.output, args.backend, args.recompute)
    else:
        replace(cfg, args.output, args.backend, args.seeds, args.channels)
