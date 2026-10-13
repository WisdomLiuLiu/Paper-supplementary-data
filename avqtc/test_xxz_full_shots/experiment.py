"""Standalone finite-shot full-space direct compilation experiment."""
import argparse
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
from xxz_core import (Engine, analytic_params, channel, digest, file_hash,
                      read_json, write_json, seed_for, stamp, model_record)
from shot_engine import hst_probability, loss_gradient, measure, hst_cost, record, wilson
from circuit_export import export_circuit

ROOT = Path(__file__).resolve().parent
SOURCES = ['experiment.py', 'shot_engine.py', 'xxz_core.py', 'checkpoint_io.py',
           'circuit_export.py', 'architecture_search.py', 'post_audit.py']


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--output', type=Path, default=ROOT/'data')
    return parser.parse_args()


def validate(cfg):
    e, m = cfg['experiment'], cfg['model']
    for key in ['repetitions', 'seeds', 'shots']:
        values = e[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f'{key} must be a nonempty list of distinct integers')
        if any(type(v) is not int or v < (1 if key == 'repetitions' else 0) for v in values):
            raise ValueError(f'Invalid {key}')
    for key in ['iterations', 'validation_interval', 'print_every', 'selection_shots_multiplier',
                'test_shots_multiplier', 'teacher_trotter_substeps']:
        if type(e[key]) is not int or e[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['min_depth']:
        if type(e[key]) is not int or e[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['extra_depth', 'random_restarts']:
        if type(e[key]) is not int or e[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if e['max_depth'] is not None and (type(e['max_depth']) is not int or e['max_depth'] < e['min_depth']):
        raise ValueError('max_depth must be null or >= min_depth')
    if e['max_depth'] is None and min(e['repetitions'])+e['extra_depth'] < e['min_depth']:
        raise ValueError('Empty depth range')
    if type(e['depth_warm_start']) is not bool:
        raise ValueError('Invalid search options')
    if e['validation_interval'] != 1:
        raise ValueError('Shot threshold must be checked after every update: validation_interval=1')
    if 'depth_override' in e:
        raise ValueError('Old fixed-depth config: replace depth_override with architecture-search options')
    if e['backend'] not in ('auto', 'cpu', 'gpu') or e['teacher_mode'] not in ('exact', 'trotter'):
        raise ValueError('Invalid backend/teacher_mode')
    if e['initialization'] != 'analytic_trotter':
        raise ValueError('Only fresh analytic_trotter initialization is supported')
    for key in ['learning_rate', 'adam_epsilon', 'jitter_scale', 'success_fidelity', 'adam_beta1', 'adam_beta2']:
        if not np.isfinite(e[key]):
            raise ValueError(f'Nonfinite {key}')
    if e['learning_rate'] <= 0 or e['adam_epsilon'] <= 0 or e['jitter_scale'] < 0:
        raise ValueError('Invalid Adam/init parameters')
    if not all(0 < e[key] < 1 for key in ['adam_beta1', 'adam_beta2']) or not 0 < e['success_fidelity'] <= 1:
        raise ValueError('Invalid beta or fidelity')
    if type(m['n']) is not int or m['n'] < 2 or m['n'] % 2 or m['dt'] <= 0 or m['gamma'] < 0:
        raise ValueError('Invalid model')
    channel(m['n'], m['dt'], m['gamma'])


def prepare(config_path, output):
    cfg = read_json(config_path)
    validate(cfg)
    e = cfg['experiment']
    jobs = []
    for seed in e['seeds']:
        for k in e['repetitions']:
            for shots in e['shots']:
                label = 'infinite' if shots == 0 else str(shots)
                jobs.append(dict(seed=seed, k=k, shots=shots,
                                 min_depth=e['min_depth'], max_depth=e['max_depth'] or k+e['extra_depth'],
                                 name=f'seed_{seed}/s{k:02d}/shots_{label}'))
    spec = dict(config=cfg, model_record=model_record(cfg), jobs=jobs,
                source_sha256={f: file_hash(ROOT/f) for f in SOURCES})
    fingerprint = digest(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'manifest.json'
    if manifest.exists():
        if read_json(manifest)['fingerprint'] != fingerprint:
            raise ValueError('Config/code changed. Use a NEW --output folder; existing results are protected.')
    else:
        write_json(manifest, dict(fingerprint=fingerprint, specification=spec, created_at=stamp(),
                                 actual_hardware_shots=0, measurement_simulation='binomial Born counts'))
    return cfg, jobs, fingerprint


def log(output, message):
    line = f'[{stamp()}] {message}'
    print(line, flush=True)
    with (Path(output)/'progress.log').open('a', encoding='utf-8') as f:
        f.write(line+'\n')


def target_for(engine, cfg, k):
    m, e = cfg['model'], cfg['experiment']
    u0, _ = engine.u0(m['dt'], m['gamma'])
    exact = engine.xp.linalg.matrix_power(u0, k)
    step, _ = channel(m['n'], m['dt'], m['gamma'])
    proxy = engine.proxy_unitary(k, step, e['teacher_trotter_substeps'])
    return (exact if e['teacher_mode'] == 'exact' else proxy), exact, proxy, step


def run_attempt(engine, cfg, job, output, fingerprint, stop_after=None, initial_params=None):
    """stop_after is used only by the interruption/resume unit test."""
    e, m = cfg['experiment'], cfg['model']
    seed, k, depth, shots = (job[key] for key in ['seed', 'k', 'depth', 'shots'])
    folder = Path(output)/'training'/job['name']
    path = folder/'checkpoint.json'
    folder.mkdir(parents=True, exist_ok=True)
    cost = hst_cost(m['n'], depth, k, e['teacher_trotter_substeps'])
    started = time.perf_counter()
    selection_shots = shots*e['selection_shots_multiplier']
    target = exact = proxy = step = None
    if path.exists():
        result = read_json(path)
        if result['fingerprint'] != fingerprint:
            raise ValueError('Checkpoint fingerprint mismatch')
        if result['status'] == 'completed':
            # Exports can be restored if interruption occurred after final checkpoint.
            export_circuit(folder/'selected_circuit', np.asarray(result['best_params']), m['n'], job)
            log(output, f'RESUME skip {job["name"]}: already completed')
            return result
        log(output, f'RESUME {job["name"]} at iteration {result["iteration"]}')
    else:
        target, exact, proxy, step = target_for(engine, cfg, k)
        params = (analytic_params(m['n'], k, depth, step, m) if initial_params is None else np.array(initial_params, copy=True))
        p = hst_probability(engine, params, target)
        measured = float(measure(p, selection_shots, np.random.default_rng(seed_for(seed, k, shots, depth, job.get('init_name'), 'select', 0))))
        result = dict(status='running', fingerprint=fingerprint, job=job, iteration=0,
                      params=params.tolist(), m=np.zeros_like(params).tolist(), v=np.zeros_like(params).tolist(),
                      best_params=params.tolist(), best_selection_fidelity=measured, selected_iteration=0,
                      initial_params=params.tolist(), best_initialization=job.get('init_name', 'analytic'),
                      history=[dict(iteration=0, measured_selection_fidelity=measured)],
                      resources_by_phase={}, cost_per_hst=cost, wall_seconds=0.0, started_at=stamp(),
                      optimizer='Adam', loss='1 - HST all-zero probability', actual_hardware_shots=0,
                      environment=engine.environment())
        record(result['resources_by_phase'], 'selection', 1, selection_shots, cost)
        write_json(path, result)
    if target is None:
        target, exact, proxy, step = target_for(engine, cfg, k)
    previous_wall = result['wall_seconds']
    params, moment, variance = (np.asarray(result[key]) for key in ['params', 'm', 'v'])
    log(output, f'START {job["name"]}: slices={depth}, parameters={params.size}, matrix_dim={engine.dim}, hardware_HST_qubits={2*m["n"]}, '
        f'shots/setting={shots or "infinite control"}, lr={e["learning_rate"]}, '
        f'settings/iteration={1+cost["gradient_settings_per_iteration"]}')
    for iteration in range(result['iteration']+1, e['iterations']+1):
        if result['best_selection_fidelity'] >= e['success_fidelity']:
            break
        iteration_started = time.perf_counter()
        rng = np.random.default_rng(seed_for(seed, k, shots, depth, job.get('init_name'), 'gradient', iteration))
        # Reuse the latest stopping measurement as the pre-update loss.
        loss = 1-result['history'][-1]['measured_selection_fidelity']
        _, grad, settings = loss_gradient(engine, params, target, shots, rng, include_forward=False)
        if settings != cost['gradient_settings_per_iteration']:
            raise AssertionError('Parameter-shift settings mismatch')
        record(result['resources_by_phase'], 'training', settings, shots, cost)
        b1, b2 = e['adam_beta1'], e['adam_beta2']
        moment = b1*moment+(1-b1)*grad
        variance = b2*variance+(1-b2)*grad**2
        params -= e['learning_rate']*(moment/(1-b1**iteration))/(np.sqrt(variance/(1-b2**iteration))+e['adam_epsilon'])
        h = dict(iteration=iteration, measured_loss_before_update=loss,
                 gradient_l2=float(np.linalg.norm(grad)), learning_rate=e['learning_rate'],
                 circuit_settings=settings, simulated_shots=settings*shots)
        if iteration == 1 or iteration % e['validation_interval'] == 0 or iteration == e['iterations']:
            p = hst_probability(engine, params, target)
            f = float(measure(p, selection_shots, np.random.default_rng(seed_for(seed, k, shots, depth, job.get('init_name'), 'select', iteration))))
            record(result['resources_by_phase'], 'selection', 1, selection_shots, cost)
            h.update(measured_selection_fidelity=f)
            # No exact probability is used to rank/select finite-shot checkpoints.
            # Ties retain the earlier checkpoint; selection data are not a test.
            if f > result['best_selection_fidelity']:
                result.update(best_selection_fidelity=f, best_params=params.tolist(), selected_iteration=iteration)
        engine.sync()
        h['wall_seconds'] = time.perf_counter()-iteration_started
        result['history'].append(h)
        result.update(iteration=iteration, params=params.tolist(), m=moment.tolist(), v=variance.tolist(),
                      wall_seconds=previous_wall+time.perf_counter()-started)
        write_json(path, result)
        if iteration == 1 or iteration % e['print_every'] == 0 or iteration == e['iterations']:
            total_shots = sum(v['simulated_measurement_shots'] for v in result['resources_by_phase'].values())
            log(output, f'{job["name"]} iter={iteration}/{e["iterations"]}, loss_hat={loss:.6f}, '
                f'best_selection={result["best_selection_fidelity"]:.6f}, '
                f'simulated_shots={total_shots:,}, elapsed={result["wall_seconds"]:.1f}s, '
                f'ETA~{h["wall_seconds"]*(e["iterations"]-iteration):.1f}s')
        if stop_after is not None and iteration >= stop_after:
            return result
    result.update(status='completed', completed_at=stamp(),
                  wall_seconds=previous_wall+time.perf_counter()-started,
                  stop_reason='shot_fidelity_threshold' if result['best_selection_fidelity'] >= e['success_fidelity'] else 'iteration_budget_exhausted')
    export_circuit(folder/'selected_circuit', np.asarray(result['best_params']), m['n'], job)
    write_json(path, result)
    log(output, f'DONE {job["name"]}: measured selection F={result["best_selection_fidelity"]:.8f}')
    return result


def audit_selected(engine, cfg, result):
    """Only called AFTER the architecture decision is durably frozen."""
    e = cfg['experiment']
    job = result['job']
    seed, k, shots = (job[key] for key in ['seed', 'k', 'shots'])
    target, exact, proxy, _ = target_for(engine, cfg, k)
    best, params = np.asarray(result['best_params']), np.asarray(result['params'])
    cost = result['cost_per_hst']
    result['initial_exact_target_fidelity'] = hst_probability(engine, np.asarray(result['initial_params']), target)
    p = hst_probability(engine, best, target)
    test_shots = shots*e['test_shots_multiplier']
    test_p = float(measure(p, test_shots, np.random.default_rng(seed_for(seed, k, shots, 'heldout'))))
    ci = wilson(round(test_p*test_shots), test_shots)
    record(result['resources_by_phase'], 'independent_test', 1, test_shots, cost)
    exact_hs = hst_probability(engine, best, exact)
    target_vs_exact = float(abs(complex((engine.xp.trace(exact.conj().T@target)/engine.dim).item()))**2)
    proxy_vs_exact = float(abs(complex((engine.xp.trace(exact.conj().T@proxy)/engine.dim).item()))**2)
    threshold = e['success_fidelity']
    result['audit'] = dict(selected_exact_target_fidelity=p, selected_exact_U0s_fidelity=exact_hs,
                           final_exact_target_fidelity=hst_probability(engine, params, target),
                           independent_test_fidelity=test_p, independent_test_shots=test_shots,
                           independent_test_wilson95=ci,
                           exact_target_threshold_pass=bool(p >= threshold),
                           exact_U0s_threshold_pass=bool(exact_hs >= threshold),
                           finite_shot_certified_pass=bool(ci is not None and ci[0] >= threshold),
                           improved_over_initial=bool(p > result['initial_exact_target_fidelity']+1e-12),
                           fidelity_gain=p-result['initial_exact_target_fidelity'],
                           teacher_vs_exact_U0s_fidelity=target_vs_exact,
                           gate_proxy_vs_exact_U0s_fidelity=proxy_vs_exact)
    return result


def train(config, output):
    from architecture_search import run_search
    cfg, jobs, fingerprint = prepare(config, output)
    engine = Engine(cfg['model']['n'], cfg['experiment']['backend'], cfg['model'], cfg['circuit'])
    log(output, f'PLAN {len(jobs)} runs; backend={engine.backend}; full-space direct HST, finite-shot parameter shifts')
    for index, job in enumerate(jobs, 1):
        log(output, f'JOB {index}/{len(jobs)}: {job["name"]}')
        try:
            run_search(engine, cfg, job, output, fingerprint)
        except BaseException:
            with (Path(output)/'errors.log').open('a', encoding='utf-8') as f:
                f.write(f'\n{stamp()} {job["name"]}\n'+traceback.format_exc())
            raise
    log(output, 'All training jobs completed. Run 03_analyze.py (run_all.ps1 does this automatically).')
