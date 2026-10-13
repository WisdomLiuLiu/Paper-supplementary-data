"""Shot-only architecture decisions; exact auditing starts after decision freeze."""
import copy
import time
from pathlib import Path
import numpy as np
from experiment import run_attempt, target_for, log
from xxz_core import analytic_params, transfer, seed_for, read_json, write_json, stamp
from shot_engine import hst_probability, measure, record, wilson
from circuit_export import export_circuit


def add_ledgers(ledgers):
    total = {}
    for ledger in ledgers:
        for phase, values in ledger.items():
            row = total.setdefault(phase, {})
            for key, value in values.items():
                row[key] = row.get(key, 0)+value
    return total


def attempt_records(folder):
    return [read_json(p) for p in sorted((Path(folder)/'attempts').glob('*/checkpoint.json'))]


def collected_resources(folder, state):
    return add_ledgers([r['resources_by_phase'] for r in attempt_records(folder)] + [state['resources_by_phase']])


def screen_candidate(engine, cfg, job, candidate):
    """Use the actual stopping measurement. No extra rescreen or exact audit."""
    e = cfg['experiment']
    shots = job['shots']*e['selection_shots_multiplier']
    measured = candidate['best_selection_fidelity']
    score = measured
    return dict(measured_fidelity=measured, shots=shots, successes=round(measured*shots) if shots else None,
                score=score, passed=bool(score >= e['success_fidelity']), rule='measured_checkpoint_threshold',
                new_measurements=0)


def run_search(engine, cfg, job, output, fingerprint, stop_after_depth=None):
    e = cfg['experiment']
    folder = Path(output)/'training'/job['name']
    folder.mkdir(parents=True, exist_ok=True)
    state_path, result_path = folder/'search.json', folder/'checkpoint.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['fingerprint'] != fingerprint:
            raise ValueError('Search fingerprint mismatch')
        if result['status']=='completed':
            log(output, f'RESUME {job["name"]}: completed search, depth={result["job"]["depth"]}')
            return result
    if state_path.exists():
        state = read_json(state_path)
        if state['fingerprint'] != fingerprint:
            raise ValueError('Search fingerprint mismatch')
    else:
        state = dict(status='searching', fingerprint=fingerprint, job=job, depths=[],
                     resources_by_phase={}, started_at=stamp(), screen_wall_seconds=0.0)
        write_json(state_path, state)
    if state['status']=='searching':
        _, _, _, step = target_for(engine, cfg, job['k'])
        for depth in range(job['min_depth'], job['max_depth']+1):
            if any(d['depth']==depth for d in state['depths']):
                continue
            base = analytic_params(engine.n, job['k'], depth, step, cfg['model'])
            candidates = [('analytic', base)]
            warm = None
            if e['depth_warm_start'] and state['depths']:
                prev = read_json(Path(output)/'training'/state['depths'][-1]['attempt']/'checkpoint.json')
                warm = transfer(np.asarray(prev['best_params']), depth)
                candidates.append(('depth_warm', warm))
            for restart in range(e['random_restarts']):
                rng = np.random.default_rng(seed_for(job['seed'], job['k'], depth, 'jitter', restart))
                center = warm if warm is not None else base
                candidates.append((f'jitter_{restart}', center+rng.normal(0, e['jitter_scale'], base.shape)))
            best = None
            log(output, f'SEARCH {job["name"]}: depth={depth}/{job["max_depth"]}, {len(candidates)} initializations')
            for init_name, params in candidates:
                candidate_job = dict(job, depth=depth, init_name=init_name,
                                     name=job['name']+f'/attempts/d{depth:02d}_{init_name}')
                result = run_attempt(engine, cfg, candidate_job, output, fingerprint, initial_params=params)
                if best is None or result['best_selection_fidelity'] > best['best_selection_fidelity']:
                    best = result
                if best['best_selection_fidelity'] >= e['success_fidelity']:
                    break
            started = time.perf_counter()
            screen = screen_candidate(engine, cfg, job, best)
            state['screen_wall_seconds'] += time.perf_counter()-started
            entry = dict(depth=depth, attempt=best['job']['name'],
                         candidate_selection_fidelity=best['best_selection_fidelity'], **screen)
            state['depths'].append(entry)
            if screen['passed'] or depth==job['max_depth']:
                chosen = entry if screen['passed'] else max(state['depths'], key=lambda d:d['score'])
                state.update(status='decision_frozen', selected_attempt=chosen['attempt'], selected_depth=chosen['depth'],
                             threshold_reached=screen['passed'], decision_frozen_at=stamp(),
                             stop_reason='first_shot_checkpoint_reaching_threshold' if screen['passed'] else 'depth_budget_exhausted_best_shot_checkpoint')
            write_json(state_path, state)
            log(output, f'SCREEN {job["name"]} depth={depth}: measured_F={screen["measured_fidelity"]:.7f}, '
                f'score={screen["score"]:.7f}, pass={screen["passed"]}; '
                f'committed shots={sum(v["simulated_measurement_shots"] for v in collected_resources(folder, state).values()):,}')
            if stop_after_depth is not None and len(state['depths'])>=stop_after_depth:
                return state
            if state['status']=='decision_frozen':
                break
    # No exact metrics or final test here. Stage 03 audits only after ALL jobs finish.
    selected = read_json(Path(output)/'training'/state['selected_attempt']/'checkpoint.json')
    final = copy.deepcopy(selected)
    final.update(job=dict(job, depth=state['selected_depth']), search=state,
                 selected_attempt=state['selected_attempt'], architecture_threshold_reached=state['threshold_reached'],
                 resources_by_phase=collected_resources(folder, state), stop_reason=state['stop_reason'])
    records = attempt_records(folder)
    final['total_optimizer_iterations'] = sum(r['iteration'] for r in records)
    final['total_attempts'] = len(records)
    final['searched_depths'] = [d['depth'] for d in state['depths']]
    final.update(status='completed', completed_at=stamp(),
                 audit_status='pending_all_training',
                 wall_seconds=sum(r['wall_seconds'] for r in records)+state['screen_wall_seconds'])
    export_circuit(folder/'selected_circuit', np.asarray(final['best_params']), engine.n, final['job'])
    write_json(result_path, final)
    log(output, f'SEARCH DONE {job["name"]}: depth={state["selected_depth"]}, '
        f'shot_threshold_pass={state["threshold_reached"]}; exact audit deferred; '
        f'{len(records)} attempts, {final["total_optimizer_iterations"]} total iterations')
    return final
