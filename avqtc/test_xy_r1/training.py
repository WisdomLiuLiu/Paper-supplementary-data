"""Four independently resumable training channels with complete attempt accounting.

NC 14, 3751 (2023), https://doi.org/10.1038/s41467-023-39381-w:
B32 minimizes product-state output infidelity; it does not estimate HS fidelity
by squaring the average of 32 overlaps. Full HS audits sum ALL d overlaps.
"""
import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from xy_core import (ROOT, Engine, analytic_params, channel, digest, file_hash,
                     gate_resources, load_config, mask, product_states, read_json,
                     seed_for, stamp, transfer, write_json)


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message):
        line = f'[{stamp()}] {message}'
        print(line, flush=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')


def proxy_cost(n, k, depth, previous_depth, route, mode, substeps):
    """Hypothetical executable teacher; never label this exact dense U0^k.

    Echo: R, teacher, V-dagger, R-dagger. HST: two registers, Bell prep/unprep.
    Depth is a conservative sequential schedule (6 primitive layers per XY
    parity layer); no transpilation, connectivity or noise corrections.
    """
    v = gate_resources(n, depth)
    target = gate_resources(n, k*substeps if route == 'direct' else substeps)
    if route == 'iterative' and k > 1:
        prev = gate_resources(n, previous_depth)
        for key in ('cnot', 'single_qubit', 'primitive_depth_upper_bound'):
            target[key] += prev[key]
    if mode == 'b32':
        overhead_cnot, overhead_single, overhead_depth, qubits = 0, 4*n, 4, n
    else:
        overhead_cnot, overhead_single, overhead_depth, qubits = 2*n, 2*n, 4, 2*n
    return {'qubits': qubits, 'target_proxy_cnot': target['cnot'],
            'target_proxy_single_qubit': target['single_qubit'],
            'target_proxy_depth_upper_bound': target['primitive_depth_upper_bound'],
            'ansatz_parameters': v['parameters'], 'echo_or_hst_cnot': target['cnot'] + v['cnot'] + overhead_cnot,
            'echo_or_hst_single_qubit_upper_bound': target['single_qubit'] + v['single_qubit'] + overhead_single,
            'echo_or_hst_depth_upper_bound': target['primitive_depth_upper_bound'] + v['primitive_depth_upper_bound'] + overhead_depth}


class Ledger:
    def __init__(self, cost, shots):
        self.cost, self.shots, self.data = cost, shots, {}

    def record(self, phase, rows, gradient=False, mode='b32', target_applications=0, settings_override=None):
        a = self.data.setdefault(phase, {
            'objective_evaluations': 0, 'gradient_evaluations': 0,
            'forward_state_vectors': 0, 'adjoint_state_vectors': 0,
            'target_state_vector_applications': 0,
            'estimated_circuit_settings': 0, 'estimated_shots': 0,
            'estimated_cnot_gate_executions': 0,
            'estimated_single_qubit_gate_executions_upper_bound': 0})
        a['objective_evaluations'] += 1
        a['gradient_evaluations'] += int(gradient)
        a['forward_state_vectors'] += rows
        a['adjoint_state_vectors'] += rows if gradient else 0
        a['target_state_vector_applications'] += target_applications
        settings = rows if mode == 'b32' else 1
        if gradient:
            # Each independent XY angle appears in RX(2 theta), RY(2 theta).
            # Gate-wise parameter shift: 2 evaluations per rotation = 4 per angle.
            # Include one unshifted objective, before reusing any measurement data.
            settings *= 1 + 4*self.cost['ansatz_parameters']
        if settings_override is not None:
            settings = settings_override
        a['estimated_circuit_settings'] += int(settings)
        a['estimated_shots'] += int(settings*self.shots)
        a['estimated_cnot_gate_executions'] += int(settings*self.shots*self.cost['echo_or_hst_cnot'])
        a['estimated_single_qubit_gate_executions_upper_bound'] += int(settings*self.shots*self.cost['echo_or_hst_single_qubit_upper_bound'])


def totals(ledgers):
    out = {}
    for ledger in ledgers:
        for phase, values in ledger.items():
            dest = out.setdefault(phase, {})
            for key, value in values.items():
                dest[key] = dest.get(key, 0) + value
    return out


def optimize_attempt(engine, target, params0, mode, cfg, seed, val_inputs, val_targets,
                     cost, path, identity, log):
    t = cfg['training']
    if path.exists():
        old = read_json(path)
        if old['status'] == 'completed':
            log(f"RESUME {identity}: completed attempt, selection F={old['best_validation_fidelity']:.8f}")
            return old
        # Keep interrupted work in total cost, even though this attempt restarts.
        archive = path.parent / 'interrupted' / (path.stem + '_' + str(time.time_ns()) + '.json')
        write_json(archive, old)
        log(f'RESTART {identity}: incomplete attempt archived; restarting this attempt')
    ledger = Ledger(cost, cfg['resource_estimate']['shots_per_circuit'])
    started = time.perf_counter()
    result = {'status': 'running', 'identity': identity, 'seed': seed,
              'initial_params': params0.tolist(), 'history': [], 'training_batch_seeds': [],
              'optimizer_iterations': 0, 'objective_evaluations': 0,
              'gradient_evaluations': 0, 'selected_iteration': 0,
              'selected_evaluation': 0, 'best_validation_fidelity': -1.0,
              'best_params': params0.tolist(), 'started_at': stamp(),
              'quantum_proxy_circuit': cost, 'actual_quantum_shots': 0,
              'optimizer': 'Adam' if mode == 'b32' else 'L-BFGS-B',
              'learning_rate': t['learning_rate'] if mode == 'b32' else None,
              'learning_rate_note': 'constant Adam base rate' if mode == 'b32' else 'line search; no fixed learning rate',
              'max_iterations': t['iterations']}

    def checkpoint():
        engine.sync()
        result['wall_seconds'] = time.perf_counter() - started
        result['resources_by_phase'] = ledger.data
        write_json(path, result)

    def validate(params, iteration, evaluation):
        if mode == 'b32':
            f = engine.fidelity(params, val_inputs, val_targets)
            ledger.record('selection', len(val_inputs), mode=mode)
        else:
            f = engine.full_metrics(params, [target], t['full_validation_chunk'])[0]['hs_fidelity']
            ledger.record('selection', engine.dim, mode=mode)
        if f > result['best_validation_fidelity']:
            result['best_validation_fidelity'] = f
            result['best_params'] = params.tolist()
            result['selected_iteration'] = iteration
            result['selected_evaluation'] = evaluation
        return f

    log(f'START {identity}, parameters={cost["ansatz_parameters"]}, optimizer={result["optimizer"]}')
    init_f = validate(params0, 0, 0)
    result['initial_validation_fidelity'] = init_f
    result['history'].append({'iteration': 0, 'evaluation': 0, 'validation_fidelity': init_f})
    checkpoint()
    if mode == 'b32':
        params = params0.copy()
        m, v = np.zeros_like(params), np.zeros_like(params)
        for iteration in range(1, t['iterations'] + 1):
            batch_seed = seed_for(seed, 'batch', iteration)
            inputs = engine.xp.asarray(product_states(engine.n, t['batch_size'], batch_seed, t['product_ensemble']))
            targets = inputs @ target.T
            loss, grad = engine.loss_gradient(params, inputs, targets, mode)
            ledger.record('training', len(inputs), gradient=True, target_applications=len(inputs))
            result['objective_evaluations'] += 1
            result['gradient_evaluations'] += 1
            result['training_batch_seeds'].append(batch_seed)
            m = t['adam_beta1']*m + (1-t['adam_beta1'])*grad
            v = t['adam_beta2']*v + (1-t['adam_beta2'])*grad**2
            alpha = t['learning_rate'] * math.sqrt(1-t['adam_beta2']**iteration)/(1-t['adam_beta1']**iteration)
            params -= alpha*m/(np.sqrt(v) + t['adam_epsilon'])
            result['optimizer_iterations'] = iteration
            h = {'iteration': iteration, 'evaluation': iteration, 'loss_before_update': loss,
                 'gradient_l2': float(np.linalg.norm(grad)), 'batch_seed': batch_seed,
                 'learning_rate': t['learning_rate'], 'bias_corrected_step_scale': alpha}
            if iteration == 1 or iteration % t['validation_interval'] == 0 or iteration == t['iterations']:
                h['validation_fidelity'] = validate(params, iteration, iteration)
            result['history'].append(h)
            checkpoint()
            if iteration == 1 or iteration % t['print_every'] == 0 or iteration == t['iterations']:
                log(f'{identity} iter={iteration}/{t["iterations"]} loss={loss:.8g} '
                    f'best_val_F={result["best_validation_fidelity"]:.8f} '
                    f'lr={t["learning_rate"]:g} elapsed={result["wall_seconds"]:.1f}s')
        result['stop_reason'] = 'fixed_iteration_budget; retained best independent selection-set checkpoint'
        result['optimizer_success'] = True
    else:
        active = mask(engine.n, params0.shape[0]-1)
        inputs = engine.xp.eye(engine.dim, dtype=complex)
        target_rows = target.T
        best_loss = float('inf')

        def objective(vector):
            nonlocal best_loss
            params = np.zeros_like(params0)
            params[active] = vector
            loss, grad = engine.loss_gradient(params, inputs, target_rows, mode)
            ledger.record('training', engine.dim, gradient=True, mode=mode)
            result['objective_evaluations'] += 1
            result['gradient_evaluations'] += 1
            ev = result['objective_evaluations']
            # loss=1-sqrt(F+1e-16); no extra forward sweep is needed here.
            f = max(0.0, min(1.0, (1-loss)**2 - 1e-16))
            if loss < best_loss:
                best_loss = loss
                if f > result['best_validation_fidelity']:
                    result['best_validation_fidelity'] = f
                    result['best_params'] = params.tolist()
                    result['selected_iteration'] = result['optimizer_iterations']
                    result['selected_evaluation'] = ev
            result['history'].append({'iteration': result['optimizer_iterations'], 'evaluation': ev,
                                      'loss': loss, 'hs_fidelity': f,
                                      'gradient_l2': float(np.linalg.norm(grad[active]))})
            checkpoint()
            if ev == 1 or ev % t['print_every'] == 0:
                log(f'{identity} evaluation={ev}, accepted_steps={result["optimizer_iterations"]}, '
                    f'loss={loss:.8g} F={f:.8f} elapsed={result["wall_seconds"]:.1f}s')
            return loss, grad[active]

        def callback(_):
            result['optimizer_iterations'] += 1

        opt = minimize(objective, params0[active], method='L-BFGS-B', jac=True, callback=callback,
                       options={'maxiter': t['iterations'], 'ftol': t['lbfgs_ftol'],
                                'gtol': t['lbfgs_gtol'], 'maxls': t['lbfgs_maxls']})
        result['optimizer_iterations'] = int(opt.nit)
        result['optimizer_success'] = bool(opt.success)
        result['stop_reason'] = str(opt.message)
        result['scipy_nfev'], result['scipy_njev'] = int(opt.nfev), int(opt.njev)
    result['status'] = 'completed'
    result['completed_at'] = stamp()
    checkpoint()
    log(f'DONE {identity}: F={result["best_validation_fidelity"]:.8f}, '
        f'iterations={result["optimizer_iterations"]}, evaluations={result["objective_evaluations"]}, '
        f'seconds={result["wall_seconds"]:.1f}')
    return result


def run_channel(cfg, mode, route, output, seeds=None, backend=None):
    t, m = cfg['training'], cfg['model']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = f'{mode}_{route}'
    engine = Engine(m['n'], backend or t['backend'])
    log = Logger(output / 'logs' / f'{name}.log')
    log(f'CHANNEL {name}; n={m["n"]}; backend={engine.backend}; seeds={seeds or t["seeds"]}')
    log('Preparing exact U0 by diagonalizing the XY Hamiltonian (setup is timed separately)')
    setup_start = time.perf_counter()
    u0, _ = engine.u0(m['dt'], m['gamma'])
    engine.sync()
    setup_seconds = time.perf_counter() - setup_start
    log(f'U0 ready in {setup_seconds:.2f}s')
    step, _ = channel(m['n'], m['dt'], m['gamma'])
    for seed in seeds or t['seeds']:
        run_dir = output / 'training' / name / f'seed_{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / 'manifest.json'
        specification = {'model': m, 'training': t, 'resource_estimate': cfg['resource_estimate'],
                         'mode': mode, 'route': route, 'seed': seed,
                         'source_hashes': {p: file_hash(ROOT / p) for p in ['training.py', 'xy_core.py']}}
        fingerprint = digest(specification)
        if manifest_path.exists():
            payload = read_json(manifest_path)
            if payload['fingerprint'] != fingerprint:
                raise ValueError(f'{run_dir}: config/code changed; choose a new --output directory')
            if payload['status'] == 'completed':
                log(f'RESUME seed={seed}: whole library already complete')
                continue
        else:
            payload = {'schema_version': 1, 'status': 'in_progress', 'fingerprint': fingerprint,
                       'specification': specification, 'environment': engine.environment(),
                       'created_at': stamp(), 'library': {}, 'setup_sessions': [],
                       'target_definition': 'exact U0=exp(-i H step); direct U0^k; iterative V_(k-1) @ U0',
                       'nc_reference': 'https://doi.org/10.1038/s41467-023-39381-w',
                       'loss_definition': '1-mean_product_states(abs(overlap)^2)' if mode == 'b32' else '1-abs(Tr(teacher^dagger V))/d',
                       'reported_hs_fidelity': 'abs(Tr(target^dagger V))^2/d^2; all d basis states, never a B32 trace estimate',
                       'selection': 'held-out product mean fidelity' if mode == 'b32' else 'full HS fidelity',
                       'test_policy': 'fresh independent product test set per k; diagnostics only, never depth/parameter selection',
                       'depth_policy': 'scan 1..k+extra_depth; minimum passing selection threshold, otherwise best available',
                       'initialization_policy': 'analytic, depth warm start with zero added angles, scaled previous-k warm start, Gaussian jitter restarts',
                       'resource_notes': {
                           'actual': 'classical exact statevector/matrix simulation with analytic adjoint gradients; zero quantum shots',
                           'estimates': 'hypothetical gate-wise parameter-shift plus echo (B32) or 2n-qubit HST (full); not measured runtime or shots',
                           'target_caveat': 'exact dense U0 targets have no specified hardware synthesis; gate costs use a separate alternating-Trotter proxy, whose fidelity is audited per selected block',
                           'preparation': 'B32 overhead <=4n single gates and depth<=4; conservative also for pauli6',
                           'excluded': 'hardware mapping, error mitigation, readout/reset time, classical-to-quantum latency',
                           'resume': 'complete attempts reused; incomplete attempts archived and counted, then restarted'}}
        payload['setup_sessions'].append({'time': stamp(), 'u0_setup_seconds': setup_seconds,
                                          'shared_across_seeds_in_invocation': True})
        write_json(manifest_path, payload)
        exact = engine.xp.eye(engine.dim, dtype=complex)
        previous_params = previous_v = None
        for k in range(1, t['max_k'] + 1):
            exact = exact @ u0
            if str(k) in payload['library']:
                previous_params = np.asarray(payload['library'][str(k)]['params'])
                previous_v = engine.unitary(previous_params)
                log(f'RESUME seed={seed} block={k}/{t["max_k"]}: selected block complete')
                continue
            block_start = time.perf_counter()
            teacher = exact if route == 'direct' else (u0 if k == 1 else previous_v @ u0)
            val_seed = seed_for(seed, name, 'selection', k)
            test_seed = seed_for(seed, name, 'test', k)
            if mode == 'b32':
                val_in = engine.xp.asarray(product_states(m['n'], t['validation_size'], val_seed, t['product_ensemble']))
                val_out = val_in @ teacher.T
            else:
                val_in = val_out = None
            best = None
            prev_depth_params = None
            candidates = []
            threshold = t[f'{route}_target_fidelity']
            previous_depth = None if previous_params is None else previous_params.shape[0]-1
            for depth in range(1, k + t[f'{route}_extra_depth'] + 1):
                base = analytic_params(m['n'], k, depth, step)
                initial = [('analytic', base)]
                depth_warm = transfer(prev_depth_params, depth)
                k_warm = transfer(previous_params, depth, k/(k-1) if k > 1 else 1)
                if depth_warm is not None:
                    initial.append(('depth_warm', depth_warm))
                if k_warm is not None:
                    initial.append(('k_warm', k_warm))
                restart_base = depth_warm if depth_warm is not None else (k_warm if k_warm is not None else base)
                rng = np.random.default_rng(seed_for(seed, name, k, depth, 'jitter'))
                for r in range(t['random_restarts']):
                    initial.append((f'jitter_{r}', restart_base + rng.normal(0, t['jitter_scale'], base.shape)*mask(m['n'], depth)))
                depth_best = None
                for init_name, params0 in initial:
                    attempt_name = f'k{k:02d}_d{depth:02d}_{init_name}'
                    path = run_dir / 'attempts' / (attempt_name + '.json')
                    cost = proxy_cost(m['n'], k, depth, previous_depth, route, mode, cfg['resource_estimate']['trotter_substeps'])
                    a = optimize_attempt(engine, teacher, params0, mode, cfg,
                                         seed_for(seed, name, attempt_name), val_in, val_out, cost, path,
                                         f'{name} seed={seed} s={k}/{t["max_k"]} d={depth} {init_name}', log)
                    item = {'fidelity': a['best_validation_fidelity'], 'params': a['best_params'],
                            'depth': depth, 'attempt': str(path.relative_to(run_dir))}
                    if depth_best is None or item['fidelity'] > depth_best['fidelity']:
                        depth_best = item
                candidates.append({key: value for key, value in depth_best.items() if key != 'params'})
                prev_depth_params = np.asarray(depth_best['params'])
                if best is None or depth_best['fidelity'] > best['fidelity']:
                    best = depth_best
                if depth_best['fidelity'] >= threshold:
                    best = depth_best
                    break
            params = np.asarray(best['params'])
            log(f'AUDIT {name} seed={seed} s={k}: full HS, independent product tests and target proxy')
            audit_start = time.perf_counter()
            metrics = engine.full_metrics(params, [teacher, exact], t['full_validation_chunk'])
            test_in = engine.xp.asarray(product_states(m['n'], t['test_size'], test_seed, t['product_ensemble']))
            test_teacher = engine.fidelity(params, test_in, test_in @ teacher.T)
            test_exact = engine.fidelity(params, test_in, test_in @ exact.T)
            r = cfg['resource_estimate']['trotter_substeps']
            proxy = engine.proxy_unitary(k if route == 'direct' else 1, step, r)
            if route == 'iterative' and k > 1:
                proxy = previous_v @ proxy
            proxy_f = float((engine.xp.abs(engine.xp.vdot(teacher, proxy))/engine.dim)**2)
            engine.sync()
            block = {'k': k, 'params': params.tolist(), 'pqc_steps': best['depth'],
                     'pqc_layers': best['depth']+1, 'selected_attempt': best['attempt'],
                     'selection_fidelity': best['fidelity'], 'target_fidelity': threshold,
                     'threshold_reached': bool(best['fidelity'] >= threshold),
                     'selection_reason': 'minimum_depth_reaching_target' if best['fidelity'] >= threshold else 'best_available_below_target',
                     'depth_candidates': candidates, 'validation_seed': val_seed, 'test_seed': test_seed,
                     'teacher_validation_hs_fidelity': metrics[0]['hs_fidelity'],
                     'exact_power_validation_hs_fidelity': metrics[1]['hs_fidelity'],
                     'teacher_computational_basis_fidelity': metrics[0]['computational_basis_fidelity'],
                     'exact_computational_basis_fidelity': metrics[1]['computational_basis_fidelity'],
                     'teacher_test_product_fidelity': test_teacher,
                     'exact_test_product_fidelity': test_exact,
                     'target_proxy_hs_fidelity_to_actual_teacher': proxy_f,
                     'gate_resources': gate_resources(m['n'], best['depth']),
                     'audit_wall_seconds': time.perf_counter()-audit_start,
                     'block_wall_seconds_this_session': time.perf_counter()-block_start,
                     'audit_actual_counts': {'ansatz_forward_state_vectors': engine.dim + 2*t['test_size'],
                                             'target_state_vector_applications': 2*t['test_size'],
                                             'selection_target_preparation_vectors': t['validation_size'] if mode == 'b32' else 0,
                                             'full_trace_comparisons': 2, 'proxy_full_trace_comparisons': 1,
                                             'actual_quantum_shots': 0}}
            # Held-out diagnostics estimates are separated from training/selection.
            audit_ledgers = []
            for target_route in [route, 'direct']:
                prod_cost = proxy_cost(m['n'], k, best['depth'], previous_depth, target_route, 'b32', r)
                aud = Ledger(prod_cost, cfg['resource_estimate']['shots_per_circuit'])
                aud.record('held_out_product_test', t['test_size'], mode='b32', target_applications=t['test_size'])
                hs = Ledger(proxy_cost(m['n'], k, best['depth'], previous_depth, target_route, 'full', r), cfg['resource_estimate']['shots_per_circuit'])
                hs.record('full_hs_audit', engine.dim, mode='full')
                audit_ledgers.extend([aud.data, hs.data])
            block['diagnostic_resource_estimates'] = totals(audit_ledgers)
            write_json(run_dir / f'block_{k:02d}.json', block)
            np.savetxt(run_dir / f'k{k:02d}_params.txt', params)
            payload['library'][str(k)] = block
            payload['completed_k'] = k
            write_json(manifest_path, payload)
            previous_params, previous_v = params, engine.unitary(params)
            log(f'SELECTED {name} seed={seed} s={k}/{t["max_k"]} depth={best["depth"]} '
                f'teacher_HS={metrics[0]["hs_fidelity"]:.8f} exact_HS={metrics[1]["hs_fidelity"]:.8f} '
                f'product_test={test_exact:.8f} passed={block["threshold_reached"]}')
        attempts = [read_json(p) for p in (run_dir / 'attempts').rglob('*.json')]
        payload['training_and_selection_resources'] = totals(a.get('resources_by_phase', {}) for a in attempts)
        payload['diagnostic_resource_estimates'] = totals(b['diagnostic_resource_estimates'] for b in payload['library'].values())
        payload['actual_optimizer_iterations_all_attempts'] = sum(a['optimizer_iterations'] for a in attempts)
        payload['actual_objective_evaluations_all_attempts'] = sum(a['objective_evaluations'] for a in attempts)
        payload['attempt_wall_seconds_including_interrupted'] = sum(a['wall_seconds'] for a in attempts)
        payload['attempt_count_including_interrupted'] = len(attempts)
        payload['status'], payload['completed_at'] = 'completed', stamp()
        write_json(manifest_path, payload)
        log(f'LIBRARY COMPLETE {name} seed={seed}; attempts={len(attempts)}; '
            f'optimizer_steps={payload["actual_optimizer_iterations_all_attempts"]}; '
            f'evaluations={payload["actual_objective_evaluations_all_attempts"]}')


def main(mode, route):
    p = argparse.ArgumentParser(description=f'Train {mode}/{route}; resumes completed attempts automatically.')
    p.add_argument('--config', type=Path, default=ROOT / 'config.json')
    p.add_argument('--output', type=Path, default=ROOT / 'data')
    p.add_argument('--seeds', type=int, nargs='+', help='Subset of the configured independent training seeds')
    p.add_argument('--backend', choices=['auto', 'cpu', 'gpu'])
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.seeds and not set(args.seeds).issubset(cfg['training']['seeds']):
        p.error('--seeds must be a subset of training.seeds in config')
    run_channel(cfg, mode, route, args.output, args.seeds, args.backend)
