"""Full-space direct training with the original XXZ optimization/search policy."""
import time
import numpy as np
from scipy.optimize import minimize
from xxz_core import (ROOT, analytic_params, channel, digest, disorder_fields,
                      file_hash, gate_resources, read_json, seed_for,
                      stamp, transfer, write_json)
from sector_engine import SectorEngine, memory_plan


def specification(cfg, n):
    model = dict(cfg['model'], n=n)
    return dict(model=model, fields=disorder_fields(model).tolist(), circuit=cfg['circuit'],
                training=cfg['training'], channel='full_direct', seed=202506,
                implementation='all_excitation_sectors_inverse_adjoint',
                source_sha256={f: file_hash(ROOT/f) for f in
                               ['sector_training.py', 'sector_engine.py', 'xxz_core.py', 'checkpoint_io.py']},
                loss='1-sqrt(abs(Tr(U0^s dagger V)/2^n)^2+1e-16)',
                hsf='abs(sum of complex sector traces / 2^n)^2',
                target_step=channel(n, model['dt'], model['gamma'])[0])


def logger(output):
    output.mkdir(parents=True, exist_ok=True)
    def log(message):
        line = f'[{stamp()}] {message}'
        print(line, flush=True)
        with (output/'progress.log').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    return log


def prepare(cfg, sizes, output):
    output.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        spec = specification(cfg, n)
        folder = output/f'n{n:02d}'
        path = folder/'specification.json'
        record = dict(specification=spec, fingerprint=digest(spec))
        if path.exists() and read_json(path) != record:
            raise ValueError(f'Configuration/code changed for {folder}; use a new output directory')
        write_json(path, record)
    write_json(output/'memory_plan.json', [memory_plan(n, cfg['training']['max_k']) for n in sizes])


def optimize(engine, params0, cfg, path, fingerprint, identity, log):
    t = cfg['training']
    if path.exists():
        old = read_json(path)
        if old['fingerprint'] != fingerprint:
            raise ValueError(f'Attempt fingerprint mismatch: {path}')
        if old['status'] == 'completed':
            log(f'RESUME {identity}: completed, F={old["best_validation_fidelity"]:.9f}')
            return old
        write_json(path.parent/'interrupted'/f'{path.stem}_{time.time_ns()}.json', old)
    started = time.perf_counter()
    result = dict(status='running', fingerprint=fingerprint, identity=identity,
                  optimizer='L-BFGS-B', learning_rate=None, step_size='adaptive line search',
                  max_iterations=t['iterations'], ftol=t['lbfgs_ftol'], gtol=t['lbfgs_gtol'], maxls=t['lbfgs_maxls'],
                  initial_params=params0.tolist(), best_params=params0.tolist(),
                  best_validation_fidelity=-1., optimizer_iterations=0, objective_evaluations=0,
                  gradient_evaluations=0, selected_iteration=0, selected_evaluation=0, history=[], actual_quantum_shots=0)
    def checkpoint():
        result['wall_seconds'] = time.perf_counter()-started
        write_json(path, result)
    log(f'START {identity}: {params0.size} parameters, at most {t["iterations"]} optimizer iterations')
    _, _, initial = engine.evaluate(params0, gradient=False)
    result['best_validation_fidelity'] = initial
    result['history'].append(dict(iteration=0, evaluation=0, validation_fidelity=initial))
    checkpoint()
    best_loss = float('inf')
    def objective(vector):
        nonlocal best_loss
        params = vector.reshape(params0.shape)
        loss, grad, fidelity = engine.evaluate(params)
        result['objective_evaluations'] += 1
        result['gradient_evaluations'] += 1
        ev = result['objective_evaluations']
        if loss < best_loss:
            best_loss = loss
            if fidelity > result['best_validation_fidelity']:
                result.update(best_validation_fidelity=fidelity, best_params=params.tolist(),
                              selected_iteration=result['optimizer_iterations'], selected_evaluation=ev)
        result['history'].append(dict(iteration=result['optimizer_iterations'], evaluation=ev,
                                     loss=loss, hs_fidelity=fidelity, gradient_l2=float(np.linalg.norm(grad))))
        checkpoint()
        if ev == 1 or ev % t['print_every'] == 0:
            log(f'{identity}: evaluations={ev}, accepted_iterations={result["optimizer_iterations"]}/{t["iterations"]}, '
                f'F={fidelity:.9f}, best_F={result["best_validation_fidelity"]:.9f}, elapsed={result["wall_seconds"]:.1f}s')
        return loss, grad.ravel()
    def callback(_):
        result['optimizer_iterations'] += 1
    opt = minimize(objective, params0.ravel(), method='L-BFGS-B', jac=True, callback=callback,
                   options=dict(maxiter=t['iterations'], ftol=t['lbfgs_ftol'], gtol=t['lbfgs_gtol'], maxls=t['lbfgs_maxls']))
    result.update(status='completed', optimizer_iterations=int(opt.nit), optimizer_success=bool(opt.success),
                  stop_reason=str(opt.message), scipy_nfev=int(opt.nfev), scipy_njev=int(opt.njev), completed_at=stamp())
    checkpoint()
    log(f'DONE {identity}: F={result["best_validation_fidelity"]:.9f}, iterations={opt.nit}, evaluations={opt.nfev}')
    return result


def train(cfg, sizes, output, backend='gpu'):
    prepare(cfg, sizes, output)
    log = logger(output)
    t, seed = cfg['training'], 202506
    for n in sizes:
        folder = output/f'n{n:02d}'
        path = folder/'manifest.json'
        spec = specification(cfg, n)
        fingerprint = digest(spec)
        if path.exists():
            payload = read_json(path)
            if payload['fingerprint'] != fingerprint:
                raise ValueError(f'Library fingerprint mismatch: {path}')
            if payload['status'] == 'completed':
                log(f'RESUME n={n}: all blocks complete')
                continue
        else:
            payload = dict(status='running', specification=spec, fingerprint=fingerprint, library={}, created_at=stamp())
            write_json(path, payload)
        model = spec['model']
        log(f'SIZE START n={n}, s=1..{t["max_k"]}, seed={seed}, fields={spec["fields"]}')
        started = time.perf_counter()
        engine = SectorEngine(model, cfg['circuit'], backend or t['backend'], log)
        payload['environment'] = engine.environment
        payload['implementation'] = 'all excitation sectors; complex128; inverse adjoint; exact full-space HSF'
        payload['effective_backend'] = engine.backend
        payload['setup_seconds'] = time.perf_counter()-started
        write_json(path, payload)
        previous = None
        step, _ = channel(n, model['dt'], model['gamma'])
        for k in range(1, t['max_k']+1):
            if str(k) in payload['library']:
                previous = np.asarray(payload['library'][str(k)]['params'])
                continue
            engine.set_target(k)
            best = None
            depth_previous = None
            candidates = []
            for depth in range(1, k+t['direct_extra_depth']+1):
                base = analytic_params(n, k, depth, step, model)
                initial = [('analytic', base)]
                dw = transfer(depth_previous, depth)
                kw = transfer(previous, depth, k/(k-1) if k > 1 else 1)
                if dw is not None:
                    initial.append(('depth_warm', dw))
                if kw is not None:
                    initial.append(('k_warm', kw))
                restart_base = dw if dw is not None else (kw if kw is not None else base)
                rng = np.random.default_rng(seed_for(seed, 'full_direct', k, depth, 'jitter'))
                for r in range(t['random_restarts']):
                    initial.append((f'jitter_{r}', restart_base+rng.normal(0, t['jitter_scale'], base.shape)))
                depth_best = None
                for init_name, params in initial:
                    name = f'k{k:02d}_d{depth:02d}_{init_name}'
                    attempt = optimize(engine, params, cfg, folder/'attempts'/f'{name}.json', fingerprint,
                                       f'n={n} s={k}/{t["max_k"]} depth={depth} {init_name}', log)
                    item = dict(fidelity=attempt['best_validation_fidelity'], params=attempt['best_params'],
                                depth=depth, attempt=f'attempts/{name}.json')
                    if depth_best is None or item['fidelity'] > depth_best['fidelity']:
                        depth_best = item
                candidates.append({key: value for key, value in depth_best.items() if key != 'params'})
                depth_previous = np.asarray(depth_best['params'])
                if best is None or depth_best['fidelity'] > best['fidelity']:
                    best = depth_best
                if depth_best['fidelity'] >= t['direct_target_fidelity']:
                    best = depth_best
                    break
            previous = np.asarray(best['params'])
            _, _, audited = engine.evaluate(previous, gradient=False)
            if abs(audited-best['fidelity']) > 1e-10:
                raise AssertionError('Stored best fidelity and independent final evaluation disagree')
            payload['library'][str(k)] = dict(k=k, params=best['params'], pqc_steps=best['depth'],
                selected_attempt=best['attempt'], selection_fidelity=best['fidelity'],
                exact_power_validation_hs_fidelity=audited, target_fidelity=t['direct_target_fidelity'],
                threshold_reached=bool(best['fidelity'] >= t['direct_target_fidelity']),
                depth_candidates=candidates, gate_resources=gate_resources(n, best['depth']))
            write_json(path, payload)
            log(f'BLOCK COMPLETE n={n} s={k}: selected_depth={best["depth"]}, HSF={audited:.9f}, '
                f'threshold_reached={payload["library"][str(k)]["threshold_reached"]}')
        records = [read_json(p) for p in (folder/'attempts').rglob('*.json')]
        payload.update(status='completed', completed_at=stamp(),
                       actual_optimizer_iterations_all_attempts=sum(a['optimizer_iterations'] for a in records),
                       actual_objective_evaluations_all_attempts=sum(a['objective_evaluations'] for a in records))
        write_json(path, payload)
        log(f'SIZE COMPLETE n={n}')
        xp = engine.xp
        del engine
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
