"""Paired M=128 trajectories; exact occupied-orbital propagation of actual circuits."""
import csv
import time
import numpy as np
from common import ROOT, config, logger, read_json, write_json, file_hash, verify_inputs, INPUTS, DATA, arguments
from xy_core import Engine, segments
from algebraic import single_particle, primitives, apply_primitives


def simulate(output=DATA):
    cfg = config()
    verify_inputs()
    lib_record = read_json(output/'library.json')
    assert lib_record['source_sha256'] == file_hash(ROOT/'algebraic.py')
    library = lib_record['library']
    branches = np.load(INPUTS['branches.npz'])['branches']
    ref = np.load(INPUTS['reference.npz'])
    baseline = np.load(INPUTS['baseline.npz'])['imbalance']
    old_counts = np.genfromtxt(INPUTS['original_resources.csv'], delimiter=',', names=True)
    n, count = cfg['model']['n'], cfg['sampling']['trajectory_m']
    original = {int(k): single_particle(n, b['original_gates']) for k, b in library.items()}
    compiled = {int(k): single_particle(n, b['compiled_gates']) for k, b in library.items()}
    weights = np.r_[np.ones(n//2), -np.ones(n//2)]
    time_grid = ref['time']
    original_values = np.full((len(time_grid), count), -1.)
    compiled_values = original_values.copy()
    rows, audits = [], []
    engine = Engine(n, 'cpu')
    started = time.perf_counter()
    for index in range(1, len(time_grid)):
        for j in range(count):
            occupied = np.eye(n, dtype=complex)[:, :n//2]
            compressed_occupied = occupied.copy()
            original_single = compressed_single = original_cnot = compressed_cnot = 0
            pieces = list(segments(branches[index-1, j, :index]))
            for branch, length in pieces:
                if branch:
                    occupied[branch-1] *= -1
                    compressed_occupied[branch-1] *= -1
                    original_single += 1
                    compressed_single += 1
                else:
                    occupied = original[length] @ occupied
                    compressed_occupied = compiled[length] @ compressed_occupied
                    block = library[str(length)]
                    original_single += block['original_counts']['single_qubit']
                    original_cnot += block['original_counts']['cnot']
                    compressed_single += block['compiled_counts']['single_qubit']
                    compressed_cnot += block['compiled_counts']['cnot']
            a = float(np.dot(weights, 1-2*np.sum(abs(occupied)**2, axis=1))/n)
            b = float(np.dot(weights, 1-2*np.sum(abs(compressed_occupied)**2, axis=1))/n)
            original_values[index, j], compiled_values[index, j] = a, b
            source = old_counts[(index-1)*count+j]
            assert (int(source['time_index']), int(source['trajectory_index'])) == (index, j)
            assert int(source['original_single_qubit']) == original_single
            assert int(source['original_cnot']) == original_cnot
            assert compressed_single <= original_single and compressed_cnot <= original_cnot
            rows.append(dict(time_index=index, time=float(time_grid[index]), trajectory_index=j,
                trajectory_seed=cfg['sampling']['stream_seed'], original_single_qubit=original_single,
                compressed_single_qubit=compressed_single, original_cnot=original_cnot,
                compressed_cnot=compressed_cnot, original_imbalance=a, compressed_imbalance=b))
            # Independent 2^10-state simulation of the exported primitive circuits.
            if index in [1, 25, 50, 75, 100] and j in [0, count-1]:
                state = engine.initial[None, :].copy()
                for branch, length in pieces:
                    if branch:
                        state *= engine.signs[:, branch-1][None, :]
                    else:
                        state = apply_primitives(state, primitives(library[str(length)]['compiled_gates']), n)
                full_value = float(np.sum(abs(state[0])**2*engine.imbalance))
                error = abs(full_value-b)
                assert error < 2e-11
                audits.append(dict(time_index=index, trajectory_index=j, full_state_error=error))
        if index == 1 or index % 10 == 0:
            logger(f'SIMULATE t={time_grid[index]:g}/10: {index*count}/12800 trajectories; '
                   f'max paired difference={np.max(abs(original_values[:index+1]-compiled_values[:index+1])):.3g}; '
                   f'elapsed={time.perf_counter()-started:.1f}s')
    mean_a, mean_b = original_values.mean(1), compiled_values.mean(1)
    baseline_error = float(np.max(abs(mean_a-baseline)))
    paired_error = float(np.max(abs(original_values-compiled_values)))
    assert baseline_error < 2e-11, baseline_error
    assert paired_error < 2e-11, paired_error
    np.savez_compressed(output/'curves.npz', time=time_grid, kraus=ref['kraus'],
        original_imbalance=mean_a, compressed_imbalance=mean_b,
        original_error_to_kraus=ref['kraus']-mean_a, compressed_error_to_kraus=ref['kraus']-mean_b,
        original_per_trajectory=original_values, compressed_per_trajectory=compiled_values)
    with (output/'resources.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    final = [r for r in rows if r['time_index'] == 100]
    final_stats = {key: float(np.mean([r[key] for r in final])) for key in
                   ['original_single_qubit', 'compressed_single_qubit', 'original_cnot', 'compressed_cnot']}
    write_json(output/'summary.json', dict(status='completed', trajectories=len(rows),
        model=cfg['model'], M=count, stream_seed=cfg['sampling']['stream_seed'],
        error_convention='Kraus imbalance minus trajectory-ensemble imbalance; signed, as original figure 3(d)',
        compression_scope='maximal consecutive U0 segments only; every Z jump is kept',
        gate_count_scope='evolution only; 4 single-qubit rotations and 2 CNOTs per XY block; each Z jump adds 1 single-qubit gate',
        simulation='Exact number-conserving occupied-orbital propagation, with independent full-state primitive-circuit audits',
        baseline_curve_max_error=baseline_error, trajectory_compression_max_error=paired_error,
        error_curve_max_difference=float(np.max(abs(mean_a-mean_b))), full_state_audits=audits,
        final_mean_counts=final_stats, elapsed_seconds=time.perf_counter()-started,
        source_sha256={f: file_hash(ROOT/f) for f in ['algebraic.py', '02_simulate.py', 'xy_core.py']},
        library_sha256=file_hash(output/'library.json')))
    logger(f'SIMULATION COMPLETE: baseline agreement={baseline_error:.3g}, compression error={paired_error:.3g}')
    logger(f'Final mean gate counts: {final_stats}')


if __name__ == '__main__':
    simulate(arguments(__doc__).output)
