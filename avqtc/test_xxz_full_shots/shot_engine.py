"""Finite-shot HST and occurrence-wise Pauli parameter-shift simulation.

HST p(0...0) = |Tr(T^dagger V)/2^n|^2. Average amplitudes BEFORE squaring.
Only binomial counts reach the finite-shot optimizer. Prefix/suffix contraction
accelerates computation of Born probabilities, not the hardware circuit count.
Reference: https://doi.org/10.1103/PhysRevA.99.032331
"""
import numpy as np
from xxz_core import operations, gate_resources


def measure(probabilities, shots, rng):
    p = np.clip(np.asarray(probabilities, dtype=float), 0, 1)
    if shots == 0:  # explicitly labelled infinite-shot control
        return p.copy()
    return rng.binomial(shots, p) / shots


def pauli(engine, states, kind, site):
    if kind in ('z', 'zz'):
        signs = engine.z_signs[:, site] if kind == 'z' else engine.zz_signs[:, site]
        return states * signs[None, :]
    ids = engine.xp.arange(engine.dim)
    permutation = ids ^ ((1 << (engine.n-1-site)) | (1 << (engine.n-2-site)))
    out = states[:, permutation]
    if kind == 'yy':
        out = out * (-engine.zz_signs[:, site])[None, :]
    elif kind != 'xx':
        raise ValueError(kind)
    return out


def hst_probability(engine, params, target):
    states = engine.pqc(engine.xp.eye(engine.dim, dtype=complex), params)
    amplitude = engine.xp.sum(engine.xp.conj(target.T)*states)/engine.dim
    return float(np.clip(abs(complex(amplitude.item()))**2, 0, 1))


def shift_probabilities(engine, params, target):
    """Return forward p, plus/minus p and chain-rule parameter occurrence map.

For exp(-i theta P), dp/dtheta=p(theta+pi/4)-p(theta-pi/4).
XY has two commuting generators XX and YY: shift each separately. Shared
half-angle parameters are differentiated occurrence by occurrence and summed.
An XY occurrence with independently shifted XX/YY angles still decomposes to
2 CX + 4 single-qubit rotations, so shifted circuits have the same gate cost.
"""
    xp = engine.xp
    states = xp.eye(engine.dim, dtype=complex)
    tape = []
    for layer, column, kind, site, scale in operations(engine.n, len(params)):
        theta = scale*params[layer, column]
        states = engine.apply_gate(states.copy(), kind, site, theta)
        tape.append((layer, column, kind, site, scale, theta, states))
    suffix_target = target.T.copy()
    amplitude = xp.sum(xp.conj(suffix_target)*states)/engine.dim
    forward = float(abs(complex(amplitude.item()))**2)
    values, mapping = [], []
    for layer, column, kind, site, scale, theta, after in reversed(tape):
        a = xp.sum(xp.conj(suffix_target)*after)/engine.dim
        for generator in (('xx', 'yy') if kind == 'xy' else (kind,)):
            b = xp.sum(xp.conj(suffix_target)*pauli(engine, after, generator, site))/engine.dim
            values.extend([xp.abs((a-1j*b)/np.sqrt(2))**2,
                           xp.abs((a+1j*b)/np.sqrt(2))**2])
            mapping.append((layer, column, scale))
        suffix_target = engine.apply_gate(suffix_target, kind, site, -theta)
    probabilities = engine.cpu(xp.stack(values))
    return float(np.clip(forward, 0, 1)), np.clip(probabilities, 0, 1), mapping


def loss_gradient(engine, params, target, shots, rng, include_forward=True):
    forward, shifted, mapping = shift_probabilities(engine, params, target)
    measured = measure(np.r_[forward, shifted] if include_forward else shifted, shots, rng)
    offset = int(include_forward)
    grad = np.zeros_like(params)
    for i, (layer, column, scale) in enumerate(mapping):
        grad[layer, column] -= scale*(measured[offset+2*i]-measured[offset+2*i+1])
    return (1-float(measured[0]) if include_forward else None), grad, len(measured)


def hst_cost(n, depth, k, substeps):
    v, t = gate_resources(n, depth), gate_resources(n, k*substeps)
    return {
        'qubits': 2*n,
        'single_qubit_gates_per_setting': v['single_qubit']+t['single_qubit']+2*n,
        'two_qubit_gates_per_setting': v['cnot']+t['cnot']+2*n,
        'depth_upper_bound': v['primitive_depth_upper_bound']+t['primitive_depth_upper_bound']+4,
        'gradient_settings_per_iteration': v['parameter_shift_gradient_settings'],
        'forward_settings_per_iteration': 0,
        'objective_settings_per_update_in_selection_phase': 1,
        'teacher': 'symmetric XXZ Trotter proxy; NOT an exact synthesis of dense U0^s',
        'teacher_slices': k*substeps,
    }


def record(ledger, phase, settings, shots, cost):
    row = ledger.setdefault(phase, dict(circuit_settings=0, simulated_measurement_shots=0,
                                       born_probability_evaluations=0, circuit_evolution_executions=0,
                                       S_per_one_shot=0, D_per_one_shot=0,
                                       single_gate_executions=0, two_gate_executions=0))
    row['circuit_settings'] += int(settings)
    row['born_probability_evaluations'] += int(settings)
    row['circuit_evolution_executions'] += int(settings*shots)
    row['simulated_measurement_shots'] += int(settings*shots)
    for letter, field, executions in [('S', 'single_qubit_gates_per_setting', 'single_gate_executions'),
                                      ('D', 'two_qubit_gates_per_setting', 'two_gate_executions')]:
        gates = int(settings*cost[field])
        row[letter+'_per_one_shot'] += gates
        row[executions] += gates*shots


def wilson(successes, trials, z=1.959963984540054):
    """95% binomial interval for a SINGLE fixed HST circuit, not selection data."""
    if trials == 0:
        return None
    p = successes/trials
    center = (p+z*z/(2*trials))/(1+z*z/trials)
    half = z*np.sqrt(p*(1-p)/trials+z*z/(4*trials**2))/(1+z*z/trials)
    return [float(max(0, center-half)), float(min(1, center+half))]
