"""Explicit RX/RY/RZ/CX decomposition, export and independent primitive simulator.

All angles are radians. RX(a)=exp(-i*a*X/2), likewise RY/RZ. Execution order
is the order of entries. Export contains evolution only, no input/readout gates.
"""
import math
from pathlib import Path

import numpy as np

from checkpoint_io import replace_with_retry


def logical_gates(params, n):
    """Merge consecutive commuting Z/ZZ gates; keep all angles, including zero."""
    from xxz_core import operations
    diagonal = {}

    def flush_diagonal():
        # All terms commute. Schedule local Z, then even ZZ, then odd ZZ,
        # so the exported list realizes the claimed parallel-depth upper bound.
        order = sorted(diagonal, key=lambda key: (0 if key[0] == 'z' else 1,
                                                  key[1] % 2 if key[0] == 'zz' else 0, key[1]))
        for gate_kind, gate_site in order:
            yield {'kind': gate_kind, 'site': gate_site, **diagonal[gate_kind, gate_site]}
    for layer, column, kind, site, scale in operations(n, len(params)):
        angle = float(scale*params[layer, column])
        term = {'slice': layer, 'column': column, 'coefficient': scale}
        if kind in ('z', 'zz'):
            entry = diagonal.setdefault((kind, site), {'angle': 0.0, 'parameter_terms': []})
            entry['angle'] += angle
            entry['parameter_terms'].append(term)
        else:
            yield from flush_diagonal()
            diagonal = {}
            yield {'kind': kind, 'site': site, 'angle': angle, 'parameter_terms': [term]}
    yield from flush_diagonal()


def primitive_for_gate(kind, site, theta):
    """Return a matrix-exact decomposition in chronological execution order."""
    def rotation(name, q, angle):
        return {'gate': name, 'qubits': [q], 'angle': float(angle)}

    def cx(a, b):
        return {'gate': 'cx', 'qubits': [a, b]}

    if kind == 'z':
        return [rotation('rz', site, 2*theta)]
    if kind == 'zz':
        return [cx(site, site+1), rotation('rz', site+1, 2*theta), cx(site, site+1)]
    if kind == 'xy':
        # exp[-i theta(XX+YY)], with the first qubit as CX control.
        return [rotation('rx', site, math.pi/2), cx(site, site+1),
                rotation('rx', site, 2*theta), rotation('ry', site+1, 2*theta),
                cx(site, site+1), rotation('rx', site, -math.pi/2)]
    raise ValueError(f'Unknown logical gate: {kind}')


def decompose(params, n):
    out = []
    for index, logical in enumerate(logical_gates(params, n)):
        for primitive in primitive_for_gate(logical['kind'], logical['site'], logical['angle']):
            out.append({**primitive, 'logical_gate_index': index})
    return out


def primitive_counts(gates, n):
    ready = [0]*n
    counts = {name: 0 for name in ('rx', 'ry', 'rz', 'cx')}
    for gate in gates:
        counts[gate['gate']] += 1
        end = 1+max(ready[q] for q in gate['qubits'])
        for q in gate['qubits']:
            ready[q] = end
    return {**counts, 'cnot': counts['cx'],
            'single_qubit': counts['rx']+counts['ry']+counts['rz'],
            'scheduled_primitive_depth': max(ready, default=0)}


def export_circuit(prefix, params, n, metadata=None):
    from xxz_core import gate_resources, write_json
    prefix = Path(prefix)
    gates = decompose(params, n)
    counts = primitive_counts(gates, n)
    expected = gate_resources(n, len(params))
    for key in ['cnot', 'single_qubit']:
        if counts[key] != expected[key]:
            raise AssertionError(f'Primitive export resource mismatch: {key}')
    if counts['scheduled_primitive_depth'] > expected['primitive_depth_upper_bound']:
        raise AssertionError('Declared depth upper bound violated by exported circuit')
    write_json(prefix.with_suffix('.json'), {
        'n': n, 'angle_units': 'radians', 'rotation_convention': 'R_P(a)=exp(-i*a*P/2)',
        'qubit_convention': 'qubit 0 is leftmost in mathematical basis; OpenQASM uses q[0]',
        'statevector_note': 'Qiskit display order reverses q[0]; account for endian order in statevector comparisons',
        'execution_order': 'first entry acts first', 'metadata': metadata or {},
        'logical_gates_after_diagonal_merge': list(logical_gates(params, n)),
        'primitive_gates': gates, 'primitive_counts': counts})
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n}];',
             '// Evolution only: no initial preparation or measurement.']
    for gate in gates:
        if gate['gate'] == 'cx':
            a, b = gate['qubits']
            lines.append(f'cx q[{a}],q[{b}];')
        else:
            lines.append(f'{gate["gate"]}({gate["angle"]:.17g}) q[{gate["qubits"][0]}];')
    path = prefix.with_suffix('.qasm')
    temporary = path.with_suffix('.qasm.tmp')
    temporary.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    replace_with_retry(temporary, path)
    return counts


def apply_primitives(inputs, gates, n):
    """CPU validation path that does not call the logical XXZ gate kernels."""
    states = np.array(inputs, dtype=complex, copy=True)
    ids = np.arange(1 << n)
    for gate in gates:
        if gate['gate'] == 'cx':
            control, target = gate['qubits']
            a, b = 1 << (n-1-control), 1 << (n-1-target)
            permutation = ids ^ np.where(ids & a, b, 0)
            states = states[:, permutation]
            continue
        q = gate['qubits'][0]
        bit = 1 << (n-1-q)
        left = ids[(ids & bit) == 0]
        right = left ^ bit
        angle = gate['angle']
        c, s = np.cos(angle/2), np.sin(angle/2)
        if gate['gate'] == 'rx':
            matrix = np.array([[c, -1j*s], [-1j*s, c]])
        elif gate['gate'] == 'ry':
            matrix = np.array([[c, -s], [s, c]])
        elif gate['gate'] == 'rz':
            matrix = np.diag([np.exp(-1j*angle/2), np.exp(1j*angle/2)])
        else:
            raise ValueError(gate['gate'])
        a, b = states[:, left].copy(), states[:, right].copy()
        states[:, left] = matrix[0, 0]*a + matrix[0, 1]*b
        states[:, right] = matrix[1, 0]*a + matrix[1, 1]*b
    return states
