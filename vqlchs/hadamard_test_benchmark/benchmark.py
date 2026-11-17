"""Reproducible HN Hadamard-test gate-count benchmark (no training).

Requires numpy, scipy, matplotlib and qiskit==2.2.3.
Counts ONE real-overlap circuit, not a complete LCHS loss evaluation.
"""
import os
os.environ.setdefault('QISKIT_PARALLEL', 'FALSE')
import sys
from pathlib import Path
import json
import csv
from collections import Counter
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.circuit.library import RXGate, RZGate, HGate, CXGate, CZGate, SdgGate, SGate
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap

OUT = Path(__file__).resolve().parent
LAYERS, DT, G, INTERACTION, K = 5, 0.2, 1.0, 1.0, 1.0
SEED, OPT = 20261117, 1
SIZES = (2, 4, 6, 8, 10)


def hn_terms(n):
    """H0-kV, omitting original real identity as in lchs_g1u1.py.

    V shift -n*sinh(g)*I is handled as an ancilla phase, never discarded.
    Deterministic product ordering: XX, YY, XY, YX, ZZ for each bond;
    then combined on-site Z terms. One first-order Trotter step.
    """
    terms = []
    for j in range(n-1):
        for axes, coeff in [('XX', np.cosh(G)/2), ('YY', np.cosh(G)/2),
                            ('XY', K*np.sinh(G)/2), ('YX', -K*np.sinh(G)/2),
                            ('ZZ', INTERACTION/4)]:
            terms.append((tuple(zip((j, j+1), axes)), float(coeff)))
    for j in range(n):
        degree = 1 if j in (0, n-1) else 2
        terms.append((((j, 'Z'),), -degree*INTERACTION/4))
    return terms, K*n*np.sinh(G)


def pqc_ops(n, parameters):
    ops, index = [], 0
    for _ in range(LAYERS):
        for j in range(n):
            for name in ('rx', 'rz', 'rx'):
                ops.append((name, (j,), float(parameters[index])))
                index += 1
        for start in (0, 1):
            for j in range(start, n-1, 2):
                ops.append(('cz', (j, j+1), None))
    return ops


def pauli_gadget(pauli, angle):
    before = []
    for j, axis in pauli:
        if axis == 'X':
            before.append(('h', (j,), None))
        elif axis == 'Y':
            before.extend([('sdg', (j,), None), ('h', (j,), None)])
    qubits = [j for j, _ in pauli]
    before.extend(('cx', (a, b), None) for a, b in zip(qubits[:-1], qubits[1:]))
    inverse = {'sdg': 's', 'h': 'h', 'cx': 'cx'}
    after = [(inverse[name], qs, value) for name, qs, value in reversed(before)]
    return before, ('rz', (qubits[-1],), angle), after


def gate(op):
    name, _, value = op
    if name == 'rx': return RXGate(value)
    if name == 'rz': return RZGate(value)
    return {'h': HGate, 'cx': CXGate, 'cz': CZGate,
            'sdg': SdgGate, 's': SGate}[name]()


def append_op(qc, op, controlled=False, offset=1):
    qs = [q + offset for q in op[1]]
    qc.append(gate(op).control(1) if controlled else gate(op), [0]+qs if controlled else qs)


def product_terms(terms, order=1, repetitions=1):
    if order == 1:
        step = list(terms)
    elif order == 2:
        halves = [(pauli, coefficient/2) for pauli, coefficient in terms[:-1]]
        step = halves + [terms[-1]] + list(reversed(halves))
    else:
        raise ValueError('Only first and second order are supported')
    return [(pauli, coefficient/repetitions) for pauli, coefficient in step]*repetitions


def circuits(n, order=1, repetitions=1, save_parameters=True):
    rng = np.random.default_rng(SEED+n)
    a, b = rng.uniform(-np.pi, np.pi, (2, 3*n*LAYERS))
    if save_parameters:
        np.savez(OUT/f'parameters_n{n}.npz', theta=a, theta_m=b)
    opa, opb = pqc_ops(n, a), pqc_ops(n, b)
    terms, shift = hn_terms(n)
    result = {}
    for variant in ('original', 'pqc_only', 'simplified_exact', 'paper_reverse'):
        qc = QuantumCircuit(n+1, name=f'{variant}_n{n}')
        qc.h(0)
        if variant == 'original':
            qc.x(0)
            for op in opa: append_op(qc, op, True)
            qc.x(0)
            for op in opb: append_op(qc, op, True)
        else:
            for left, right in zip(opa, opb):
                append_op(qc, left)
                if left[2] is not None:
                    append_op(qc, (left[0], left[1], right[2]-left[2]), True)
        for pauli, coeff in product_terms(terms, order, repetitions):
            pre, rotation, post = pauli_gadget(pauli, 2*DT*coeff)
            if variant in ('original', 'pqc_only'):
                for op in pre + [rotation] + post: append_op(qc, op, True)
            else:
                # Exact controlled-Pauli construction cancels control on
                # fixed conjugating gates. The manuscript's dt/2dt pairing
                # additionally inserts the unconditional rotation.
                for op in pre: append_op(qc, op)
                if variant == 'paper_reverse': append_op(qc, rotation)
                append_op(qc, rotation, True)
                for op in post: append_op(qc, op)
        qc.p(-DT*shift, 0)
        qc.h(0)
        result[variant] = qc
    # Simple product state for counts; no occupation preparation included.
    return result, (opa, opb, terms, shift)


def counts(qc):
    arity = Counter(len(ins.qubits) for ins in qc.data)
    return {'single': arity[1], 'double': arity[2], 'triple': arity[3],
            'depth': qc.depth(), 'ops': dict(qc.count_ops())}


def ancilla_x(qc, system_index, imaginary=False):
    # Final H already maps X measurement to Z.
    if imaginary:
        qc = qc.copy()
        qc.data.pop()  # replace final H by Sdg,H to measure Y
        qc.sdg(0)
        qc.h(0)
    sv = Statevector.from_int(2*system_index, 2**qc.num_qubits).evolve(qc)
    prob = np.abs(sv.data)**2
    return float(prob[::2].sum()-prob[1::2].sum())


def fermion_validation(n, terms, shift):
    # Independent occupation-basis hopping matrix verifies signs/JW terms.
    dim = 2**n
    h = np.zeros((dim, dim), complex)
    for x in range(dim):
        for j in range(n-1):
            left, right = (x >> j)&1, (x >> (j+1))&1
            h[x, x] += INTERACTION*left*right
            if left != right:
                y = x ^ (1 << j) ^ (1 << (j+1))
                h[y, x] += np.exp(G) if left else np.exp(-G)
    h -= (n-1)*INTERACTION/4*np.eye(dim)
    expected = (h+h.conj().T)/2-K*(h-h.conj().T)/(2j)+shift*np.eye(dim)
    paulis = []
    for term, coeff in terms:
        label = ['I']*n
        for j, axis in term: label[n-1-j] = axis
        paulis.append((''.join(label), coeff))
    paulis.append(('I'*n, shift))
    actual = SparsePauliOp.from_list(paulis).to_matrix()
    error = float(np.max(np.abs(expected-actual)))
    assert error < 1e-12, error
    return error


def main():
    rows, checks, diagnostics = [], [], []
    for n in SIZES:
        circuits_n, (_, _, terms, shift) = circuits(n)
        if n <= 4:
            checks.append({'n': n, 'jw_matrix_error': fermion_validation(n, terms, shift)})
            error = float(np.max(np.abs(Operator(circuits_n['original']).data-
                                        Operator(circuits_n['simplified_exact']).data)))
            assert error < 1e-10, error
            checks.append({'n': n, 'full_operator_error': error})
        sample = ((1 << (n//2-1)) | (1 << (n//2)))
        max_err, reverse_err = 0.0, 0.0
        for state in (0, sample):
            for imaginary in (False, True):
                original = ancilla_x(circuits_n['original'], state, imaginary)
                simple = ancilla_x(circuits_n['simplified_exact'], state, imaginary)
                reverse = ancilla_x(circuits_n['paper_reverse'], state, imaginary)
                max_err = max(max_err, abs(original-simple))
                reverse_err = max(reverse_err, abs(original-reverse))
        assert max_err < 1e-10, max_err
        checks.append({'n': n, 'overlap_component_max_error': max_err,
                       'paper_reverse_overlap_component_max_difference': reverse_err})
        coupling = CouplingMap.from_line(n+1, bidirectional=True)
        for variant, qc in circuits_n.items():
            before = counts(qc)
            compiled = transpile(qc, basis_gates=['u', 'cx'], coupling_map=coupling,
                                 initial_layout=list(range(n+1)), routing_method='sabre',
                                 seed_transpiler=SEED, optimization_level=OPT,
                                 num_processes=1)
            after = counts(compiled)
            assert after['triple'] == 0
            for ins in compiled.data:
                if len(ins.qubits) == 2:
                    ids = [compiled.find_bit(q).index for q in ins.qubits]
                    assert abs(ids[0]-ids[1]) == 1
            # Routing can move the logical ancilla; use final_index_layout.
            physical_ancilla = compiled.layout.final_index_layout()[0]
            sv = Statevector.from_int(0, 2**(n+1)).evolve(compiled)
            prob = np.abs(sv.data)**2
            z = float(sum(p*(1-2*((i >> physical_ancilla)&1)) for i,p in enumerate(prob)))
            compile_err = abs(z-ancilla_x(qc, 0))
            assert compile_err < 1e-10, compile_err
            row = {'system_qubits': n, 'total_qubits': n+1, 'variant': variant,
                   **{'logical_'+k: before[k] for k in ('single','double','triple','depth')},
                   **{'line_'+k: after[k] for k in ('single','double','depth')},
                   'compile_overlap_error': compile_err}
            rows.append(row)
            diagnostics.append({'n': n, 'variant': variant, 'logical': before,
                                'compiled': after, 'physical_ancilla': physical_ancilla})
            qasm2.dump(compiled, str(OUT/f'{variant}_n{n}_line.qasm'))
            print(row, flush=True)
    with (OUT/'gate_counts.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {'qiskit': qiskit.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
              'layers': LAYERS, 'dt': DT, 'g': G, 'interaction': INTERACTION, 'k': K,
              'trotter_steps': 1, 'trotter_order': 1, 'basis': ['u','cx'],
              'optimization_level': OPT, 'seed_transpiler': SEED,
              'connectivity': 'bidirectional line, ancilla initially at endpoint 0',
              'scope': 'one real overlap, ancilla H gates included; no shots, penalties, or state-preparation X gates'}
    (OUT/'details.json').write_text(json.dumps({'config': config, 'checks': checks,
                                               'counts': diagnostics}, indent=2), encoding='utf-8')
    print('VALIDATION', json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
