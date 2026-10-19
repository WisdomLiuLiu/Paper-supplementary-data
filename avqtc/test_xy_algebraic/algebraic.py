"""Isotropic XY specialization of F3C fusion/commutation/turnover compression.

Triangle absorption and triangle-to-square indexing are adapted from
QuantumComputingLab/f3c, commit 6b29c65a8b193234090be035dacd159663bc4940,
+f3c/TriangleCircuit.m. See F3C_LICENSE.txt for the retained license.
F3C Copyright (c) 2021, The Regents of the University of California,
through Lawrence Berkeley National Laboratory. All rights reserved.

Our local turnover uses a three-mode SO(3) Euler identity. It specializes the
paper's B-block rule to exp[-i theta(XX+YY)], without fitting/optimization.
The diagonal change of basis diag(1,i,i^2,...) is only used to derive the
classical identity; it does not add physical gates to the output circuit.
"""
import math
import numpy as np


def original_gates(n, length, step):
    """Preserve the original alternating-Trotter convention exactly."""
    return [(bond, step*(1 if layer in (0, length) else 2))
            for layer in range(length+1)
            for bond in range(1-layer % 2, n-1, 2)] if length else []


def single_particle(n, gates):
    out = np.eye(n, dtype=complex)
    for bond, theta in gates:
        a, b = out[bond].copy(), out[bond+1].copy()
        c, s = math.cos(2*theta), -1j*math.sin(2*theta)
        out[bond], out[bond+1] = c*a+s*b, s*a+c*b
    return out


def turnover(g1, g2, g3):
    """Chronological (i,j,i) -> (j,i,j), |i-j|=1; exact to roundoff."""
    i, a = g1
    j, b = g2
    assert g3[0] == i and abs(i-j) == 1
    low = min(i, j)
    matrix = np.eye(3)
    for bond, theta in [g1, g2, g3]:
        q = bond-low
        c, s = math.cos(2*theta), math.sin(2*theta)
        u, v = matrix[q].copy(), matrix[q+1].copy()
        matrix[q], matrix[q+1] = c*u+s*v, -s*u+c*v
    reverse = i > j
    if reverse:
        matrix = matrix[::-1, ::-1]
    sine = math.hypot(matrix[0, 1], matrix[0, 2])
    middle = math.atan2(sine, matrix[0, 0])
    if sine > 1e-15:
        first = math.atan2(matrix[0, 2], matrix[0, 1])
        last = math.atan2(matrix[2, 0], -matrix[1, 0])
    elif matrix[0, 0] >= 0:
        first, last = math.atan2(matrix[1, 2], matrix[1, 1]), 0.
    else:
        first, last = math.atan2(-matrix[1, 2], -matrix[1, 1]), 0.
    sign = -.5 if reverse else .5
    return (j, sign*first), (i, sign*middle), (j, sign*last)


def triangle(n, gates):
    tri = {(q, layer): (q, 0.) for layer in range(n-1) for q in range(n-2, layer-1, -1)}
    turns = 0
    for gate in gates:
        layer = n-2
        for q in range(gate[0], n-2):
            asc = layer+q-n+2
            gate, tri[q, asc], tri[q+1, asc+1] = turnover(tri[q, asc], tri[q+1, asc+1], gate)
            layer -= 1
            turns += 1
        old = tri[n-2, layer]
        assert old[0] == gate[0]
        tri[n-2, layer] = (old[0], old[1]+gate[1])
    return tri, turns


def square(n, tri):
    """F3C triangle-to-square transformation; n parallel nearest-neighbor layers."""
    assert n % 2 == 0
    tri = dict(tri)
    out = {}
    turns = 0
    for moving_layer in range(n-2, 1, -2):
        for q in range(n-2, moving_layer-1, -1):
            gate = tri[q, moving_layer]
            initial_q = gate[0]
            for layer in range(moving_layer-1, -1, -1):
                a, b = (gate[0], layer), (gate[0]-1, layer)
                gate, tri[a], tri[b] = turnover(tri[a], tri[b], gate)
                turns += 1
            out[n-initial_q-2, gate[0]] = gate
    for q in range(n-2, -1, -1):
        out[n-q-2, q] = tri[q, 0]
    for moving_layer in range(1, n-2, 2):
        for q in range(n-2, moving_layer-1, -1):
            out[n-q+moving_layer-1, q] = tri[q, moving_layer]
    expected = {(layer, q) for layer in range(n) for q in range(layer % 2, n-1, 2)}
    assert set(out) == expected
    return [out[key] for key in sorted(out)], turns


def compress(n, gates):
    cap = n*(n-1)//2
    if len(gates) <= cap:
        return list(gates), dict(method='unchanged_short_segment', turnovers=0, fusions=0)
    tri, absorption = triangle(n, gates)
    result, conversion = square(n, tri)
    assert len(result) == cap
    np.testing.assert_allclose(single_particle(n, result), single_particle(n, gates), atol=2e-11, rtol=0)
    return result, dict(method='F3C_XY_triangle_to_square', turnovers=absorption+conversion, fusions=len(gates))


def primitives(gates):
    out = []
    for q, angle in gates:
        out += [('rx', q, math.pi/2), ('cx', q, q+1), ('rx', q, 2*angle),
                ('ry', q+1, 2*angle), ('cx', q, q+1), ('rx', q, -math.pi/2)]
    return out


def apply_primitives(inputs, gates, n):
    """Independent full-state RX/RY/CNOT simulator for exported-circuit tests."""
    states = np.array(inputs, dtype=complex, copy=True)
    ids = np.arange(1 << n)
    for kind, q, value in gates:
        if kind == 'cx':
            control, target = 1 << (n-1-q), 1 << (n-1-int(value))
            states = states[:, ids ^ np.where(ids & control, target, 0)]
        else:
            bit = 1 << (n-1-q)
            left = ids[(ids & bit) == 0]
            right = left ^ bit
            a, b = states[:, left].copy(), states[:, right].copy()
            c, s = math.cos(value/2), math.sin(value/2)
            if kind == 'rx':
                states[:, left], states[:, right] = c*a-1j*s*b, -1j*s*a+c*b
            elif kind == 'ry':
                states[:, left], states[:, right] = c*a-s*b, s*a+c*b
            else:
                raise ValueError(kind)
    return states


def counts(gates, n):
    sequence = primitives(gates)
    ready, single, two = [0]*n, 0, 0
    for kind, q, value in sequence:
        sites = [q, int(value)] if kind == 'cx' else [q]
        end = 1+max(ready[s] for s in sites)
        for site in sites:
            ready[site] = end
        two += kind == 'cx'
        single += kind != 'cx'
    return dict(xy_blocks=len(gates), single_qubit=single, cnot=two, primitive_depth=max(ready))


def export_qasm(path, n, gates):
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n}];']
    for kind, q, value in primitives(gates):
        lines.append(f'cx q[{q}],q[{int(value)}];' if kind == 'cx'
                     else f'{kind}({value:.17g}) q[{q}];')
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
