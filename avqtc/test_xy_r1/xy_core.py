"""10-qubit XY experiment primitives; big-endian computational basis.

Independent of the old experiment. NumPy and CuPy execute identical kernels.
Gate convention: exp(-i theta (XX+YY)), 2 CNOT + 4 rotations per bond.
"""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    from checkpoint_io import write_json as write_checkpoint
    write_checkpoint(path, obj)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(master, *labels):
    return int.from_bytes(hashlib.sha256(repr((int(master), labels)).encode()).digest()[:4], 'little')


def load_config(path=None):
    c = read_json(path or ROOT / 'config.json')
    m, t, s, r = (c[k] for k in ('model', 'training', 'sampling', 'resource_estimate'))
    if m['n'] < 2 or m['n'] % 2 or m['gamma'] < 0 or m['dt'] <= 0 or m['total_time'] <= 0:
        raise ValueError('Require even n >= 2, dt,T > 0 and gamma >= 0')
    if not np.isclose(m['total_time'] / m['dt'], round(m['total_time'] / m['dt'])):
        raise ValueError('total_time must be an integer multiple of dt')
    channel(m['n'], m['dt'], m['gamma'])
    for k in ('max_k', 'batch_size', 'validation_size', 'test_size', 'iterations',
              'validation_interval', 'print_every', 'full_validation_chunk'):
        if t[k] < 1:
            raise ValueError(f'training.{k} must be positive')
    if t['product_ensemble'] not in ('haar', 'pauli6') or t['backend'] not in ('auto', 'cpu', 'gpu'):
        raise ValueError('Unknown ensemble or backend')
    if not t['seeds'] or len(set(t['seeds'])) != len(t['seeds']):
        raise ValueError('Training seeds must be nonempty and distinct')
    if t['learning_rate'] <= 0 or t['random_restarts'] < 0 or t['jitter_scale'] < 0:
        raise ValueError('Invalid learning rate/restarts/jitter')
    if not 0 < t['adam_beta1'] < 1 or not 0 < t['adam_beta2'] < 1 or t['adam_epsilon'] <= 0:
        raise ValueError('Invalid Adam parameters')
    for route in ('direct', 'iterative'):
        if not 0 < t[f'{route}_target_fidelity'] <= 1 or t[f'{route}_extra_depth'] < 0:
            raise ValueError('Invalid fidelity/depth search settings')
    if min(s['m_values']) < 1 or s['trajectory_m'] < 1 or s['repeats'] < 2:
        raise ValueError('Require positive sampling sizes and >=2 repeats')
    if not s['iterative_cutoffs'] or min(s['iterative_cutoffs']) < 1:
        raise ValueError('Require positive iterative cutoffs')
    if r['shots_per_circuit'] < 1 or r['trotter_substeps'] < 1:
        raise ValueError('Resource assumptions must be positive')
    return c


def channel(n, dt, gamma):
    g = n * gamma
    if 2 - g * dt <= 0:
        raise ValueError('2 - n*gamma*dt must be positive')
    step = 2 * dt / (2 - g * dt)
    p0 = 1 - g * dt / (1 + (g * dt) ** 2 / 4)
    return step, np.array([p0] + [(1 - p0) / n] * n)


def schedule(depth):
    return [1 - j % 2 for j in range(depth + 1)]


def mask(n, depth):
    out = np.zeros((depth + 1, n - 1), dtype=bool)
    for layer, parity in enumerate(schedule(depth)):
        out[layer, parity::2] = True
    return out


def analytic_params(n, k, depth, step):
    out = np.zeros((depth + 1, n - 1))
    for layer, parity in enumerate(schedule(depth)):
        out[layer, parity::2] = k * step / depth * (1 if layer in (0, depth) else 2)
    return out


def transfer(params, depth, scale=1.0):
    if params is None or params.shape[0] > depth + 1:
        return None
    out = np.zeros((depth + 1, params.shape[1]))
    out[:params.shape[0]] = params * scale
    return out


def gate_resources(n, depth):
    bonds = int(mask(n, depth).sum())
    return {'parameters': bonds, 'cnot': 2 * bonds, 'single_qubit': 4 * bonds,
            'parity_layers': depth + 1, 'primitive_depth_upper_bound': 6 * (depth + 1)}


def product_states(n, count, seed, ensemble='haar'):
    rng = np.random.default_rng(seed)
    if ensemble == 'haar':
        # z=cos(theta) uniform, not theta uniform. Rz(phi) Ry(theta)|0> up to phase.
        z = rng.uniform(-1, 1, (count, n))
        phi = rng.uniform(0, 2 * np.pi, (count, n))
        local = np.stack([np.sqrt((1 + z) / 2),
                          np.exp(1j * phi) * np.sqrt((1 - z) / 2)], axis=-1)
    elif ensemble == 'pauli6':
        states = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 1j], [1, -1j]], complex)
        states[2:] /= np.sqrt(2)
        local = states[rng.integers(0, 6, (count, n))]
    else:
        raise ValueError(ensemble)
    result = np.ones((count, 1), complex)
    for j in range(n):
        result = (result[:, :, None] * local[:, None, j, :]).reshape(count, -1)
    return result


class Engine:
    def __init__(self, n, backend='auto'):
        self.n, self.dim = n, 1 << n
        self.xp = np
        if backend != 'cpu':
            try:
                import cupy
                cupy.zeros(1).sum().item()
                self.xp = cupy
            except Exception:
                if backend == 'gpu':
                    raise
        self.backend = 'cpu' if self.xp is np else 'gpu'
        self.pairs = []
        ids = np.arange(self.dim)
        for b in range(n - 1):
            i, j = 1 << (n - 1 - b), 1 << (n - 2 - b)
            left = ids[((ids & i) == 0) & ((ids & j) != 0)]
            self.pairs.append((self.xp.asarray(left), self.xp.asarray(left ^ i ^ j)))
        bits = ((ids[:, None] >> np.arange(n - 1, -1, -1)) & 1)
        self.signs = 1 - 2 * bits
        self.imbalance = self.xp.asarray((self.signs[:, :n//2].sum(1) - self.signs[:, n//2:].sum(1)) / n)
        self.initial = self.xp.zeros(self.dim, dtype=complex)
        self.initial[((1 << (n//2)) - 1) << (n//2)] = 1

    def cpu(self, a):
        return np.asarray(a) if self.xp is np else self.xp.asnumpy(a)

    def sync(self):
        if self.backend == 'gpu':
            self.xp.cuda.Stream.null.synchronize()

    def environment(self):
        out = {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
               'platform': platform.platform(), 'backend': self.backend,
               'dtype': 'complex128', 'actual_quantum_shots': 0}
        if self.backend == 'gpu':
            out['cupy'] = self.xp.__version__
            out['gpu'] = self.xp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        return out

    def bond(self, states, b, theta):
        x, y = self.pairs[b]
        a, v = states[:, x].copy(), states[:, y].copy()
        c, s = self.xp.cos(2 * theta), -1j * self.xp.sin(2 * theta)
        states[:, x], states[:, y] = c*a + s*v, s*a + c*v
        return states

    def pqc(self, inputs, params):
        out = self.xp.asarray(inputs).copy()
        for layer, parity in enumerate(schedule(params.shape[0] - 1)):
            for b in range(parity, self.n - 1, 2):
                self.bond(out, b, params[layer, b])
        return out

    def unitary(self, params):
        return self.pqc(self.xp.eye(self.dim, dtype=complex), params).T.copy()

    def u0(self, dt, gamma):
        h = np.zeros((self.dim, self.dim))
        for a, b in self.pairs:
            a, b = self.cpu(a), self.cpu(b)
            h[a, b] += 2
            h[b, a] += 2
        vals, vecs = eigh(h)
        step, _ = channel(self.n, dt, gamma)
        u = (vecs * np.exp(-1j * step * vals)) @ vecs.T
        return self.xp.asarray(u), h

    def fidelity(self, params, inputs, targets):
        overlaps = self.xp.sum(self.xp.conj(targets) * self.pqc(inputs, params), axis=1)
        return float(self.xp.mean(self.xp.abs(overlaps)**2).item())

    def full_metrics(self, params, targets, chunk=64):
        # targets is a list of full matrices. Accumulate trace BEFORE abs-square.
        traces = np.zeros(len(targets), complex)
        basis_f = np.zeros(len(targets))
        for start in range(0, self.dim, chunk):
            stop = min(start + chunk, self.dim)
            states = self.xp.zeros((stop-start, self.dim), dtype=complex)
            states[self.xp.arange(stop-start), self.xp.arange(start, stop)] = 1
            out = self.pqc(states, params)
            for j, target in enumerate(targets):
                overlaps = self.xp.sum(self.xp.conj(target[:, start:stop].T) * out, axis=1)
                traces[j] += complex(overlaps.sum().item())
                basis_f[j] += float((self.xp.abs(overlaps)**2).sum().item())
        return [{'hs_fidelity': float(abs(t/self.dim)**2),
                 'computational_basis_fidelity': float(b/self.dim)} for t, b in zip(traces, basis_f)]

    def loss_gradient(self, params, inputs, targets, mode):
        xp = self.xp
        states, tape = xp.asarray(inputs).copy(), []
        for layer, parity in enumerate(schedule(params.shape[0] - 1)):
            for bond in range(parity, self.n - 1, 2):
                before = states
                theta = params[layer, bond]
                states = self.bond(states.copy(), bond, theta)
                tape.append((layer, bond, theta, before))
        overlap = xp.sum(xp.conj(targets) * states, axis=1)
        size = len(inputs)
        if mode == 'full':
            mean = xp.mean(overlap)
            mag = xp.sqrt(xp.abs(mean)**2 + 1e-16)
            loss = 1 - mag
            adjoint = -(mean / (2 * mag * size)) * targets
        elif mode == 'b32':
            loss = 1 - xp.mean(xp.abs(overlap)**2)
            adjoint = -(overlap[:, None] / size) * targets
        else:
            raise ValueError(mode)
        gradient = xp.zeros(params.shape)
        for layer, bond, theta, before in reversed(tape):
            x, y = self.pairs[bond]
            c, s = -2*xp.sin(2*theta), -2j*xp.cos(2*theta)
            dx = c*before[:, x] + s*before[:, y]
            dy = s*before[:, x] + c*before[:, y]
            gradient[layer, bond] = 2*xp.real(xp.sum(xp.conj(adjoint[:, x])*dx) + xp.sum(xp.conj(adjoint[:, y])*dy))
            adjoint = self.bond(adjoint.copy(), bond, -theta)
        return float(loss.item()), self.cpu(gradient)

    def original_run(self, states, k, step):
        if k:
            # Same alternating first-order Trotter/merging convention as old 02.
            self._apply_layers(states, k, step)
        return states

    def _apply_layers(self, states, k, step):
        for layer, parity in enumerate(schedule(k)):
            theta = step * (1 if layer in (0, k) else 2)
            for b in range(parity, self.n - 1, 2):
                self.bond(states, b, theta)

    def proxy_unitary(self, k, step, substeps=1):
        out = self.xp.eye(self.dim, dtype=complex)
        self.original_run(out, k*substeps, step/substeps)
        return out.T.copy()


def segments(branches):
    pos = 0
    while pos < len(branches):
        branch = int(branches[pos])
        end = pos + 1
        if branch == 0:
            while end < len(branches) and branches[end] == 0:
                end += 1
        yield branch, end-pos
        pos = end


def trajectory(engine, branches, step, library=None, cutoff=None, exact_u0=None):
    state = engine.initial[None, :].copy()
    original_cnot = replaced_cnot = original_single = replaced_single = 0
    original_depth = replaced_depth = 0
    blocks = []
    for branch, length in segments(branches):
        if branch:
            state *= engine.xp.asarray(engine.signs[:, branch-1])[None, :]
            original_single += 1
            replaced_single += 1
            original_depth += 1
            replaced_depth += 1
        else:
            res = gate_resources(engine.n, length)
            original_cnot += res['cnot']
            original_single += res['single_qubit']
            original_depth += res['primitive_depth_upper_bound']
            if library is not None:
                parts = [cutoff]*(length//cutoff) + ([length % cutoff] if length % cutoff else [])
                for k in parts:
                    state = engine.pqc(state, library[k])
                    r = gate_resources(engine.n, library[k].shape[0]-1)
                    replaced_cnot += r['cnot']
                    replaced_single += r['single_qubit']
                    replaced_depth += r['primitive_depth_upper_bound']
                blocks.extend(parts)
            else:
                if exact_u0 is None:
                    engine.original_run(state, length, step)
                else:
                    for _ in range(length):
                        state = state @ exact_u0.T
                replaced_cnot += res['cnot']
                replaced_single += res['single_qubit']
                replaced_depth += res['primitive_depth_upper_bound']
    value = float(engine.xp.sum(engine.xp.abs(state[0])**2 * engine.imbalance).item())
    return value, {'original_cnot': original_cnot, 'replaced_cnot': replaced_cnot,
                   'original_single_qubit': original_single, 'replaced_single_qubit': replaced_single,
                   'original_depth_upper_bound': original_depth, 'replaced_depth_upper_bound': replaced_depth,
                   'pqc_block_count': len(blocks), 'largest_pqc_k': max(blocks, default=0),
                   'blocks': blocks}
