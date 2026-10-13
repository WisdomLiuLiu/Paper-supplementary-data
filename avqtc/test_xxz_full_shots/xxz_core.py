"""Disordered XXZ, big-endian basis; NC-inspired symmetric Hamiltonian ansatz.

H = Jxy sum(XX+YY) + Delta sum(ZZ) + sum(h_i Z_i).
One slice: D/2, E/2, O, E/2, D/2 in execution order; D contains Z and ZZ,
E/O contain even/odd XY bonds. Each slice has independently trainable angles.
This adapts NC 2023 Eq. (13), not its different Hamiltonian or parameter tying.
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
    if not all(np.isfinite(m[key]) for key in ('jxy', 'delta', 'disorder_strength')) or m['disorder_strength'] < 0:
        raise ValueError('XXZ couplings must be finite and disorder strength nonnegative')
    if m['initial_state'] not in ('domain_wall', 'neel'):
        raise ValueError('initial_state must be domain_wall or neel')
    disorder_fields(m)
    if c['circuit']['ansatz'] != 'nc_inspired_symmetric_xxz':
        raise ValueError('Unknown ansatz')
    if c['circuit']['baseline_trotter_substeps'] < 1:
        raise ValueError('baseline_trotter_substeps must be positive')
    if max(s['iterative_cutoffs']) > t['max_k']:
        raise ValueError('iterative_cutoffs must be covered by max_k')
    for value in [m['n'], m['disorder_seed'], t['max_k'], t['iterations'],
                  r['trotter_substeps'], c['circuit']['baseline_trotter_substeps']]:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('Sizes, step counts and disorder_seed must be integers')
    return c


def disorder_fields(model):
    """One fixed realization shared by every optimization seed and channel."""
    if model.get('fields') is not None:
        fields = np.asarray(model['fields'], dtype=float)
        if fields.shape != (model['n'],) or not np.isfinite(fields).all():
            raise ValueError('Explicit fields must contain n finite numbers')
        return fields
    return np.random.default_rng(model['disorder_seed']).uniform(
        -model['disorder_strength'], model['disorder_strength'], model['n'])


def model_record(cfg):
    return {'model': cfg['model'], 'fields': disorder_fields(cfg['model']).tolist(),
            'circuit': cfg['circuit'],
            'hamiltonian': 'Jxy*sum(XX+YY) + delta*sum(ZZ) + sum(h_i*Z_i); open boundaries',
            'disorder_policy': 'fixed realization; training seeds do not resample disorder',
            'gamma_convention': 'L_i=sqrt(gamma)*Z_i',
            'pauli_convention': 'Pauli eigenvalues +/-1; not spin operators sigma/2',
            'parameter_order': 'per slice: n Z, n-1 ZZ, n-1 XY angles',
            'ansatz_reference': 'NC 14, 3751 (2023), Eq. (13), adapted to XXZ with independent slices'}


def source_hashes():
    return {name: file_hash(ROOT/name) for name in
            ['training.py', 'xxz_core.py', 'checkpoint_io.py', 'circuit_export.py']}


def save_model_record(cfg, output):
    path = Path(output)/'model.json'
    record = model_record(cfg)
    if path.exists() and read_json(path) != record:
        raise ValueError('Model/disorder differs from stored model.json; choose another output directory')
    if not path.exists():
        write_json(path, record)
    return record


def channel(n, dt, gamma):
    g = n * gamma
    if 2 - g * dt <= 0:
        raise ValueError('2 - n*gamma*dt must be positive')
    step = 2 * dt / (2 - g * dt)
    p0 = 1 - g * dt / (1 + (g * dt) ** 2 / 4)
    return step, np.array([p0] + [(1 - p0) / n] * n)


def operations(n, depth):
    """(slice, parameter-column, kind, site/bond, angle multiplier)."""
    for layer in range(depth):
        for j in range(n):
            yield layer, j, 'z', j, .5
        for b in range(n-1):
            yield layer, n+b, 'zz', b, .5
        for b in range(0, n-1, 2):
            yield layer, 2*n-1+b, 'xy', b, .5
        for b in range(1, n-1, 2):
            yield layer, 2*n-1+b, 'xy', b, 1.0
        for b in range(0, n-1, 2):
            yield layer, 2*n-1+b, 'xy', b, .5
        for b in range(n-1):
            yield layer, n+b, 'zz', b, .5
        for j in range(n):
            yield layer, j, 'z', j, .5


def mask(n, depth):
    return np.ones((depth, 3*n-2), dtype=bool)


def analytic_params(n, k, depth, step, model):
    if depth < 1:
        raise ValueError('A circuit needs at least one symmetric slice')
    row = np.concatenate([disorder_fields(model),
                          np.full(n-1, model['delta']), np.full(n-1, model['jxy'])])
    return np.tile(row * (k*step/depth), (depth, 1))


def transfer(params, depth, scale=1.0):
    if params is None or params.shape[0] > depth:
        return None
    out = np.zeros((depth, params.shape[1]))
    out[:params.shape[0]] = params * scale
    return out


def gate_resources(n, depth):
    """Count a fixed decomposition, merging adjacent commuting D half-slices.

XY: 2 CX+4 rotations; ZZ: 2 CX+1 RZ; Z: 1 RZ. Do not remove small angles.
Gradient estimates conservatively shift unmerged rotation occurrences, using
the chain rule for repeated parameters. These are hypothetical hardware costs.
"""
    if depth < 1:
        raise ValueError('depth must be positive')
    even, odd = n//2, (n-1)//2
    xy = depth*(2*even+odd)
    zz, z = (depth+1)*(n-1), (depth+1)*n
    raw_zz, raw_z = 2*depth*(n-1), 2*depth*n
    # An XY matching uses <=6 primitive layers; ZZ even+odd <=6; local Z <=1.
    primitive_depth = 18*depth + 7*(depth+1)
    return {'parameters': depth*(3*n-2), 'cnot': 2*(xy+zz),
            'single_qubit': 4*xy+zz+z, 'xy_gates': xy, 'zz_gates': zz, 'z_gates': z,
            'symmetric_slices': depth, 'diagonal_layers_after_merge': depth+1,
            'primitive_depth_upper_bound': primitive_depth,
            'parameter_shift_gradient_settings': 4*xy + 2*raw_zz + 2*raw_z,
            'gradient_settings_convention': 'gate-wise shifts on unmerged occurrences; shared-angle chain rule'}


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
    def __init__(self, n, backend='auto', model=None, circuit=None):
        if model is None:
            raise ValueError('Pass an explicit XXZ model, including its disorder seed/fields')
        if model['n'] != n:
            raise ValueError('Model size mismatch')
        self.model, self.fields = model, disorder_fields(model)
        self.baseline_substeps = (circuit or {}).get('baseline_trotter_substeps', 1)
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
        self.z_signs = self.xp.asarray(self.signs)
        self.zz_signs = self.z_signs[:, :-1]*self.z_signs[:, 1:]
        initial_bits = ([1]*(n//2)+[0]*(n//2) if model['initial_state'] == 'domain_wall'
                        else [1, 0]*(n//2))
        weights = 2*np.asarray(initial_bits)-1
        self.imbalance = self.xp.asarray(self.signs @ weights / n)
        self.initial = self.xp.zeros(self.dim, dtype=complex)
        self.initial[int(''.join(map(str, initial_bits)), 2)] = 1

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

    def apply_gate(self, states, kind, site, theta):
        if kind == 'xy':
            return self.bond(states, site, theta)
        signs = self.z_signs[:, site] if kind == 'z' else self.zz_signs[:, site]
        states *= self.xp.exp(-1j*theta*signs)[None, :]
        return states

    def pqc(self, inputs, params):
        out = self.xp.asarray(inputs).copy()
        if params.ndim != 2 or params.shape[1] != 3*self.n-2:
            raise ValueError('XXZ parameter array must have shape (slices, 3*n-2)')
        for layer, column, kind, site, scale in operations(self.n, len(params)):
            self.apply_gate(out, kind, site, scale*params[layer, column])
        return out

    def unitary(self, params):
        return self.pqc(self.xp.eye(self.dim, dtype=complex), params).T.copy()

    def u0(self, dt, gamma):
        h = np.zeros((self.dim, self.dim))
        for a, b in self.pairs:
            a, b = self.cpu(a), self.cpu(b)
            h[a, b] += 2*self.model['jxy']
            h[b, a] += 2*self.model['jxy']
        diagonal = (self.model['delta']*(self.signs[:, :-1]*self.signs[:, 1:]).sum(1)
                    + self.signs @ self.fields)
        h[np.diag_indices(self.dim)] = diagonal
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
        for layer, column, kind, site, scale in operations(self.n, len(params)):
            before = states
            theta = scale*params[layer, column]
            states = self.apply_gate(states.copy(), kind, site, theta)
            tape.append((layer, column, kind, site, scale, theta, before))
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
        for layer, column, kind, site, scale, theta, before in reversed(tape):
            if kind == 'xy':
                x, y = self.pairs[site]
                c, s = -2*xp.sin(2*theta), -2j*xp.cos(2*theta)
                dx = c*before[:, x] + s*before[:, y]
                dy = s*before[:, x] + c*before[:, y]
                contribution = xp.sum(xp.conj(adjoint[:, x])*dx) + xp.sum(xp.conj(adjoint[:, y])*dy)
            else:
                signs = self.z_signs[:, site] if kind == 'z' else self.zz_signs[:, site]
                derivative = (-1j*signs*xp.exp(-1j*theta*signs))[None, :]*before
                contribution = xp.sum(xp.conj(adjoint)*derivative)
            # A parameter can occur twice within the symmetric slice.
            gradient[layer, column] += scale*2*xp.real(contribution)
            adjoint = self.apply_gate(adjoint.copy(), kind, site, -theta)
        return float(loss.item()), self.cpu(gradient)

    def original_run(self, states, k, step):
        if k:
            params = analytic_params(self.n, k, k*self.baseline_substeps, step, self.model)
            states[:] = self.pqc(states, params)
        return states

    def proxy_unitary(self, k, step, substeps=1):
        return self.unitary(analytic_params(self.n, k, k*substeps, step, self.model))


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
            res = gate_resources(engine.n, length*engine.baseline_substeps)
            original_cnot += res['cnot']
            original_single += res['single_qubit']
            original_depth += res['primitive_depth_upper_bound']
            if library is not None:
                parts = [cutoff]*(length//cutoff) + ([length % cutoff] if length % cutoff else [])
                for k in parts:
                    state = engine.pqc(state, library[k])
                    r = gate_resources(engine.n, library[k].shape[0])
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
