"""Exact full-space trace and adjoint gradient using all excitation sectors.

The Hamiltonian and every ansatz gate conserve excitation number. Traces and
their complex derivatives are summed over ALL sectors before normalization,
absolute value or squaring. No stochastic trace estimate is used.
"""
import math
import time
import numpy as np
from scipy.linalg import eigh
from xxz_core import Engine, channel, disorder_fields


def memory_plan(n, max_depth=20):
    largest = math.comb(n, n//2)
    entries = math.comb(2*n, n)
    gates = max_depth*(2*(2*n-1)+2*(n//2)+(n-1)//2)
    return dict(n=n, dimension=2**n, sectors=n+1, largest_sector=largest,
                full_complex_matrix_GiB=16*4**n/2**30,
                original_forward_tape_GiB=16*4**n*gates/2**30,
                all_sector_targets_GiB=16*entries/2**30,
                estimated_working_GiB=(24*entries+8*16*largest**2)/2**30+1)


class SectorEngine:
    def __init__(self, model, circuit, backend='auto', log=print):
        self.model, self.n = model, model['n']
        self.dim = 1 << self.n
        base = Engine(self.n, backend, model, circuit)
        self.xp, self.backend = base.xp, base.backend
        self.environment = base.environment()
        self.log, self.sectors = log, []
        xp = self.xp
        if self.backend == 'gpu':
            free, _ = xp.cuda.runtime.memGetInfo()
            free += xp.get_default_memory_pool().free_bytes()
            estimate = memory_plan(self.n)['estimated_working_GiB']
            if free/2**30 < estimate:
                raise MemoryError(f'Estimated {estimate:.2f} GiB needed, only {free/2**30:.2f} GiB free; '
                                  'close other GPU jobs before retrying')
        fields = disorder_fields(model)
        all_ids = np.arange(self.dim)
        counts = np.array([int(i).bit_count() if hasattr(int, 'bit_count') else bin(int(i)).count('1')
                           for i in all_ids])
        for particles in range(self.n+1):
            started = time.perf_counter()
            ids = all_ids[counts == particles]
            signs = 1-2*((ids[:, None] >> np.arange(self.n-1, -1, -1)) & 1)
            diagonal_signs = np.concatenate([signs, signs[:, :-1]*signs[:, 1:]], axis=1).astype(float)
            inverse = np.full(self.dim, -1, dtype=int)
            inverse[ids] = np.arange(len(ids))
            pairs = []
            h = np.diag(signs@fields+model['delta']*(signs[:, :-1]*signs[:, 1:]).sum(1)).astype(float)
            for bond in range(self.n-1):
                a, b = 1 << (self.n-1-bond), 1 << (self.n-2-bond)
                left_ids = ids[((ids & a) == 0) & ((ids & b) != 0)]
                left, right = inverse[left_ids], inverse[left_ids ^ a ^ b]
                h[left, right] += 2*model['jxy']
                h[right, left] += 2*model['jxy']
                pairs.append((xp.asarray(left), xp.asarray(right)))
            log(f'n={self.n}: diagonalizing sector {particles}/{self.n}, dimension={len(ids)}')
            vals, vecs = eigh(h, overwrite_a=True, check_finite=False)
            self.sectors.append(dict(size=len(ids), ids=ids, pairs=pairs,
                                     signs=xp.asarray(diagonal_signs), vals=xp.asarray(vals),
                                     vecs=xp.asarray(vecs), target=None))
            self.sync()
            log(f'n={self.n}: eigensystem sector {particles}/{self.n}, dimension={len(ids)}, '
                f'elapsed={time.perf_counter()-started:.2f}s')
        self.k = None

    def sync(self):
        if self.backend == 'gpu':
            self.xp.cuda.Stream.null.synchronize()

    def cpu(self, value):
        return np.asarray(value) if self.backend == 'cpu' else self.xp.asnumpy(value)

    def set_target(self, k):
        if self.k == k:
            return
        xp = self.xp
        step, _ = channel(self.n, self.model['dt'], self.model['gamma'])
        for sector in self.sectors:
            # Eigenphases implement exact U0^k, without a Trotter target proxy.
            sector['target'] = None
            v = sector['vecs']
            sector['target'] = (v*xp.exp(-1j*k*step*sector['vals']))@v.T
        self.k = k
        self.sync()
        self.log(f'n={self.n}, s={k}: all {self.n+1} exact target sectors ready')

    def _bond(self, states, pair, theta):
        x, y = pair
        if not len(x):
            return
        xp = self.xp
        a, b = states[:, x].copy(), states[:, y].copy()
        c, s = xp.cos(2*theta), -1j*xp.sin(2*theta)
        states[:, x], states[:, y] = c*a+s*b, s*a+c*b

    def _schedule(self, params, sector):
        schedule = []
        for layer in range(len(params)):
            phase = self.xp.exp(-.5j*(sector['signs']@self.xp.asarray(params[layer, :2*self.n-1])))
            schedule.append(('diag', layer, None, .5, phase))
            for bonds, scale in [(range(0, self.n-1, 2), .5),
                                 (range(1, self.n-1, 2), 1.), (range(0, self.n-1, 2), .5)]:
                for bond in bonds:
                    schedule.append(('xy', layer, bond, scale, scale*params[layer, 2*self.n-1+bond]))
            schedule.append(('diag', layer, None, .5, phase))
        return schedule

    def evaluate(self, params, gradient=True):
        if self.k is None:
            raise RuntimeError('Call set_target(s) first')
        xp = self.xp
        trace = xp.asarray(0j)
        derivative = xp.zeros(params.shape, dtype=complex) if gradient else None
        started = last_progress = time.perf_counter()
        for sector_index, sector in enumerate(self.sectors):
            states = xp.eye(sector['size'], dtype=complex)
            schedule = self._schedule(params, sector)
            for kind, layer, bond, scale, value in schedule:
                if kind == 'diag':
                    states *= value[None, :]
                else:
                    self._bond(states, sector['pairs'][bond], value)
            # Rows are evolved computational basis inputs: V.T and target.T.
            target_rows = sector['target'].T
            trace += xp.vdot(target_rows, states)
            if not gradient:
                if time.perf_counter()-last_progress >= 30:
                    self.sync()
                    self.log(f'n={self.n}, s={self.k}: fidelity sector {sector_index}/{self.n}, '
                             f'elapsed={time.perf_counter()-started:.1f}s')
                    last_progress = time.perf_counter()
                continue
            adjoint = target_rows.copy()
            for kind, layer, bond, scale, value in reversed(schedule):
                if kind == 'diag':
                    weights = xp.sum(xp.conj(adjoint)*states, axis=0)
                    derivative[layer, :2*self.n-1] += -1j*scale*(sector['signs'].T@weights)
                    states *= xp.conj(value)[None, :]
                    adjoint *= xp.conj(value)[None, :]
                else:
                    x, y = sector['pairs'][bond]
                    if len(x):
                        term = -2j*(xp.vdot(adjoint[:, x], states[:, y])+xp.vdot(adjoint[:, y], states[:, x]))
                        derivative[layer, 2*self.n-1+bond] += scale*term
                        self._bond(states, sector['pairs'][bond], -value)
                        self._bond(adjoint, sector['pairs'][bond], -value)
            # Reconstruct preceding states by inverse gates; no per-gate tape.
            if time.perf_counter()-last_progress >= 30:
                self.sync()
                self.log(f'n={self.n}, s={self.k}: gradient sector {sector_index}/{self.n}, '
                         f'elapsed={time.perf_counter()-started:.1f}s')
                last_progress = time.perf_counter()
        normalized = complex(trace.item())/self.dim
        fidelity = float(abs(normalized)**2)
        magnitude = np.sqrt(fidelity+1e-16)
        grad = None if not gradient else -np.real(np.conj(normalized)*self.cpu(derivative)/self.dim)/magnitude
        return float(1-magnitude), grad, fidelity
