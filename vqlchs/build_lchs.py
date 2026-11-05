"""Construct the Cauchy-quadrature LCHS matrices used in the model tests.

The sum is a non-unitary propagator approximation, despite the historical
filename 'unitary_lchs'. Hermitian branch exponentials are evaluated exactly
numerically (up to floating-point error), without a Trotter approximation.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.linalg import expm


def ising_generators(n, gi, gr=2.0, coupling=1.0):
    size = 2**n
    h0 = np.zeros((size, size), complex)
    v = np.zeros_like(h0)
    for state in range(size):
        bits = [(state >> (n-1-j)) & 1 for j in range(n)]
        h0[state, state] = -coupling*sum((1-2*bits[j])*(1-2*bits[j+1]) for j in range(n-1))
        for j in range(n):
            target = state ^ (1 << (n-1-j))
            h0[target, state] = -gr
            v[target, state] = -gi
    # Scalar shift makes the anti-Hermitian generator negative semidefinite.
    v -= abs(gi)*n*np.eye(size)
    return h0, v, np.arange(size)


def hn_generators(n=10, g=1.0, interaction=1.0, reverse=False, sector=None):
    basis = np.array([s for s in range(2**n) if sector is None or bin(s).count('1')==sector])
    lookup = {s:i for i,s in enumerate(basis)}
    h = np.zeros((len(basis), len(basis)), complex)
    right, left = np.exp(g), np.exp(-g)
    if reverse: right, left = left, right
    for col, state in enumerate(basis):
        bits = [(state >> (n-1-j)) & 1 for j in range(n)]
        h[col,col] = interaction*sum(bits[j]*bits[j+1] for j in range(n-1))
        for j in range(n-1):
            if bits[j] != bits[j+1]:
                target = state ^ (1 << (n-1-j)) ^ (1 << (n-2-j))
                h[lookup[target],col] += right if bits[j] else left
    # Match the original JW construction: omit the scalar identity term.
    h -= interaction*(n-1)/4*np.eye(len(basis))
    h0 = (h+h.conj().T)/2
    v = (h-h.conj().T)/(2j)
    v -= abs(np.sinh(g))*n*np.eye(len(basis))
    return h0, v, basis


def lchs_matrix(h0, v, dt, cutoff, spacing):
    nodes = np.arange(-cutoff, cutoff+spacing, spacing)
    result = np.zeros_like(h0, dtype=complex)
    for k in nodes:
        result += spacing/(np.pi*(1+k*k))*expm(-1j*(h0-k*v)*dt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ising','hn'], required=True)
    parser.add_argument('--gi', type=float, default=0.5)
    parser.add_argument('--K', type=float, default=100)
    parser.add_argument('--dk', type=float)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--sector', type=int, help='HN fixed-particle-number block for a smaller verification calculation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path, help='Existing full matrix to compare with the reconstructed matrix/block')
    args=parser.parse_args()
    if args.model=='ising':
        h0,v,basis=ising_generators(6,args.gi)
        dt=0.1 if args.dt is None else args.dt
        dk=1.0 if args.dk is None else args.dk
    else:
        h0,v,basis=hn_generators(reverse=args.reverse,sector=args.sector)
        dt=0.2 if args.dt is None else args.dt
        dk=0.5 if args.dk is None else args.dk
    result=lchs_matrix(h0,v,dt,args.K,dk)
    report={'model':args.model,'K':args.K,'delta_k':dk,'delta_t':dt,'shape':list(result.shape),'nodes':len(np.arange(-args.K,args.K+dk,dk))}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savetxt(args.output,result)
    if args.verify:
        saved=np.loadtxt(args.verify,dtype=complex)
        expected=saved[np.ix_(basis,basis)]
        report['maximum_absolute_difference']=float(np.max(abs(result-expected)))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
