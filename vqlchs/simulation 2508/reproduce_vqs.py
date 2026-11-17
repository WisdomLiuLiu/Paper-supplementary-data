"""Reconstruct archived HN VQS occupations without training or hardware calls.

Run from any directory. Dependencies: NumPy only.
The archived curves are read and checked, never overwritten.
"""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
N, LAYERS = 10, 5
BITS = ((np.arange(2**N)[:,None] >> np.arange(N-1,-1,-1)) & 1)


def state(theta, initial):
    psi = np.zeros(2**N,complex)
    psi[int(initial,2)] = 1
    ix=0
    for _ in range(LAYERS):
        for j in range(N):
            for axis in ('X','Z','X'):
                angle=theta[ix];ix+=1
                if axis=='Z':
                    psi*=np.exp(-.5j*angle*(1-2*BITS[:,j]))
                else:
                    flip=np.arange(2**N)^(1<<(N-1-j))
                    psi=np.cos(angle/2)*psi-1j*np.sin(angle/2)*psi[flip]
        for j in range(N-1):
            psi[(BITS[:,j]*BITS[:,j+1]).astype(bool)]*=-1
    return psi


def main():
    results={}
    cases=[('g1u1','param_layer_5_g1u1.txt','exp_vqa_g1u1.txt','0000110000'),
           ('LR_switch','param_switch.txt','exp_vqa_switch.txt','1111001111')]
    for directory,parameters,curve,initial in cases:
        path=ROOT/directory/parameters
        theta=np.loadtxt(path)
        states=[state(row,initial) for row in theta]
        occupation=np.column_stack([abs(s)**2@BITS for s in states])
        archived=np.loadtxt(ROOT/directory/curve)
        assert occupation.shape==archived.shape
        error=float(np.max(abs(occupation-archived)))
        results[directory]={'shape':list(occupation.shape),
            'parameters_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'max_occupation_difference':error,
            'max_norm_error':float(max(abs(np.vdot(s,s)-1) for s in states))}
        assert error<1e-10, results[directory]
    (ROOT/'vqs_reconstruction_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
