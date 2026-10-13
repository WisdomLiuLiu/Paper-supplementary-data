"""Exact evaluation of the selected PQC circuits on the original paired trajectories.

Cache each PQC unitary to accelerate classical simulation. This changes neither
the circuit nor its primitive gate counts. No new measurement shot noise.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from xxz_core import segments, gate_resources
from checkpoint_io import replace_with_retry


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **values)
    replace_with_retry(temp, path)


def tokens_for(branches, cutoff):
    tokens = []
    for branch, length in segments(branches):
        if branch:
            tokens.append(-branch)
        else:
            tokens += [cutoff]*(length//cutoff)
            if length % cutoff:
                tokens.append(length % cutoff)
    return tokens


def evaluate_batch(engine, matrices, sequences):
    xp = engine.xp
    states = xp.tile(engine.initial[None, :], (len(sequences), 1))
    for position in range(max(map(len, sequences), default=0)):
        groups = defaultdict(list)
        for j, seq in enumerate(sequences):
            if position < len(seq):
                groups[seq[position]].append(j)
        for token, indices in groups.items():
            ids = xp.asarray(indices)
            if token < 0:
                states[ids] = states[ids]*engine.z_signs[:, -token-1][None, :]
            else:
                states[ids] = states[ids] @ matrices[token].T
    probabilities = xp.abs(states)**2
    norms = engine.cpu(probabilities.sum(axis=1))
    values = engine.cpu(probabilities @ engine.imbalance)
    return values, float(np.max(np.abs(norms-1)))


def resource_row(engine, branches, tokens, library):
    counts = dict(original_cnot=0, original_single_qubit=0, replaced_cnot=0, replaced_single_qubit=0)
    for branch, length in segments(branches):
        if branch:
            counts['original_single_qubit'] += 1
        else:
            cost = gate_resources(engine.n, length*engine.baseline_substeps)
            counts['original_cnot'] += cost['cnot']
            counts['original_single_qubit'] += cost['single_qubit']
    for token in tokens:
        if token < 0:
            counts['replaced_single_qubit'] += 1
        else:
            cost = gate_resources(engine.n, len(library[token]))
            counts['replaced_cnot'] += cost['cnot']
            counts['replaced_single_qubit'] += cost['single_qubit']
    return dict(**counts, u0_count=int(np.count_nonzero(np.asarray(branches)==0)),
                z_count=int(np.count_nonzero(np.asarray(branches)>0)),
                pqc_block_count=sum(t>0 for t in tokens), largest_pqc_k=max([t for t in tokens if t>0], default=0))


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def accuracy(values, reference):
    error = np.asarray(reference)-np.asarray(values)
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                max_abs_error=float(np.max(np.abs(error))), final_signed_error=float(error[-1]))
