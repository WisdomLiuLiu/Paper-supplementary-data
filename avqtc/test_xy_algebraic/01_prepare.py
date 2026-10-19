"""Reuse original XY samples/references and algebraically compile each U0 run."""
import time
import numpy as np
from common import ROOT, config, logger, read_json, write_json, file_hash, verify_inputs, INPUTS, DATA, arguments
from xy_core import channel, digest, segments, gate_resources
from algebraic import original_gates, compress, single_particle, counts, export_qasm


def prepare(output=DATA):
    cfg = config()
    verify_inputs()
    logger('Using verified paired samples and references from test_xy_r1/data')
    table = np.load(INPUTS['branches.npz'])
    branches = table['branches']
    signature = digest(dict(model=cfg['model'], M=128, stream_seed=cfg['sampling']['stream_seed']))
    assert str(table['signature']) == signature and branches.shape == (100, 128, 100)
    step, probs = channel(10, .1, .06)
    rng = np.random.default_rng(cfg['sampling']['stream_seed'])
    used = set()
    for index in range(1, 101):
        expected = rng.choice(11, size=(128, index), p=probs)
        np.testing.assert_array_equal(branches[index-1, :, :index], expected)
        for row in expected:
            used.update(length for branch, length in segments(row) if branch == 0)
    logger(f'Verified all 12,800 paired trajectories; {len(used)} distinct U0-run lengths, maximum={max(used)}')
    dest = output/'circuits'
    dest.mkdir(parents=True, exist_ok=True)
    library = {}
    started = time.perf_counter()
    for length in sorted(used):
        original = original_gates(10, length, step)
        compiled, info = compress(10, original)
        original_count, compiled_count = counts(original, 10), counts(compiled, 10)
        assert original_count['cnot'] == gate_resources(10, length)['cnot']
        assert original_count['single_qubit'] == gate_resources(10, length)['single_qubit']
        error = float(np.max(abs(single_particle(10, compiled)-single_particle(10, original))))
        block = dict(length=length, **info, original_gates=original, compiled_gates=compiled,
                     original_counts=original_count, compiled_counts=compiled_count, single_particle_max_error=error)
        library[str(length)] = block
        write_json(dest/f'k{length:03d}.json', block)
        export_qasm(dest/f'k{length:03d}_compressed.qasm', 10, compiled)
        if length % 10 == 0 or length == max(used):
            logger(f'Compiled run length={length}: XY blocks {len(original)} -> {len(compiled)}, error={error:.3g}')
    write_json(output/'library.json', dict(n=10, block='exp[-i theta(XX+YY)]',
        method='XY B-block fusion, commutation and SO(3) turnover; triangle-to-square',
        short_segment_policy='keep original if it has at most n(n-1)/2 XY blocks',
        step=step, branch_signature=signature, actual_quantum_shots=0, optimizer_evaluations=0,
        compile_seconds=time.perf_counter()-started, source_sha256=file_hash(ROOT/'algebraic.py'), library=library))
    logger('PREPARATION COMPLETE')


if __name__ == '__main__':
    prepare(arguments(__doc__).output)
