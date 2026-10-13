"""Reproduce the five XXZ figures using the published training seed 202506."""
import argparse
from pathlib import Path
import numpy as np
from xxz_core import ROOT, load_config, read_json, file_hash
from workflows import ensure_reference, paired_branch_table
from paper_figures import figures


def main(output):
    cfg = load_config()
    seed = 202506
    reference, _ = ensure_reference(cfg, output)
    _, signature = paired_branch_table(cfg, output)
    if read_json(output/'baseline/metadata.json')['signature'] != signature:
        raise ValueError('Baseline trajectory signature mismatch')
    baseline = np.load(output/'baseline/baseline.npz')['imbalance']
    manifests, results = {}, {}
    for channel in ['full_direct', 'full_iterative', 'b32_direct', 'b32_iterative']:
        path = output/'training'/channel/f'seed_{seed}'/'manifest.json'
        run = read_json(path)
        if run['status'] != 'completed' or run['specification']['training'] != cfg['training']:
            raise ValueError(f'Incomplete or incompatible library: {path}')
        manifests[channel, seed] = run
        cutoffs = [cfg['training']['max_k']] if channel.endswith('direct') else cfg['sampling']['iterative_cutoffs']
        for cutoff in cutoffs:
            folder = output/'replacement'/channel/f'seed_{seed}'/f'K{cutoff}'
            summary = read_json(folder/'summary.json')
            if summary['branch_signature'] != signature or summary['training_manifest_sha256'] != file_hash(path):
                raise ValueError(f'Replacement provenance mismatch: {folder}')
            results[channel, seed, cutoff] = folder
    dest = output/'figures'
    dest.mkdir(parents=True, exist_ok=True)
    figures(cfg, dest, reference, baseline, manifests, results, seed, lambda text: print(text, flush=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data')
    main(parser.parse_args().output)
