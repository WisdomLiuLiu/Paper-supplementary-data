"""Shared paths, source provenance and progress reporting."""
import argparse
from pathlib import Path
from xy_core import read_json, write_json, file_hash, stamp, load_config

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'
INPUTS = {name: ROOT.parent/record['source']
          for name, record in read_json(ROOT/'input_checksums.json')['files'].items()}


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output', type=Path, default=DATA)
    return parser.parse_args()


def logger(message):
    line = f'[{stamp()}] {message}'
    print(line, flush=True)


def config():
    cfg = load_config(ROOT/'config.json')
    if cfg['model'] != dict(n=10, dt=.1, gamma=.06, total_time=10.):
        raise ValueError('This supplementary test uses the original 10-qubit XY model')
    if cfg['sampling']['trajectory_m'] != 128 or cfg['sampling']['stream_seed'] != 1225499654:
        raise ValueError('Use the original paired M=128 stream')
    return cfg


def verify_inputs():
    meta = read_json(ROOT/'input_checksums.json')
    for filename, record in meta['files'].items():
        if file_hash(INPUTS[filename]) != record['sha256']:
            raise ValueError(f'Input checksum mismatch: {filename}')
    return meta
