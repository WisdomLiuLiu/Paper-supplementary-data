"""Reproduce the eleven numerical figures.

Default (no training):
    python reproduce.py
    python reproduce.py --example xy
    python reproduce.py --example xxz
    python reproduce.py --example shots

Recompute data in a NEW directory (potentially long; prints progress):
    python reproduce.py --mode recompute --example xy --output /path/to/new_xy
    python reproduce.py --mode recompute --example xxz --output /path/to/new_xxz
    python reproduce.py --mode recompute --example shots --output /path/to/new_shots

Recompute trains only seed 202506. For shots, insertion uses the distributed
XXZ paired trajectories. The original data are never overwritten in this mode.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FOLDERS = dict(xy='test_xy_r1', xxz='test_xxz_r1', shots='test_xxz_full_shots')


def run(folder, script, *args):
    command = [sys.executable, '-B', '-u', str(folder/script), *map(str, args)]
    print(f'Running {folder.name}/{script} {" ".join(map(str, args))}', flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    subprocess.run(command, cwd=folder, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--example', choices=['all', *FOLDERS], default='all')
    parser.add_argument('--mode', choices=['plots', 'recompute'], default='plots')
    parser.add_argument('--output', type=Path, help='New data directory for recompute mode')
    args = parser.parse_args()
    if args.mode == 'recompute':
        if args.example == 'all' or args.output is None:
            parser.error('Recompute requires one --example and a new --output directory')
        output = args.output.resolve()
        if output.exists():
            parser.error('--output must not exist; choose a new directory')
        # Do not place a potentially large recomputation inside the publication subset.
        if output == ROOT or ROOT in output.parents:
            parser.error('--output must be outside this publication folder')
    elif args.output is not None:
        parser.error('--output is used only with --mode recompute')
    examples = list(FOLDERS) if args.example == 'all' else [args.example]
    for example in examples:
        folder = ROOT/FOLDERS[example]
        data = output if args.mode == 'recompute' else folder/'data'
        if args.mode == 'recompute':
            if example == 'shots':
                run(folder, '02_train.py', '--output', data)
                run(folder, '06_insert_pqc.py', '--training-data', data,
                    '--source-data', ROOT/'test_xxz_r1/data', '--output', data/'insert')
            else:
                run(folder, '01_prepare_reference.py', '--output', data, '--recompute')
                run(folder, '02_original_circuit.py', '--output', data, '--recompute')
                for stage in ['03_train_full_direct.py', '04_train_full_iterative.py',
                              '05_train_b32_direct.py', '06_train_b32_iterative.py', '07_replace.py']:
                    run(folder, stage, '--output', data, '--seeds', '202506')
        if example == 'shots':
            run(folder, '07_plot_insert.py', '--output', data/'insert')
        else:
            run(folder, '08_plot.py', '--output', data)
    print('Requested figures complete.', flush=True)


if __name__ == '__main__':
    main()
