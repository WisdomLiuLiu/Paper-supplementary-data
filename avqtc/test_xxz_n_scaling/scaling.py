"""Size scan invoking the unchanged original test_xxz_r1 training code."""
import argparse
import copy
import ctypes
import os
from pathlib import Path
from xxz_core import ROOT, Engine, file_hash, load_config, operations, read_json, write_json, stamp
import training


def arguments(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=ROOT/'config.json')
    p.add_argument('--output', type=Path, default=ROOT/'data')
    p.add_argument('--sizes', type=int, nargs='+')
    p.add_argument('--backend', choices=['auto', 'cpu', 'gpu'])
    return p


def settings(config, sizes=None):
    cfg = load_config(config)
    selected = sizes if sizes is not None else cfg['size_scan']['n_values']
    if not selected or len(set(selected)) != len(selected) or any(n < 2 or n % 2 for n in selected):
        raise ValueError('Supply distinct even n >= 2')
    if cfg['training']['seeds'] != [202506] or cfg['model']['fields'] is not None:
        raise ValueError('Use seed 202506 and the fixed disorder RNG for this scan')
    return cfg, selected


def config_for(cfg, n):
    result = copy.deepcopy(cfg)
    result.pop('size_scan')
    result['model']['n'] = n
    return result


def memory_plan(n, depth=1):
    gates = len(list(operations(n, depth)))
    return dict(n=n, depth=depth, full_complex_matrix_GiB=16*4**n/2**30,
                forward_tape_GiB=16*4**n*gates/2**30,
                gradient_additional_estimate_GiB=16*4**n*(gates+8)/2**30)


def cpu_available_bytes():
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ['total_phys', 'avail_phys', 'total_page',
                                                       'avail_page', 'total_virtual', 'avail_virtual', 'extended']]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Cannot read available physical memory')
        return status.avail_phys
    return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')


class MemoryCheckedEngine(Engine):
    """Only check allocations; numerical operations use the original Engine."""
    def u0(self, dt, gamma):
        # eigh and exact target construction use CPU RAM even on the GPU backend.
        needed = 64*self.dim**2/2**30
        available = cpu_available_bytes()/2**30
        if needed > .9*available:
            raise MemoryError(f'n={self.n}: dense target setup estimated at {needed:.2f} GiB RAM; '
                              f'available {available:.2f} GiB. No alternative implementation selected.')
        return super().u0(dt, gamma)

    def loss_gradient(self, params, inputs, targets, mode):
        if mode == 'full':
            needed = memory_plan(self.n, len(params))['gradient_additional_estimate_GiB']
            if self.backend == 'cpu':
                available = cpu_available_bytes()/2**30
            else:
                free, _ = self.xp.cuda.runtime.memGetInfo()
                available = (free+self.xp.get_default_memory_pool().free_bytes())/2**30
            if needed > .9*available:
                raise MemoryError(f'Original dense implementation: n={self.n}, depth={len(params)} '
                                  f'needs about {needed:.2f} GiB additional working memory; '
                                  f'available {self.backend} memory={available:.2f} GiB. '
                                  'Completed attempts are preserved. No sector implementation was substituted.')
        return super().loss_gradient(params, inputs, targets, mode)


def logger(output):
    output.mkdir(parents=True, exist_ok=True)
    def log(message):
        line = f'[{stamp()}] {message}'
        print(line, flush=True)
        with (output/'progress.log').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    return log


def library_path(output, n):
    sector = output/f'sector_gpu/n{n:02d}/manifest.json'
    if sector.exists():
        return sector
    local = output/f'n{n:02d}/training/full_direct/seed_202506/manifest.json'
    if n == 8 and output.resolve() == (ROOT/'data').resolve() and not local.exists():
        return ROOT.parent/'test_xxz_r1/data/training/full_direct/seed_202506/manifest.json'
    return local


def validate_library(run, cfg, n):
    spec = run['specification']
    if spec.get('implementation') == 'all_excitation_sectors_inverse_adjoint':
        from sector_training import specification
        from xxz_core import digest
        expected = specification(cfg, n)
        if spec != expected or run['fingerprint'] != digest(expected):
            raise ValueError(f'n={n}: sector settings/source fingerprint mismatch')
        return
    expected = config_for(cfg, n)
    actual_t, expected_t = copy.deepcopy(spec['training']), copy.deepcopy(expected['training'])
    for key in ['seeds', 'backend']:
        actual_t.pop(key); expected_t.pop(key)
    if (spec['model'] != expected['model'] or spec['circuit'] != expected['circuit']
            or actual_t != expected_t or spec['seed'] != 202506
            or spec['mode'] != 'full' or spec['route'] != 'direct'):
        raise ValueError(f'n={n}: saved results differ from requested settings')
    for name, sha in spec['source_hashes'].items():
        if file_hash(ROOT/name) != sha:
            raise ValueError(f'n={n}: original numerical source differs: {name}')


def prepare(cfg, sizes, output):
    output.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        path = output/f'n{n:02d}/config.json'
        expected = config_for(cfg, n)
        if path.exists() and read_json(path) != expected:
            raise ValueError(f'Configuration differs: {path}; choose a fresh output directory')
        write_json(path, expected)
    write_json(output/'implementation.json', dict(implementation='original dense matrix and stored forward tape',
               default_backend=cfg['training']['backend'],
               source_sha256={f: file_hash(ROOT/f) for f in ['training.py', 'xxz_core.py', 'circuit_export.py', 'checkpoint_io.py']},
               allocation_guard_sha256=file_hash(ROOT/'scaling.py')))
    write_json(output/'memory_plan.json', [memory_plan(n, d) for n in sizes for d in [1, 10, 20]])


def import_n8(cfg, output, log):
    source = ROOT.parent/'test_xxz_r1/data/training/full_direct/seed_202506/manifest.json'
    if not cfg['size_scan']['reuse_existing_n8'] or not source.exists():
        return False
    run = read_json(source)
    validate_library(run, cfg, 8)
    if run['status'] != 'completed' or len(run['library']) != cfg['training']['max_k']:
        raise ValueError('Original n=8 library is incomplete')
    target = output/'n08/imported_manifest.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    log('n=8: imported original dense/GPU full-direct seed=202506 result (20 blocks)')
    return True


def train(cfg, sizes, output, backend=None):
    prepare(cfg, sizes, output)
    log = logger(output)
    training.Engine = MemoryCheckedEngine
    for n in sizes:
        path = library_path(output, n)
        if path.exists():
            run = read_json(path)
            validate_library(run, cfg, n)
            if run['status'] == 'completed':
                log(f'RESUME n={n}: original dense library already complete')
                continue
        elif n == 8 and import_n8(cfg, output, log):
            continue
        effective_backend = backend or cfg['training']['backend']
        if effective_backend == 'cpu':
            # Only check the size about to run; larger sizes do not block n<=10.
            minimum = memory_plan(n, 1)['gradient_additional_estimate_GiB']
            available = cpu_available_bytes()/2**30
            if minimum > .9*available:
                raise MemoryError(f'n={n}: depth-1 dense gradient needs about {minimum:.1f} GiB; '
                                  f'available RAM {available:.1f} GiB. Earlier sizes are preserved.')
        log(f'START n={n}, backend={effective_backend}: original test_xxz_r1 full_direct implementation')
        training.run_channel(config_for(cfg, n), 'full', 'direct', output/f'n{n:02d}', [202506], effective_backend)
        log(f'COMPLETE n={n}')
