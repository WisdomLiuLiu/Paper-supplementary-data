"""Reuse all five shot-selected libraries in the original XXZ replacement task."""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
from experiment import ROOT, log
from xxz_core import (Engine, channel, digest, file_hash, model_record, read_json,
                      write_json, stamp, trajectory)
from insertion import atomic_npz, tokens_for, evaluate_batch, resource_row, write_csv, accuracy


def main(args):
    training = read_json(args.training_data/'manifest.json')
    cfg = training['specification']['config']
    if cfg['experiment']['teacher_mode']!='exact':
        raise ValueError('This figure compares direct compilation of exact U0^s.')
    source_model = read_json(args.source_data/'model.json')
    if source_model != model_record(cfg):
        raise ValueError('The shot experiment and original paired trajectories have different models.')
    base_meta = read_json(args.source_data/'baseline/metadata.json')
    ref_meta = read_json(args.source_data/'reference/metadata.json')
    for name, expected in base_meta['files_sha256'].items():
        if file_hash(args.source_data/'baseline'/name)!=expected:
            raise ValueError(f'Original baseline checksum mismatch: {name}')
    if file_hash(args.source_data/'reference/reference.npz')!=ref_meta['npz_sha256']:
        raise ValueError('Reference checksum mismatch')
    if ref_meta['specification']['resolved_model']!=source_model:
        raise ValueError('Reference physical model mismatch')
    if file_hash(ROOT/'xxz_core.py')!=base_meta['generator_source_hashes']['xxz_core.py']:
        raise ValueError('Original trajectory/PQC kernels differ from the current snapshot')
    seed = args.seed
    if seed not in cfg['experiment']['seeds']:
        raise ValueError('Requested seed was not trained')
    libraries = {}
    inputs = {}
    for shots in cfg['experiment']['shots']:
        label = 'infinite' if shots==0 else str(shots)
        blocks = {}
        for k in range(1,args.cutoff+1):
            path = args.training_data/'training'/f'seed_{seed}'/f's{k:02d}'/f'shots_{label}'/'checkpoint.json'
            result = read_json(path)
            if result['status']!='completed' or result.get('audit_status')!='completed':
                raise ValueError(f'Incomplete selected circuit: {path}')
            if result['fingerprint']!=training['fingerprint']:
                raise ValueError('Training provenance mismatch')
            blocks[str(k)] = dict(params=result['best_params'], depth=result['job']['depth'],
                                  hs_fidelity=result['audit']['selected_exact_U0s_fidelity'],
                                  sampled_stopping_fidelity=result['best_selection_fidelity'],
                                  source_sha256=file_hash(path))
        libraries[label] = dict(shots=shots, seed=seed, cutoff=args.cutoff, blocks=blocks)
    ref_path = args.source_data/'reference/reference.npz'
    branch_path = args.source_data/'baseline/branches.npz'
    baseline_path = args.source_data/'baseline/baseline.npz'
    dependencies = {str(p):file_hash(p) for p in [ref_path, branch_path, baseline_path]}
    spec = dict(model=source_model, seed=seed, cutoff=args.cutoff, libraries=libraries,
                training_fingerprint=training['fingerprint'], input_sha256=dependencies,
                code_sha256={name:file_hash(ROOT/name) for name in ['06_insert_pqc.py','insertion.py','xxz_core.py']},
                replacement_rule='split each consecutive no-jump run into K blocks plus remainder; retain every Z jump',
                error_definition='Kraus minus replacement; signed',
                measurement_policy='shots affect trained PQCs only; evaluate trajectory expectation values exactly')
    fingerprint = digest(spec)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists() and read_json(manifest_path)['fingerprint']!=fingerprint:
        raise ValueError('Insertion inputs/code changed; choose a new --output directory')
    write_json(manifest_path, dict(status='running',fingerprint=fingerprint,specification=spec))
    snapshot = args.output/'inputs'
    snapshot.mkdir(exist_ok=True)
    for path in [ref_path, branch_path, baseline_path]:
        target = snapshot/path.name
        if not target.exists():
            shutil.copy2(path,target)
        if file_hash(target)!=file_hash(path):
            raise ValueError('Input snapshot mismatch')
    write_json(snapshot/'model.json',source_model)
    write_json(snapshot/'selected_libraries.json',libraries)
    with np.load(ref_path) as archive:
        ref = {k:archive[k] for k in ['time','kraus','exact','same_seed_exact']}
    with np.load(branch_path) as archive:
        branches = archive['branches']
        if str(archive['signature'])!=base_meta['signature']:
            raise ValueError('Branch signature mismatch')
    with np.load(baseline_path) as archive:
        baseline = archive['imbalance']
        np.testing.assert_array_equal(archive['time'],ref['time'])
    grid = ref['time']
    count = branches.shape[1]
    if branches.shape[0]!=len(grid)-1:
        raise ValueError('Trajectory time-grid mismatch')
    engine = Engine(cfg['model']['n'],args.backend,cfg['model'],cfg['circuit'])
    step,_ = channel(engine.n,cfg['model']['dt'],cfg['model']['gamma'])
    plans = [[tokens_for(branches[i-1,j,:i],args.cutoff) for j in range(count)] for i in range(1,len(grid))]
    log(args.output,f'INSERT START: {len(libraries)} libraries, M={count}, T={grid[-1]}, K={args.cutoff}; backend={engine.backend}')
    comparisons=[]
    for label, library_record in libraries.items():
        dest = args.output/f'shots_{label}'
        dest.mkdir(exist_ok=True)
        summary_path=dest/'summary.json'
        if summary_path.exists():
            old=read_json(summary_path)
            if old['fingerprint']!=fingerprint:
                raise ValueError('Result fingerprint mismatch')
            for filename,sha in old['files_sha256'].items():
                if file_hash(dest/filename)!=sha:
                    raise ValueError('Completed insertion result checksum mismatch')
            comparisons.append(old)
            log(args.output,f'RESUME shots={label}: complete, RMSE={old["accuracy_to_kraus"]["rmse"]:.7f}')
            continue
        library={int(k):np.asarray(v['params']) for k,v in library_record['blocks'].items()}
        matrices={k:engine.unitary(params) for k,params in library.items()}
        # Independently compare cached-matrix propagation with original gate-by-gate trajectory().
        checks=[]
        for index,j in [(1,0),(37,7),(len(grid)-1,0),(len(grid)-1,count-1)]:
            index=min(index,len(grid)-1);j=min(j,count-1)
            fast,_=evaluate_batch(engine,matrices,[plans[index-1][j]])
            slow,_=trajectory(engine,branches[index-1,j,:index],step,library,args.cutoff)
            checks.append(float(abs(fast[0]-slow)))
        if max(checks)>2e-11:
            raise AssertionError('Cached unitary differs from original trajectory implementation')
        values=np.full((len(grid),count),np.nan)
        values[0]=float(engine.cpu(engine.imbalance)[np.argmax(engine.cpu(abs(engine.initial)))])
        completed=0;previous_wall=0.;max_norm_error=0.
        partial=dest/'partial.npz'
        if partial.exists():
            with np.load(partial) as a:
                if str(a['fingerprint'])!=fingerprint:
                    raise ValueError('Partial result fingerprint mismatch')
                values=a['trajectory_imbalance'];completed=int(a['completed']);previous_wall=float(a['wall_seconds']);max_norm_error=float(a['max_norm_error'])
        started=time.perf_counter()
        for index in range(completed+1,len(grid)):
            values[index],norm_error=evaluate_batch(engine,matrices,plans[index-1])
            max_norm_error=max(max_norm_error,norm_error)
            if max_norm_error>2e-10:
                raise AssertionError('Trajectory norm drift')
            if index==1 or index%5==0 or index==len(grid)-1:
                elapsed=previous_wall+time.perf_counter()-started
                atomic_npz(partial,trajectory_imbalance=values,completed=index,wall_seconds=elapsed,
                           max_norm_error=max_norm_error,fingerprint=fingerprint)
                log(args.output,f'shots={label}, step={index}/{len(grid)-1}, t={grid[index]:.1f}, '
                    f'I={values[index].mean():.7f}, elapsed={elapsed:.1f}s')
        curve=values.mean(axis=1)
        atomic_npz(dest/'curves.npz',time=grid,imbalance=curve,trajectory_imbalance=values,
                   trajectory_std=values.std(axis=1,ddof=1),error_to_kraus=ref['kraus']-curve,
                   error_to_exact=ref['exact']-curve,error_to_same_seed_exact=ref['same_seed_exact']-curve,
                   error_to_baseline=baseline-curve)
        rows=[];usage={str(k):0 for k in library}
        for index in range(1,len(grid)):
            for j in range(count):
                tokens=plans[index-1][j]
                for token in tokens:
                    if token>0:usage[str(token)]+=1
                rows.append(dict(time_index=index,time=grid[index],trajectory_index=j,imbalance=values[index,j],
                                 **resource_row(engine,branches[index-1,j,:index],tokens,library)))
        write_csv(dest/'resources.csv',rows)
        summary=dict(fingerprint=fingerprint,shots=library_record['shots'],M=count,cutoff=args.cutoff,
                     wall_seconds=previous_wall+time.perf_counter()-started,
                     accuracy_to_kraus=accuracy(curve,ref['kraus']),accuracy_to_same_seed_exact=accuracy(curve,ref['same_seed_exact']),
                     accuracy_to_baseline=accuracy(curve,baseline),block_usage=usage,
                     cached_vs_original_max_error=max(checks),max_norm_error=max_norm_error,
                     files_sha256={name:file_hash(dest/name) for name in ['curves.npz','resources.csv']})
        write_json(summary_path,summary);comparisons.append(summary)
        log(args.output,f'COMPLETE shots={label}: RMSE(Kraus)={summary["accuracy_to_kraus"]["rmse"]:.8f}')
    write_csv(args.output/'comparison.csv',[dict(shots=s['shots'],**s['accuracy_to_kraus'],
                rmse_to_same_seed_exact=s['accuracy_to_same_seed_exact']['rmse']) for s in comparisons])
    write_json(manifest_path,dict(status='completed',fingerprint=fingerprint,specification=spec,completed_at=stamp(),
               environment=engine.environment(),M=count,time_points=len(grid),
               baseline_accuracy=accuracy(baseline,ref['kraus']),same_seed_exact_accuracy=accuracy(ref['same_seed_exact'],ref['kraus'])))
    log(args.output,'INSERT COMPLETE. Run 07_plot_insert.py to generate the full-width three-panel figure.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--training-data',type=Path,default=ROOT/'data')
    p.add_argument('--source-data',type=Path,default=ROOT.parent/'test_xxz_r1'/'data')
    p.add_argument('--output',type=Path,default=ROOT/'data'/'insert')
    p.add_argument('--backend',choices=['auto','cpu','gpu'],default='auto')
    p.add_argument('--seed',type=int,default=202506)
    p.add_argument('--cutoff',type=int,default=20)
    main(p.parse_args())
