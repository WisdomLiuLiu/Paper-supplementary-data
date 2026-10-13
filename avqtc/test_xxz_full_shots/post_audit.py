"""Post-training-only exact audits and independent final shot measurements."""
import time
from pathlib import Path
from experiment import audit_selected, log
from xxz_core import Engine, read_json, write_json, stamp


def audit_all_if_ready(cfg, jobs, output):
    paths = [Path(output)/'training'/j['name']/'checkpoint.json' for j in jobs]
    if any(not p.exists() or read_json(p)['status']!='completed' for p in paths):
        log(output, 'Exact HSF audit deferred: not all architecture searches have completed.')
        return False
    pending = [p for p in paths if read_json(p).get('audit_status')!='completed']
    if not pending:
        return True
    engine = Engine(cfg['model']['n'], cfg['experiment']['backend'], cfg['model'], cfg['circuit'])
    for index, path in enumerate(pending, 1):
        result = read_json(path)
        state = read_json(path.parent/'search.json')
        if state['status']!='decision_frozen':
            raise ValueError('Exact audit requires a durably frozen architecture decision')
        started = time.perf_counter()
        audit_selected(engine, cfg, result)
        result.update(audit_status='completed', audit_completed_at=stamp(), audit_wall_seconds=time.perf_counter()-started)
        write_json(path, result)
        log(output, f'POST AUDIT {index}/{len(pending)} {result["job"]["name"]}: depth={result["job"]["depth"]}, '
            f'measured_stop_F={result["best_selection_fidelity"]:.7f}, '
            f'exact_HSF={result["audit"]["selected_exact_target_fidelity"]:.7f}')
    return True
