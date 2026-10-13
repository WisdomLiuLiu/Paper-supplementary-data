"""Atomic checkpoint output with bounded retries for Windows file locks."""
import json
import os
from pathlib import Path
import sys
import time


def replace_with_retry(source, destination, max_attempts=12):
    """Keep both files intact on failure; never unlink the old checkpoint."""
    for attempt in range(1, max_attempts + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            retryable = getattr(exc, 'winerror', None) in (5, 32, 33)
            if not retryable or attempt == max_attempts:
                if retryable:
                    print(f'CHECKPOINT ERROR: replacement still blocked after {attempt} attempts; '
                          f'old checkpoint and temporary file preserved: {destination}',
                          file=sys.stderr, flush=True)
                raise
            delay = min(0.05 * 2**(attempt - 1), 1.0)
            if attempt == 1 or attempt % 5 == 0:
                print(f'CHECKPOINT RETRY {attempt}/{max_attempts}: Windows temporarily denied '
                      f'file replacement; waiting {delay:.2f}s: {destination}',
                      file=sys.stderr, flush=True)
            time.sleep(delay)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    # Serialize before opening the temporary file so invalid data cannot truncate it.
    serialized = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.write_text(serialized, encoding='utf-8')
    replace_with_retry(temporary, path)
