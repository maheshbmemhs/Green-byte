"""Standalone standard-library helper, sent to the remote host over SSH."""
import json
from pathlib import Path
import select
import subprocess
import sys
import time


def main():
    config = json.loads(sys.stdin.readline())
    path = Path(config['path']).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = path.with_suffix('.stop')
    ready = path.with_suffix('.ready')
    # Unique paths prevent stale flags and concurrent measurement collisions.
    if any(p.exists() for p in (path, stop, ready)):
        raise RuntimeError('Profiler output already exists')
    child_code = (
        'from pathlib import Path; import time; '
        f'Path({str(ready)!r}).touch(); end=time.monotonic()+240\n'
        f'while not Path({str(stop)!r}).exists() and time.monotonic()<end: time.sleep(0.05)'
    )
    command = config['command'] + ['--gpu', '-o', str(path), '--', 'python3', '-c', child_code]
    process = subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr)
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() or not path.exists() or path.stat().st_size == 0:
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('EnergiBridge failed to start or write samples')
            time.sleep(0.1)
        print(json.dumps({'status': 'ready', 'time': time.time(), 'pid': process.pid}), flush=True)
        deadline = time.monotonic() + 240
        while process.poll() is None:
            if select.select([sys.stdin], [], [], 0.2)[0]:
                sys.stdin.readline()  # STOP or EOF
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Profiler watchdog expired')
    finally:
        stop.touch()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
    if process.returncode:
        raise RuntimeError(f'EnergiBridge exited {process.returncode}')
    print(json.dumps({'status': 'done', 'time': time.time()}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
        sys.exit(1)
