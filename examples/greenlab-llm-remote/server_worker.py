"""Standard-library remote lifecycle agent. JSON-line RPC over an owned SSH channel."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

PORTS = {'gemma4b': 8083, 'qwen3b': 8081, 'qwen7b': 8082}
REPOS = {'gemma4b': 'ggml-org/gemma-3-4b-it-GGUF',
         'qwen3b': 'ggml-org/Qwen2.5-VL-3B-Instruct-GGUF',
         'qwen7b': 'ggml-org/Qwen2.5-VL-7B-Instruct-GGUF'}
LEGACY = ('qwen3b', 'qwen7b', 'gemma')


def process_identity(pid):
    try:
        stat = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': stat[19], 'ppid': int(stat[1])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def is_same_process(identity):
    current = process_identity(identity['pid'])
    return current is not None and current['start_ticks'] == identity['start_ticks']


def stop_identities(identities, grace=15):
    # Start ticks prevent signaling a recycled PID. Recheck before every signal.
    for identity in identities:
        if is_same_process(identity):
            try:
                os.kill(identity['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace
    while any(is_same_process(i) for i in identities) and time.monotonic() < deadline:
        time.sleep(0.1)
    for identity in identities:
        if is_same_process(identity):
            try:
                os.kill(identity['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    while any(is_same_process(i) for i in identities) and time.monotonic() < deadline:
        time.sleep(0.1)
    if any(is_same_process(i) for i in identities):
        raise RuntimeError('Recorded model process did not exit')


def ports_free():
    for port in PORTS.values():
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', port))
                sock.listen()
            except OSError as exc:
                raise RuntimeError(f'Remote port {port} still occupied; refusing to stop an unrelated listener') from exc


def tmux(*args, check=True):
    return subprocess.run(['tmux', *args], check=check, capture_output=True, text=True, timeout=10)


def has_session(name):
    result = tmux('has-session', '-t', '=' + name, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.returncode == 0


def session_id(name):
    # display-message -t resolves a target pane, not an exact target session.
    # Listing sessions avoids an empty format result for '=session-name'.
    result = tmux('list-sessions', '-F', '#{session_name}\t#{session_id}', check=False)
    for line in result.stdout.splitlines():
        fields = line.split('\t', 1)
        if len(fields) == 2 and fields[0] == name:
            return fields[1]
    return None


class Agent:
    def __init__(self):
        self.lock = None
        self.current = None
        self.records = []
        self.root = None

    def save(self):
        if self.root:
            (self.root / 'lifecycle.json').write_text(json.dumps(self.records, indent=2))

    def initialize(self, config):
        if self.lock:
            raise RuntimeError('Agent is already initialized')
        if not shutil.which('tmux'):
            raise RuntimeError('tmux is not installed')
        self.llama_dir = Path(config['llama_dir']).expanduser().resolve()
        self.binary = self.llama_dir / 'build-cuda/bin/llama-server'
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise RuntimeError(f'Executable missing: {self.binary}')
        lock_path = Path.home() / '.cache/greenlab/remote-experiment.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('Another experiment holds the remote account lock')
        self.lock = lock
        self.config = config
        self.root = (Path(config['log_dir']).expanduser() / config['session']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock.seek(0)
        self.lock.truncate()
        self.lock.write(json.dumps({'agent_pid': os.getpid(), 'session': config['session'], 'root': str(self.root)}))
        self.lock.flush()
        return {'root': str(self.root), 'agent': process_identity(os.getpid())}

    def cleanup_legacy(self):
        if not self.lock:
            raise RuntimeError('Remote lock required')
        for name in LEGACY:
            if not has_session(name):
                continue
            panes = [int(pid) for pid in tmux('list-panes', '-s', '-t', '=' + name, '-F', '#{pane_pid}').stdout.split()]
            tree = {}
            for entry in Path('/proc').iterdir():
                if entry.name.isdigit():
                    identity = process_identity(int(entry.name))
                    if identity:
                        tree[identity['pid']] = identity
            owned = set(panes)
            while True:
                expanded = owned | {pid for pid, info in tree.items() if info['ppid'] in owned}
                if expanded == owned:
                    break
                owned = expanded
            identities = [tree[pid] for pid in owned if pid in tree]
            record = {'legacy_session': name, 'processes': identities, 'stopping_at': time.time()}
            self.records.append(record)
            self.save()
            stop_identities(identities)
            if has_session(name):
                tmux('kill-session', '-t', '=' + name)
            if any(is_same_process(i) for i in identities):
                raise RuntimeError(f'Legacy session {name} did not stop')
            record['stopped_at'] = time.time()
            self.save()
        ports_free()
        return self.records

    def start(self, model):
        if not self.lock or self.current:
            raise RuntimeError('Remote lock required and previous model must be stopped')
        ports_free()
        directory = self.root / model
        directory.mkdir(exist_ok=False)
        name = 'greenlab-' + self.config['session'] + '-' + model
        command = [str(self.binary), '-hf', REPOS[model], '--host', '127.0.0.1', '--port', str(PORTS[model]),
                   '-ngl', str(self.config['gpu_layers_by_model'][model]), '-c', str(self.config['ctx_size']),
                   '--no-context-shift', '-np', '1']
        record = {'model': model, 'tmux_session': name, 'command': command,
                  'log': str(directory / 'server.log'), 'identity_file': str(directory / 'process.json'),
                  'starting_at': time.time()}
        self.current = record
        self.records.append(record)
        self.save()
        # Record PID/start ticks before exec; exec preserves both and avoids shell PID ambiguity.
        bootstrap = (
            "import os,json; from pathlib import Path; "
            "s=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split(); "
            f"Path({record['identity_file']!r}).write_text(json.dumps({{'pid':os.getpid(),'start_ticks':s[19]}})); "
            f"os.chdir({str(self.llama_dir)!r}); os.execv({str(self.binary)!r}, {command!r})"
        )
        launch = shlex.join(['python3', '-c', bootstrap]) + ' > ' + shlex.quote(record['log']) + ' 2>&1'
        record['tmux_id'] = tmux('new-session', '-d', '-P', '-F', '#{session_id}', '-s', name, 'exec ' + launch).stdout.strip()
        self.save()
        deadline = time.monotonic() + 10
        while not Path(record['identity_file']).exists():
            if not has_session(name) or time.monotonic() > deadline:
                raise RuntimeError(f'Model failed to start; see {record["log"]}')
            time.sleep(0.1)
        record['process'] = json.loads(Path(record['identity_file']).read_text())
        self.save()
        self.status()
        return record

    def status(self):
        if not self.current or not self.current.get('process') or not is_same_process(self.current['process']):
            raise RuntimeError('Owned model exited or its process identity changed')
        return self.current

    def stop(self):
        if not self.current:
            return None
        record = self.current
        # Covers a lost start reply or an interruption while the PID file was being written.
        if not record.get('process') and Path(record['identity_file']).exists():
            record['process'] = json.loads(Path(record['identity_file']).read_text())
        if has_session(record['tmux_session']):
            actual_id = session_id(record['tmux_session'])
            if actual_id is not None and actual_id != record.get('tmux_id'):
                raise RuntimeError('tmux session identity changed; refusing to stop unrelated session')
        if record.get('process'):
            current = process_identity(record['process']['pid'])
            if current and not is_same_process(record['process']):
                raise RuntimeError('PID was reused; refusing to stop unrelated process')
            stop_identities([record['process']])
        if has_session(record['tmux_session']):
            tmux('kill-session', '-t', '=' + record['tmux_session'])
        if record.get('process') and is_same_process(record['process']):
            raise RuntimeError('Owned model did not exit')
        record['stopped_at'] = time.time()
        self.save()
        self.current = None
        ports_free()
        return record

    def close(self):
        try:
            self.stop()
        finally:
            if self.lock:
                self.lock.close()
                self.lock = None


def main():
    agent = Agent()
    try:
        for line in sys.stdin:
            request_id = None
            try:
                request = json.loads(line)
                request_id = request.get('request_id')
                operation = request['op']
                if operation == 'init':
                    result = agent.initialize(request['config'])
                elif operation == 'legacy':
                    result = agent.cleanup_legacy()
                elif operation == 'start':
                    result = agent.start(request['model'])
                elif operation == 'status':
                    result = agent.status()
                elif operation == 'stop':
                    result = agent.stop()
                elif operation == 'close':
                    agent.close()
                    print(json.dumps({'result': None, 'request_id': request_id}), flush=True)
                    return
                else:
                    raise ValueError('Unknown operation')
                print(json.dumps({'result': result, 'request_id': request_id}), flush=True)
            except Exception as exc:
                print(json.dumps({'error': str(exc), 'current': agent.current, 'request_id': request_id}), flush=True)
    finally:
        agent.close()  # EOF/SSH disconnect also stops the owned model.


if __name__ == '__main__':
    main()
