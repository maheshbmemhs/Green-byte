"""Remote experiment I/O; never starts or stops model servers."""
import importlib.util
import json
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'experiment-runner'))
spec = importlib.util.spec_from_file_location('greenlab_local_config', HERE.parent / 'greenlab-llm-local/RunnerConfig.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
LocalConfig = local.RunnerConfig
NETWORKS = {'normal': {}, 'degraded': {'delay': '50ms', 'rate': '20mbit'},
            'bad': {'delay': '100ms', 'rate': '5mbit'}}
ENDPOINTS = {'gemma4b': 'http://127.0.0.1:8083', 'qwen3b': 'http://127.0.0.1:8081',
             'qwen7b': 'http://127.0.0.1:8082'}
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3']
HANDLE = '7a10:'


def ssh(host, command):
    return ['ssh', *SSH_OPTIONS, host, shlex.join(command)]


def fetch_remote_file(host, remote_path, destination):
    """Retry only downloads; publish a file only after successful SSH completion."""
    destination = Path(destination)
    command = ssh(host, ['python3', '-c',
        'import sys; from pathlib import Path; sys.stdout.buffer.write(Path(sys.argv[1]).expanduser().read_bytes())',
        str(remote_path)])
    for attempt in range(1, 4):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent,
                                             prefix=destination.name + '.', suffix='.part',
                                             delete=False) as out:
                temporary = Path(out.name)
                subprocess.run(command, stdout=out, check=True, timeout=120)
            temporary.replace(destination)
            return
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            if attempt == 3:
                raise RuntimeError(
                    f'Download failed after 3 attempts: {host}:{remote_path} -> {destination}: {exc}'
                ) from exc
            print(f'Download attempt {attempt}/3 failed for {remote_path}: {exc}; retrying in 5 seconds',
                  file=sys.stderr, flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        time.sleep(5)


def network_state(interface):
    return json.loads(subprocess.check_output(['tc', '-j', 'qdisc', 'show', 'dev', interface], text=True))


def reset_network(interface):
    # Never remove a qdisc belonging to another application.
    if any(q.get('root') and q.get('handle') == HANDLE for q in network_state(interface)):
        subprocess.run(['sudo', '-n', 'tc', 'qdisc', 'del', 'dev', interface, 'root', 'handle', HANDLE], check=True, timeout=10)


def check_network(interface):
    roots = [q for q in network_state(interface) if q.get('root')]
    if any(q.get('kind') not in ('noqueue', 'mq', 'fq_codel', 'pfifo_fast') or q.get('handle') != '0:' for q in roots):
        raise RuntimeError('Interface has a custom root qdisc; refusing to overwrite it')


def set_network(interface, label):
    reset_network(interface)
    check_network(interface)
    condition = NETWORKS[label]
    if condition:
        subprocess.run(['sudo', '-n', 'tc', 'qdisc', 'replace', 'dev', interface,
                        'root', 'handle', HANDLE, 'netem', 'delay', condition['delay'],
                        'rate', condition['rate']], check=True, timeout=10)


class Profiler:
    """Keep an SSH channel open; EOF or a watchdog stops the profiling helper."""
    def __init__(self, host, command, path, log_path):
        self.host, self.command, self.path = host, command, str(path)
        self.log_path = log_path
        self.process = None
        self.log = None

    def read_reply(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError('Profiler helper timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Profiler helper exited; see {self.log_path}')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    def start(self):
        source = (HERE / 'profiler_worker.py').read_text()
        command = ['python3', '-u', '-c', source]
        if self.host:
            command = ssh(self.host, command)
        self.log = self.log_path.open('w')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True)
        self.process.stdin.write(json.dumps({'command': self.command, 'path': self.path}) + '\n')
        self.process.stdin.flush()
        reply = self.read_reply(30)
        if reply.get('status') != 'ready':
            raise RuntimeError('Profiler did not become ready')
        return reply

    def stop(self):
        try:
            if self.process:
                if self.process.poll() is None:
                    self.process.stdin.close()  # EOF stops the remote watchdog too.
                reply = self.read_reply(30)
                self.process.wait(timeout=10)
                if self.process.returncode or reply.get('status') != 'done':
                    raise RuntimeError('Profiler failed to finish')
                return reply
        finally:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            if self.log:
                self.log.close()
            if self.process:
                self.process.stdout.close()
                if not self.process.stdin.closed:
                    self.process.stdin.close()

    def fetch(self, destination):
        if self.host:
            fetch_remote_file(self.host, self.path, destination)
