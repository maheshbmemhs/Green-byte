"""Client for an SSH lifecycle agent that holds the remote account lock."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

from remote_support import HERE, ssh, fetch_remote_file


class RemoteLifecycle:
    def __init__(self, args, session):
        self.args, self.session = args, session
        self.process = None
        self.log = None
        self.broken = False
        self.current = None
        self.events = []
        self.sequence = 0
        self.response_buffer = b''

    def save(self):
        (self.session / 'server_lifecycle.json').write_text(json.dumps(self.events, indent=2))

    def rpc(self, op, timeout=35, **kwargs):
        if self.broken:
            raise RuntimeError('Remote control channel unavailable; shutdown cannot be confirmed')
        try:
            self.sequence += 1
            request_id = self.sequence
            self.process.stdin.write(json.dumps({'op': op, 'request_id': request_id, **kwargs}) + '\n')
            self.process.stdin.flush()
            deadline = time.monotonic() + timeout
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while True:
                    while b'\n' not in self.response_buffer:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise TimeoutError('Remote lifecycle response timed out')
                        chunk = os.read(self.process.stdout.fileno(), 65536)
                        if not chunk:
                            raise RuntimeError('Remote lifecycle SSH connection closed')
                        self.response_buffer += chunk
                    line, self.response_buffer = self.response_buffer.split(b'\n', 1)
                    reply = json.loads(line)
                    if reply.get('request_id') == request_id:
                        break
                    # A signal may have interrupted an earlier RPC; consume its
                    # reply without mistaking it for confirmation of shutdown.
        except (OSError, ValueError, RuntimeError) as exc:
            self.broken = True
            self.events.append({'op': op, 'time': time.time(), 'error': str(exc), 'shutdown_confirmed': False})
            self.save()
            raise RuntimeError('Remote control failed; shutdown cannot be confirmed. See server_lifecycle.json for recovery identifiers.') from exc
        if reply.get('current'):
            self.current = reply['current']
        if 'error' in reply:
            self.events.append({'op': op, 'time': time.time(), **reply})
            self.save()
            raise RuntimeError(reply['error'])
        if op != 'status':
            self.events.append({'op': op, 'time': time.time(), 'result': reply['result']})
            self.save()
        return reply['result']

    def open(self):
        source = (HERE / 'server_worker.py').read_text()
        self.log = (self.session / 'server_control.log').open('w')
        self.process = subprocess.Popen(ssh(self.args.remote_host, ['python3', '-u', '-c', source]),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
        self.rpc('init', config={'llama_dir': self.args.remote_llama_dir, 'log_dir': self.args.remote_log_dir,
                                'session': self.session.name, 'ctx_size': self.args.ctx_size,
                                'gpu_layers_by_model': self.args.gpu_layers_by_model})

    def cleanup_legacy(self):
        return self.rpc('legacy', timeout=90)

    def start(self, model):
        # Persist predictable recovery identifiers even if the SSH start reply is lost.
        self.events.append({'op': 'start_requested', 'model': model,
                            'tmux_session': 'greenlab-' + self.session.name + '-' + model,
                            'remote_directory': self.args.remote_log_dir + '/' + self.session.name + '/' + model,
                            'time': time.time()})
        self.save()
        self.current = self.rpc('start', model=model)
        return self.current

    def status(self):
        return self.rpc('status')

    def stop(self):
        result = self.rpc('stop')
        if result:
            self.current = result
        return result

    def fetch_log(self, model):
        if not self.current or self.current.get('model') != model:
            return
        directory = self.session / model
        directory.mkdir(exist_ok=True)
        fetch_remote_file(self.args.remote_host, self.current['log'], directory / 'server.log')

    def close(self):
        try:
            if self.process and not self.broken and self.process.poll() is None:
                self.rpc('close')
        finally:
            if self.process:
                if not self.process.stdin.closed:
                    self.process.stdin.close()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                self.process.stdout.close()
            if self.log:
                self.log.close()
