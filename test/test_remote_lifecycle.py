import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parents[1] / 'examples/greenlab-llm-remote'
sys.path.insert(0, str(HERE))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

worker = load('server_worker_test', HERE / 'server_worker.py')
launcher = load('lifecycle_launcher_test', HERE / 'run_experiments.py')
from server_lifecycle import RemoteLifecycle


class LifecycleTests(unittest.TestCase):
    def test_server_log_download_uses_shared_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = RemoteLifecycle(SimpleNamespace(remote_host='host'), Path(tmp))
            manager.current = {'model': 'qwen7b', 'log': '/remote/server.log'}
            with patch('server_lifecycle.fetch_remote_file') as fetch:
                manager.fetch_log('qwen7b')
                fetch.assert_called_once_with('host', '/remote/server.log', Path(tmp) / 'qwen7b/server.log')

    def test_rpc_does_not_confuse_old_reply_with_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = RemoteLifecycle(SimpleNamespace(remote_host='unused'), Path(tmp))
            code = '''import json,sys
for line in sys.stdin:
    r=json.loads(line)
    print(json.dumps({'request_id': r['request_id']-1, 'result': 'stale'}), flush=True)
    print(json.dumps({'request_id': r['request_id'], 'result': r['op']}), flush=True)
    if r['op']=='close': break
'''
            manager.process = subprocess.Popen([sys.executable, '-u', '-c', code],
                                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(manager.rpc('stop'), 'stop')
            finally:
                manager.close()

    def test_missing_forward_fails_before_remote_changes(self):
        with patch.object(launcher.socket, 'create_connection', side_effect=ConnectionRefusedError), \
             patch.object(launcher, 'RemoteLifecycle') as manager:
            with self.assertRaisesRegex(RuntimeError, 'SSH forward on localhost:8081'):
                launcher.check_forwarded_ports(['qwen3b'])
            manager.assert_not_called()

    def test_start_command_and_exclusive_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = worker.Agent()
            agent.root = Path(tmp)
            agent.lock = Mock()
            agent.config = {'session': 'session', 'ctx_size': 8192, 'gpu_layers_by_model': {'qwen3b': 99}}
            agent.llama_dir = Path(tmp) / 'llama'
            agent.binary = agent.llama_dir / 'build-cuda/bin/llama-server'
            def start_tmux(*args, **kwargs):
                Path(agent.current['identity_file']).write_text(json.dumps({'pid': 123, 'start_ticks': '5'}))
                return SimpleNamespace(stdout='$7')
            with patch.object(worker, 'ports_free') as ports, patch.object(worker, 'tmux', side_effect=start_tmux), \
                 patch.object(agent, 'status'):
                result = agent.start('qwen3b')
                command = result['command']
                self.assertEqual(command[command.index('-hf') + 1], 'ggml-org/Qwen2.5-VL-3B-Instruct-GGUF')
                self.assertEqual(command[command.index('--port') + 1], '8081')
                self.assertEqual(command[command.index('-c') + 1], '8192')
                self.assertEqual(command[command.index('-ngl') + 1], '99')
                self.assertEqual(command[command.index('-np') + 1], '1')
                self.assertIn('--no-context-shift', command)
                with self.assertRaisesRegex(RuntimeError, 'previous model must be stopped'):
                    agent.start('qwen7b')
                ports.assert_called_once()

    def test_default_gpu_layers_reach_remote_commands(self):
        for model, count in {'gemma4b': 999, 'qwen3b': 999, 'qwen7b': 20}.items():
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(remote_host='unused', remote_llama_dir='~/llama.cpp',
                                       remote_log_dir='logs', ctx_size=8192,
                                       gpu_layers_by_model=dict(launcher.RunnerConfig.GPU_LAYERS))
                manager = RemoteLifecycle(args, Path(tmp))
                with patch('server_lifecycle.subprocess.Popen'), patch.object(manager, 'rpc') as rpc:
                    manager.open()
                    config = rpc.call_args.kwargs['config']
                manager.log.close()
                agent = worker.Agent()
                agent.root = Path(tmp)
                agent.lock = Mock()
                agent.config = config
                agent.llama_dir = Path(tmp)
                agent.binary = Path(tmp) / 'llama-server'
                def start_tmux(*args, **kwargs):
                    Path(agent.current['identity_file']).write_text(
                        json.dumps({'pid': 123, 'start_ticks': '5'}))
                    return SimpleNamespace(stdout='$7')
                with patch.object(worker, 'ports_free'), patch.object(worker, 'tmux', side_effect=start_tmux), patch.object(agent, 'status'):
                    command = agent.start(model)['command']
                self.assertEqual(command[command.index('-ngl') + 1], str(count))

    def test_stops_actual_owned_process(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            identity = worker.process_identity(process.pid)
            self.assertTrue(worker.is_same_process(identity))
            worker.stop_identities([identity], grace=1)
            process.wait(timeout=3)
            self.assertFalse(worker.is_same_process(identity))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_recycled_pid_is_not_signaled(self):
        with patch.object(worker, 'process_identity', return_value={'pid': 50, 'start_ticks': 'new'}), \
             patch.object(worker.os, 'kill') as kill:
            worker.stop_identities([{'pid': 50, 'start_ticks': 'old'}], grace=0)
            kill.assert_not_called()

    def test_force_stop_only_owned_identity(self):
        identity = {'pid': 50, 'start_ticks': 'same'}
        alive = [True]
        def kill(pid, sig):
            if sig == worker.signal.SIGKILL:
                alive[0] = False
        with patch.object(worker, 'is_same_process', side_effect=lambda _: alive[0]), \
             patch.object(worker.os, 'kill', side_effect=kill) as signal:
            worker.stop_identities([identity], grace=0)
            self.assertEqual([c.args for c in signal.call_args_list],
                             [(50, worker.signal.SIGTERM), (50, worker.signal.SIGKILL)])

    def test_remote_lock_prevents_second_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            binary = home / 'llama/build-cuda/bin/llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o700)
            config = dict(llama_dir=str(home / 'llama'), log_dir=str(home / 'logs'), session='session', ctx_size=8192, gpu_layers=99)
            first, second = worker.Agent(), worker.Agent()
            with patch.object(worker.Path, 'home', return_value=home), patch.object(worker.shutil, 'which', return_value='/tmux'):
                try:
                    first.initialize(config)
                    with self.assertRaisesRegex(RuntimeError, 'holds the remote account lock'):
                        second.initialize(config)
                finally:
                    first.close()
                    second.close()

    def test_legacy_cleanup_only_known_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = worker.Agent()
            agent.lock = Mock()
            agent.root = Path(tmp)
            live = set(worker.LEGACY)
            calls = []
            def tmux(*args, **kwargs):
                calls.append(args)
                if args[0] == 'list-panes':
                    return SimpleNamespace(stdout=str(os.getpid()))
                if args[0] == 'kill-session':
                    live.remove(args[-1][1:])
                return SimpleNamespace(stdout='')
            with patch.object(worker, 'has_session', side_effect=lambda name: name in live), \
                 patch.object(worker, 'tmux', side_effect=tmux), patch.object(worker, 'stop_identities') as stop, \
                 patch.object(worker, 'is_same_process', return_value=False), patch.object(worker, 'ports_free') as free:
                agent.cleanup_legacy()
            self.assertEqual(stop.call_count, 3)
            self.assertEqual([c[-1] for c in calls if c[0] == 'kill-session'], ['=qwen3b', '=qwen7b', '=gemma'])
            free.assert_called_once()

    def test_busy_port_does_not_kill_anything(self):
        sock = Mock()
        sock.bind.side_effect = OSError('occupied')
        factory = Mock()
        factory.__enter__ = Mock(return_value=sock)
        factory.__exit__ = Mock(return_value=False)
        with patch.object(worker.socket, 'socket', return_value=factory), patch.object(worker.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'unrelated listener'):
                worker.ports_free()
            kill.assert_not_called()

    def test_managed_failure_always_stops_and_fetches(self):
        for failure in (RuntimeError('inference failed'), KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as tmp:
                manager = Mock()
                manager.start.return_value = {'process': {'pid': 5}}
                manager.stop.return_value = {'stopped_at': 123}
                settings = {'servers': {}, 'contexts': {}}
                args = SimpleNamespace(interface='wifi', skip_warmup=False)
                with patch.object(launcher, 'wait_model_ready', return_value={'context_size': 8192}), \
                     patch.object(launcher, 'reset_network'), patch.object(launcher, 'run_phase', side_effect=failure):
                    with self.assertRaises(type(failure)):
                        launcher.run_managed_model(manager, Path(tmp), 'qwen3b', args, settings)
                manager.stop.assert_called_once()
                manager.fetch_log.assert_called_once_with('qwen3b')

    def test_start_failure_still_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = Mock()
            manager.start.side_effect = RuntimeError('startup failed')
            manager.stop.return_value = None
            with patch.object(launcher, 'reset_network'):
                with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                    launcher.run_managed_model(manager, Path(tmp), 'qwen3b', SimpleNamespace(interface='wifi', skip_warmup=False), {})
            manager.stop.assert_called_once()

    def test_readiness_timeout_and_context_mismatch(self):
        manager = Mock()
        args = SimpleNamespace(startup_timeout=1, ctx_size=8192)
        with patch.object(launcher.time, 'monotonic', side_effect=[0, 0, 2]), \
             patch.object(launcher.time, 'sleep'), patch.object(launcher, 'inspect_server', side_effect=OSError('loading')):
            with self.assertRaises(TimeoutError):
                launcher.wait_model_ready(manager, 'qwen3b', 'url', args)
        with patch.object(launcher, 'inspect_server', return_value={'context_size': 4096}):
            with self.assertRaisesRegex(RuntimeError, 'actual context'):
                launcher.wait_model_ready(manager, 'qwen3b', 'url', args)

    def test_changed_tmux_identity_not_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = worker.Agent()
            agent.root = Path(tmp)
            agent.current = {'tmux_session': 'owned', 'tmux_id': '$1',
                             'process': {'pid': 99, 'start_ticks': '1'}, 'identity_file': str(Path(tmp) / 'pid')}
            with patch.object(worker, 'has_session', return_value=True), \
                 patch.object(worker, 'tmux', return_value=SimpleNamespace(stdout='owned\t$2\n')), \
                 patch.object(worker, 'stop_identities') as stop:
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    agent.stop()
                stop.assert_not_called()

    def test_exact_session_identity_lookup(self):
        with patch.object(worker, 'tmux', return_value=SimpleNamespace(stdout='owned-other\t$1\nowned\t$7\n')):
            self.assertEqual(worker.session_id('owned'), '$7')
            self.assertIsNone(worker.session_id('missing'))


if __name__ == '__main__':
    unittest.main()
