import csv
import importlib.util
import io
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
spec = importlib.util.spec_from_file_location('remote_launcher_test', HERE / 'run_experiments.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
Config = launcher.RunnerConfig
support = sys.modules['remote_support']
module = launcher.config_module


class RemoteTests(unittest.TestCase):
    def test_server_comparison_ignores_volatile_metadata(self):
        before = {'context_size': 8192, 'models': {'data': [
            {'id': 'qwen-model', 'created': 100}, {'id': 'alias', 'created': 100}]}}
        after = {'context_size': 8192, 'models': {'data': [
            {'id': 'alias', 'created': 200}, {'id': 'qwen-model', 'created': 200}]}}
        self.assertTrue(launcher.same_server_configuration(after, before))
        after['context_size'] = 4096
        self.assertFalse(launcher.same_server_configuration(after, before))
        after['context_size'] = 8192
        after['models']['data'][1]['id'] = 'different-model'
        self.assertFalse(launcher.same_server_configuration(after, before))

    def test_matrix(self):
        for reps, count, measured in ((10, 180, False), (30, 540, True), (1, 18, False), (2, 36, True)):
            with patch('random.shuffle') as shuffle:
                rows = Config.build_run_table_model('qwen3b', reps, measured).generate_experiment_run_table()
            self.assertEqual(len(rows), count)
            self.assertEqual(shuffle.call_count, int(measured))
            self.assertEqual({r['network_condition'] for r in rows}, {'normal', 'degraded', 'bad'})
            self.assertEqual(len({(r['prompt_type'], r['max_tokens'], r['network_condition']) for r in rows}), 18)

    def test_network_ownership(self):
        with patch.object(support, 'network_state', return_value=[{'root': True, 'kind': 'netem', 'handle': '123:'}]), \
             patch.object(support.subprocess, 'run') as run:
            support.reset_network('wifi')
            run.assert_not_called()
            with self.assertRaises(RuntimeError):
                support.check_network('wifi')
        with patch.object(support, 'network_state', return_value=[{'root': True, 'kind': 'fq_codel', 'handle': '0:'}]), \
             patch.object(support.subprocess, 'run') as run:
            support.set_network('wifi', 'bad')
            self.assertIn('100ms', run.call_args.args[0])
            self.assertIn('5mbit', run.call_args.args[0])
        with patch.object(support, 'network_state', return_value=[{'root': True, 'kind': 'netem', 'handle': support.HANDLE}]), \
             patch.object(support.subprocess, 'run') as run:
            support.reset_network('wifi')
            self.assertIn('del', run.call_args.args[0])

    def test_endpoint_identity(self):
        replies = [{'status': 'ok'}, {'modalities': {'vision': True}, 'default_generation_settings': {'n_ctx': 8192}},
                   {'data': [{'id': 'ggml-org/Qwen2.5-VL-3B-Instruct-GGUF'}]}]
        with patch.object(launcher, 'get_json', side_effect=replies):
            self.assertEqual(launcher.inspect_server('qwen3b', 'http://localhost')['context_size'], 8192)
        with patch.object(launcher, 'get_json', side_effect=replies):
            with self.assertRaises(RuntimeError):
                launcher.inspect_server('qwen7b', 'http://localhost')

    def make_config(self, tmp, phase):
        settings = {'endpoints': {'qwen3b': 'http://localhost:8081'}, 'interface': 'wifi',
                    'contexts': {'qwen3b': 8192}, 'remote_host': 'user@host', 'remote_log_dir': 'logs',
                    'local_energy_command': ['energibridge'], 'remote_energy_command': ['energibridge']}
        (Path(tmp) / 'session.json').write_text(json.dumps(settings))
        env = dict(GREENLAB_MODEL='qwen3b', GREENLAB_PHASE=phase, GREENLAB_SESSION_DIR=tmp,
                   GREENLAB_SERVER_PID=str(os.getpid()), GREENLAB_IMAGE='/unused.png', GREENLAB_CTX_SIZE='8192')
        with patch.dict(os.environ, env):
            return Config()

    def test_request_cleanup_and_measurement(self):
        for phase, fail, rollover in (('warmup', False, False), ('main', False, False),
                                       ('main', True, False), ('main', False, True)):
            with self.subTest(phase=phase, fail=fail, rollover=rollover), tempfile.TemporaryDirectory() as tmp:
                config = self.make_config(tmp, phase)
                directory = Path(tmp) / 'run'
                context = SimpleNamespace(run_dir=directory, execute_run={
                    'model': 'qwen3b', 'prompt_type': 'text', 'max_tokens': 120, 'network_condition': 'degraded'})
                csv_data = 'CPU_USAGE_0,USED_MEMORY,PACKAGE_ENERGY (J)\n10,1048576,1\n20,2097152,3\n'
                if rollover:
                    csv_data = csv_data.replace('1048576,1', '1048576,262143')
                instances = []
                def profiler(host, command, path, log):
                    obj = Mock()
                    obj.start.return_value = {'status': 'ready'}
                    obj.stop.return_value = {'status': 'done'}
                    if not host:
                        Path(path).write_text(csv_data)
                    obj.fetch.side_effect = lambda destination: destination.write_text(csv_data)
                    instances.append(obj)
                    return obj
                def request(*a, **kw):
                    if fail:
                        raise subprocess.TimeoutExpired('curl', 180)
                    (directory / 'response.json').write_text(json.dumps({'usage': {'prompt_tokens': 50, 'completion_tokens': 120},
                        'choices': [{'message': {'content': 'Photosynthesis'}}]}))
                with patch.object(config, 'server_ready', return_value=True), \
                     patch.object(module, 'set_network') as shape, patch.object(module, 'reset_network') as reset, \
                     patch.object(module, 'Profiler', side_effect=profiler) as prof, \
                     patch.object(module.time, 'sleep'), patch.object(module.subprocess, 'run', side_effect=request):
                    if fail:
                        with self.assertRaises(subprocess.TimeoutExpired):
                            config.interact(context)
                        self.assertTrue((Path(tmp) / 'failure.json').exists())
                    else:
                        config.interact(context)
                    shape.assert_called_once_with('wifi', 'degraded')
                    reset.assert_called_once_with('wifi')
                    self.assertEqual(prof.call_count, 2 if phase == 'main' else 0)
                    for obj in instances:
                        obj.stop.assert_called_once()
                if phase == 'main' and not fail:
                    self.assertEqual(config.metrics['remote_mean_energy_usage'], '' if rollover else 2)
                    if rollover:
                        self.assertIn('rollover/reset', config.metrics['remote_energy_status'])
                    self.assertTrue((directory / 'remote_energy.csv').exists())
                payload = json.loads((directory / 'response.request.json').read_text())
                self.assertFalse(payload['cache_prompt'])
                self.assertIn('100 words', payload['messages'][0]['content'])

    def test_helper_with_fake_profiler(self):
        # Exercise the real helper protocol and EOF cleanup without privileges or SSH.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fake = directory / 'fake.py'
            fake.write_text("import sys, subprocess\nfrom pathlib import Path\na=sys.argv\nPath(a[a.index('-o')+1]).write_text('Time,PACKAGE_ENERGY (J)\\n0,1\\n200,2\\n')\nsys.exit(subprocess.call(a[a.index('--')+1:]))\n")
            profiler = support.Profiler(None, [sys.executable, str(fake)], directory / 'energy.csv', directory / 'log')
            self.assertEqual(profiler.start()['status'], 'ready')
            self.assertEqual(profiler.stop()['status'], 'done')
            self.assertEqual(profiler.process.returncode, 0)

    def test_completion_network_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = Config.build_run_table_model('qwen3b', 1).generate_experiment_run_table()
            for row in rows:
                row['__done'] = 'DONE'
            def write():
                with (Path(tmp) / 'run_table.csv').open('w') as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            launcher.verify_completion(Path(tmp), 'qwen3b', 1)
            rows[0]['network_condition'] = 'bad'
            write()
            with self.assertRaises(RuntimeError):
                launcher.verify_completion(Path(tmp), 'qwen3b', 1)

    def test_setup_failure_still_resets_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(tmp, 'main')
            context = SimpleNamespace(run_dir=Path(tmp) / 'run', execute_run={
                'model': 'qwen3b', 'prompt_type': 'text', 'max_tokens': 120, 'network_condition': 'bad'})
            with patch.object(config, 'server_ready', return_value=True), \
                 patch.object(module, 'set_network', side_effect=RuntimeError('tc failed')), \
                 patch.object(module, 'reset_network') as reset, patch.object(module, 'Profiler') as profiler:
                with self.assertRaisesRegex(RuntimeError, 'tc failed'):
                    config.interact(context)
                reset.assert_called_once()
                profiler.assert_not_called()
            self.assertTrue((Path(tmp) / 'failure.json').exists())

    def test_launcher_selected_models_and_phase_order(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'image.png'
            Image.new('RGB', (8, 8)).save(image)
            args = ['launcher', '--image', str(image), '--models', 'qwen3b', 'qwen7b',
                    '--results-dir', str(Path(tmp) / 'results'), '--interface', 'testwifi']
            server = {'context_size': 8192, 'models': {'data': []}, 'props': {}}
            with patch.object(sys, 'argv', args), patch.object(launcher, 'inspect_server', return_value=server), \
                 patch.object(launcher, 'check_network'), patch.object(launcher, 'reset_network'), \
                 patch.object(launcher.time, 'sleep'), patch.object(launcher, 'run_phase') as phase:
                launcher.main()
            self.assertEqual([(c.args[1], c.args[2]) for c in phase.call_args_list],
                             [('qwen3b', 'warmup'), ('qwen3b', 'main'), ('qwen7b', 'warmup'), ('qwen7b', 'main')])
            session = next((Path(tmp) / 'results').iterdir())
            settings = json.loads((session / 'session.json').read_text())
            self.assertEqual(settings['status'], 'complete')
            self.assertEqual(settings['model_order'], ['qwen3b', 'qwen7b'])
            self.assertEqual(settings['main_repetitions'], 30)


if __name__ == '__main__':
    unittest.main()
