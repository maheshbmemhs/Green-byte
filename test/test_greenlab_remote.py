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

    def test_remote_word_targets_and_payloads(self):
        self.assertEqual(support.LocalConfig.TOKEN_LIMITS, Config.TOKEN_LIMITS)
        self.assertEqual(support.LocalConfig.WORD_TARGETS, Config.WORD_TARGETS)
        self.assertEqual(support.LocalConfig.PROMPTS, Config.PROMPTS)
        self.assertEqual(Config.TOKEN_LIMITS, [2100, 1100, 300])
        with tempfile.TemporaryDirectory() as tmp:
            for phase in ('warmup', 'main'):
                config = self.make_config(tmp, phase)
                config.image_path = Path(tmp) / 'image.png'
                config.image_path.write_bytes(b'image fixture')
                for prompt_type in ('text', 'image'):
                    for tokens, words in ((2100, 1000), (1100, 500), (300, 100)):
                        with self.subTest(phase=phase, prompt_type=prompt_type, tokens=tokens):
                            path = config.send_request(prompt_type, tokens, Path(tmp) / 'response.json')
                            payload = json.loads(path.read_text())
                            content = payload['messages'][0]['content']
                            text = content if prompt_type == 'text' else content[0]['text']
                            if prompt_type == 'image':
                                self.assertIn(f'Aim for {words} words, up to 10% more.', text)
                                self.assertIn('the inputs and outputs, the stages of the process', text)
                                self.assertIn('additional scientific explanation', text)
                                self.assertEqual(content[1]['image_url']['url'],
                                                 'data:image/png;base64,aW1hZ2UgZml4dHVyZQ==')
                            else:
                                self.assertEqual(text, support.LocalConfig.PROMPTS['text'].format(word_count=words))
                                self.assertIn(f'{words} words', text)
                            self.assertNotIn('{word_count}', text)
                            self.assertEqual(payload['max_tokens'], tokens)
                            self.assertFalse(payload['cache_prompt'])

    def test_download_retries_and_atomic_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'energy.csv'
            destination.write_bytes(b'previous valid data')
            attempts = []
            def transfer(command, stdout, check, timeout):
                self.assertEqual(timeout, 120)
                self.assertTrue(check)
                self.assertIn('ConnectTimeout=30', command)
                self.assertEqual(destination.read_bytes(), b'previous valid data')
                attempts.append(command)
                stdout.write(b'partial' if len(attempts) < 3 else b'complete csv')
                if len(attempts) == 1:
                    raise subprocess.TimeoutExpired(command, timeout)
                if len(attempts) == 2:
                    raise subprocess.CalledProcessError(255, command)
            with patch.object(support.subprocess, 'run', side_effect=transfer), patch.object(support.time, 'sleep') as sleep:
                support.fetch_remote_file('host', 'remote.csv', destination)
            self.assertEqual(len(attempts), 3)
            self.assertEqual([c.args for c in sleep.call_args_list], [(5,), (5,)])
            self.assertEqual(destination.read_bytes(), b'complete csv')
            self.assertEqual(list(Path(tmp).glob('*.part')), [])

    def test_download_exhaustion_preserves_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'energy.csv'
            destination.write_bytes(b'valid')
            with patch.object(support.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ssh', 120)) as run, patch.object(support.time, 'sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
                    support.fetch_remote_file('host', 'remote.csv', destination)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(destination.read_bytes(), b'valid')
            self.assertEqual(list(Path(tmp).glob('*.part')), [])

    def test_download_interruption_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'energy.csv'
            with patch.object(support.subprocess, 'run', side_effect=KeyboardInterrupt), patch.object(support.time, 'sleep') as sleep:
                with self.assertRaises(KeyboardInterrupt):
                    support.fetch_remote_file('host', 'remote.csv', destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])
            sleep.assert_not_called()

    def test_profiler_download_uses_shared_transfer(self):
        with patch.object(support, 'fetch_remote_file') as fetch:
            profiler = support.Profiler('host', [], 'remote.csv', Path('unused.log'))
            profiler.fetch(Path('energy.csv'))
            fetch.assert_called_once_with('host', 'remote.csv', Path('energy.csv'))

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
                    'model': 'qwen3b', 'prompt_type': 'text', 'max_tokens': 300, 'network_condition': 'degraded'})
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
                'model': 'qwen3b', 'prompt_type': 'text', 'max_tokens': 300, 'network_condition': 'bad'})
            with patch.object(config, 'server_ready', return_value=True), \
                 patch.object(module, 'set_network', side_effect=RuntimeError('tc failed')), \
                 patch.object(module, 'reset_network') as reset, patch.object(module, 'Profiler') as profiler:
                with self.assertRaisesRegex(RuntimeError, 'tc failed'):
                    config.interact(context)
                reset.assert_called_once()
                profiler.assert_not_called()
            self.assertTrue((Path(tmp) / 'failure.json').exists())

    def test_launcher_selected_models_and_phase_order(self):
        self.check_launcher_gpu_layers(None)

    def test_launcher_gpu_layers_override(self):
        for override in (0, 17):
            with self.subTest(override=override):
                self.check_launcher_gpu_layers(override)

    def test_launcher_skip_warmup(self):
        for smoke in (False, True):
            with self.subTest(smoke=smoke):
                self.check_launcher_gpu_layers(None, skip_warmup=True, smoke=smoke)

    def check_launcher_gpu_layers(self, override, skip_warmup=False, smoke=False):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'image.png'
            Image.new('RGB', (8, 8)).save(image)
            args = ['launcher', '--image', str(image), '--models', 'qwen3b', 'qwen7b',
                    '--results-dir', str(Path(tmp) / 'results'), '--interface', 'testwifi']
            if skip_warmup:
                args.append('--skip-warmup')
            if smoke:
                args.append('--smoke')
            if override is not None:
                args.extend(['--gpu-layers', str(override)])
            server = {'context_size': 8192, 'models': {'data': []}, 'props': {}}
            manager = Mock()
            manager.cleanup_legacy.return_value = []
            manager.start.return_value = {'process': {'pid': 123, 'start_ticks': '1'}}
            manager.stop.return_value = {'stopped_at': 1}
            with patch.object(sys, 'argv', args), patch.object(launcher, 'inspect_server', return_value=server), \
                 patch.object(launcher, 'check_network'), patch.object(launcher, 'reset_network'), \
                 patch.object(launcher, 'check_forwarded_ports'), \
                 patch.object(launcher.time, 'sleep') as sleep, patch.object(launcher, 'run_phase') as phase, \
                 patch.object(launcher, 'RemoteLifecycle', return_value=manager):
                launcher.main()
            expected_phases = [('qwen3b', 'main'), ('qwen7b', 'main')] if skip_warmup else [
                ('qwen3b', 'warmup'), ('qwen3b', 'main'), ('qwen7b', 'warmup'), ('qwen7b', 'main')]
            self.assertEqual([(c.args[1], c.args[2]) for c in phase.call_args_list], expected_phases)
            self.assertEqual(sleep.call_count, 20)
            self.assertTrue(all(c.args == (1,) for c in sleep.call_args_list))
            session = next((Path(tmp) / 'results').iterdir())
            settings = json.loads((session / 'session.json').read_text())
            self.assertEqual(settings['status'], 'complete')
            self.assertEqual(settings['model_order'], ['qwen3b', 'qwen7b'])
            self.assertEqual(settings['main_repetitions'], 2 if smoke else 30)
            self.assertEqual(settings['skip_warmup'], skip_warmup)
            self.assertEqual(settings['warmup_repetitions'], 0 if skip_warmup else (1 if smoke else 10))
            self.assertEqual(settings['stabilization_s'], 10)
            self.assertEqual(settings['gpu_layers'], override)
            expected = {'qwen3b': 999, 'qwen7b': 20} if override is None else {'qwen3b': override, 'qwen7b': override}
            self.assertEqual(settings['gpu_layers_by_model'], expected)
            self.assertEqual(manager.start.call_count, 2)
            self.assertEqual(manager.stop.call_count, 2)
            manager.cleanup_legacy.assert_called_once()
            manager.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
