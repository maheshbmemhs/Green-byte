import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test.test_greenlab_remote import Config, launcher, support
import resume_support as resume
import resume_worker as worker

CSV = 'CPU_USAGE_0,USED_MEMORY,PACKAGE_ENERGY (J)\n10,1048576,1\n20,2097152,3\n'


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = Path(self.tmp.name)
        self.original = (list(Config.TOKEN_LIMITS), dict(Config.WORD_TARGETS), dict(Config.PROMPTS),
                         copy.deepcopy(support.NETWORKS))
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.restore)
        image = self.session / 'image.png'
        image.write_bytes(b'saved image bytes')
        self.settings = dict(
            models={'qwen7b': resume.REPOS['qwen7b']}, model_order=['qwen7b'],
            max_tokens=[2100, 1100, 300], word_targets={'2100': 1000, '1100': 500, '300': 100},
            prompts={'text': 'Original text {word_count} words', 'image': 'Original image {word_count} words'},
            image={'path': 'image.png', 'sha256': hashlib.sha256(image.read_bytes()).hexdigest()},
            endpoints={'qwen7b': 'http://127.0.0.1:8082'}, contexts={'qwen7b': 8192}, servers={},
            ctx_size=8192, cache_prompt=False, network_direction='client_egress',
            network_conditions=copy.deepcopy(support.NETWORKS), main_repetitions=1,
            warmup_repetitions=0, skip_warmup=True, main_delay_s=10, warmup_delay_s=0,
            stabilization_s=10, remote_host='user@host', interface='testresume',
            remote_log_dir='logs', remote_llama_dir='~/llama.cpp', startup_timeout=300,
            gpu_layers_by_model={'qwen7b': 20}, local_energy_command=['energy'],
            remote_energy_command=['energy'], status='failed')
        Config.restore_settings(self.settings)
        self.table = self.session / 'qwen7b/main/run_table.csv'
        self.table.parent.mkdir(parents=True)
        self.rows = Config.build_run_table_model('qwen7b', 1, True).generate_experiment_run_table()
        for row in self.rows:
            row['__done'] = 'DONE'
        # Keep two pending rows in deliberately non-generated order.
        self.rows.sort(key=lambda r: r['__run_id'], reverse=True)
        for row in self.rows[:2]:
            row['__done'] = 'TODO'
        self.fields = list(self.rows[0])
        self.save()
        self.attempt = self.session / 'attempts/resume-test'
        self.attempt.mkdir(parents=True)

    def restore(self):
        Config.TOKEN_LIMITS, Config.WORD_TARGETS, Config.PROMPTS, networks = self.original
        support.NETWORKS.clear()
        support.NETWORKS.update(networks)

    def save(self):
        resume.write_json(self.session / 'session.json', self.settings)
        resume.write_table(self.table, self.fields, self.rows)

    def artifacts(self, row):
        directory = self.table.parent / row['__run_id']
        directory.mkdir(exist_ok=True)
        config = object.__new__(Config)
        config.image_path = self.session / 'image.png'
        config.send_request(row['prompt_type'], int(row['max_tokens']), directory / 'response.json')
        (directory / 'response.json').write_text(json.dumps({
            'usage': {'prompt_tokens': 20, 'completion_tokens': 100},
            'choices': [{'message': {'content': 'A valid answer'}, 'finish_reason': 'stop'}]}))
        metadata = {k: row[k] for k in resume.FACTORS}
        metadata.update(phase='main', cache_prompt=False, start_ts=10, end_ts=12,
                        remote_csv='logs/original/remote_energy.csv',
                        profiling_windows={label: {'start': {'status': 'ready'}, 'stop': {'status': 'done'}}
                                           for label in ('local', 'remote')})
        resume.write_json(directory / 'run_metadata.json', metadata)
        (directory / 'local_energy.csv').write_text(CSV)
        (directory / 'remote_energy.csv').write_bytes(b'')
        return directory

    def test_validation_restores_settings_and_rejects_changed_image_or_matrix(self):
        Config.PROMPTS = {'text': 'changed', 'image': 'changed'}
        resume.validate_session(self.session, self.settings, Config)
        self.assertEqual(Config.PROMPTS, self.settings['prompts'])
        self.rows[0]['max_tokens'] = '999'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Incorrect saved configuration'):
            resume.validate_session(self.session, self.settings, Config)
        (self.session / 'image.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            resume.validate_session(self.session, self.settings, Config)

    def test_recover_missing_csv_without_inference_preserves_order_and_done_rows(self):
        directory = self.artifacts(self.rows[0])
        before = resume.read_table(self.table)[1]
        with patch.object(resume, 'fetch_remote_file',
                          side_effect=lambda host, source, target: target.write_text(CSV)) as fetch,              patch.object(Config, 'interact') as inference:
            resume.recover_tables(self.session, self.settings, Config, self.attempt)
        after = resume.read_table(self.table)[1]
        self.assertEqual([r['__run_id'] for r in before], [r['__run_id'] for r in after])
        self.assertEqual(after[0]['__done'], 'DONE')
        self.assertEqual(after[0]['remote_mean_energy_usage'], '2.0')
        self.assertEqual(after[1:], before[1:])
        self.assertEqual(json.loads((directory / 'metrics.json').read_text())['duration_source'], 'saved_wall_clock')
        fetch.assert_called_once()
        inference.assert_not_called()

    def test_failed_download_leaves_run_pending(self):
        self.artifacts(self.rows[0])
        before = self.table.read_bytes()
        with patch.object(resume, 'fetch_remote_file', side_effect=RuntimeError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                resume.recover_tables(self.session, self.settings, Config, self.attempt)
        self.assertEqual(self.table.read_bytes(), before)

    def test_recovery_requires_completed_profilers(self):
        directory = self.artifacts(self.rows[0])
        path = directory / 'run_metadata.json'
        metadata = json.loads(path.read_text())
        metadata['profiling_windows']['remote']['stop'] = {}
        resume.write_json(path, metadata)
        with patch.object(resume, 'fetch_remote_file') as fetch:
            self.assertFalse(resume.recover_run(object.__new__(Config), self.settings, self.rows[0], directory))
        fetch.assert_not_called()

    def test_invalid_remote_csv_cannot_mark_done(self):
        directory = self.artifacts(self.rows[0])
        with patch.object(resume, 'fetch_remote_file', side_effect=lambda h, s, p: p.write_text('broken')):
            with self.assertRaisesRegex(ValueError, 'invalid'):
                resume.recover_run(object.__new__(Config), self.settings, self.rows[0], directory)
        self.assertEqual(self.rows[0]['__done'], 'TODO')

    def test_worker_skips_done_rows_and_archives_incomplete_artifacts(self):
        old = self.table.parent / self.rows[0]['__run_id']
        old.mkdir()
        (old / 'old.txt').write_text('preserve')
        done = self.table.parent / self.rows[2]['__run_id']
        done.mkdir()
        (done / 'response.json').write_text('completed response')
        config = SimpleNamespace(results_output_path=self.session, model='qwen7b', phase='main',
                                 settings=self.settings, metrics={}, start_run=Mock())
        seen = []
        def interact(context):
            seen.append(context.execute_run['__run_id'])
            context.run_dir.mkdir()
            config.metrics = {'duration_s': 2}
        config.interact = interact
        with patch.object(worker, 'RunnerConfig', return_value=config),              patch.dict(os.environ, GREENLAB_RESUME_DIR=str(self.attempt)),              patch.object(worker.time, 'sleep') as sleep, patch.object(worker.signal, 'signal'):
            worker.main()
        self.assertEqual(seen, [r['__run_id'] for r in self.rows[:2]])
        self.assertTrue(all(r['__done'] == 'DONE' for r in resume.read_table(self.table)[1]))
        self.assertEqual((done / 'response.json').read_text(), 'completed response')
        self.assertEqual((self.attempt / 'failed_runs/qwen7b/main' / self.rows[0]['__run_id'] / 'old.txt').read_text(), 'preserve')
        self.assertEqual([c.args for c in sleep.call_args_list], [(10,), (10,)])

    def test_worker_interruption_preserves_progress(self):
        config = SimpleNamespace(results_output_path=self.session, model='qwen7b', phase='main',
                                 settings=self.settings, metrics={}, start_run=Mock())
        config.interact = Mock(side_effect=[None, KeyboardInterrupt()])
        with patch.object(worker, 'RunnerConfig', return_value=config),              patch.dict(os.environ, GREENLAB_RESUME_DIR=str(self.attempt)),              patch.object(worker.time, 'sleep'), patch.object(worker.signal, 'signal'):
            with self.assertRaises(KeyboardInterrupt):
                worker.main()
        rows = resume.read_table(self.table)[1]
        self.assertEqual(rows[0]['__done'], 'DONE')
        self.assertEqual(rows[1]['__done'], 'TODO')
        self.assertTrue((self.session / 'failure.json').exists())

    def test_resume_parent_recovers_then_runs_only_pending_models_and_cleans_up(self):
        self.artifacts(self.rows[0])
        (self.session / 'failure.json').write_text('old failure')
        args = SimpleNamespace(resume=self.session, skip_warmup=False)
        manager = Mock()
        manager.cleanup_legacy.return_value = []
        old_endpoints = dict(launcher.ENDPOINTS)
        self.addCleanup(lambda: launcher.ENDPOINTS.update(old_endpoints))
        def run_model(manager, session, model, args, settings):
            self.assertFalse((session / 'failure.json').exists())
            self.assertTrue(args.skip_warmup)
            self.assertEqual(args.gpu_layers_by_model, {'qwen7b': 20})
            fields, rows = resume.read_table(self.table)
            self.assertEqual(rows[0]['__done'], 'DONE')
            for row in rows:
                row['__done'] = 'DONE'
            resume.write_table(self.table, fields, rows)
        with patch.object(sys, 'argv', ['launcher', '--resume', str(self.session)]),              patch.object(launcher, 'RemoteLifecycle', return_value=manager),              patch.object(launcher, 'check_network'), patch.object(launcher, 'reset_network'),              patch.object(launcher, 'check_forwarded_ports'),              patch.object(launcher, 'run_managed_model', side_effect=run_model) as run,              patch.object(resume, 'fetch_remote_file', side_effect=lambda h, s, p: p.write_text(CSV)):
            resume.resume_session(launcher, args, Mock())
        run.assert_called_once()
        manager.close.assert_called_once()
        saved = json.loads((self.session / 'session.json').read_text())
        self.assertEqual(saved['status'], 'complete')
        self.assertEqual(len(saved['resume_attempts']), 1)

    def test_resume_warmup_is_separate_and_uses_saved_prompt(self):
        self.settings['warmup_repetitions'] = 1
        self.settings['skip_warmup'] = False
        self.save()
        original_table = self.table.read_bytes()
        original_warmup = self.session / 'qwen7b/warmup'
        original_warmup.mkdir()
        (original_warmup / 'sentinel').write_text('original warmup')
        env = dict(GREENLAB_MODEL='qwen7b', GREENLAB_PHASE='warmup',
                   GREENLAB_SESSION_DIR=str(self.session), GREENLAB_SERVER_PID=str(os.getpid()),
                   GREENLAB_IMAGE=str(self.session / 'image.png'), GREENLAB_CTX_SIZE='8192',
                   GREENLAB_RESUME_ATTEMPT=self.attempt.name, GREENLAB_RESUME_DIR=str(self.attempt))
        seen = []
        def interact(config, context):
            seen.append(context.execute_run['__run_id'])
            self.assertEqual(config.PROMPTS, self.settings['prompts'])
            self.assertTrue(context.run_dir.is_relative_to(self.attempt))
        with patch.dict(os.environ, env), patch.object(worker, 'RunnerConfig', Config), \
             patch.object(Config, 'interact', interact), patch.object(worker.time, 'sleep'), \
             patch.object(worker.signal, 'signal'):
            worker.main()
        self.assertEqual(len(seen), 18)
        self.assertEqual(self.table.read_bytes(), original_table)
        self.assertEqual((original_warmup / 'sentinel').read_text(), 'original warmup')
        self.assertTrue((self.attempt / 'qwen7b/warmup/run_table.csv').exists())
        self.assertEqual(list(self.attempt.rglob('*energy.csv')), [])

    def test_phase_uses_resume_worker_and_keeps_saved_table(self):
        for row in self.rows:
            row['__done'] = 'DONE'
        self.save()
        args = SimpleNamespace(image=self.session / 'image.png', smoke=False,
                               interface='testresume', resume_attempt=self.attempt)
        process = Mock(pid=123, returncode=0)
        process.poll.return_value = 0
        with patch.object(launcher.subprocess, 'Popen', return_value=process) as popen, \
             patch.object(launcher.os, 'killpg'), patch.object(launcher, 'reset_network'):
            launcher.run_phase(self.session, 'qwen7b', 'main', args, self.settings, Mock())
        self.assertTrue(popen.call_args.args[0][-1].endswith('resume_worker.py'))
        self.assertEqual(popen.call_args.kwargs['env']['GREENLAB_RESUME_ATTEMPT'], self.attempt.name)

    def test_complete_session_does_not_start_server(self):
        for row in self.rows:
            row['__done'] = 'DONE'
        self.save()
        with patch.object(sys, 'argv', ['launcher', '--resume', str(self.session)]),              patch.object(launcher, 'RemoteLifecycle') as manager:
            resume.resume_session(launcher, SimpleNamespace(resume=self.session, skip_warmup=False), Mock())
        manager.assert_not_called()

if __name__ == '__main__':
    unittest.main()
