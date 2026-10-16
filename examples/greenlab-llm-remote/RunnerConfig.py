"""Remote inference through existing SSH tunnels, launched by run_experiments.py."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_support import LocalConfig, NETWORKS, Profiler, set_network, reset_network
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel


class RunnerConfig(LocalConfig):
    TOKEN_LIMITS = [2100, 1100, 300]
    WORD_TARGETS = {2100: 1000, 1100: 500, 300: 100}
    PROMPTS = {
        **LocalConfig.PROMPTS,
        "image": (
            "Describe and explain the photosynthesis shown in this image.\n\n"
            "1. Title: A concise title.\n"
            "2. Main text: Aim for {word_count} words, up to 10% more.\n\n"
            "Describe the visible elements, then explain their roles in photosynthesis, "
            "the inputs and outputs, the stages of the process, and its biological importance. "
            "Scale the level of detail to the requested word target. "
            "Clearly distinguish what is visible in the image from additional scientific explanation.\n\n"
            "Count only the main text toward the word target."
        ),
    }

    NETWORK_CONDITIONS = NETWORKS

    def __init__(self):
        super().__init__()
        self.settings = json.loads((self.results_output_path / 'session.json').read_text())
        self.URL = self.settings['endpoints'][self.model]
        self.interface = self.settings['interface']
        self.context_size = self.settings['contexts'][self.model]

    def server_ready(self):
        try:
            with urllib.request.urlopen(self.URL + '/health', timeout=10) as response:
                return json.load(response).get('status') == 'ok'
        except Exception:
            return False

    @classmethod
    def build_run_table_model(cls, model, repetitions, measured=False):
        return RunTableModel(
            factors=[FactorModel('model', [model]), FactorModel('prompt_type', cls.PROMPT_TYPES),
                     FactorModel('max_tokens', cls.TOKEN_LIMITS),
                     FactorModel('network_condition', list(NETWORKS))],
            repetitions=repetitions, shuffle=measured,
            data_columns=[] if not measured else [
                'duration_s', 'prompt_tokens', 'completion_tokens', 'output_bytes',
                'tokens_per_second', 'start_ts', 'end_ts',
                'local_mean_cpu_usage', 'local_mean_memory_usage', 'local_mean_energy_usage',
                'remote_mean_cpu_usage', 'remote_mean_memory_usage', 'remote_mean_energy_usage',
                'local_energy_status', 'remote_energy_status'])

    def before_experiment(self):
        pass

    def start_run(self, context):
        self.metrics = {}
        # Failure cleanup belongs inside interact, including network setup.

    def interact(self, context):
        directory = context.run_dir
        directory.mkdir(parents=True, exist_ok=True)
        row = context.execute_run
        tokens, prompt_type = int(row['max_tokens']), row['prompt_type']
        payload = self.send_request(prompt_type, tokens, directory / 'response.json')
        metadata = {key: row[key] for key in ('model', 'prompt_type', 'max_tokens', 'network_condition')}
        metadata.update(phase=self.phase, target_words=self.WORD_TARGETS[tokens],
                        context_size=self.context_size, cache_prompt=False,
                        profiling_windows={}, network_direction='client_egress')
        profilers = []
        error = None
        try:
            if not self.server_ready():
                raise RuntimeError('Remote model endpoint is not healthy')
            set_network(self.interface, row['network_condition'])
            time.sleep(1)  # Preserve the existing network settling interval.
            if self.phase == 'main':
                remote_path = (f"{self.settings['remote_log_dir']}/{self.results_output_path.name}/"
                               f"{self.model}/{self.phase}/{directory.name}/remote_energy.csv")
                metadata['remote_csv'] = remote_path
                for label, profiler in (
                    ('remote', Profiler(self.settings['remote_host'], self.settings['remote_energy_command'],
                                        remote_path, directory / 'remote_profiler.log')),
                    ('local', Profiler(None, self.settings['local_energy_command'],
                                       directory / 'local_energy.csv', directory / 'local_profiler.log')),
                ):
                    profilers.append((label, profiler))
                    metadata['profiling_windows'][label] = {'start': profiler.start()}
            start_ts = time.time()
            start = time.perf_counter()
            subprocess.run(['curl', '--fail-with-body', '--silent', '--show-error', '--max-time', '180',
                            '-o', str(directory / 'response.json'), '-H', 'Content-Type: application/json',
                            '--data-binary', '@' + str(payload), self.URL + '/v1/chat/completions'],
                           check=True, timeout=190)
            elapsed = time.perf_counter() - start
            end_ts = time.time()
            metadata.update(start_ts=start_ts, end_ts=end_ts)
        except BaseException as exc:
            error = exc
        finally:
            for label, profiler in reversed(profilers):
                try:
                    metadata['profiling_windows'].setdefault(label, {})['stop'] = profiler.stop()
                except Exception as exc:
                    metadata.setdefault('cleanup_errors', []).append(str(exc))
                    error = error or exc
            try:
                reset_network(self.interface)
            except Exception as exc:
                error = error or exc
            for label, profiler in profilers:
                if label == 'remote':
                    try:
                        profiler.fetch(directory / 'remote_energy.csv')
                    except Exception as exc:
                        error = error or exc
            (directory / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
        try:
            if error:
                raise error
            response = json.loads((directory / 'response.json').read_text())
            usage = response['usage']
            prompt_tokens = int(usage['prompt_tokens'])
            if (response.get('__verbose', {}).get('truncated') or response.get('truncated')
                    or prompt_tokens + tokens + 1 >= self.context_size):
                raise ValueError('Context overflow: configure the remote server with a larger context')
            content = response['choices'][0]['message']['content']
            generated = int(usage['completion_tokens'])
            if not isinstance(content, str) or not content or generated <= 0:
                raise ValueError('Empty inference response')
            if self.phase == 'warmup':
                return
            self.metrics = dict(duration_s=round(elapsed, 4), prompt_tokens=prompt_tokens,
                                completion_tokens=generated, output_bytes=len(content.encode()),
                                tokens_per_second=round(generated / elapsed, 4), start_ts=start_ts, end_ts=end_ts)
            for label in ('local', 'remote'):
                path = directory / f'{label}_energy.csv'
                with path.open() as handle:
                    rows = list(csv.DictReader(handle))
                if len(rows) < 2:
                    raise ValueError(f'{label} profiler produced fewer than two samples')
                # Preserve valid inference and raw data on counter rollover; never fabricate energy.
                try:
                    energy_col = next((c for c in ('PACKAGE_ENERGY (J)', 'CPU_ENERGY (J)') if c in rows[0]), None)
                    if energy_col and any(float(b[energy_col]) < float(a[energy_col]) for a, b in zip(rows, rows[1:])):
                        raise ValueError('Energy counter decreased: rollover/reset; raw CSV retained')
                    metrics = self.calculate_resource_metrics(path)
                    self.metrics.update({label + '_' + k: v for k, v in metrics.items()})
                    self.metrics[label + '_energy_status'] = 'ok'
                except ValueError as exc:
                    for key in ('mean_cpu_usage', 'mean_memory_usage', 'mean_energy_usage'):
                        self.metrics[label + '_' + key] = ''
                    self.metrics[label + '_energy_status'] = str(exc)
            (directory / 'metrics.json').write_text(json.dumps({**metadata, **self.metrics}, indent=2))
        except BaseException as exc:
            (self.results_output_path / 'failure.json').write_text(json.dumps({'run': str(directory), 'error': str(exc)}))
            raise
