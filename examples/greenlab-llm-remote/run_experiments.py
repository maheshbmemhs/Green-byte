"""Run warmup/main phases against existing remote model servers and SSH tunnels."""
import argparse
import csv
import fcntl
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone

from remote_support import HERE, ROOT, ENDPOINTS, NETWORKS, check_network, reset_network
spec = importlib.util.spec_from_file_location('greenlab_remote_config', HERE / 'RunnerConfig.py')
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
RunnerConfig = config_module.RunnerConfig


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def inspect_server(model, url):
    if get_json(url + '/health').get('status') != 'ok':
        raise RuntimeError(f'{model} is not healthy')
    props = get_json(url + '/props')
    if props.get('modalities', {}).get('vision') is not True:
        raise RuntimeError(f'{model} does not report vision support')
    context = props.get('default_generation_settings', {}).get('n_ctx')
    if not isinstance(context, int) or context <= max(RunnerConfig.TOKEN_LIMITS) + 1:
        raise RuntimeError(f'{model} context is missing or too small')
    models = get_json(url + '/v1/models')
    identifiers = [item.get('id', '') for item in models.get('data', [])]
    expected = {'gemma4b': 'gemma34b', 'qwen3b': 'qwen25vl3b', 'qwen7b': 'qwen25vl7b'}[model]
    if not any(expected in re.sub('[^a-z0-9]', '', name.lower()) for name in identifiers):
        raise RuntimeError(f'{model} endpoint reports unexpected models: {identifiers}')
    return {'props': props, 'models': models, 'context_size': context}


def same_server_configuration(current, previous):
    # /v1/models may include per-request timestamps or other volatile metadata.
    def model_ids(snapshot):
        return sorted(item['id'] for item in snapshot['models'].get('data', []))

    return (model_ids(current) == model_ids(previous)
            and current['context_size'] == previous['context_size'])


def verify_completion(directory, model, repetitions):
    expected = {r['__run_id']: r for r in RunnerConfig.build_run_table_model(model, repetitions).generate_experiment_run_table()}
    with (directory / 'run_table.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected) or {r['__run_id'] for r in rows} != set(expected):
        raise RuntimeError(f'Unexpected matrix in {directory}')
    for row in rows:
        if row['__done'] != 'DONE' or any(str(row[k]) != str(expected[row['__run_id']][k]) for k in
                                        ('model', 'prompt_type', 'max_tokens', 'network_condition')):
            raise RuntimeError(f"Incomplete or incorrect run: {directory}/{row['__run_id']}")


def run_phase(session, model, phase, args, settings):
    env = os.environ.copy()
    env.update(GREENLAB_MODEL=model, GREENLAB_PHASE=phase, GREENLAB_SESSION_DIR=str(session),
               GREENLAB_SERVER_PID=str(os.getpid()), GREENLAB_IMAGE=str(args.image),
               GREENLAB_CTX_SIZE=str(settings['contexts'][model]), GREENLAB_SMOKE='1' if args.smoke else '0')
    print(f'{model}: starting {phase}', flush=True)
    process = subprocess.Popen([sys.executable, str(ROOT / 'experiment-runner'), str(HERE / 'RunnerConfig.py')],
                               env=env, cwd=ROOT, start_new_session=True)
    try:
        while process.poll() is None:
            if (session / 'failure.json').exists():
                raise RuntimeError((session / 'failure.json').read_text())
            time.sleep(0.2)
        if process.returncode:
            raise RuntimeError(f'{model} {phase} failed')
        repetitions = (1 if args.smoke else 10) if phase == 'warmup' else (2 if args.smoke else 30)
        verify_completion(session / model / phase, model, repetitions)
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        reset_network(args.interface)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=list(ENDPOINTS), default=list(ENDPOINTS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--interface', default='wlp0s20f3')
    parser.add_argument('--remote-host', default='gl_greenbyte@glg3')
    parser.add_argument('--remote-log-dir', default='logs/greenlab-remote')
    parser.add_argument('--results-dir', type=Path, default=Path.home() / 'GreenLab/results-remote')
    parser.add_argument('--local-energy-command', default='sudo -n energibridge',
                        help='Command prefix; use e.g. sudo -n /path/to/energibridge if needed')
    parser.add_argument('--remote-energy-command', default='energibridge')
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error('Duplicate models')
    if args.remote_host.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.@-]+', args.remote_host):
        parser.error('Invalid SSH host')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.interface):
        parser.error('Invalid network interface')
    if not args.remote_log_dir or args.remote_log_dir.startswith('-'):
        parser.error('Invalid remote log directory')
    from PIL import Image
    image_bytes = args.image.expanduser().read_bytes()
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.format not in ('PNG', 'JPEG') or getattr(image, 'n_frames', 1) != 1:
            parser.error('Image must be a single-frame PNG or JPEG')
        fmt, dimensions = image.format, image.size
        image.verify()
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
    # Exclusive advisory lock protects against another remote launcher on this interface.
    with open('/tmp/greenlab-remote-' + args.interface + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_network(args.interface)
        servers = {m: inspect_server(m, ENDPOINTS[m]) for m in args.models}
        session = args.results_dir.expanduser().resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        session.mkdir(parents=True, exist_ok=False)
        (session / 'inputs').mkdir()
        args.image = session / 'inputs' / ('photosynthesis.png' if fmt == 'PNG' else 'photosynthesis.jpg')
        args.image.write_bytes(image_bytes)
        settings = dict(models={m: RunnerConfig.MODELS[m] for m in args.models}, model_order=args.models,
                        endpoints={m: ENDPOINTS[m] for m in args.models}, servers=servers,
                        contexts={m: servers[m]['context_size'] for m in args.models},
                        interface=args.interface, network_conditions=NETWORKS, network_direction='client_egress',
                        remote_host=args.remote_host, remote_log_dir=args.remote_log_dir,
                        local_energy_command=shlex.split(args.local_energy_command),
                        remote_energy_command=shlex.split(args.remote_energy_command),
                        prompts=RunnerConfig.PROMPTS, max_tokens=RunnerConfig.TOKEN_LIMITS,
                        word_targets=RunnerConfig.WORD_TARGETS, cache_prompt=False,
                        image={'path': str(args.image.relative_to(session)), 'sha256': hashlib.sha256(image_bytes).hexdigest(), 'dimensions': dimensions},
                        warmup_repetitions=1 if args.smoke else 10, main_repetitions=2 if args.smoke else 30,
                        main_delay_s=10, stabilization_s=10, warmup_delay_s=0, status='running')
        path = session / 'session.json'
        path.write_text(json.dumps(settings, indent=2))
        print(f'Results: {session}', flush=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            for model in args.models:
                # Recheck identities between models; servers are externally managed.
                current = inspect_server(model, ENDPOINTS[model])
                if not same_server_configuration(current, servers[model]):
                    raise RuntimeError(f'{model} server properties changed since preflight')
                run_phase(session, model, 'warmup', args, settings)
                time.sleep(10)
                run_phase(session, model, 'main', args, settings)
            settings['status'] = 'complete'
        except BaseException as exc:
            settings.update(status='failed', error=str(exc))
            raise
        finally:
            try:
                reset_network(args.interface)
            finally:
                signal.signal(signal.SIGTERM, previous)
                path.write_text(json.dumps(settings, indent=2))


if __name__ == '__main__':
    main()
