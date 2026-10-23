"""Run remote models sequentially through existing SSH tunnels."""
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
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone

from server_lifecycle import RemoteLifecycle
from server_worker import REPOS
from remote_support import HERE, ROOT, ENDPOINTS, NETWORKS, check_network, reset_network
spec = importlib.util.spec_from_file_location('greenlab_remote_config', HERE / 'RunnerConfig.py')
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
RunnerConfig = config_module.RunnerConfig


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def check_forwarded_ports(models):
    from urllib.parse import urlparse
    for model in models:
        endpoint = urlparse(ENDPOINTS[model])
        try:
            with socket.create_connection((endpoint.hostname, endpoint.port), timeout=2):
                pass
        except OSError as exc:
            raise RuntimeError(f'{model}: SSH forward on localhost:{endpoint.port} is not open; '
                               'start your SSH port-forwarding command before launching') from exc


def inspect_server(model, url):
    if get_json(url + '/health').get('status') != 'ok':
        raise ConnectionError(f'{model} is not healthy yet')
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


def run_phase(session, model, phase, args, settings, manager=None):
    env = os.environ.copy()
    env.update(GREENLAB_MODEL=model, GREENLAB_PHASE=phase, GREENLAB_SESSION_DIR=str(session),
               GREENLAB_SERVER_PID=str(os.getpid()), GREENLAB_IMAGE=str(args.image),
               GREENLAB_CTX_SIZE=str(settings['contexts'][model]), GREENLAB_SMOKE='1' if args.smoke else '0')
    print(f'{model}: starting {phase}', flush=True)
    command = [sys.executable, str(ROOT / 'experiment-runner'), str(HERE / 'RunnerConfig.py')]
    phase_directory = session / model / phase
    if getattr(args, 'resume_attempt', None):
        env.update(GREENLAB_RESUME_ATTEMPT=args.resume_attempt.name, GREENLAB_RESUME_DIR=str(args.resume_attempt))
        command = [sys.executable, str(HERE / 'resume_worker.py')]
        if phase == 'warmup':
            phase_directory = args.resume_attempt / model / phase
    process = subprocess.Popen(command,
                               env=env, cwd=ROOT, start_new_session=True)
    try:
        next_check = 0
        while process.poll() is None:
            if manager and time.monotonic() >= next_check:
                manager.status()
                next_check = time.monotonic() + 2
            if (session / 'failure.json').exists():
                raise RuntimeError((session / 'failure.json').read_text())
            time.sleep(0.2)
        if manager:
            manager.status()
        if process.returncode:
            raise RuntimeError(f'{model} {phase} failed')
        repetitions = settings[phase + '_repetitions']
        verify_completion(phase_directory, model, repetitions)
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


def wait_model_ready(manager, model, url, args):
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        manager.status()
        try:
            info = inspect_server(model, url)
        except (OSError, ValueError):
            time.sleep(1)
            continue
        if info['context_size'] != args.ctx_size:
            raise RuntimeError(f"{model} actual context does not match --ctx-size {args.ctx_size}")
        manager.status()
        return info
    raise TimeoutError(f'{model} did not become ready within {args.startup_timeout} seconds')


def run_managed_model(manager, session, model, args, settings):
    error = None
    cleanup_errors = []
    try:
        record = manager.start(model)
        settings.setdefault('server_lifecycle', {})[model] = record
        (session / 'session.json').write_text(json.dumps(settings, indent=2))
        info = wait_model_ready(manager, model, ENDPOINTS[model], args)
        settings['servers'][model] = info
        settings['contexts'][model] = info['context_size']
        (session / 'session.json').write_text(json.dumps(settings, indent=2))
        if not args.skip_warmup:
            run_phase(session, model, 'warmup', args, settings, manager)
        for _ in range(settings.get('stabilization_s', 10)):
            manager.status()
            time.sleep(1)
        current = inspect_server(model, ENDPOINTS[model])
        if not same_server_configuration(current, info):
            raise RuntimeError(f'{model} server configuration changed between phases')
        manager.status()
        run_phase(session, model, 'main', args, settings, manager)
    except BaseException as exc:
        error = exc
    finally:
        # Network must be restored before SSH shutdown/log retrieval.
        try:
            reset_network(args.interface)
        except Exception as exc:
            cleanup_errors.append(str(exc))
        try:
            record = manager.stop()
            if record:
                settings.setdefault('server_lifecycle', {})[model] = record
        except Exception as exc:
            cleanup_errors.append(str(exc))
        try:
            manager.fetch_log(model)
        except Exception as exc:
            cleanup_errors.append('Server log retrieval: ' + str(exc))
        if cleanup_errors:
            settings.setdefault('cleanup_errors', []).extend(cleanup_errors)
        (session / 'session.json').write_text(json.dumps(settings, indent=2))
    if cleanup_errors:
        raise RuntimeError(f"{model}: {error or 'cleanup failed'}; {'; '.join(cleanup_errors)}") from error
    if error:
        raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--resume', type=Path, help='Resume an existing session using its saved settings')
    parser.add_argument('--models', nargs='+', choices=list(ENDPOINTS), default=list(ENDPOINTS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--skip-warmup', action='store_true',
                        help='Skip warmup requests; retain the 10-second stabilization wait')
    parser.add_argument('--interface', default='wlp0s20f3')
    parser.add_argument('--remote-host', default='gl_greenbyte@glg3')
    parser.add_argument('--remote-log-dir', default='logs/greenlab-remote')
    parser.add_argument('--results-dir', type=Path, default=Path.home() / 'GreenLab/results-remote')
    parser.add_argument('--local-energy-command', default='sudo -n energibridge',
                        help='Command prefix; use e.g. sudo -n /path/to/energibridge if needed')
    parser.add_argument('--remote-energy-command', default='energibridge')
    parser.add_argument('--remote-llama-dir', default='~/llama.cpp')
    parser.add_argument('--ctx-size', type=int, default=8192)
    parser.add_argument('--gpu-layers', type=int, default=None,
                        help='Override GPU layers for all selected models; defaults match local per-model settings')
    parser.add_argument('--startup-timeout', type=int, default=300)
    args = parser.parse_args()
    if args.resume:
        from resume_support import resume_session
        resume_session(sys.modules[__name__], args, parser)
        return
    if args.image is None:
        parser.error('--image is required unless --resume is supplied')
    if args.ctx_size <= max(RunnerConfig.TOKEN_LIMITS) + 1 or (args.gpu_layers is not None and args.gpu_layers < 0) or args.startup_timeout <= 0:
        parser.error('Invalid context size, GPU layer count, or startup timeout')
    if len(set(args.models)) != len(args.models):
        parser.error('Duplicate models')
    if args.remote_host.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.@-]+', args.remote_host):
        parser.error('Invalid SSH host')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.interface):
        parser.error('Invalid network interface')
    if not args.remote_log_dir or args.remote_log_dir.startswith('-'):
        parser.error('Invalid remote log directory')
    args.gpu_layers_by_model = {
        model: RunnerConfig.GPU_LAYERS[model] if args.gpu_layers is None else args.gpu_layers
        for model in args.models
    }
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
        check_forwarded_ports(args.models)
        servers = {}
        session = args.results_dir.expanduser().resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        session.mkdir(parents=True, exist_ok=False)
        (session / 'inputs').mkdir()
        args.image = session / 'inputs' / ('photosynthesis.png' if fmt == 'PNG' else 'photosynthesis.jpg')
        args.image.write_bytes(image_bytes)
        settings = dict(models={m: REPOS[m] for m in args.models}, model_order=args.models,
                        endpoints={m: ENDPOINTS[m] for m in args.models}, servers=servers,
                        contexts={}, remote_llama_dir=args.remote_llama_dir, ctx_size=args.ctx_size,
                        gpu_layers=args.gpu_layers, gpu_layers_by_model=args.gpu_layers_by_model, startup_timeout=args.startup_timeout,
                        interface=args.interface, network_conditions=NETWORKS, network_direction='client_egress',
                        remote_host=args.remote_host, remote_log_dir=args.remote_log_dir,
                        local_energy_command=shlex.split(args.local_energy_command),
                        remote_energy_command=shlex.split(args.remote_energy_command),
                        prompts=RunnerConfig.PROMPTS, max_tokens=RunnerConfig.TOKEN_LIMITS,
                        word_targets=RunnerConfig.WORD_TARGETS, cache_prompt=False,
                        image={'path': str(args.image.relative_to(session)), 'sha256': hashlib.sha256(image_bytes).hexdigest(), 'dimensions': dimensions},
                        skip_warmup=args.skip_warmup,
                        warmup_repetitions=0 if args.skip_warmup else (1 if args.smoke else 10),
                        main_repetitions=2 if args.smoke else 30,
                        main_delay_s=10, stabilization_s=10, warmup_delay_s=0, status='running')
        path = session / 'session.json'
        path.write_text(json.dumps(settings, indent=2))
        print(f'Results: {session}', flush=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        previous = signal.signal(signal.SIGTERM, interrupted)
        manager = RemoteLifecycle(args, session)
        error = None
        try:
            manager.open()
            settings['legacy_cleanup'] = manager.cleanup_legacy()
            path.write_text(json.dumps(settings, indent=2))
            for model in args.models:
                run_managed_model(manager, session, model, args, settings)
            settings['status'] = 'complete'
        except BaseException as exc:
            error = exc
            settings.update(status='failed', error=str(exc))
        finally:
            try:
                reset_network(args.interface)
            except Exception as exc:
                error = error or exc
                settings.setdefault('cleanup_errors', []).append(str(exc))
            try:
                manager.close()
            except Exception as exc:
                error = error or exc
                settings.setdefault('cleanup_errors', []).append(str(exc))
            if error:
                settings.update(status='failed', error=str(error))
            signal.signal(signal.SIGTERM, previous)
            path.write_text(json.dumps(settings, indent=2))
        if error:
            raise error


if __name__ == '__main__':
    main()
