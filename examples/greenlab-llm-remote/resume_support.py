"""Validated recovery and execution of saved remote experiment tables."""
import base64
import csv
import math
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import uuid

from remote_support import fetch_remote_file
from server_worker import REPOS

FACTORS = ('model', 'prompt_type', 'max_tokens', 'network_condition')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def read_table(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_table(path, fields, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_session(session, settings, config):
    if settings.get('cache_prompt') is not False:
        raise ValueError('Resume requires the saved disabled prompt-cache policy')
    if settings.get('network_direction') != 'client_egress':
        raise ValueError('Unsupported saved network direction')
    models = settings['model_order']
    if not models or len(set(models)) != len(models) or any(m not in REPOS for m in models):
        raise ValueError('Invalid saved model order')
    if any(settings['models'][m] != REPOS[m] for m in models):
        raise ValueError('Saved model repositories do not match this launcher')
    if set(settings['prompts']) != {'text', 'image'}:
        raise ValueError('Invalid saved prompt types')
    if any('{word_count}' not in prompt for prompt in settings['prompts'].values()):
        raise ValueError('Missing word target in saved prompts')
    if settings['main_repetitions'] < 1 or settings['warmup_repetitions'] < 0:
        raise ValueError('Invalid saved repetition counts')
    image = (session / settings['image']['path']).resolve()
    if not image.is_relative_to(session) or hashlib.sha256(image.read_bytes()).hexdigest() != settings['image']['sha256']:
        raise ValueError('Saved image is missing, outside the session, or its SHA-256 changed')
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', settings['remote_host']) or settings['remote_host'].startswith('-'):
        raise ValueError('Invalid saved SSH host')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', settings['interface']):
        raise ValueError('Invalid saved network interface')
    config.restore_settings(settings)
    if set(config.TOKEN_LIMITS) != set(config.WORD_TARGETS):
        raise ValueError('Saved token limits and word targets disagree')
    if settings['ctx_size'] <= max(config.TOKEN_LIMITS) + 1:
        raise ValueError('Saved context is too small')
    for model in models:
        path = session / model / 'main/run_table.csv'
        if not path.exists():
            if (path.parent).exists():
                raise ValueError(f'Measurement directory has no run table: {path.parent}')
            continue
        fields, rows = read_table(path)
        expected = config.build_run_table_model(model, settings['main_repetitions'], True).generate_experiment_run_table()
        expected = {r['__run_id']: r for r in expected}
        if set(fields) != set(next(iter(expected.values()))) or len(rows) != len(expected):
            raise ValueError(f'Unexpected saved table schema or size: {path}')
        if len({r['__run_id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate run IDs: {path}')
        for row in rows:
            match = expected.get(row['__run_id'])
            if not match or any(str(match[k]) != row[k] for k in FACTORS) or row['__done'] not in ('TODO', 'DONE'):
                raise ValueError(f'Incorrect saved configuration: {path}: {row}')
    return image


def pending_models(session, settings):
    pending = []
    for model in settings['model_order']:
        table = session / model / 'main/run_table.csv'
        if not table.exists() or any(r['__done'] != 'DONE' for r in read_table(table)[1]):
            pending.append(model)
    return pending


def valid_energy_csv(path):
    try:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < 2:
            return False
        cpu = [key for key in rows[0] if key.startswith('CPU_USAGE_')]
        energy = next((key for key in ('PACKAGE_ENERGY (J)', 'CPU_ENERGY (J)') if key in rows[0]), None)
        if not cpu or not energy:
            return False
        return all(math.isfinite(float(row[key])) for row in rows
                   for key in cpu + ['USED_MEMORY', energy])
    except (OSError, ValueError, TypeError, KeyError, csv.Error):
        return False


def recover_run(config, settings, row, directory):
    """Only salvage requests with valid responses and confirmed profiler shutdown."""
    metadata_path = directory / 'run_metadata.json'
    if not metadata_path.exists() or not (directory / 'response.json').exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    if any(str(metadata.get(k)) != str(row[k]) for k in FACTORS):
        raise ValueError(f'Run metadata conflicts with table: {directory}')
    if metadata.get('phase') != 'main' or metadata.get('cache_prompt') is not False:
        return False
    windows = metadata.get('profiling_windows', {})
    if metadata.get('cleanup_errors') or any(
        windows.get(label, {}).get('start', {}).get('status') != 'ready' or
        windows.get(label, {}).get('stop', {}).get('status') != 'done'
        for label in ('local', 'remote')
    ):
        return False
    # Reject malformed/invalid artifacts before requesting a remote download.
    try:
        response = json.loads((directory / 'response.json').read_text())
        content = response['choices'][0]['message']['content']
        elapsed = metadata.get('duration_s', metadata['end_ts'] - metadata['start_ts'])
        if not isinstance(content, str) or not content or not math.isfinite(elapsed) or elapsed <= 0:
            return False
        if int(response['usage']['completion_tokens']) <= 0:
            return False
        request = json.loads((directory / 'response.request.json').read_text())
        if request.get('cache_prompt') is not False or request.get('max_tokens') != int(row['max_tokens']):
            return False
        expected_prompt = settings['prompts'][row['prompt_type']].format(
            word_count=settings['word_targets'][str(row['max_tokens'])])
        content_request = request['messages'][0]['content']
        prompt = content_request if isinstance(content_request, str) else content_request[0]['text']
        if prompt != expected_prompt:
            raise ValueError(f'Saved request does not match session prompt: {directory}')
        if row['prompt_type'] == 'image':
            encoded = content_request[1]['image_url']['url'].split(',', 1)[1]
            if hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest() != settings['image']['sha256']:
                raise ValueError(f'Saved request image differs from the session image: {directory}')
        if not valid_energy_csv(directory / 'local_energy.csv'):
            return False
    except (KeyError, TypeError, FileNotFoundError, json.JSONDecodeError):
        return False
    remote = directory / 'remote_energy.csv'
    if not valid_energy_csv(remote):
        if not metadata.get('remote_csv'):
            return False
        # Transfer failure aborts resume, rather than repeating a successful request.
        fetch_remote_file(settings['remote_host'], metadata['remote_csv'], remote)
    if not valid_energy_csv(remote):
        raise ValueError(f'Recovered remote energy CSV is invalid: {remote}')
    config.context_size = settings['contexts'][row['model']]
    config.phase = 'main'
    config.metrics = {}
    recovered_metadata = dict(metadata, recovered_at=time.time(),
                              duration_source='saved_monotonic' if 'duration_s' in metadata else 'saved_wall_clock')
    config.summarize_run(directory, recovered_metadata, elapsed)
    row.update({k: v for k, v in config.metrics.items() if k in row})
    row['__done'] = 'DONE'
    return True


def recover_tables(session, settings, config_class, attempt):
    events = []
    config = object.__new__(config_class)
    for model in settings['model_order']:
        path = session / model / 'main/run_table.csv'
        if not path.exists():
            continue
        fields, rows = read_table(path)
        for row in rows:
            if row['__done'] == 'DONE':
                continue
            if recover_run(config, settings, row, path.parent / row['__run_id']):
                write_table(path, fields, rows)
                events.append({'model': model, 'run': row['__run_id'], 'recovered_at': time.time()})
                write_json(attempt / 'recovered_runs.json', events)


def resume_session(launcher, args, parser):
    # Saved settings are authoritative: do not silently mix CLI changes into old data.
    allowed = {'--resume', '--skip-warmup'}
    for value in sys.argv[1:]:
        if value.startswith('--') and value.split('=')[0] not in allowed:
            parser.error('--resume accepts only --skip-warmup; all other settings come from session.json')
    session = args.resume.expanduser().resolve()
    settings = json.loads((session / 'session.json').read_text())
    image = validate_session(session, settings, launcher.RunnerConfig)
    if not pending_models(session, settings):
        print(f'All measurements already complete: {session}')
        return
    for name in ('remote_host', 'remote_log_dir', 'remote_llama_dir', 'ctx_size', 'startup_timeout', 'interface'):
        setattr(args, name, settings[name])
    args.image = image
    args.gpu_layers_by_model = settings.get('gpu_layers_by_model') or {
        m: settings['gpu_layers'] for m in settings['model_order']}
    if any(not isinstance(v, int) or v < 0 for v in args.gpu_layers_by_model.values()):
        raise ValueError('Missing or invalid saved GPU layer settings')
    args.skip_warmup = args.skip_warmup or settings['warmup_repetitions'] == 0
    args.smoke = settings['main_repetitions'] == 2
    launcher.ENDPOINTS.update(settings['endpoints'])
    # Same interface lock as fresh launches, plus a session lock for this results directory.
    with (session / '.resume.lock').open('a') as session_lock, open(
            '/tmp/greenlab-remote-' + args.interface + '.lock', 'a') as interface_lock:
        fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(interface_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launcher.check_network(args.interface)
        attempt = session / 'attempts' / ('resume-' + uuid.uuid4().hex[:12])
        attempt.mkdir(parents=True)
        args.resume_attempt = attempt
        shutil.copy2(session / 'session.json', attempt / 'session.before.json')
        for model in settings['model_order']:
            table = session / model / 'main/run_table.csv'
            if table.exists():
                target = attempt / model
                target.mkdir()
                shutil.copy2(table, target / 'run_table.before.csv')
        event = {'id': attempt.name, 'started_at': time.time(), 'skip_warmup': args.skip_warmup,
                 'warmup_repetitions': 0 if args.skip_warmup else settings['warmup_repetitions']}
        settings.setdefault('resume_attempts', []).append(event)
        settings.update(status='running')
        settings.pop('error', None)
        manager = launcher.RemoteLifecycle(args, attempt)
        error = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            manager.open()  # Hold the account lock even while recovering files.
            recover_tables(session, settings, launcher.RunnerConfig, attempt)
            models = pending_models(session, settings)
            failure = session / 'failure.json'
            if failure.exists():
                failure.rename(attempt / 'failure.before.json')
            if models:
                launcher.check_forwarded_ports(models)
                event['legacy_cleanup'] = manager.cleanup_legacy()
                write_json(session / 'session.json', settings)
                for model in models:
                    launcher.run_managed_model(manager, session, model, args, settings)
            event['status'] = 'complete'
            settings['status'] = 'complete'
        except BaseException as exc:
            error = exc
            event.update(status='failed', error=str(exc))
        finally:
            for cleanup in (lambda: launcher.reset_network(args.interface), manager.close):
                try:
                    cleanup()
                except Exception as exc:
                    error = error or exc
                    event.setdefault('cleanup_errors', []).append(str(exc))
            if error:
                settings.update(status='failed', error=str(error))
                event['status'] = 'failed'
            event['ended_at'] = time.time()
            signal.signal(signal.SIGTERM, previous)
            write_json(session / 'session.json', settings)
        if error:
            raise error
        print(f'Resumed session complete: {session}')
