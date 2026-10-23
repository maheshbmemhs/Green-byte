"""Execute saved pending rows in order, using the normal remote interaction hooks."""
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from RunnerConfig import RunnerConfig
from resume_support import read_table, write_table

def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    config = RunnerConfig()
    attempt = Path(os.environ['GREENLAB_RESUME_DIR'])
    directory = (config.results_output_path / config.model / 'main' if config.phase == 'main'
                 else attempt / config.model / 'warmup')
    directory.mkdir(parents=True, exist_ok=True)
    table = directory / 'run_table.csv'
    repetitions = config.settings[config.phase + '_repetitions']
    if not table.exists():
        rows = config.build_run_table_model(config.model, repetitions, config.phase == 'main').generate_experiment_run_table()
        for row in rows:
            row['__done'] = 'TODO'
        write_table(table, list(rows[0]), rows)
    fields, rows = read_table(table)
    try:
        for row in rows:
            if row['__done'] == 'DONE':
                continue
            run_dir = directory / row['__run_id']
            if run_dir.exists():
                archive = attempt / 'failed_runs' / config.model / config.phase / row['__run_id']
                archive.parent.mkdir(parents=True, exist_ok=True)
                run_dir.rename(archive)
            context = SimpleNamespace(run_dir=run_dir, execute_run=row)
            print(f"Resuming {config.model}/{config.phase}: {row['__run_id']}", flush=True)
            config.start_run(context)
            config.interact(context)
            row.update({k: v for k, v in config.metrics.items() if k in row})
            row['__done'] = 'DONE'
            write_table(table, fields, rows)
            time.sleep(config.settings[config.phase + '_delay_s'])
    except BaseException as exc:
        (config.results_output_path / 'failure.json').write_text(json.dumps({
            'run': str(run_dir), 'error': str(exc), 'resume_attempt': attempt.name}))
        raise

if __name__ == '__main__':
    main()
