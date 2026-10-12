"""Own one model server across separate warmup and measured Runner executions."""

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiment-runner"))
from RunnerConfig import RunnerConfig


def check_port(port):
    with socket.socket() as sock:
        # A stopped HTTP server can leave accepted connections in TIME_WAIT.
        # Allow those, but never share an active listener (no SO_REUSEPORT).
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            sock.listen()
        except OSError as exc:
            raise RuntimeError(f"Port {port} is occupied; refusing to reuse a server") from exc


def stop_process(process):
    """Stop the owned process group, including Runner's forked workers."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The group leader may exit before its children.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def require_alive(server):
    if server.poll() is not None:
        raise RuntimeError("Model server exited; aborting this session")


def wait_ready(server, url, timeout=300):
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require_alive(server)
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as response:
                ready = json.load(response).get("status") == "ok"
            require_alive(server)
            if ready:
                return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("Model server did not become ready within 300 seconds")


def verify_completion(directory, model, repetitions):
    with (directory / "run_table.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        f"run_{i}_repetition_{j}": tokens
        for j in range(repetitions)
        for i, tokens in enumerate(RunnerConfig.TOKEN_LIMITS)
    }
    if len(rows) != len(expected) or {row["__run_id"] for row in rows} != set(expected):
        raise RuntimeError(f"Unexpected run matrix in {directory}")
    for row in rows:
        if (row["__done"] != "DONE" or row["model"] != model
                or int(row["max_tokens"]) != expected[row["__run_id"]]):
            raise RuntimeError(f"Incomplete or incorrect run in {directory}: {row['__run_id']}")


def run_phase(server, session, model, phase, args):
    env = os.environ.copy()
    env.update({
        "GREENLAB_MODEL": model,
        "GREENLAB_PHASE": phase,
        "GREENLAB_SESSION_DIR": str(session),
        "GREENLAB_SERVER_PID": str(server.pid),
        "GREENLAB_URL": f"http://127.0.0.1:{args.port}",
        "GREENLAB_SMOKE": "1" if args.smoke else "0",
        "GREENLAB_ENERGIBRIDGE": str(args.energibridge),
    })
    print(f"{model}: starting {phase}", flush=True)
    runner = None
    try:
        runner = subprocess.Popen(
            [sys.executable, str(ROOT / "experiment-runner"), str(HERE / "RunnerConfig.py")],
            env=env, cwd=ROOT, start_new_session=True,
        )
        while runner.poll() is None:
            require_alive(server)
            time.sleep(0.2)
        require_alive(server)
        if runner.returncode:
            raise RuntimeError(f"{model} {phase} Runner failed ({runner.returncode})")
        repetitions = (1 if args.smoke else 10) if phase == "warmup" else (2 if args.smoke else 30)
        verify_completion(session / model / phase, model, repetitions)
    finally:
        stop_process(runner)


def run_model(model, session, args):
    check_port(args.port)
    directory = session / model
    directory.mkdir()
    command = [
        str(args.llama), "-hf", RunnerConfig.MODELS[model],
        "--host", "127.0.0.1", "--port", str(args.port),
        "-c", "2048", "-t", "8", "-np", "1",
        "-ngl", str(RunnerConfig.GPU_LAYERS[model]),
    ]
    server = None
    with (directory / "server.log").open("w") as log:
        try:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (directory / "server.json").write_text(json.dumps({"pid": server.pid, "command": command}, indent=2))
            wait_ready(server, f"http://127.0.0.1:{args.port}")
            run_phase(server, session, model, "warmup", args)
            print(f"{model}: waiting 5 seconds before measurement", flush=True)
            for _ in range(5):
                require_alive(server)
                time.sleep(1)
            run_phase(server, session, model, "main", args)
        finally:
            stop_process(server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="1 warmup and 2 measured repetitions per configuration")
    parser.add_argument("--llama", type=Path, default=RunnerConfig.LLAMA)
    parser.add_argument("--energibridge", type=Path, default=RunnerConfig.ENERGIBRIDGE)
    parser.add_argument("--results-dir", type=Path, default=Path.home() / "GreenLab" / "results")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    for name in ("llama", "energibridge"):
        path = getattr(args, name)
        if path is None or not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f"Provide an executable --{name} path")
        setattr(args, name, path.resolve())
    check_port(args.port)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    session = args.results_dir.expanduser().resolve() / session_id
    session.mkdir(parents=True, exist_ok=False)
    metadata = {
        "session": session_id, "models": RunnerConfig.MODELS,
        "max_tokens": RunnerConfig.TOKEN_LIMITS,
        "warmup_repetitions": 1 if args.smoke else 10,
        "main_repetitions": 2 if args.smoke else 30,
        "warmup_shuffle": False, "main_shuffle": True,
        "warmup_delay_s": 0, "main_delay_s": 5, "stabilization_s": 5,
        "prompt": RunnerConfig.PROMPT, "energibridge": str(args.energibridge),
        "status": "running",
    }
    metadata_path = session / "session.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Results: {session}", flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        for model in RunnerConfig.MODELS:
            run_model(model, session, args)
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
