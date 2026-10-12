"""Own one model server across separate warmup and measured Runner executions."""

import argparse
import csv
import hashlib
import io
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


def read_image(path):
    """Decode for validation only; retain the exact original file bytes."""
    from PIL import Image

    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in ("PNG", "JPEG") or getattr(image, "n_frames", 1) != 1:
            raise ValueError("--image must be a single-frame PNG or JPEG")
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        metadata = {"original_filename": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                    "width": image.width, "height": image.height, "format": image.format}
    return data, metadata


def validate_vision(url, context_size):
    import urllib.request

    with urllib.request.urlopen(url + "/props", timeout=5) as response:
        props = json.load(response)
    if props.get("modalities", {}).get("vision") is not True:
        raise RuntimeError("Server does not report vision support; check the model's multimodal projector")
    actual_context = props.get("default_generation_settings", {}).get("n_ctx")
    if actual_context != context_size:
        raise RuntimeError(f"Server context {actual_context} does not match requested --ctx-size {context_size}")
    return props


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
    expected = {row["__run_id"]: row for row in
                RunnerConfig.build_run_table_model(model, repetitions).generate_experiment_run_table()}
    if len(rows) != len(expected) or {row["__run_id"] for row in rows} != set(expected):
        raise RuntimeError(f"Unexpected run matrix in {directory}")
    for row in rows:
        if (row["__done"] != "DONE" or row["model"] != model
                or any(str(row[key]) != str(expected[row["__run_id"]][key])
                       for key in ("prompt_type", "max_tokens"))):
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
        "GREENLAB_IMAGE": str(args.image),
        "GREENLAB_CTX_SIZE": str(args.ctx_size),
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
        "-c", str(args.ctx_size), "--no-context-shift", "-t", "8", "-np", "1",
        "-ngl", str(RunnerConfig.GPU_LAYERS[model]),
    ]
    server = None
    with (directory / "server.log").open("w") as log:
        try:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (directory / "server.json").write_text(json.dumps({"pid": server.pid, "command": command}, indent=2))
            wait_ready(server, f"http://127.0.0.1:{args.port}")
            props = validate_vision(f"http://127.0.0.1:{args.port}", args.ctx_size)
            (directory / "server_props.json").write_text(json.dumps(props, indent=2))
            require_alive(server)
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
    parser.add_argument("--image", type=Path, required=True, help="Fixed photosynthesis image (PNG or JPEG)")
    parser.add_argument("--ctx-size", type=int, default=8192, help="Shared context size for text and image requests")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.ctx_size <= max(RunnerConfig.TOKEN_LIMITS) + 1:
        parser.error("--ctx-size must leave room for input plus 1000 output tokens")
    try:
        image_bytes, image_metadata = read_image(args.image.expanduser())
    except (OSError, ValueError) as exc:
        parser.error(f"Invalid --image: {exc}")
    for name in ("llama", "energibridge"):
        path = getattr(args, name)
        if path is None or not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f"Provide an executable --{name} path")
        setattr(args, name, path.resolve())
    check_port(args.port)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    session = args.results_dir.expanduser().resolve() / session_id
    session.mkdir(parents=True, exist_ok=False)
    inputs = session / "inputs"
    inputs.mkdir()
    args.image = inputs / ("photosynthesis.png" if image_metadata["format"] == "PNG" else "photosynthesis.jpg")
    args.image.write_bytes(image_bytes)
    image_metadata["session_path"] = str(args.image.relative_to(session))
    metadata = {
        "session": session_id, "models": RunnerConfig.MODELS,
        "max_tokens": RunnerConfig.TOKEN_LIMITS,
        "warmup_repetitions": 1 if args.smoke else 10,
        "main_repetitions": 2 if args.smoke else 30,
        "warmup_shuffle": False, "main_shuffle": True,
        "warmup_delay_s": 0, "main_delay_s": 5, "stabilization_s": 5,
        "prompts": RunnerConfig.PROMPTS, "prompt_types": RunnerConfig.PROMPT_TYPES,
        "image": image_metadata, "context_size": args.ctx_size, "cache_prompt": False,
        "energibridge": str(args.energibridge),
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
