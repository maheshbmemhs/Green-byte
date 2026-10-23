import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "greenlab-llm-local"
sys.path.insert(0, str(EXAMPLE))
spec = importlib.util.spec_from_file_location("greenlab_launcher", EXAMPLE / "run_experiments.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
Config = launcher.RunnerConfig


class GreenLabTests(unittest.TestCase):
    def config(self, directory, phase, model="gemma4b", smoke=False):
        env = {
            "GREENLAB_MODEL": model, "GREENLAB_PHASE": phase,
            "GREENLAB_SESSION_DIR": str(directory), "GREENLAB_SERVER_PID": str(os.getpid()),
            "GREENLAB_SMOKE": "1" if smoke else "0",
            "GREENLAB_IMAGE": str(Path(directory) / "input.png"), "GREENLAB_CTX_SIZE": "8192",
        }
        with patch.dict(os.environ, env):
            return Config()

    def test_matrix_and_shuffle(self):
        for model in Config.MODELS:
            for phase, count, delay in (("warmup", 60, 0), ("main", 180, 10000)):
                config = self.config("/tmp", phase, model)
                with patch("random.shuffle") as shuffle:
                    rows = config.create_run_table_model().generate_experiment_run_table()
                    self.assertEqual(shuffle.call_count, int(phase == "main"))
                    if phase == "main":
                        self.assertEqual(len(shuffle.call_args.args[0]), 180)
                self.assertEqual(len(rows), count)
                self.assertEqual({row["model"] for row in rows}, {model})
                self.assertEqual({row["max_tokens"] for row in rows}, {2100, 1100, 300})
                self.assertEqual({row["prompt_type"] for row in rows}, {"text", "image"})
                self.assertEqual(config.time_between_runs_in_ms, delay)

    def test_smoke_counts(self):
        for phase, count in (("warmup", 6), ("main", 12)):
            config = self.config("/tmp", phase, smoke=True)
            self.assertEqual(len(config.create_run_table_model().generate_experiment_run_table()), count)

    def test_structured_prompts_for_all_combinations(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (8, 8), "green").save(Path(tmp) / "input.png")
            for prompt_type in ("text", "image"):
                for tokens, words in ((2100, 1000), (1100, 500), (300, 100)):
                    payloads = []
                    for phase in ("warmup", "main"):
                        config = self.config(tmp, phase)
                        path = config.send_request(prompt_type, tokens, Path(tmp) / "response.json")
                        payload = json.loads(path.read_text())
                        payloads.append(payload)
                        content = payload["messages"][0]["content"]
                        prompt = content if prompt_type == "text" else content[0]["text"]
                        self.assertIn("1. Title: A concise title.", prompt)
                        self.assertIn(f"2. Main text: Aim for {words} words, up to 10% more." if prompt_type == "image"
                                      else f"2. Main text: Exactly {words} words or slightly more", prompt)
                        self.assertIn("Count only the main text toward the word target.", prompt)
                        self.assertIn("shown in this image" if prompt_type == "image" else "Describe photosynthesis using", prompt)
                        self.assertNotIn("{word_count}", prompt)
                        self.assertEqual(payload["max_tokens"], tokens)
                        self.assertFalse(payload["cache_prompt"])
                    self.assertEqual(payloads[0], payloads[1])

    def test_real_http_warmup_and_mocked_profiler(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading

        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices": [{"message": {"content": "Photosynthesis"}}],
                                   "usage": {"prompt_tokens": 20, "completion_tokens": 3}}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        real_run = subprocess.run
        try:
            with tempfile.TemporaryDirectory() as tmp:
                from PIL import Image
                Image.new("RGB", (8, 8), "green").save(Path(tmp) / "input.png")
                for phase in ("warmup", "main"):
                    config = self.config(tmp, phase)
                    config.URL = f"http://127.0.0.1:{server.server_port}"
                    directory = Path(tmp) / phase
                    context = SimpleNamespace(run_dir=directory, execute_run={"model": "gemma4b", "prompt_type": "image", "max_tokens": 300})

                    def execute(command, **kwargs):
                        if phase == "main":
                            import shlex
                            self.assertEqual(command[:4], ["sudo", "-n", "sg", "msr"])
                            wrapped = shlex.split(command[-1])
                            Path(wrapped[wrapped.index("-o") + 1]).write_text(
                                "CPU_USAGE_0,USED_MEMORY,PACKAGE_ENERGY (J)\n10,1048576,1\n20,2097152,3\n")
                            command = wrapped[wrapped.index("--") + 1:]
                        else:
                            self.assertEqual(command[0], "curl")
                        return real_run(command, **kwargs)

                    with patch("subprocess.run", side_effect=execute):
                        config.interact(context)
                    self.assertTrue((directory / "response.json").exists())
                    metadata = json.loads((directory / "run_metadata.json").read_text())
                    self.assertEqual(metadata["target_words"], 100)
                    self.assertEqual((directory / "energibridge.csv").exists(), phase == "main")
                    self.assertEqual((directory / "metrics.json").exists(), phase == "main")
                self.assertEqual(len(received), 2)
                self.assertTrue(all(item["max_tokens"] == 300 for item in received))
                self.assertTrue(all(item["cache_prompt"] is False for item in received))
                self.assertEqual(received[0]["messages"][0]["content"][0]["text"], Config.PROMPTS["image"].format(word_count=100))
                import base64
                encoded = received[0]["messages"][0]["content"][1]["image_url"]["url"]
                self.assertEqual(base64.b64decode(encoded.split(",", 1)[1]), (Path(tmp) / "input.png").read_bytes())
                # Exercise the actual CLI and forked Runner lifecycle, not just hooks.
                env = os.environ.copy()
                env.update({
                    "GREENLAB_MODEL": "gemma4b", "GREENLAB_PHASE": "warmup",
                    "GREENLAB_SESSION_DIR": str(Path(tmp) / "integration"),
                    "GREENLAB_SERVER_PID": str(os.getpid()), "GREENLAB_SMOKE": "1",
                    "GREENLAB_URL": f"http://127.0.0.1:{server.server_port}",
                    "GREENLAB_IMAGE": str(Path(tmp) / "input.png"), "GREENLAB_CTX_SIZE": "8192",
                })
                result = real_run(
                    [sys.executable, str(launcher.ROOT / "experiment-runner"), str(EXAMPLE / "RunnerConfig.py")],
                    env=env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                launcher.verify_completion(Path(tmp) / "integration/gemma4b/warmup", "gemma4b", 1)
                self.assertEqual([item["max_tokens"] for item in received[2:]], [2100, 1100, 300] * 2)
                self.assertEqual(received[2]["messages"][0]["content"], Config.PROMPTS["text"].format(word_count=1000))
                self.assertFalse(list((Path(tmp) / "integration").rglob("energibridge.csv")))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_invalid_response_fails_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp, "warmup")
            context = SimpleNamespace(run_dir=Path(tmp), execute_run={"prompt_type": "text", "max_tokens": 300})
            with patch("subprocess.run", side_effect=lambda *a, **k: (Path(tmp) / "response.json").write_text('{"error":"bad"}')):
                with self.assertRaises(ValueError):
                    config.interact(context)

    def test_completion_requires_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp, "warmup", smoke=True)
            rows = config.create_run_table_model().generate_experiment_run_table()
            for row in rows:
                row["__done"] = "DONE"

            def save():
                with (Path(tmp) / "run_table.csv").open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            save()
            launcher.verify_completion(Path(tmp), "gemma4b", 1)
            rows[0]["__done"] = "TODO"
            save()
            with self.assertRaises(RuntimeError):
                launcher.verify_completion(Path(tmp), "gemma4b", 1)

    def test_server_spans_phases_and_cleanup_on_failure(self):
        for failure in (None, RuntimeError("warmup failed"), KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as tmp:
                server = Mock(pid=123)
                server.poll.return_value = None
                events = []

                def phase(process, session, model, name, args):
                    self.assertIs(process, server)
                    events.append(name)
                    if failure:
                        raise failure

                args = SimpleNamespace(port=8080, llama=Path("llama"), ctx_size=8192)
                with patch.object(launcher, "check_port"), patch.object(launcher, "wait_ready"), \
                     patch.object(launcher, "validate_vision", return_value={}), \
                     patch.object(launcher.subprocess, "Popen", return_value=server) as start, \
                     patch.object(launcher, "run_phase", side_effect=phase), \
                     patch.object(launcher.time, "sleep", side_effect=lambda n: events.append(n)), \
                     patch.object(launcher, "stop_process") as stop:
                    if failure:
                        with self.assertRaises(type(failure)):
                            launcher.run_model("gemma4b", Path(tmp), args)
                    else:
                        launcher.run_model("gemma4b", Path(tmp), args)
                        self.assertEqual(events, ["warmup"] + [1] * 10 + ["main"])
                    start.assert_called_once()
                    command = start.call_args.args[0]
                    self.assertEqual(command[command.index("-c") + 1], "8192")
                    self.assertIn("--no-context-shift", command)
                    stop.assert_called_once_with(server)

    def test_image_validation(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            for format, suffix in (("PNG", "png"), ("JPEG", "jpg")):
                path = Path(tmp) / f"input.{suffix}"
                Image.new("RGB", (12, 8), "green").save(path, format=format)
                data, metadata = launcher.read_image(path)
                self.assertEqual(data, path.read_bytes())
                self.assertEqual((metadata["width"], metadata["height"]), (12, 8))
                self.assertEqual(metadata["format"], format)
            with self.assertRaises(FileNotFoundError):
                launcher.read_image(Path(tmp) / "missing.png")
            invalid = Path(tmp) / "bad.png"
            invalid.write_bytes(b"not an image")
            with self.assertRaises(OSError):
                launcher.read_image(invalid)
            Image.new("RGB", (8, 8)).save(invalid, format="GIF")
            with self.assertRaises(ValueError):
                launcher.read_image(invalid)

    def test_context_overflow_rejected(self):
        for phase in ("warmup", "main"):
            for extra, prompt_tokens in (({}, 8150), ({"__verbose": {"truncated": True}}, 10)):
                with tempfile.TemporaryDirectory() as tmp:
                    config = self.config(tmp, phase)
                    context = SimpleNamespace(run_dir=Path(tmp), execute_run={"prompt_type": "text", "max_tokens": 300})
                    response = {"choices": [{"message": {"content": "text"}}],
                                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2}, **extra}
                    with patch("subprocess.run", side_effect=lambda *a, **k: (Path(tmp) / "response.json").write_text(json.dumps(response))):
                        with self.assertRaisesRegex(ValueError, "Insufficient context"):
                            config.interact(context)
                    self.assertFalse((Path(tmp) / "metrics.json").exists())

    def test_vision_and_context_validation(self):
        import io
        for props, succeeds in (
            ({"modalities": {"vision": True}, "default_generation_settings": {"n_ctx": 8192}}, True),
            ({"modalities": {"vision": False}}, False),
            ({}, False),
            ({"modalities": {"vision": True}, "default_generation_settings": {"n_ctx": 2048}}, False),
        ):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(props).encode())):
                if succeeds:
                    self.assertEqual(launcher.validate_vision("http://localhost", 8192), props)
                else:
                    with self.assertRaises(RuntimeError):
                        launcher.validate_vision("http://localhost", 8192)

    def test_session_image_snapshot(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "diagram.png"
            Image.new("RGB", (12, 8), "green").save(source)
            original = source.read_bytes()
            sessions = Path(tmp) / "sessions"

            def model_run(model, session, args):
                self.assertEqual(args.image.read_bytes(), original)
                # Subsequent runs must use the session copy, not the original file.
                source.write_bytes(b"changed")

            argv = ["launcher", "--image", str(source), "--results-dir", str(sessions),
                    "--llama", sys.executable, "--energibridge", sys.executable, "--smoke"]
            with patch.object(sys, "argv", argv), patch.object(launcher, "check_port"), \
                 patch.object(launcher, "run_model", side_effect=model_run) as run:
                launcher.main()
            self.assertEqual(run.call_count, 3)
            session = next(sessions.iterdir())
            metadata = json.loads((session / "session.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["prompts"], Config.PROMPTS)
            self.assertEqual(metadata["word_targets"], {"2100": 1000, "1100": 500, "300": 100})
            self.assertFalse(metadata["cache_prompt"])
            import hashlib
            self.assertEqual(metadata["image"]["sha256"], hashlib.sha256(original).hexdigest())

    def test_occupied_port_and_dead_server(self):
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            with self.assertRaises(RuntimeError):
                launcher.check_port(sock.getsockname()[1])
        with self.assertRaises(RuntimeError):
            launcher.wait_ready(Mock(poll=Mock(return_value=1)), "http://127.0.0.1:1")

    def test_selected_models_and_order(self):
        from PIL import Image
        for selected in (["qwen3b", "qwen7b"], ["qwen7b", "qwen3b"], ["qwen3b"]):
            with self.subTest(models=selected), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "diagram.png"
                Image.new("RGB", (8, 8)).save(source)
                sessions = Path(tmp) / "results"
                argv = ["launcher", "--image", str(source), "--results-dir", str(sessions),
                        "--llama", sys.executable, "--energibridge", sys.executable,
                        "--models", *selected]
                with patch.object(sys, "argv", argv), patch.object(launcher, "check_port"), \
                     patch.object(launcher, "run_model") as run:
                    launcher.main()
                self.assertEqual([call.args[0] for call in run.call_args_list], selected)
                metadata = json.loads((next(sessions.iterdir()) / "session.json").read_text())
                self.assertEqual(list(metadata["models"]), selected)
                self.assertEqual(metadata["model_order"], selected)
                self.assertEqual(metadata["warmup_repetitions"], 10)
                self.assertEqual(metadata["main_repetitions"], 30)

    def test_invalid_model_selection(self):
        import io
        for selected in (["unknown"], ["qwen3b", "qwen3b"], []):
            with patch.object(sys, "argv", ["launcher", "--image", "unused.png", "--models", *selected]), \
                 patch("sys.stderr", new_callable=io.StringIO), \
                 patch.object(launcher, "run_model") as run:
                with self.assertRaises(SystemExit) as error:
                    launcher.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_port_check_allows_closed_server_connections(self):
        import socket

        # Close the accepted server connection first to leave server-side TIME_WAIT.
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                connection, _ = listener.accept()
                connection.close()
                self.assertEqual(client.recv(1), b"")
        launcher.check_port(port)

    def test_port_check_rejects_reusable_active_listener(self):
        import socket

        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaises(RuntimeError):
                launcher.check_port(listener.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
