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
        }
        with patch.dict(os.environ, env):
            return Config()

    def test_matrix_and_shuffle(self):
        for model in Config.MODELS:
            for phase, count, delay in (("warmup", 30, 0), ("main", 90, 5000)):
                config = self.config("/tmp", phase, model)
                with patch("random.shuffle") as shuffle:
                    rows = config.create_run_table_model().generate_experiment_run_table()
                    self.assertEqual(shuffle.call_count, int(phase == "main"))
                    if phase == "main":
                        self.assertEqual(len(shuffle.call_args.args[0]), 90)
                self.assertEqual(len(rows), count)
                self.assertEqual({row["model"] for row in rows}, {model})
                self.assertEqual({row["max_tokens"] for row in rows}, {1000, 500, 100})
                self.assertEqual(config.time_between_runs_in_ms, delay)

    def test_smoke_counts(self):
        for phase, count in (("warmup", 3), ("main", 6)):
            config = self.config("/tmp", phase, smoke=True)
            self.assertEqual(len(config.create_run_table_model().generate_experiment_run_table()), count)

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
                for phase in ("warmup", "main"):
                    config = self.config(tmp, phase)
                    config.URL = f"http://127.0.0.1:{server.server_port}"
                    directory = Path(tmp) / phase
                    context = SimpleNamespace(run_dir=directory, execute_run={"model": "gemma4b", "max_tokens": 100})

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
                    self.assertEqual((directory / "energibridge.csv").exists(), phase == "main")
                    self.assertEqual((directory / "metrics.json").exists(), phase == "main")
                self.assertEqual(len(received), 2)
                self.assertTrue(all(item["max_tokens"] == 100 for item in received))
                # Exercise the actual CLI and forked Runner lifecycle, not just hooks.
                env = os.environ.copy()
                env.update({
                    "GREENLAB_MODEL": "gemma4b", "GREENLAB_PHASE": "warmup",
                    "GREENLAB_SESSION_DIR": str(Path(tmp) / "integration"),
                    "GREENLAB_SERVER_PID": str(os.getpid()), "GREENLAB_SMOKE": "1",
                    "GREENLAB_URL": f"http://127.0.0.1:{server.server_port}",
                })
                result = real_run(
                    [sys.executable, str(launcher.ROOT / "experiment-runner"), str(EXAMPLE / "RunnerConfig.py")],
                    env=env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                launcher.verify_completion(Path(tmp) / "integration/gemma4b/warmup", "gemma4b", 1)
                self.assertEqual([item["max_tokens"] for item in received[2:]], [1000, 500, 100])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_invalid_response_fails_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp, "warmup")
            context = SimpleNamespace(run_dir=Path(tmp), execute_run={"max_tokens": 100})
            with patch("subprocess.run", side_effect=lambda *a, **k: (Path(tmp) / "response.json").write_text('{"error":"bad"}')):
                with self.assertRaises(KeyError):
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

                args = SimpleNamespace(port=8080, llama=Path("llama"))
                with patch.object(launcher, "check_port"), patch.object(launcher, "wait_ready"), \
                     patch.object(launcher.subprocess, "Popen", return_value=server) as start, \
                     patch.object(launcher, "run_phase", side_effect=phase), \
                     patch.object(launcher.time, "sleep", side_effect=lambda n: events.append(n)), \
                     patch.object(launcher, "stop_process") as stop:
                    if failure:
                        with self.assertRaises(type(failure)):
                            launcher.run_model("gemma4b", Path(tmp), args)
                    else:
                        launcher.run_model("gemma4b", Path(tmp), args)
                        self.assertEqual(events, ["warmup", 1, 1, 1, 1, 1, "main"])
                    start.assert_called_once()
                    stop.assert_called_once_with(server)

    def test_occupied_port_and_dead_server(self):
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            with self.assertRaises(RuntimeError):
                launcher.check_port(sock.getsockname()[1])
        with self.assertRaises(RuntimeError):
            launcher.wait_ready(Mock(poll=Mock(return_value=1)), "http://127.0.0.1:1")

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
