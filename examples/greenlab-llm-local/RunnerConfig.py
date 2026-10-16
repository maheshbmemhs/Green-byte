
import json
import base64
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.OperationType import OperationType
import shutil
import shlex
import csv
import statistics


class RunnerConfig:
    operation_type = OperationType.AUTO
    self_measure = False
    TOKEN_LIMITS = [1250, 650, 120]
    WORD_TARGETS = {1250: 1000, 650: 500, 120: 100}
    PROMPT_TYPES = ["text", "image"]

    LLAMA = Path(
        "/media/mahesh/New Volume/courses/"
        "GreenLab/repo/llama.cpp/build-cuda/bin/llama-server"
    )

    ENERGIBRIDGE = shutil.which("energibridge")

    MODELS = {
        "gemma4b": "ggml-org/gemma-3-4b-it-GGUF",
        "qwen3b": (
            "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M"
        ),
        "qwen7b": (
            "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M"
        ),
    }

    GPU_LAYERS = {
        "gemma4b": 999,
        "qwen3b": 999,
        "qwen7b": 20,
    }

    URL = "http://127.0.0.1:8080"

    PROMPTS = {
        "text": (
            "Describe photosynthesis using the following fields:\n"
            "1. Title: A concise title.\n"
            "2. Main text: Exactly {word_count} words or slightly more explaining photosynthesis.\n"
            "Count only the main text toward the word target."
        ),
        "image": (
            "Describe the image using the following fields:\n"
            "1. Title: A concise title.\n"
            "2. Main text: Exactly {word_count} words or slightly more describing and explaining the image.\n"
            "Count only the main text toward the word target."
        ),
    }


    def __init__(self):
        try:
            self.model = os.environ["GREENLAB_MODEL"]
            self.phase = os.environ["GREENLAB_PHASE"]
            self.results_output_path = Path(os.environ["GREENLAB_SESSION_DIR"])
            self.server_pid = int(os.environ["GREENLAB_SERVER_PID"])
            self.image_path = Path(os.environ["GREENLAB_IMAGE"])
            self.context_size = int(os.environ["GREENLAB_CTX_SIZE"])
        except KeyError as exc:
            raise RuntimeError("Use run_experiments.py to start this configuration") from exc
        if self.model not in self.MODELS or self.phase not in ("warmup", "main"):
            raise ValueError("Invalid GreenLab model or phase")
        self.name = f"{self.model}/{self.phase}"
        self.URL = os.environ.get("GREENLAB_URL", self.URL)
        self.ENERGIBRIDGE = os.environ.get("GREENLAB_ENERGIBRIDGE", self.ENERGIBRIDGE)
        smoke = os.environ.get("GREENLAB_SMOKE") == "1"
        self.repetitions = (1 if smoke else 10) if self.phase == "warmup" else (2 if smoke else 30)
        self.time_between_runs_in_ms = 0 if self.phase == "warmup" else 10000
        self.metrics = {}

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment),
        ])

    def calculate_resource_metrics(self, csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError("EnergiBridge CSV is empty")

        # CPU usage: average across logical CPUs and samples
        cpu_columns = [
            col for col in rows[0]
            if col.startswith("CPU_USAGE_")
        ]

        if not cpu_columns:
            raise ValueError("CPU usage columns not found")

        cpu_values = [
            statistics.mean(float(row[col]) for col in cpu_columns)
            for row in rows
        ]

        mean_cpu = statistics.mean(cpu_values)

        # System memory usage in MiB
        memory_values = [
            float(row["USED_MEMORY"]) / (1024 ** 2)
            for row in rows
        ]

        mean_memory = statistics.mean(memory_values)

        # CPU package energy in joules
        energy_col = next(
            (
                col for col in [
                    "PACKAGE_ENERGY (J)",
                    "CPU_ENERGY (J)"
                ]
                if col in rows[0]
            ),
            None
        )

        if energy_col is None:
            raise ValueError(
                "No supported cumulative energy column found"
            )

        energy_start = float(rows[0][energy_col])
        energy_end = float(rows[-1][energy_col])

        total_energy = energy_end - energy_start

        if total_energy < 0:
            raise ValueError(
                "Energy counter decreased; check counter wrap/reset"
            )

        return {
            "mean_cpu_usage": round(mean_cpu, 2),
            "mean_memory_usage": round(mean_memory, 2),
            "mean_energy_usage": round(total_energy, 4),
        }


    @classmethod
    def build_run_table_model(cls, model, repetitions, measured=False):
        return RunTableModel(
            factors=[
                FactorModel("model", [model]),
                FactorModel("prompt_type", cls.PROMPT_TYPES),
                FactorModel("max_tokens", cls.TOKEN_LIMITS)
            ],
            repetitions=repetitions,
            data_columns=[] if not measured else [
                "duration_s",
                "prompt_tokens",
                "completion_tokens",
                "output_bytes",
                "tokens_per_second",
                "mean_cpu_usage",
                "mean_memory_usage",
                "mean_energy_usage"
            ],
            shuffle=measured
        )

    def create_run_table_model(self):
        self.run_table_model = self.build_run_table_model(
            self.model, self.repetitions, self.phase == "main"
        )
        return self.run_table_model

    def before_experiment(self):
        self.results_output_path.mkdir(
            parents=True, exist_ok=True
        )

        if self.phase == "main" and not self.ENERGIBRIDGE:
            raise FileNotFoundError("EnergiBridge executable not found in PATH")

    def before_run(self):
        pass

    def server_ready(self):
        try:
            with urllib.request.urlopen(
                self.URL + "/health", timeout=3
            ) as response:
                data = json.load(response)
                return data.get("status") == "ok"
        except Exception:
            return False

    def start_run(self, context):
        self.metrics = {}
        if context.execute_run["model"] != self.model:
            raise RuntimeError("Run model does not match launcher model")
        os.kill(self.server_pid, 0)
        if not self.server_ready():
            raise RuntimeError("Launcher-owned model server is not healthy")

    def send_request(self, prompt_type, tokens, output_path):
        content = self.PROMPTS[prompt_type].format(word_count=self.WORD_TARGETS[int(tokens)])
        if prompt_type == "image":
            mime = "image/png" if self.image_path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(self.image_path.read_bytes()).decode("ascii")
            content = [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ]
        payload = {
            "messages": [
                {"role": "user", "content": content}
            ],
            "max_tokens": int(tokens),
            "temperature": 0,
            "stream": False,
            "cache_prompt": False,
            "verbose": True,
        }

        payload_path = output_path.with_suffix(".request.json")
        payload_path.write_text(json.dumps(payload))

        return payload_path

    def start_measurement(self, context):
        # EnergiBridge is started in interact() so it
        # wraps precisely the HTTP request.
        pass

    def interact(self, context):
        tokens = int(context.execute_run["max_tokens"])
        prompt_type = context.execute_run["prompt_type"]
        run_dir = context.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        response_path = run_dir / "response.json"
        energy_path = run_dir / "energibridge.csv"

        payload_path = self.send_request(
            prompt_type, tokens,
            response_path
        )
        (run_dir / "run_metadata.json").write_text(json.dumps({
            "model": self.model, "phase": self.phase,
            "prompt_type": prompt_type, "max_tokens": tokens,
            "target_words": self.WORD_TARGETS[tokens],
            "context_size": self.context_size, "cache_prompt": False,
        }, indent=2))

        curl_command = [
            "curl",
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--max-time", "180",
            "-o", str(response_path),
            "-H", "Content-Type: application/json",
            "--data-binary", "@" + str(payload_path),
            self.URL + "/v1/chat/completions",
        ]

        command = curl_command
        if self.phase == "main":
            energy_command = [
                str(self.ENERGIBRIDGE), "--gpu", "--summary",
                "-o", str(energy_path), "--", *curl_command,
            ]
            command = ["sudo", "-n", "sg", "msr", "-c", shlex.join(energy_command)]

        start = time.perf_counter()

        subprocess.run(
            command,
            check=True,
            timeout=210,
        )

        elapsed = time.perf_counter() - start

        response = json.loads(response_path.read_text())
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
            raise ValueError("Response is missing a valid prompt_tokens count")
        if (response.get("truncated") or response.get("__verbose", {}).get("truncated")
                or prompt_tokens + tokens + 1 >= self.context_size):
            raise ValueError("Insufficient context for input plus output limit; increase --ctx-size for a new session")

        output_text = response["choices"][0]["message"]["content"]
        generated = int(usage.get("completion_tokens", 0))
        if not isinstance(output_text, str) or not output_text or generated <= 0:
            raise ValueError("Inference returned no generated text/tokens")
        if self.phase == "warmup":
            return


        self.metrics = {
            "duration_s": round(elapsed, 4),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": generated,
            "output_bytes": len(output_text.encode("utf-8")),
            "tokens_per_second": (
                round(generated / elapsed, 4)
                if elapsed > 0 else 0
            ),
        }

        resource_metrics = self.calculate_resource_metrics(energy_path)

        self.metrics.update(resource_metrics)

        (run_dir / "metrics.json").write_text(
            json.dumps({**self.metrics, "model": self.model, "prompt_type": prompt_type,
                        "max_tokens": tokens}, indent=2)
        )

    def stop_measurement(self, context):
        pass

    def stop_run(self, context):
        pass

    def populate_run_data(self, context):
        return self.metrics

    def after_experiment(self):
        pass

    experiment_path = None
