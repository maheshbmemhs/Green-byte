# Remote text/image experiments

Open the existing forwarding connection in another terminal. You no longer need
to start the model servers manually:

```bash
ssh -N -L 8081:localhost:8081 -L 8082:localhost:8082 -L 8083:localhost:8083 gl_greenbyte@glg3
```

From the repository root:

```bash
python -m pip install -r requirements.txt
python examples/greenlab-llm-remote/run_experiments.py --image /path/to/photosynthesis.png --smoke
```

Add `--skip-warmup` to omit warmup requests for every selected model. The server
still loads and passes readiness checks, followed by the 10-second stabilization
wait before measurement. Measurement counts, network conditions, and profiling
are unchanged. This also works with `--smoke`. Session metadata records
`skip_warmup: true` and `warmup_repetitions: 0`; no warmup run directories are created.

```bash
python examples/greenlab-llm-remote/run_experiments.py --image /path/to/photosynthesis.png --models qwen7b --skip-warmup
```

Warmups remain enabled by default. Remove `--smoke` for the full experiment. To select models:

```bash
python examples/greenlab-llm-remote/run_experiments.py \
  --image /path/to/photosynthesis.png --models qwen3b qwen7b
```

The launcher owns model startup and shutdown through SSH and tmux. On startup it
checks that the selected localhost forwards are open, then
stops only your legacy `qwen3b`, `qwen7b`, and `gemma` sessions (even when selecting
a subset), records their process identities, and verifies all three remote model
ports are free. An unrelated listener is an error; it is never killed by port or
process-name matching.

Default order is Gemma 4B (8083), Qwen 3B (8081), Qwen 7B (8082). Each model is loaded
once, kept running through warmup and measurement, and stopped before the next is
loaded. Models are stopped at the end; the manually opened tunnel remains running.
Startup uses the exact remote repositories from your tmux commands (without adding
quantization suffixes), `--host 127.0.0.1`, `--no-context-shift`, and one parallel slot.
The launch command and actual `/props` and `/v1/models` responses are recorded.

Lifecycle options:

| Option | Default |
|---|---|
| `--remote-llama-dir` | `~/llama.cpp` (uses `build-cuda/bin/llama-server`) |
| `--ctx-size` | `8192` |
| `--gpu-layers` | Per model: Gemma 4B `999`, Qwen 3B `999`, Qwen 7B `20` |
| `--startup-timeout` | `300` seconds |

GPU-layer defaults reuse the local `RunnerConfig.GPU_LAYERS` mapping. Supplying
`--gpu-layers N` overrides all selected models (including `0`). Session metadata
records the explicit override in `gpu_layers` (null when omitted) and the resolved
counts in `gpu_layers_by_model`; lifecycle logs also retain each launch command.

The agent holds a remote account lock at `~/.cache/greenlab/remote-experiment.lock`
before changing any servers. New tmux sessions are named
`greenlab-<session-id>-<model>`. PID/start-time and tmux session-ID checks protect
against signaling replacement processes. Shutdown requests SIGTERM, waits up to
15 seconds, then uses SIGKILL only for the recorded identity if required. The next
model cannot start until shutdown and port checks succeed.

Health, model identity, vision support, and the configured context are checked
after startup. The agent monitors the same PID throughout both phases, and the
launcher checks its status every two seconds during runs. This control traffic
is included in host activity during energy measurement. A failed server is never
automatically reloaded. Startup time is outside energy measurement.

On failure or interruption, the launcher stops Runner activity, restores network
shaping, stops the owned model, and attempts to retrieve its log. EOF on the SSH
control channel also triggers remote model cleanup. If connectivity is lost,
shutdown cannot be guaranteed: inspect `server_lifecycle.json` for the remote
session name and recorded PID/start time, and `server_control.log` for SSH errors.
Remote `lifecycle.json` retains records even if the client misses an SSH reply.

## Matrix and phases

The remote config reuses the local text prompt and image request construction.
Its image prompt is defined in the remote `RunnerConfig.PROMPTS` and asks for a
description plus scientific explanation, distinguishing visible details from
additional knowledge. It targets the requested word count with up to 10% extra,
scaling detail to the target. Word counts are instructions, not guarantees.
Local and remote defaults use identical prompts, word targets, and token ceilings:

* Models: `gemma4b`, `qwen3b`, `qwen7b`.

* Prompt types: text and image; fixed original image snapshot per session.

* Output ceilings: 2100, 1100, 300; main-text word targets: 1000, 500, 100.

* Upload network conditions: normal, degraded (50 ms / 20 Mbit/s), bad (100 ms / 5 Mbit/s).

There are 54 configurations. For each model, run all 18 combinations with 10
warmup repetitions, wait 10 seconds, then run 30 measured repetitions shuffled
across combinations. Warmup has no inter-run cooldown; measurement has 10 seconds.
The existing one-second network settling pause remains before each request.
Full totals: 540 warmup requests and 1620 measured requests. Smoke totals: 54 warmup
and 108 measured requests (one and two repetitions). Prompt caching is disabled.
Warmup sends actual requests but never starts an energy profiler.

## Network and profiler requirements

The default interface is `wlp0s20f3`; override with `--interface`. Shaping affects
all outgoing traffic on that interface, including SSH and TCP acknowledgements.
It is client egress shaping, not a symmetric bandwidth/RTT guarantee. No ingress
shaping is installed. Root netem is removed after each request, on failure, and
on normal interruption; custom existing root qdiscs are rejected. An advisory
lock prevents two launchers from shaping the same interface. Only the launcher's
reserved qdisc handle `7a10:` is removed. SIGKILL or machine failure can require
manual inspection and cleanup of that handle.

Local requirements: `tc`, `curl`, SSH, Python dependencies, and passwordless
`sudo -n tc` plus EnergiBridge permissions. Both hosts need Python 3 and a GPU-enabled
EnergiBridge executable with permission to read their energy counters. Control SSH
uses batch authentication and must work without a password prompt. Defaults:

* `--remote-host gl_greenbyte@glg3`

* `--local-energy-command 'sudo -n energibridge'`

* `--remote-energy-command 'energibridge'`

* `--remote-log-dir logs/greenlab-remote`

These energy options accept command prefixes, not shell scripts. For example,
`--remote-energy-command 'sudo -n /path/to/energibridge'`. No permission settings are changed automatically. The helper is sent over SSH and requires no
installation on the server.

Each measured request starts remote profiling, then local profiling, performs the
HTTP request, stops local and remote profiling, restores the network, and retrieves
the remote CSV. Profiler setup, image encoding, and CSV transfer are outside the
reported HTTP `duration_s`. Energy windows enclose the request but include small
startup/stop and SSH coordination overheads. The two windows are not exactly
synchronized; host-local timestamps are saved separately and are only comparable
across hosts if their clocks are synchronized. Energy includes other host activity.
This differs from the local run's profiler-wrapper duration definition.

Profiler helpers stop on control-channel EOF and have a 240-second watchdog.
Request timeout remains 180 seconds. Failed inference/profiling aborts the session
rather than continuing into the next model. The owned model is stopped; existing
tunnels remain running. Raw remote files remain available if retrieval fails.

## Results

Fresh sessions are saved under `~/GreenLab/results-remote` (`--results-dir` overrides
this). Existing experiments and the old `results/` CSV are untouched.

```text
<session>/
  session.json
  server_lifecycle.json
  server_control.log
  inputs/photosynthesis.png
  <model>/server.log
  <model>/warmup/run_table.csv
  <model>/main/run_table.csv
  <model>/main/<run_id>/
    response.request.json
    response.json
    run_metadata.json
    local_energy.csv
    remote_energy.csv
    local_profiler.log
    remote_profiler.log
    metrics.json
```

Run tables identify all four factors explicitly. Per-run metadata records the
remote file path, request timestamps, and profiler readiness/stop acknowledgements.
Energy summaries have `local_` and `remote_` prefixes. `mean_energy_usage` retains
the local naming convention: it is CPU-package energy delta in joules, not an
average, GPU energy, or whole-machine energy. GPU samples remain in the raw CSVs.
Counter decreases or unsupported summary columns leave summary fields blank and
set an explicit `*_energy_status`; they never fabricate energy or discard the
successful inference/raw CSV. This does not repair earlier local results.

No automatic resume: use `--models` to run remaining models in a new session.

Tests (fake profilers and mocked network/SSH; no remote changes):

```bash
python -m unittest test.test_greenlab_remote test.test_remote_lifecycle -v
```

Remote energy CSV and server-log downloads allow 120 seconds per attempt, with
up to three attempts and 5-second waits between failures. SSH connection setup
keeps its 30-second timeout. Files are downloaded to temporary files and replace
the destination only after a successful transfer. Exhausted retries still fail
the experiment and preserve existing files; inference requests are never retried
by this mechanism. These changes do not resume or repair earlier failed sessions.

## Resume a failed session

Keep the manual SSH tunnel open, then run:

```bash
python examples/greenlab-llm-remote/run_experiments.py \
  --resume /home/mahesh/GreenLab/results-remote/20261016T092308Z-a930744a
```

No image argument is needed. Resume validates the saved image SHA-256 and full
configuration matrix, restores the saved prompts, word targets, token ceilings,
network conditions, model order, GPU settings, profiling commands, and delays.
Completed rows are skipped; remaining rows retain their saved order. Other
configuration flags are rejected to prevent mixing new conditions into old data.
The original session directory remains the results destination.

Before loading models, resume tries to recover pending measurements that have a
valid response and confirmed successful shutdown of both profilers. Missing or
invalid remote CSV downloads are fetched from the exact path in the original run
metadata. Validated artifacts are summarized with the same metric calculations,
and only then is the row marked DONE. A download failure aborts recovery without
repeating inference. Older runs lack a saved monotonic duration, so recovered
metrics use the saved end/start timestamps and record this in
`metrics.json.duration_source`. Counter rollovers retain the existing blank
energy values and explanatory status.

Models with no pending rows are not loaded. For remaining models, warmups follow
the original session's repetition count; a session created with
`--skip-warmup` still skips them. You can explicitly omit new warmups with
`--resume /path/to/session --skip-warmup`. The saved stabilization wait is
retained. A server restart is recorded; resumed measurements therefore span
separate model lifetimes.

Each restart creates `attempts/resume-<id>/` containing the prior session metadata
and run tables, recovery records, separate new warmup directories, lifecycle
records, and server logs. Incomplete run directories are moved under that
attempt's `failed_runs/` before re-execution; completed run artifacts remain in
place. New remote profiling files also use the attempt ID, preserving originals.
Unstarted models receive a new shuffled table once; subsequent resumes retain it.

Resume uses both local locks and the remote account lock. It stops the owned
model and resets network shaping on errors or interruption. Repeated resumes
skip every row already marked DONE. This command performs the remaining
experiment; it is not a dry run.
