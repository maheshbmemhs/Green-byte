# Remote text/image experiments

Keep the remote llama servers running and open the existing forwarding connection
in another terminal:

```bash
ssh -N -L 8081:localhost:8081 -L 8082:localhost:8082 -L 8083:localhost:8083 gl_greenbyte@glg3
```

From the repository root:

```bash
python -m pip install -r requirements.txt
python examples/greenlab-llm-remote/run_experiments.py --image /path/to/photosynthesis.png --smoke
```

Remove `--smoke` for the full experiment. To select models:

```bash
python examples/greenlab-llm-remote/run_experiments.py \
  --image /path/to/photosynthesis.png --models qwen3b qwen7b
```

The launcher does not start, unload, or stop your model servers or tunnels.
Default order is Gemma 4B (8083), Qwen 3B (8081), Qwen 7B (8082). Every selected
endpoint must report the expected model ID, vision support, and a usable context
via `/health`, `/v1/models`, and `/props`. It reads each server's actual context
size; input plus requested output must fit. Start servers with `--no-context-shift`
and sufficient context (8192 is the local experiment setting). Unexpected model
aliases are rejected; use names containing the actual model family and size.
Keep servers running without reloads throughout each model's phases. If all three
remain resident, memory and idle-power conditions differ from the local setup.

## Matrix and phases

The remote config reuses the current local config's prompts, word targets, image
request construction, and token limits, including your 1250-token edit:

* Models: `gemma4b`, `qwen3b`, `qwen7b`.

* Prompt types: text and image; fixed original image snapshot per session.

* Output ceilings: 1250, 650, 120; main-text word targets: 1000, 500, 100.

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
`--remote-energy-command 'sudo -n /path/to/energibridge'`. No permission or server
settings are changed automatically. The helper is sent over SSH and requires no
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
rather than continuing into the next model. Models and existing tunnels remain
running. Raw remote files remain available if retrieval fails.

## Results

Fresh sessions are saved under `~/GreenLab/results-remote` (`--results-dir` overrides
this). Existing experiments and the old `results/` CSV are untouched.

```text
<session>/
  session.json
  inputs/photosynthesis.png
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
python -m unittest test.test_greenlab_remote -v
```
