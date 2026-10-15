# Local LLM experiment

Run from the repository root after installing `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python examples/greenlab-llm-local/run_experiments.py --image /path/to/photosynthesis.png --smoke
python examples/greenlab-llm-local/run_experiments.py --image /path/to/photosynthesis.png
```

The first command performs real inference: one warmup and two measured repetitions
for every configuration (18 warmup and 36 measured requests total). The second runs
the full experiment (180 warmup and 540 measured requests).

Models run sequentially: Gemma 4B, Qwen 3B, then Qwen 7B. Each model is loaded once,
warmed up and measured at `max_tokens` 1250, 650, and 120 for both `prompt_type`
values, `text` and `image`, and then unloaded. There are 18 configurations total.
Full runs use 10 warmup repetitions and 30 measured repetitions per combination:
60 warmup and 180 measured requests per model. Measured repetitions are shuffled
together across all six combinations within each model.
Warmup has no inter-run delay. A fixed 10-second pause separates phases, and the
runner waits 10 seconds after each measured repetition (including the final one).
This fixed pause does not verify thermal equilibrium.

The launcher owns the server across both Runner executions. Invoke the launcher,
not `RunnerConfig.py` directly. An occupied port is an error. A server exit or
incomplete phase aborts the session; it does not silently reload the model.

To run only the remaining Qwen models after a Gemma session:

```bash
python examples/greenlab-llm-local/run_experiments.py \
  --image /path/to/photosynthesis.png --models qwen3b qwen7b
```

Use the same image (the previous session's `inputs/` copy can be supplied).
This creates a new session and preserves earlier results. Each selected model
gets its full warmup and measured phases: 120 warmup and 360 measured requests
total for these two models. `--smoke` also works with `--models`.
Models run in the given order; unknown names and duplicates are rejected.
Omitting `--models` runs all three models as before. Session metadata records
only the selected models and their execution order. This option does not repair
failed results or change energy-counter rollover handling.

## Machine setup

The existing model repository identifiers, GPU layer counts, thread count (8), and
single parallel slot are preserved. Context size defaults to 8192 for both prompt
types; override it with `--ctx-size`. The launcher checks `/props` for vision support
and the requested context size before warmup. The `-hf` model loader must load the
matching vision projector. Missing vision support aborts the session.

`max_tokens` is an output ceiling; a model can stop before reaching it. Input size
is not a factor. Both prompt types request a concise **Title** and **Main text**.
Text requests explain photosynthesis; image requests describe and explain the
photosynthesis shown in the same supplied image for every run. The explicit
`WORD_TARGETS` mapping in `RunnerConfig.py` sets the approximate main-text length:

| Maximum output tokens | Requested main-text words |
|---:|---:|
| 1250 | 1000 |
| 650 | 500 |
| 120 | 100 |

Only the main text counts toward the word target. The output token ceiling covers
the title and formatting too, so a response may stop before the requested length.
There are no automatic retries or token-limit increases to enforce word counts.
Warmup and measurement use the same rendered prompts.
These are related tasks, not identical input information. Temperature is zero,
streaming is disabled, and `cache_prompt=false` is explicit for every request.
The model remains loaded, but prompt KV-cache reuse is disabled.

Supply a valid, single-frame PNG or JPEG. Pillow validates the image without
resizing or re-encoding it; the session stores and uses a byte-for-byte copy.
The model's internal image preprocessing and image-token count can differ by model.
Image base64 encoding and request-file preparation happen before profiling;
the measured HTTP call includes server-side vision processing and generation.

Context shifting is disabled. HTTP errors, reported truncation, and insufficient
space for the reported prompt tokens plus the requested output limit fail the run
rather than contributing a successful result. Increase `--ctx-size` for a new
session if necessary; the launcher never adjusts context or GPU offload mid-session.

The configured llama-server default retains the previous machine-specific path.
Override executable paths, results directory, or local port as needed:

```bash
python examples/greenlab-llm-local/run_experiments.py --smoke \
  --image /path/to/photosynthesis.png --ctx-size 8192 \
  --llama /path/to/llama-server \
  --energibridge /path/to/energibridge \
  --results-dir /path/to/results --port 8080
```

`curl` must be installed. Measured requests retain the existing
`sudo -n sg msr -c ...` EnergiBridge invocation, so the machine must already allow
that command without an interactive password and support EnergiBridge's `--gpu`
option. Warmup executes curl directly and does not profile energy.

## Results

Each invocation creates a fresh timestamp/UUID session beneath `~/GreenLab/results`
(or `--results-dir`). `session.json` records settings, both instructions, context
size, cache policy, token-to-word mapping (`word_targets`), image
filename/dimensions/SHA-256, and final status. JSON mapping keys are token limits
encoded as strings; `prompts` stores the templates, and each request file stores
the fully rendered instruction. `inputs/`
holds the original image snapshot. Each model has `server.log`, `server.json`
(command and PID), `server_props.json`, and separate `warmup/` and `main/`
Runner outputs. Warmup keeps request/response diagnostics and completion records,
but no energy CSV or metric files. Analyze only `main/` measurement outputs.

Both run tables include `model`, `prompt_type`, and `max_tokens`. Each run has
`run_metadata.json` identifying its phase, configuration, and `target_words`. Measured runs also
have `metrics.json` with configuration identifiers and the existing metrics, plus
the raw `energibridge.csv`. Run indices 0–2 represent text at 1250/650/120 tokens;
indices 3–5 represent image at 1250/650/120. Prefer the factor columns when analyzing.

Measured metrics retain their existing definitions: duration includes the profiler
wrapper, and `mean_energy_usage` is the CPU package energy delta in joules, not
total system/GPU energy. Raw EnergiBridge CSVs are retained.

No automatic resume is provided: rerunning creates a new session and repeats
warmup. Failed session output is preserved for inspection. The launcher validates
every expected run-table row before advancing because Runner exit status alone
does not guarantee that its child runs succeeded.

Run isolated tests without models or EnergiBridge:

```bash
python -m unittest test.test_greenlab_phases -v
```
