# Local LLM experiment

Run from the repository root after installing `requirements.txt`:

```bash
python examples/greenlab-llm-local/run_experiments.py --smoke
python examples/greenlab-llm-local/run_experiments.py
```

The first command performs real inference: one warmup and two measured repetitions
for every configuration (9 warmup and 18 measured requests total). The second runs
the full experiment (90 warmup and 270 measured requests).

Models run sequentially: Gemma 4B, Qwen 3B, then Qwen 7B. Each model is loaded once,
warmed up at `max_tokens` 1000, 500, and 100, measured at those same limits, and
unloaded. Full runs use 10 warmup repetitions and 30 measured repetitions per
token limit. Measured repetitions are shuffled together within each model.
Warmup has no inter-run delay. A fixed 5-second pause separates phases, and the
runner waits 5 seconds after each measured repetition (including the final one).
This fixed pause does not verify thermal equilibrium.

The launcher owns the server across both Runner executions. Invoke the launcher,
not `RunnerConfig.py` directly. An occupied port is an error. A server exit or
incomplete phase aborts the session; it does not silently reload the model.

## Machine setup

The existing model repository identifiers, GPU layer counts, context size (2048),
thread count (8), and single parallel slot are preserved. `max_tokens` is an output
ceiling; a model can stop before reaching it. The prompt is fixed photosynthesis
text, temperature is zero, and streaming is disabled.

The configured llama-server default retains the previous machine-specific path.
Override executable paths, results directory, or local port as needed:

```bash
python examples/greenlab-llm-local/run_experiments.py --smoke \
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
(or `--results-dir`). `session.json` records settings and final status. Each model
has `server.log`, `server.json` (command and PID), and separate `warmup/` and `main/`
Runner outputs. Warmup keeps request/response diagnostics and completion records,
but no energy CSV or metric files. Analyze only `main/` measurement outputs.

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
