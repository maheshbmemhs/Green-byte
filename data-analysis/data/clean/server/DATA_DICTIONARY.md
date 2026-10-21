# Data dictionary: server_runs_clean.csv

One row = one measured remote run. 1620 rows = 3 models × 2 modalities × 3 content sizes × 3 network conditions × 30 repetitions. No missing values in the metrics.

Load in R:

```r
d <- read.csv("server_runs_clean.csv")
d$content_size <- factor(d$content_size, levels = c("short", "medium", "long"))
d$network_condition <- factor(d$network_condition, levels = c("standard", "degraded", "bad"))
```

| column | unit | description |
|---|---|---|
| `run_id` |  | Experiment Runner run folder name (run_<treatment>_repetition_<n>) |
| `model` |  | Model key: qwen3b, gemma4b, qwen7b |
| `model_label` |  | Model name for tables and plots |
| `input_modality` |  | text or image |
| `content_size` |  | short (50-100 words), medium (300-500), long (800-1000) |
| `target_words` | words | Word target in the prompt: 100, 500, 1000 |
| `max_tokens` | tokens | Generation limit sent to llama-server |
| `network_condition` |  | standard (no shaping), degraded (20 Mbit/s + 50 ms), bad (5 Mbit/s + 100 ms) |
| `repetition` |  | Repetition index 0-29 within the treatment |
| `e_total_j` | J | Total energy = client_energy_j + server_energy_j (main metric) |
| `client_energy_j` | J | Laptop energy = client_cpu_j + client_gpu_j |
| `server_energy_j` | J | Server energy = server_cpu_j + server_gpu_j |
| `client_cpu_j` | J | Laptop CPU package energy (RAPL, overflow-corrected) |
| `client_gpu_j` | J | Laptop GPU energy (integral of sampled GPU power) |
| `server_cpu_j` | J | Server CPU package energy (RAPL, overflow-corrected) |
| `server_gpu_j` | J | Server GPU energy (integral of sampled GPU power) |
| `e_per_word_j` | J/word | e_total_j / output_words |
| `duration_s` | s | Execution time: request sent until full response received (laptop clock) |
| `throughput_bytes_per_s` | B/s | output_bytes / duration_s |
| `output_words` | words | Words in the response (whitespace split, title included) |
| `output_bytes` | B | UTF-8 size of the response text |
| `input_bytes` | B | Prompt text bytes + raw image bytes (image runs) |
| `prompt_tokens` | tokens | Prompt tokens reported by llama-server |
| `completion_tokens` | tokens | Generated tokens reported by llama-server |
| `finish_reason` |  | stop = model ended by itself; length = cut off at max_tokens |
| `within_word_range` |  | TRUE if output_words is inside the requested range |
| `client_window_s` | s | Length of the laptop EnergiBridge measurement window |
| `server_window_s` | s | Length of the server EnergiBridge measurement window |
| `flag_rapl_corrected` |  | TRUE if a RAPL counter overflow was corrected in this run |
| `flag_runner_incomplete` |  | TRUE if the runner marked the run TODO (it crashed on the overflow); recovered from raw logs |
| `flag_truncated_at_max_tokens` |  | TRUE if finish_reason == length |
| `had_warmup` |  | TRUE if the session ran warm-up requests |
| `session` |  | Raw session folder the run comes from |
| `raw_run_dir` |  | Path of the raw run folder (for tracing back) |
