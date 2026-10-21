#!/usr/bin/env python3
"""Writes data/clean/server/ and data/clean/local/ (CSV + data dictionary) for R."""
from pathlib import Path

import pandas as pd

from prepare_data import to_r_csv

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"

COMMON_FIRST = [
    ("run_id", "", "Experiment Runner run folder name (run_<treatment>_repetition_<n>)"),
    ("model", "", "Model key: qwen3b, gemma4b, qwen7b"),
    ("model_label", "", "Model name for tables and plots"),
    ("input_modality", "", "text or image"),
    ("content_size", "", "short (50-100 words), medium (300-500), long (800-1000)"),
    ("target_words", "words", "Word target in the prompt: 100, 500, 1000"),
    ("max_tokens", "tokens", "Generation limit sent to llama-server"),
]
NETWORK = [("network_condition", "", "standard (no shaping), degraded (20 Mbit/s + 50 ms), bad (5 Mbit/s + 100 ms)")]
COMMON_MID = [("repetition", "", "Repetition index 0-29 within the treatment")]
ENERGY_SERVER = [
    ("e_total_j", "J", "Total energy = client_energy_j + server_energy_j (main metric)"),
    ("client_energy_j", "J", "Laptop energy = client_cpu_j + client_gpu_j"),
    ("server_energy_j", "J", "Server energy = server_cpu_j + server_gpu_j"),
    ("client_cpu_j", "J", "Laptop CPU package energy (RAPL, overflow-corrected)"),
    ("client_gpu_j", "J", "Laptop GPU energy (integral of sampled GPU power)"),
    ("server_cpu_j", "J", "Server CPU package energy (RAPL, overflow-corrected)"),
    ("server_gpu_j", "J", "Server GPU energy (integral of sampled GPU power)"),
]
ENERGY_LOCAL = [
    ("e_total_j", "J", "Total energy = laptop CPU + laptop GPU (main metric; no server involved)"),
    ("client_cpu_j", "J", "Laptop CPU package energy (RAPL, overflow-corrected)"),
    ("client_gpu_j", "J", "Laptop GPU energy (integral of sampled GPU power)"),
]
COMMON_LAST = [
    ("e_per_word_j", "J/word", "e_total_j / output_words"),
    ("duration_s", "s", "Execution time: request sent until full response received (laptop clock)"),
    ("throughput_bytes_per_s", "B/s", "output_bytes / duration_s"),
    ("output_words", "words", "Words in the response (whitespace split, title included)"),
    ("output_bytes", "B", "UTF-8 size of the response text"),
    ("input_bytes", "B", "Prompt text bytes + raw image bytes (image runs)"),
    ("prompt_tokens", "tokens", "Prompt tokens reported by llama-server"),
    ("completion_tokens", "tokens", "Generated tokens reported by llama-server"),
    ("finish_reason", "", "stop = model ended by itself; length = cut off at max_tokens"),
    ("within_word_range", "", "TRUE if output_words is inside the requested range"),
]
WINDOWS_SERVER = [
    ("client_window_s", "s", "Length of the laptop EnergiBridge measurement window"),
    ("server_window_s", "s", "Length of the server EnergiBridge measurement window"),
]
WINDOWS_LOCAL = [
    ("client_window_s", "s", "Length of the laptop EnergiBridge measurement window"),
    ("duration_source", "", "run_table, or imputed (window + per-model offset) for runs the runner left unfinished"),
]
FLAGS = [
    ("flag_rapl_corrected", "", "TRUE if a RAPL counter overflow was corrected in this run"),
    ("flag_runner_incomplete", "", "TRUE if the runner marked the run TODO (it crashed on the overflow); recovered from raw logs"),
    ("flag_truncated_at_max_tokens", "", "TRUE if finish_reason == length"),
    ("had_warmup", "", "TRUE if the session ran warm-up requests"),
    ("session", "", "Raw session folder the run comes from"),
    ("raw_run_dir", "", "Path of the raw run folder (for tracing back)"),
]
COLUMNS = {
    "server": COMMON_FIRST + NETWORK + COMMON_MID + ENERGY_SERVER + COMMON_LAST + WINDOWS_SERVER + FLAGS,
    "local": COMMON_FIRST + COMMON_MID + ENERGY_LOCAL + COMMON_LAST + WINDOWS_LOCAL + FLAGS,
}


def dictionary(side, df, cols):
    shape = ("3 models × 2 modalities × 3 content sizes × 3 network conditions × 30 repetitions"
             if side == "server" else "3 models × 2 modalities × 3 content sizes × 30 repetitions")
    L = [f"# Data dictionary: {side}_runs_clean.csv", "",
         f"One row = one measured {'remote' if side == 'server' else 'on-device'} run. "
         f"{len(df)} rows = {shape}. No missing values in the metrics.", "",
         "Load in R:", "", "```r", f'd <- read.csv("{side}_runs_clean.csv")',
         'd$content_size <- factor(d$content_size, levels = c("short", "medium", "long"))']
    if side == "server":
        L.append('d$network_condition <- factor(d$network_condition, levels = c("standard", "degraded", "bad"))')
    L += ["```", "", "| column | unit | description |", "|---|---|---|"]
    L += [f"| `{c}` | {u} | {d} |" for c, u, d in cols]
    return "\n".join(L) + "\n"


def main():
    runs = pd.read_csv(CLEAN / "all_runs_clean.csv")

    for side, approach in (("server", "remote"), ("local", "local")):
        out = CLEAN / side
        out.mkdir(parents=True, exist_ok=True)
        cols = COLUMNS[side]
        d = runs[runs.approach == approach]
        to_r_csv(d[[c for c, _, _ in cols]], out / f"{side}_runs_clean.csv")
        (out / "DATA_DICTIONARY.md").write_text(dictionary(side, d, cols), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}/ ({len(d)} rows, {len(cols)} columns)")


if __name__ == "__main__":
    main()
