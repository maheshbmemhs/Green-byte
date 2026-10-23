# Replication package: data cleaning and analysis

Green Lab 2026/2027, team Green-Byte: *Comparing On-Device and Remote LLM Inference Across
Workload and Network Conditions*.

This folder reproduces every number, table and figure of the Results section from the raw
experiment data. The experiment itself (Experiment Runner configurations) is in `../examples/`.

## Reproduce the results

From this folder (`data-analysis/`):

```bash
./run_all.sh
```

This runs three steps (about 30 seconds in total):

```bash
python3 cleaning/prepare_data.py        # data/raw    -> data/clean
python3 cleaning/export_clean_data.py   # data/clean  -> data/clean/server, data/clean/local
Rscript analysis/run_all.R              # data/clean  -> results/
```

Each R module can also be run on its own, e.g. `Rscript analysis/04_rq1_3_network_energy.R`.

### Requirements

- Python 3.10+ with `pandas` and `numpy`
- R 4.x with `ggplot2`, `ARTool` and `effsize`:
  `install.packages(c("ggplot2", "ARTool", "effsize"))`

Tested with Python 3.14, R 4.6.1, ggplot2 4.0.3, ARTool 0.11.2, effsize 0.8.1 (macOS).

## Folders

| Folder | Content |
|---|---|
| `data/raw/local_exp_v2/` | On-device runs (laptop), Experiment Runner output as collected |
| `data/raw/remote_model/` | Remote runs (laptop + server), Experiment Runner output as collected |
| `data/clean/` | Clean data built from the raw data (do not edit by hand; rerun the scripts) |
| `cleaning/` | Python scripts that build the clean data set |
| `analysis/` | R analysis, one script per research question; shared code in `analysis/R/` |
| `results/` | Generated output: `figures/`, `tables/` (LaTeX), `stats/` (CSV) |

The raw data is never modified by the scripts.

## Raw data

Each run is one folder `.../<model>/main/run_<k>_repetition_<n>/` containing:

- `run_metadata.json`: treatment (model, input modality, max_tokens, word target, network condition)
- `response.request.json` / `response.json`: the request sent to llama-server and its answer
- `energibridge.csv` (on-device) or `local_energy.csv` + `remote_energy.csv` (remote):
  EnergiBridge samples every 200 ms (CPU package counter in J, GPU power in mW, ...)

`warmup/` folders hold the warm-up runs; they are not used in the analysis. In
`remote_model/remote_model1/attempts/` are failed resume attempts of the Qwen2.5-VL-7B session;
the runs were repeated and are in the main folder, so `attempts/` is excluded.

## How the clean data is built (`cleaning/`)

- **Energy per machine** = CPU package energy + GPU energy, recomputed from the raw EnergiBridge logs.
  - CPU package: sum of consecutive RAPL counter differences. The counter overflows at
    262,144 J; a drop of more than half that range is corrected by adding 262,144 J
    (15 runs were affected).
  - GPU: trapezoidal integral of the sampled `GPU0_POWER` over time.
  - DRAM is not included: EnergiBridge reports 0 J for it on both machines.
- **E_total** = laptop + server energy (on-device: laptop only).
- **E_word** = E_total / output words; **throughput** = output bytes / execution time.
- **Execution time** = `duration_s` from the run table. Two on-device runs that the runner left
  unfinished (it stopped on the counter overflow) get the energy window + the per-model median
  offset between window and duration (about 0.44 s).
- **R_E, R_T**: each remote run divided by the on-device run of the same model, modality, content
  size and repetition.
- No runs or outliers are removed.

Note: the run table column `mean_energy_usage` is not a mean; it is the uncorrected CPU package energy only. The cleaning scripts do not use it except as a consistency check.

Details per run are in `data/clean/cleaning_report.md`; column descriptions are in
`data/clean/server/DATA_DICTIONARY.md` and `data/clean/local/DATA_DICTIONARY.md`.

## Analysis (`analysis/`)

| Script | Report part |
|---|---|
| `01_descriptives.R` | Descriptive statistics (mean, median, SD, IQR, variance, min, max) |
| `02_normality.R` | Shapiro-Wilk per treatment, Q-Q plots of model residuals |
| `03_rq1_local_vs_remote.R` | RQ1, RQ1.1, RQ1.2 |
| `04_rq1_3_network_energy.R` | RQ1.3 |
| `05_rq2_performance.R` | RQ2 |
| `06_validity_checks.R` | Output length vs word target, RAPL overflow example |
| `07_robustness.R` | Diagnostics, parametric vs ART, cluster-robust check, sensitivity, warm-up/order check |
| `08_hypothesis_summary.R` | All hypotheses in one table, network-effects table |

Methods: each LLM is analysed separately (blocking factor). Runs are treated as independent
observations. Factorial tests use the Aligned Rank Transform (ART) ANOVA, because the residuals
are not normal and the spread differs strongly between treatments; a parametric factorial ANOVA
is reported as a check. Post-hoc tests: Wilcoxon rank-sum (paired only for R_E / R_T across
network conditions, which share the on-device run), Benjamini-Hochberg correction, effect sizes
partial eta squared, Cliff's delta and Cohen's d.

## Where each result in the report comes from

| In the report | File |
|---|---|
| Descriptives table | `results/tables/tab_descriptives_compact.tex` |
| Hypothesis summary table | `results/tables/tab_hypothesis_summary.tex` |
| RQ1.1/RQ1.2 ANOVA and post-hoc | `results/tables/tab_rq1_1_2_anova.tex`, `tab_rq1_posthoc.tex` |
| Network effects (RQ1.3, RQ2) | `results/tables/tab_network_effects.tex` |
| Trade-off table (RQ2) | `results/tables/tab_rq2_tradeoff.tex` |
| Figures | `results/figures/paper/*.pdf` |
| All test statistics | `results/stats/*.csv` |
