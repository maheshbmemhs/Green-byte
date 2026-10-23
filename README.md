# Green-Byte: Comparing On-Device and Remote LLM Inference

This repository contains the replication package for our Green Lab study:
**Comparing On-Device and Remote LLM Inference Across Workload and Network Conditions**.

We compare the energy consumption and execution time of running LLMs locally
and on a remote server, using different models, text and image prompts,
output lengths, and network conditions.

## Repository Structure

- [Local experiments](examples/greenlab-llm-local/README.md):
  Setup and instructions for on-device inference.
- [Remote experiments](examples/greenlab-llm-remote/README.md):
  Server setup, network configuration, and remote inference instructions.
- [Data analysis](data-analysis/README.md):
  Data cleaning, statistical analysis, and result generation.
- `experiment-runner/`: Framework used to execute the experiments.

## Requirements

- Python 3 and the dependencies in `requirements.txt`
- llama.cpp with `llama-server` and the required models
- EnergiBridge for energy measurements
- SSH access to the server for remote experiments

See the local and remote READMEs for machine setup and permissions.

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Running the Experiments

Complete the setup in the corresponding README before running these commands.

### Local Inference

```bash
python examples/greenlab-llm-local/run_experiments.py \
  --image /path/to/photosynthesis.png
```

See the [greenlab-llm-local README](examples/greenlab-llm-local/README.md).

### Remote Inference

Start the SSH forwarding connection described in the remote README, then run:

```bash
python examples/greenlab-llm-remote/run_experiments.py \
  --image /path/to/photosynthesis.png
```

See the [greenlab-llm-remote README](examples/greenlab-llm-remote/README.md).

Add `--smoke` to either command for a shorter experiment.

## Results

By default, experiment results are saved in:

- Local: `~/GreenLab/results`
- Remote: `~/GreenLab/results-remote`

Each session contains experiment settings, model responses, run tables,
and energy measurements.

## Data Analysis

Install the analysis dependencies listed in the
[data-analysis README](data-analysis/README.md), then run:

```bash
cd data-analysis
./run_all.sh
```

Generated figures, tables, and statistics are saved in `data-analysis/results/`.

## Acknowledgements

The experiments use the
[Experiment Runner](https://github.com/S2-group/experiment-runner) framework.
