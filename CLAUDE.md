# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLSys 2026 artifact evaluation for the paper *Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token*. Contains scripts, data, and pre-built figures to reproduce every figure, table, and inline number in the paper. The `stream2llm/` directory is a modified vLLM engine with streaming input support. The `data/` directory is a git submodule (HuggingFace dataset) that holds all large data for the artifact.

## Setup

```bash
conda create -n stream2llm python=3.10.9 -y && conda activate stream2llm
pip install -r requirements.txt
# Optional: huggingface-cli login (for tokenizer access in workload stats)
```

## Key Commands

### Reproduce all paper artifacts at once
```bash
bash reproduce_artifacts.sh
```

### Run experiments (requires NVIDIA GPU with compute capability >= 7.0)
```bash
# Single experiment: bash experiments/{crawler,anns}/run_scheduler_experiments.sh <config_dir> single <scheduler> <arrival_time>
# All schedulers at one arrival time: ... compare <arrival_time>
# All arrival times for one scheduler: ... scheduler <scheduler_name>
# Full sweep: ... all
```

### Generate plots from run logs
```bash
bash experiments/crawler/generate_all_plots.sh <log_dir> [output_dir]
bash experiments/anns/generate_all_plots.sh <log_dir> [output_dir]
```

### Analysis scripts
```bash
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir <log_dir> --output-dir tables --dataset-name <name> --max-qps <N>
python scripts/utils/analysis/analyze_preemptions.py --log-dir <log_dir> --output-dir tables
```

## Architecture

- **`stream2llm/`** — Fork of vLLM 0.8.1 with streaming input support. Installed in dev mode (`pip install -e .` from that directory, requires pre-downloaded wheel and CUDA GPU). Key modifications are in `stream2llm/vllm/core/scheduler.py`.
- **`data/`** — Git submodule (HuggingFace dataset `rbachkaniwala3/stream2llm-data`). Contains `run_log/`, workload traces (`anns/`, `crawl/`), and `perf_model/` JSONs. Clone with `--recurse-submodules`.
- **`scripts/`** — Python plotting and analysis scripts, organized by workload:
  - `scripts/crawler/plotter_utils/` and `scripts/anns/plotter_utils/` — Per-workload figure generation
  - `scripts/utils/analysis/` — Shared analysis (`compute_scheduler_improvements.py`, `analyze_preemptions.py`)
  - `scripts/utils/plotting/` — Shared plotting (`plot_recomp_vs_swap_clean.py`)
- **`experiments/`** — Experiment driver scripts, YAML configs, and SLURM batch files. Configs are in `experiments/{crawler,anns}/configs/`. Output goes to `data/run_log/`.
- **`figures/`** — Generated plots; `figures/reference/` has pre-built reference copies.
- **`tables/`** — Generated table data and analysis results (`.txt` files).

## Scheduler Variants

Schedulers tested: `default_vllm`, `fcfs`, `lcas`, `mcps`. Config naming: `*_full` = all mechanisms, `*_delay_N` = artificial delay, `*_recomp_only` / `*_swap_only` = ablations.

## Data Conventions

- Run logs contain `run_metrics.csv` + `config_*.yaml` per experiment
- Arrival times use underscores for decimals (e.g., `0_25` = 0.25 seconds)
- Max QPS: 4 for crawler workloads, 2 for ANNS workloads
