# Re-running Experiments

This directory contains driver scripts and configs to reproduce experiment run logs from scratch on GPU hardware. Pre-computed run logs are available in `data/run_log/` for convenience.

Experiment configs write output directly to `data/run_log/`, so the same plot commands work whether using pre-computed data or freshly generated results.

## Prerequisites

1. Install the Stream2LLM engine (see [Building Stream2LLM Engine](../README.md#building-stream2llm-engine))
2. Install Python dependencies: `pip install -r requirements.txt`
3. Access to NVIDIA GPUs (2x H200 or 2x H100 with tensor parallelism)

## Experiment Configurations

### Config-to-Data Mapping

| Workload | Config Directory | Output / Pre-computed Data | Hardware | Description |
|----------|-----------------|---------------------------|----------|-------------|
| Crawler | `crawler/configs/H200_enhanced_schedulers_v1_full` | `data/run_log/crawler/H200_enhanced_schedulers_v1_full` | H200 | Main experiment |
| Crawler | `crawler/configs/H200_enhanced_schedulers_v1_full_delay_10` | `data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10` | H200 | 10x page-scrape delay |
| Crawler | `crawler/configs/H200_enhanced_schedulers_v1_full_delay_10_recomp_only` | `data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only` | H200 | Recompute-only ablation |
| Crawler | `crawler/configs/H200_enhanced_schedulers_v1_full_delay_10_swap_only` | `data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_swap_only` | H200 | Swap-only ablation |
| Crawler | `crawler/configs/H100_enhanced_schedulers_v1_full` | `data/run_log/crawler/H100_enhanced_schedulers_v1_full` | H100 | Hardware comparison |
| ANNS | `anns/configs/H200_enhanced_schedulers_v1_full` | `data/run_log/anns/H200_enhanced_schedulers_v1_full` | H200 | Main experiment |
| ANNS | `anns/configs/H200_enhanced_schedulers_v1_500q_delay_30` | `data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30` | H200 | 30x pipeline delay |
| ANNS | `anns/configs/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only` | `data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only` | H200 | Recompute-only ablation |
| ANNS | `anns/configs/H200_enhanced_schedulers_v1_500q_delay_30_swap_only` | `data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_swap_only` | H200 | Swap-only ablation |
| ANNS | `anns/configs/H100_enhanced_schedulers_v1_full` | `data/run_log/anns/H100_enhanced_schedulers_v1_full` | H100 | Hardware comparison |

### Config Variant Naming

- **`*_full`** — All schedulers with both recomputation and swap enabled
- **`*_delay_10`** / **`*_delay_30`** — Artificial delay multiplier simulating slower streaming sources (10x for crawler page-scrape, 30x for ANNS pipeline)
- **`*_recomp_only`** — Swap disabled, only recomputation
- **`*_swap_only`** — Recomputation disabled, only swap

### Available Schedulers

| Scheduler | Crawler | ANNS | Description |
|-----------|---------|------|-------------|
| `default_vllm` | Yes | Yes | Unmodified vLLM baseline |
| `fcfs` | Yes | Yes | First-come-first-served with LRU eviction |
| `lcas` | Yes | Yes | Least-context-aware scheduling with C+P priority |
| `mcps` | Yes | Yes | Most-computed-prefix scheduling with least-cache eviction |

### Available Arrival Times

**Crawler (`*_full` configs):** 12 levels — `0_0625`, `0_125`, `0_166667`, `0_2`, `0_25`, `0_5`, `1`, `2`, `4`, `8`, `16`, `32` (seconds between arrivals)

**Crawler (delay/ablation configs):** 1 level — `0_25`

**ANNS (`*_full` configs):** 7 levels — `0_0625`, `0_125`, `0_25`, `0_5`, `1`, `2`, `4`

**ANNS (delay/ablation configs):** 1 level — `0_5`

---

## Crawler Experiments

All commands are run from the repository root.

### Run a Single Experiment

```bash
# Syntax: bash experiments/crawler/run_scheduler_experiments.sh <config_dir> single <scheduler> <arrival_time>

# Example: run default_vllm at arrival_time=1 on H200
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full single default_vllm 1
```

### Compare All Schedulers at a Specific Arrival Time

```bash
# Syntax: bash experiments/crawler/run_scheduler_experiments.sh <config_dir> compare <arrival_time>

# Example: compare all schedulers at arrival_time=0_25 on H200
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full compare 0_25
```

### Run All Experiments for a Specific Scheduler

```bash
# Syntax: bash experiments/crawler/run_scheduler_experiments.sh <config_dir> scheduler <scheduler>

# Example: run fcfs at all arrival times on H200
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full scheduler fcfs
```

### Run All Experiments (All Schedulers x All Arrival Times)

```bash
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full all
```

### Run Ablation Experiments

```bash
# Recompute-only ablation (H200, 10x delay)
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10_recomp_only compare 0_25

# Swap-only ablation (H200, 10x delay)
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10_swap_only compare 0_25

# Delayed streaming (H200, 10x delay, both mechanisms)
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10 compare 0_25
```

### Run on H100 Hardware

```bash
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H100_enhanced_schedulers_v1_full all
```

### Generate Plots from Run Logs

```bash
# Syntax: bash experiments/crawler/generate_all_plots.sh <log_dir> [output_dir]

# From pre-computed data (H200 main experiment)
bash experiments/crawler/generate_all_plots.sh \
  data/run_log/crawler/H200_enhanced_schedulers_v1_full

# With custom output directory
bash experiments/crawler/generate_all_plots.sh \
  data/run_log/crawler/H200_enhanced_schedulers_v1_full \
  experiments/crawler/analysis_results/H200_main

# From H100 data
bash experiments/crawler/generate_all_plots.sh \
  data/run_log/crawler/H100_enhanced_schedulers_v1_full

# From ablation data
bash experiments/crawler/generate_all_plots.sh \
  data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only
```

---

## ANNS Experiments

All commands are run from the repository root.

### Run a Single Experiment

```bash
# Syntax: bash experiments/anns/run_scheduler_experiments.sh <config_dir> single <scheduler> <arrival_time>

# Example: run default_vllm at arrival_time=1 on H200
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_full single default_vllm 1
```

### Compare All Schedulers at a Specific Arrival Time

```bash
# Syntax: bash experiments/anns/run_scheduler_experiments.sh <config_dir> compare <arrival_time>

# Example: compare all schedulers at arrival_time=0_5 on H200
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_full compare 0_5
```

### Run All Experiments for a Specific Scheduler

```bash
# Syntax: bash experiments/anns/run_scheduler_experiments.sh <config_dir> scheduler <scheduler>

# Example: run mcps at all arrival times on H200
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_full scheduler mcps
```

### Run All Experiments (All Schedulers x All Arrival Times)

```bash
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_full all
```

### Run Ablation Experiments

```bash
# Recompute-only ablation (H200, 30x delay)
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only compare 0_5

# Swap-only ablation (H200, 30x delay)
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_500q_delay_30_swap_only compare 0_5

# Delayed streaming (H200, 30x delay, both mechanisms)
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_500q_delay_30 compare 0_5
```

### Run on H100 Hardware

```bash
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H100_enhanced_schedulers_v1_full all
```

### Generate Plots from Run Logs

```bash
# Syntax: bash experiments/anns/generate_all_plots.sh <log_dir> [output_dir]

# From pre-computed data (H200 main experiment)
bash experiments/anns/generate_all_plots.sh \
  data/run_log/anns/H200_enhanced_schedulers_v1_full

# With custom output directory
bash experiments/anns/generate_all_plots.sh \
  data/run_log/anns/H200_enhanced_schedulers_v1_full \
  experiments/anns/analysis_results/H200_main

# From H100 data
bash experiments/anns/generate_all_plots.sh \
  data/run_log/anns/H100_enhanced_schedulers_v1_full

# From ablation data
bash experiments/anns/generate_all_plots.sh \
  data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only
```

---

## SLURM Cluster Submission

Batch job files are provided for running full experiment suites on SLURM-based HPC clusters. Each job runs all schedulers across all arrival times and generates plots.

### Main Experiments

```bash
# H200 crawler — all schedulers x all arrival times (~48h)
sbatch experiments/crawler/H200_batch_job_full.sbatch

# H200 ANNS — all schedulers x all arrival times (~36h)
sbatch experiments/anns/H200_batch_job_full.sbatch

# H100 crawler
sbatch experiments/crawler/H100_batch_job_full.sbatch

# H100 ANNS
sbatch experiments/anns/H100_batch_job_full.sbatch
```

### Available Batch Jobs

| File | Hardware | Workload | Description |
|------|----------|----------|-------------|
| `crawler/H200_batch_job_full.sbatch` | 2x H200 | Crawler | Main experiment, all schedulers |
| `crawler/H100_batch_job_full.sbatch` | 2x H100 | Crawler | Main experiment, all schedulers |
| `anns/H200_batch_job_full.sbatch` | 2x H200 | ANNS | Main experiment, all schedulers |
| `anns/H100_batch_job_full.sbatch` | 2x H100 | ANNS | Main experiment, all schedulers |
| `anns/H200_batch_job_500q.sbatch` | 2x H200 | ANNS | 500-query variant |
| `anns/H100_batch_job_500q.sbatch` | 2x H100 | ANNS | 500-query variant |
| `anns/H200_batch_job_500q_delay_5.sbatch` | 2x H200 | ANNS | 500-query with 5x delay |
| `anns/H100_batch_job_500q_delay_5.sbatch` | 2x H100 | ANNS | 500-query with 5x delay |

---

## Reproducing All Paper Experiments End-to-End

To reproduce all 10 experiment configurations from scratch:

```bash
# --- Crawler experiments ---

# H200 main experiment (all schedulers x 12 arrival times)
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full all

# H200 delay + ablation experiments (4 schedulers x 1 arrival time each)
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10 compare 0_25
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10_recomp_only compare 0_25
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10_swap_only compare 0_25

# H100 hardware comparison (all schedulers x 12 arrival times)
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H100_enhanced_schedulers_v1_full all

# --- ANNS experiments ---

# H200 main experiment (all schedulers x 7 arrival times)
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_full all

# H200 delay + ablation experiments (4 schedulers x 1 arrival time each)
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_500q_delay_30 compare 0_5
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only compare 0_5
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_500q_delay_30_swap_only compare 0_5

# H100 hardware comparison (all schedulers x 7 arrival times)
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H100_enhanced_schedulers_v1_full all

# --- Generate all plots ---

bash experiments/crawler/generate_all_plots.sh data/run_log/crawler/H200_enhanced_schedulers_v1_full
bash experiments/crawler/generate_all_plots.sh data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10
bash experiments/crawler/generate_all_plots.sh data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only
bash experiments/crawler/generate_all_plots.sh data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_swap_only
bash experiments/crawler/generate_all_plots.sh data/run_log/crawler/H100_enhanced_schedulers_v1_full
bash experiments/anns/generate_all_plots.sh data/run_log/anns/H200_enhanced_schedulers_v1_full
bash experiments/anns/generate_all_plots.sh data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30
bash experiments/anns/generate_all_plots.sh data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only
bash experiments/anns/generate_all_plots.sh data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_swap_only
bash experiments/anns/generate_all_plots.sh data/run_log/anns/H100_enhanced_schedulers_v1_full
```
