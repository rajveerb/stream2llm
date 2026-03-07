# StreamLLM Artifact

Reproducibility artifact for the StreamLLM paper: *Streaming Prompt Inference for LLM Serving*.

This repository contains all scripts, data, and pre-built figures needed to reproduce every figure, table, and inline number in the paper.

## Quick Start

```bash
# Clone with data submodule
git clone --recurse-submodules https://github.com/rajveerb/stream2llm.git
cd stream2llm

# Create and activate conda environment
conda create -n stream2llm python=3.10.9 -y
conda activate stream2llm

# Install pinned Python dependencies
pip install -r requirements.txt

# (Optional) HuggingFace login for tokenizer access (needed for workload stats only)
huggingface-cli login
```

## Repository Structure

| Directory | Contents |
|-----------|----------|
| `stream2llm/` | Modified vLLM engine with streaming input support |
| `experiments/` | Experiment driver scripts, configs, and SLURM job files |
| `data/` | HuggingFace submodule with all large data (run logs, workload traces, perf models) |
| `scripts/` | Plotting and analysis scripts |
| `figures/` | Pre-built reference figures from the paper |

## Reproducing Paper Artifacts

### Summary Table

| Paper Artifact | Command |
|----------------|---------|
| **fig:perf-model** (Perf model comparison) | `python scripts/utils/plotting/plot_recomp_vs_swap_clean.py --recomp_input data/perf_model/recomputation/H200_tp2_recomputation_latency.json --swap_input data/perf_model/swap/H200_tp2_swap_kernel_latency.json --recomp_input_2 data/perf_model/recomputation/A40_recomputation_latency.json --swap_input_2 data/perf_model/swap/A40_swap_kernel_latency.json --output_dir figures --output_prefix hardware_comparison --title_1 "H200 TP=2" --title_2 "A40"` |
| **fig:ttft** — Crawler row | `python scripts/crawler/plotter_utils/plot_ttft_ccdf_combined_1x4.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures --hardware H200` |
| **fig:ttft** — ANNS row | `python scripts/crawler/plotter_utils/plot_ttft_ccdf_anns_1x4.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures` |
| **fig:completion** — Crawler | `python scripts/crawler/plotter_utils/plot_trace_completion_time.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures` |
| **fig:completion** — ANNS | `python scripts/anns/plotter_utils/plot_trace_completion_time.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures` |
| **fig:appendix-crawler-ttft-qps** | `python scripts/crawler/plotter_utils/plot_ttft_qps_comparison.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures -p 95` |
| **fig:appendix-anns-ttft-qps** | `python scripts/anns/plotter_utils/plot_ttft_qps_comparison.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures -p 95` |
| **fig:tokens-invalidated-ccdf** | `python scripts/anns/plotter_utils/plot_tokens_invalidated_aggregated.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --min-qps 0.25 --max-qps 2.0` |
| **tab:workload-characteristics** — ANNS | `cd data/anns && python compute_workload_stats.py --corpus-prefix retrieved_corpus_content --query-map query_trace_map_5k.json --trace-dir res --max-queries 500 --tokenizer-model meta-llama/Llama-3.1-8B-Instruct` |
| **tab:workload-characteristics** — Crawler | `cd data/crawl && python compute_workload_stats.py --input-dir traces/simpleQA_ALL --tokenizer-model meta-llama/Llama-3.1-8B-Instruct --cores 100` |

## Building StreamLLM Engine

The `stream2llm/` directory contains the modified vLLM engine with streaming input support.

```bash
# Install StreamLLM (requires CUDA-capable GPU)
cd stream2llm
pip install -e .
cd ..
```

**Hardware requirements:** NVIDIA GPU with compute capability >= 7.0 (e.g., A40, H100, H200). Tensor parallelism requires multiple GPUs.

## Re-running Experiments

The `experiments/` directory contains driver scripts and configs to reproduce experiment run logs from scratch on GPU hardware. Pre-computed run logs are available in `data/run_log/` for convenience. Experiment configs write output directly to `data/run_log/`, so the same plot commands work whether using pre-computed data or freshly generated results — no path translation needed.

### Experiment Configurations

The table below maps each experiment configuration to its pre-computed data directory:

| Workload | Config Directory | Pre-computed Data | Hardware | Description |
|----------|-----------------|-------------------|----------|-------------|
| Crawler | `experiments/crawler/configs/H200_enhanced_schedulers_v1_full` | `data/run_log/crawler/H200_enhanced_schedulers_v1_full` | H200 | Main experiment |
| Crawler | `experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10` | `data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10` | H200 | 10x page-scrape delay |
| Crawler | `experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10_recomp_only` | `data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only` | H200 | Recompute-only ablation |
| Crawler | `experiments/crawler/configs/H200_enhanced_schedulers_v1_full_delay_10_swap_only` | `data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_swap_only` | H200 | Swap-only ablation |
| Crawler | `experiments/crawler/configs/H100_enhanced_schedulers_v1_full` | `data/run_log/crawler/H100_enhanced_schedulers_v1_full` | H100 | Hardware comparison |
| ANNS | `experiments/anns/configs/H200_enhanced_schedulers_v1_full` | `data/run_log/anns/H200_enhanced_schedulers_v1_full` | H200 | Main experiment |
| ANNS | `experiments/anns/configs/H200_enhanced_schedulers_v1_500q_delay_30` | `data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30` | H200 | 30x pipeline delay |
| ANNS | `experiments/anns/configs/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only` | `data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only` | H200 | Recompute-only ablation |
| ANNS | `experiments/anns/configs/H200_enhanced_schedulers_v1_500q_delay_30_swap_only` | `data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_swap_only` | H200 | Swap-only ablation |
| ANNS | `experiments/anns/configs/H100_enhanced_schedulers_v1_full` | `data/run_log/anns/H100_enhanced_schedulers_v1_full` | H100 | Hardware comparison |

#### Config Variant Naming

- **`*_full`** — All schedulers with both recomputation and swap enabled
- **`*_delay_10`** / **`*_delay_30`** — Artificial delay multiplier simulating slower streaming sources (10x for crawler page-scrape, 30x for ANNS pipeline)
- **`*_recomp_only`** — Swap disabled, only recomputation
- **`*_swap_only`** — Recomputation disabled, only swap

### Crawler Experiments

```bash
# Run a single scheduler experiment
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full single default_vllm 1

# Run all scheduler comparisons at a specific QPS
bash experiments/crawler/run_scheduler_experiments.sh \
  experiments/crawler/configs/H200_enhanced_schedulers_v1_full compare 1

# Generate all analysis plots from run logs
bash experiments/crawler/generate_all_plots.sh \
  data/run_log/crawler/H200_enhanced_schedulers_v1_full
```

### ANNS Experiments

```bash
# Run a single scheduler experiment
bash experiments/anns/run_scheduler_experiments.sh \
  experiments/anns/configs/H200_enhanced_schedulers_v1_full single default_vllm 1

# Generate all analysis plots from run logs
bash experiments/anns/generate_all_plots.sh \
  data/run_log/anns/H200_enhanced_schedulers_v1_full
```

### SLURM Cluster Submission

Each experiment directory includes `*.sbatch` files for SLURM-based HPC clusters:

```bash
sbatch experiments/crawler/H200_batch_job_full.sbatch
sbatch experiments/anns/H200_batch_job_full.sbatch
```

## Data Organization

The `data/` submodule ([rbachkaniwala3/stream2llm-data](https://huggingface.co/datasets/rbachkaniwala3/stream2llm-data)) contains:

- **`run_log/crawler/`** and **`run_log/anns/`**: Experiment run logs (`run_metrics.csv` + `config_*.yaml`) for 10 configurations across H200 and H100 hardware
- **`anns/`**: ANNS workload data (corpus content, query trace map, 4997 pipeline traces)
- **`crawl/`**: Crawler workload data (4322 trace CSVs)
- **`perf_model/`**: Performance model JSONs (7 recomputation + 11 swap)

## License

This project is licensed under the [MIT License](LICENSE).
