# Stream2LLM Artifact

Reproducibility artifact for the Stream2LLM paper: *Streaming Prompt Inference for LLM Serving*.

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
| `figures/` | Generated plots and figures from the paper |
| `figures/reference/` | Pre-built reference figures from the paper (for comparison) |
| `tables/` | Generated table data and analysis results |

## Reproducing Paper Artifacts

### Summary Table

| Paper Artifact | Command |
|----------------|---------|
| **fig:perf-model** (Perf model comparison) | `python scripts/utils/plotting/plot_recomp_vs_swap_clean.py --recomp_input data/perf_model/recomputation/H200_tp2_recomputation_latency.json --swap_input data/perf_model/swap/H200_tp2_swap_kernel_latency.json --recomp_input_2 data/perf_model/recomputation/A40_recomputation_latency.json --swap_input_2 data/perf_model/swap/A40_swap_kernel_latency.json --output_dir figures --output_prefix hardware_comparison --title_1 "H200 TP=2" --title_2 "A40"` |
| **fig:ttft** — Crawler row | `python scripts/crawler/plotter_utils/plot_ttft_ccdf_combined_1x4.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_ccdf_crawler --hardware H200` |
| **fig:ttft** — ANNS row | `python scripts/crawler/plotter_utils/plot_ttft_ccdf_anns_1x4.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_ccdf_anns` |
| **fig:completion** — Crawler | `python scripts/crawler/plotter_utils/plot_trace_completion_time.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix trace_completion_time_crawler --max-qps 4` |
| **fig:completion** — ANNS | `python scripts/anns/plotter_utils/plot_trace_completion_time.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix trace_completion_time_anns --max-qps 2` |
| **fig:appendix-crawler-ttft-qps** | `python scripts/crawler/plotter_utils/plot_ttft_qps_comparison.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_qps_comparison_crawler -p 95 --max-rate 4` |
| **fig:appendix-anns-ttft-qps** | `python scripts/anns/plotter_utils/plot_ttft_qps_comparison.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_qps_comparison_anns -p 95 --max-rate 2` |
| **fig:tokens-invalidated-ccdf** | `python scripts/anns/plotter_utils/plot_tokens_invalidated_aggregated.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --table-output-dir tables --min-qps 0.25 --max-qps 2.0` |
| **tab:workload-characteristics** — ANNS | `cd data/anns && python compute_workload_stats.py --corpus-prefix retrieved_corpus_content --query-map query_trace_map_5k.json --trace-dir res --max-queries 500 --tokenizer-model meta-llama/Llama-3.1-8B-Instruct` |
| **tab:workload-characteristics** — Crawler | `cd data/crawl && python compute_workload_stats.py --input-dir traces/simpleQA_ALL --tokenizer-model meta-llama/Llama-3.1-8B-Instruct --cores 100` |
| **tab:eviction-ablation-combined** | See [Detailed Reproduction Commands](#detailed-reproduction-commands) |
| **tab:preemption-stats-combined** | See [Detailed Reproduction Commands](#detailed-reproduction-commands) |
| **Inline evaluation numbers** | See [Detailed Reproduction Commands](#detailed-reproduction-commands) |

### Detailed Reproduction Commands

#### tab:eviction-ablation-combined

Ablation study table (scheduler + eviction strategy speedups). Run `compute_scheduler_improvements.py` against all 6 delay/ablation directories — 3 Crawler + 3 ANNS. Each invocation produces a `.txt` file in the output directory.

**Crawler:**
```bash
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10 --output-dir tables --dataset-name H200_crawler_cost_based --max-qps 4
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only --output-dir tables --dataset-name H200_crawler_recomp_only --max-qps 4
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_swap_only --output-dir tables --dataset-name H200_crawler_swap_only --max-qps 4
```

**ANNS:**
```bash
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30 --output-dir tables --dataset-name H200_anns_cost_based --max-qps 2
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only --output-dir tables --dataset-name H200_anns_recomp_only --max-qps 2
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_swap_only --output-dir tables --dataset-name H200_anns_swap_only --max-qps 2
```

#### tab:preemption-stats-combined

Preemption statistics table. Run `analyze_preemptions.py` against the same 6 directories (output is printed to stdout).

```bash
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_swap_only
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_swap_only
```

#### Inline evaluation numbers

All speedup ratios cited in the paper body (Sections 3.1–3.6). Run `compute_scheduler_improvements.py` against the standard (no pressure) H200 and H100 run logs.

```bash
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir tables --dataset-name H200_crawler --max-qps 4
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir tables --dataset-name H200_anns --max-qps 2
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H100_enhanced_schedulers_v1_full --output-dir tables --dataset-name H100_crawler --max-qps 4
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H100_enhanced_schedulers_v1_full --output-dir tables --dataset-name H100_anns --max-qps 2
```

## Re-running Experiments

### Building Stream2LLM Engine

The `stream2llm/` directory contains the modified vLLM engine with streaming input support.

```bash
# Install Stream2LLM (requires CUDA-capable GPU)
cd stream2llm
wget https://files.pythonhosted.org/packages/c4/9d/64e107313a19327b049a2267871cceb9b0415f79ee5c00dc360099f929e8/vllm-0.8.1-cp38-abi3-manylinux1_x86_64.whl

# Environment variables (update with each vLLM release)
export VLLM_VERSION=0.8.1
export VLLM_PRECOMPILED_WHEEL_LOCATION=${PWD}/vllm-0.8.1-cp38-abi3-manylinux1_x86_64.whl

# Install in development mode
pip install -e .
cd ..
```

**Hardware requirements:** NVIDIA GPU with compute capability >= 7.0 (e.g., A40, H100, H200). Tensor parallelism requires multiple GPUs.

See [`experiments/README.md`](experiments/README.md) for detailed instructions with exact commands for running all 10 experiment configurations, ablation studies, SLURM submission, and plot generation.

## Data Organization

The `data/` submodule ([rbachkaniwala3/stream2llm-data](https://huggingface.co/datasets/rbachkaniwala3/stream2llm-data)) contains:

- **`run_log/crawler/`** and **`run_log/anns/`**: Experiment run logs (`run_metrics.csv` + `config_*.yaml`) for 10 configurations across H200 and H100 hardware
- **`anns/`**: ANNS workload data (corpus content, query trace map, 4997 pipeline traces)
- **`crawl/`**: Crawler workload data (4322 trace CSVs)
- **`perf_model/`**: Performance model JSONs (7 recomputation + 11 swap)

## License

This project is licensed under the [MIT License](LICENSE).
