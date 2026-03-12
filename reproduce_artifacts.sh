#!/bin/bash
# Stream2LLM Artifact Reproduction Script
# Runs all commands from the "Reproducing Paper Artifacts" summary table

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "Starting Stream2LLM artifact reproduction..."
echo "=============================================="

# Figure 4 (Perf model comparison)
echo "Running: Figure 4 — Perf model comparison"
python scripts/utils/plotting/plot_recomp_vs_swap_clean.py --recomp_input data/perf_model/recomputation/H200_tp2_recomputation_latency.json --swap_input data/perf_model/swap/H200_tp2_swap_kernel_latency.json --recomp_input_2 data/perf_model/recomputation/A40_recomputation_latency.json --swap_input_2 data/perf_model/swap/A40_swap_kernel_latency.json --output_dir figures --output_prefix hardware_comparison --title_1 "H200 TP=2" --title_2 "A40"
echo "✓ Complete"
echo ""

# Figure 5 — TTFT CCDF (Crawler + ANNS stacked 2x4)
echo "Running: Figure 5 — TTFT CCDF (Crawler + ANNS stacked 2x4)"
python scripts/utils/plotting/plot_ttft_ccdf_stacked_2x4.py --crawler-log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --anns-log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures
echo "✓ Complete"
echo ""

# Figure 11 — Trace completion (Crawler + ANNS)
echo "Running: Figure 11 — Trace completion (Crawler + ANNS)"
python scripts/utils/plotting/plot_trace_completion_combined.py --crawler-log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --anns-log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures
echo "✓ Complete"
echo ""

# Figure 6 — Crawler TTFT vs QPS
echo "Running: Figure 6 — Crawler TTFT vs QPS"
python scripts/crawler/plotter_utils/plot_ttft_qps_comparison.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_qps_comparison_crawler -p 95 --max-rate 4
echo "✓ Complete"
echo ""

# Figure 7 — ANNS TTFT vs QPS
echo "Running: Figure 7 — ANNS TTFT vs QPS"
python scripts/anns/plotter_utils/plot_ttft_qps_comparison.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_qps_comparison_anns -p 95 --max-rate 2
echo "✓ Complete"
echo ""

# Figure 12 — Tokens invalidated CCDF
echo "Running: Figure 12 — Tokens invalidated CCDF"
python scripts/anns/plotter_utils/plot_tokens_invalidated_aggregated.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --table-output-dir tables --min-qps 0.25 --max-qps 2.0
echo "✓ Complete"
echo ""

# Table 2 — ANNS workload stats
echo "Running: Table 2 — ANNS workload stats"
python data/anns/compute_workload_stats.py --corpus-prefix data/anns/retrieved_corpus_content --query-map data/anns/query_trace_map_5k.json --trace-dir data/anns/res --max-queries 500 --tokenizer-model meta-llama/Llama-3.1-8B-Instruct --output-dir tables
echo "✓ Complete"
echo ""

# Table 2 — Crawler workload stats
echo "Running: Table 2 — Crawler workload stats"
python data/crawl/compute_workload_stats.py --input-dir data/crawl/traces/simpleQA_ALL --tokenizer-model meta-llama/Llama-3.1-8B-Instruct --cores $(nproc) --output-dir tables
echo "✓ Complete"
echo ""

# Figures 8–10 — Chunk arrival characterization
echo "Running: Figures 8–10 — Chunk arrival characterization"
python scripts/utils/analysis/chunk_arrival_characterization.py --anns-dir data/anns/res --crawler-dir data/crawl/traces/simpleQA_ALL --output-dir figures --table-dir tables
echo "✓ Complete"
echo ""

# Detailed Reproduction Commands

# Table 3 — Eviction ablation (Crawler)
echo "Running: Table 3 — Eviction ablation — Crawler (cost-based)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10 --output-dir tables --dataset-name H200_crawler_cost_based --max-qps 4
echo "✓ Complete"
echo ""

echo "Running: Table 3 — Eviction ablation — Crawler (recomp-only)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only --output-dir tables --dataset-name H200_crawler_recomp_only --max-qps 4
echo "✓ Complete"
echo ""

echo "Running: Table 3 — Eviction ablation — Crawler (swap-only)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_swap_only --output-dir tables --dataset-name H200_crawler_swap_only --max-qps 4
echo "✓ Complete"
echo ""

# Table 3 — Eviction ablation (ANNS)
echo "Running: Table 3 — Eviction ablation — ANNS (cost-based)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30 --output-dir tables --dataset-name H200_anns_cost_based --max-qps 2
echo "✓ Complete"
echo ""

echo "Running: Table 3 — Eviction ablation — ANNS (recomp-only)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only --output-dir tables --dataset-name H200_anns_recomp_only --max-qps 2
echo "✓ Complete"
echo ""

echo "Running: Table 3 — Eviction ablation — ANNS (swap-only)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_swap_only --output-dir tables --dataset-name H200_anns_swap_only --max-qps 2
echo "✓ Complete"
echo ""

# Table 4 — Preemption stats (Crawler)
echo "Running: Table 4 — Preemption stats — Crawler (cost-based)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10 --output-dir tables
echo "✓ Complete"
echo ""

echo "Running: Table 4 — Preemption stats — Crawler (recomp-only)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only --output-dir tables
echo "✓ Complete"
echo ""

echo "Running: Table 4 — Preemption stats — Crawler (swap-only)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_swap_only --output-dir tables
echo "✓ Complete"
echo ""

# Table 4 — Preemption stats (ANNS)
echo "Running: Table 4 — Preemption stats — ANNS (cost-based)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30 --output-dir tables
echo "✓ Complete"
echo ""

echo "Running: Table 4 — Preemption stats — ANNS (recomp-only)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only --output-dir tables
echo "✓ Complete"
echo ""

echo "Running: Table 4 — Preemption stats — ANNS (swap-only)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_swap_only --output-dir tables
echo "✓ Complete"
echo ""

# Inline evaluation numbers
echo "Running: Inline evaluation numbers — H200 Crawler"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir tables --dataset-name H200_crawler --max-qps 4
echo "✓ Complete"
echo ""

echo "Running: Inline evaluation numbers — H200 ANNS"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir tables --dataset-name H200_anns --max-qps 2
echo "✓ Complete"
echo ""

echo "Running: Inline evaluation numbers — H100 Crawler"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H100_enhanced_schedulers_v1_full --output-dir tables --dataset-name H100_crawler --max-qps 4
echo "✓ Complete"
echo ""

echo "Running: Inline evaluation numbers — H100 ANNS"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H100_enhanced_schedulers_v1_full --output-dir tables --dataset-name H100_anns --max-qps 2
echo "✓ Complete"
echo ""

# Scheduler sorting + budget allocation latency benchmark
echo "Running: Scheduler sorting latency benchmark — ANNS"
python scripts/utils/analysis/benchmark_scheduler_latency.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir tables --dataset-name anns
echo "✓ Complete"
echo ""

echo "Running: Scheduler sorting latency benchmark — Crawler"
python scripts/utils/analysis/benchmark_scheduler_latency.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir tables --dataset-name crawler
echo "✓ Complete"
echo ""

echo "=============================================="
echo "All artifact reproduction commands completed!"
