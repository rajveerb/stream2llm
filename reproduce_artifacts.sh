#!/bin/bash
# Stream2LLM Artifact Reproduction Script
# Runs all commands from the "Reproducing Paper Artifacts" summary table

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "Starting Stream2LLM artifact reproduction..."
echo "=============================================="

# fig:perf-model (Perf model comparison)
echo "Running: fig:perf-model (Perf model comparison)"
python scripts/utils/plotting/plot_recomp_vs_swap_clean.py --recomp_input data/perf_model/recomputation/H200_tp2_recomputation_latency.json --swap_input data/perf_model/swap/H200_tp2_swap_kernel_latency.json --recomp_input_2 data/perf_model/recomputation/A40_recomputation_latency.json --swap_input_2 data/perf_model/swap/A40_swap_kernel_latency.json --output_dir figures --output_prefix hardware_comparison --title_1 "H200 TP=2" --title_2 "A40"
echo "✓ Complete"
echo ""

# fig:ttft — Crawler row
echo "Running: fig:ttft — Crawler row"
python scripts/crawler/plotter_utils/plot_ttft_ccdf_combined_1x4.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_ccdf_crawler --hardware H200
echo "✓ Complete"
echo ""

# fig:ttft — ANNS row
echo "Running: fig:ttft — ANNS row"
python scripts/crawler/plotter_utils/plot_ttft_ccdf_anns_1x4.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_ccdf_anns
echo "✓ Complete"
echo ""

# fig:completion — Crawler
echo "Running: fig:completion — Crawler"
python scripts/crawler/plotter_utils/plot_trace_completion_time.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix trace_completion_time_crawler --max-qps 4
echo "✓ Complete"
echo ""

# fig:completion — ANNS
echo "Running: fig:completion — ANNS"
python scripts/anns/plotter_utils/plot_trace_completion_time.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix trace_completion_time_anns --max-qps 2
echo "✓ Complete"
echo ""

# fig:appendix-crawler-ttft-qps
echo "Running: fig:appendix-crawler-ttft-qps"
python scripts/crawler/plotter_utils/plot_ttft_qps_comparison.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_qps_comparison_crawler -p 95 --max-rate 4
echo "✓ Complete"
echo ""

# fig:appendix-anns-ttft-qps
echo "Running: fig:appendix-anns-ttft-qps"
python scripts/anns/plotter_utils/plot_ttft_qps_comparison.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --output-prefix ttft_qps_comparison_anns -p 95 --max-rate 2
echo "✓ Complete"
echo ""

# fig:tokens-invalidated-ccdf
echo "Running: fig:tokens-invalidated-ccdf"
python scripts/anns/plotter_utils/plot_tokens_invalidated_aggregated.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_full --output-dir figures --table-output-dir tables --min-qps 0.25 --max-qps 2.0
echo "✓ Complete"
echo ""

# tab:workload-characteristics — ANNS
echo "Running: tab:workload-characteristics — ANNS"
python data/anns/compute_workload_stats.py --corpus-prefix data/anns/retrieved_corpus_content --query-map data/anns/query_trace_map_5k.json --trace-dir data/anns/res --max-queries 500 --tokenizer-model meta-llama/Llama-3.1-8B-Instruct --output-dir tables
echo "✓ Complete"
echo ""

# tab:workload-characteristics — Crawler
echo "Running: tab:workload-characteristics — Crawler"
python data/crawl/compute_workload_stats.py --input-dir data/crawl/traces/simpleQA_ALL --tokenizer-model meta-llama/Llama-3.1-8B-Instruct --cores 100 --output-dir tables
echo "✓ Complete"
echo ""

# Chunk arrival characterization
echo "Running: Chunk arrival characterization"
python scripts/utils/analysis/chunk_arrival_characterization.py --anns-dir data/anns/res --crawler-dir data/crawl/traces/simpleQA_ALL --output-dir figures --table-dir tables
echo "✓ Complete"
echo ""

# Detailed Reproduction Commands

# tab:eviction-ablation-combined (Crawler)
echo "Running: tab:eviction-ablation-combined — Crawler (cost-based)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10 --output-dir tables --dataset-name H200_crawler_cost_based --max-qps 4
echo "✓ Complete"
echo ""

echo "Running: tab:eviction-ablation-combined — Crawler (recomp-only)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only --output-dir tables --dataset-name H200_crawler_recomp_only --max-qps 4
echo "✓ Complete"
echo ""

echo "Running: tab:eviction-ablation-combined — Crawler (swap-only)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_swap_only --output-dir tables --dataset-name H200_crawler_swap_only --max-qps 4
echo "✓ Complete"
echo ""

# tab:eviction-ablation-combined (ANNS)
echo "Running: tab:eviction-ablation-combined — ANNS (cost-based)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30 --output-dir tables --dataset-name H200_anns_cost_based --max-qps 2
echo "✓ Complete"
echo ""

echo "Running: tab:eviction-ablation-combined — ANNS (recomp-only)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only --output-dir tables --dataset-name H200_anns_recomp_only --max-qps 2
echo "✓ Complete"
echo ""

echo "Running: tab:eviction-ablation-combined — ANNS (swap-only)"
python scripts/utils/analysis/compute_scheduler_improvements.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_swap_only --output-dir tables --dataset-name H200_anns_swap_only --max-qps 2
echo "✓ Complete"
echo ""

# tab:preemption-stats-combined (Crawler)
echo "Running: tab:preemption-stats-combined — Crawler (cost-based)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10 --output-dir tables
echo "✓ Complete"
echo ""

echo "Running: tab:preemption-stats-combined — Crawler (recomp-only)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_recomp_only --output-dir tables
echo "✓ Complete"
echo ""

echo "Running: tab:preemption-stats-combined — Crawler (swap-only)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/crawler/H200_enhanced_schedulers_v1_full_delay_10_swap_only --output-dir tables
echo "✓ Complete"
echo ""

# tab:preemption-stats-combined (ANNS)
echo "Running: tab:preemption-stats-combined — ANNS (cost-based)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30 --output-dir tables
echo "✓ Complete"
echo ""

echo "Running: tab:preemption-stats-combined — ANNS (recomp-only)"
python scripts/utils/analysis/analyze_preemptions.py --log-dir data/run_log/anns/H200_enhanced_schedulers_v1_500q_delay_30_recomp_only --output-dir tables
echo "✓ Complete"
echo ""

echo "Running: tab:preemption-stats-combined — ANNS (swap-only)"
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

echo "=============================================="
echo "All artifact reproduction commands completed!"
