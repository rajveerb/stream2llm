#!/bin/bash

# Script to generate all plots for ANNS analysis
# Usage: bash experiments/anns/generate_all_plots.sh <log_dir> [output_base_dir]

set -e  # Exit on any error

# Check if log directory is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <log_dir> [output_base_dir]"
    echo ""
    echo "Arguments:"
    echo "  log_dir         - Input log directory containing run results"
    echo "  output_base_dir - Base output directory (default: experiments/anns/analysis_results)"
    echo ""
    echo "Example:"
    echo "  $0 data/run_log/anns/test"
    echo "  $0 data/run_log/anns/test my_analysis"
    exit 1
fi

LOG_DIR="$1"
OUTPUT_BASE_DIR="${2:-experiments/anns/analysis_results}"

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory '$LOG_DIR' does not exist"
    exit 1
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

echo "========================================="
echo "ANNS Plot Generation Script"
echo "========================================="
echo "Log directory: $LOG_DIR"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo ""

# Extract log directory name for output subdirectory
LOG_DIR_NAME=$(basename "$LOG_DIR")
OUTPUT_DIR="$OUTPUT_BASE_DIR/$LOG_DIR_NAME"

echo "=== 1. Generating TTFT CDF Plots ==="
python scripts/anns/plotter_utils/plot_ttft_cdf_percentiles.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/ttft_cdf_percentiles"
echo "✓ TTFT CDF plots saved to: $OUTPUT_DIR/ttft_cdf_percentiles"
echo ""

echo "=== 2. Generating TTFT vs QPS Comparison Plots ==="
python scripts/anns/plotter_utils/plot_ttft_qps_comparison.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/ttft_qps_comparison"
echo "✓ TTFT vs QPS comparison plots saved to: $OUTPUT_DIR/ttft_qps_comparison"
echo ""

echo "=== 3. Generating Engine Event Duration Plots ==="
python scripts/anns/plotter_utils/plot_engine_event_durations.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/engine_event_durations"
echo "✓ Engine event duration plots saved to: $OUTPUT_DIR/engine_event_durations"
echo ""

echo "=== 4. Generating Preemption Analysis Plots ==="
python scripts/anns/plotter_utils/plot_preemptions.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/preemption_analysis"
echo "✓ Preemption analysis plots saved to: $OUTPUT_DIR/preemption_analysis"
echo ""

echo "=== 5. Generating Queue Box Plot Per Query ==="
python scripts/anns/plotter_utils/plot_queue_boxplot_per_query.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/queue_boxplot"
echo "✓ Queue box plot per query saved to: $OUTPUT_DIR/queue_boxplot"
echo ""

echo "=== 6. Generating Tokens Invalidated Plots ==="
python scripts/anns/plotter_utils/plot_tokens_invalidated.py \
    --log_dir "$LOG_DIR" \
    --output_dir "$OUTPUT_DIR/tokens_invalidated"
echo "✓ Tokens invalidated plots saved to: $OUTPUT_DIR/tokens_invalidated"
echo ""

echo "=== 7. Generating Trace Completion Time vs QPS Plots ==="
python scripts/anns/plotter_utils/plot_trace_completion_scheduler_comparison.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/trace_completion_comparison"
echo "✓ Trace completion plots saved to: $OUTPUT_DIR/trace_completion_comparison"
echo ""

echo "========================================="
echo "All plots generated successfully!"
echo "========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated plot categories:"
echo "  1. TTFT CDF percentiles"
echo "     - Streaming vs non-streaming comparison"
echo "     - Improvement ratio tables"
echo "     - Multiple QPS values"
echo ""
echo "  2. TTFT vs QPS comparison"
echo "     - Mean and P95 latency trends"
echo "     - Scaling behavior analysis"
echo ""
echo "  3. Engine event durations"
echo "     - Box plots for QUEUED, SCHEDULED, KV_ON_GPU events"
echo "     - QPS vs duration analysis"
echo "     - Streaming vs non-streaming separation"
echo ""
echo "  4. Preemption analysis"
echo "     - SWAP and RECOMPUTE preemption counts"
echo "     - Stacked bar charts by QPS and delay multiplier"
echo "     - Combined and individual preemption type plots"
echo ""
echo "  5. Queue analysis"
echo "     - Box plots of queue depths per query"
echo "     - Distribution analysis"
echo ""
echo "  6. Tokens invalidated"
echo "     - CDF and histogram of invalidation patterns"
echo "     - ANNS-specific streaming analysis"
echo ""
echo "  7. Trace completion time vs QPS"
echo "     - Combined and speedup plots"
echo "     - Scheduler comparison"
echo ""
echo "To view results:"
echo "  ls -la $OUTPUT_DIR/"
echo ""
echo "For detailed usage of individual scripts, see:"
echo "  scripts/anns/plotter_utils/README.md"
echo "========================================="
