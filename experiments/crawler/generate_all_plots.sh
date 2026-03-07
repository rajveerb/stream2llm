#!/bin/bash

# Script to generate all plots for crawler analysis
# Usage: bash experiments/crawler/generate_all_plots.sh <log_dir> [output_base_dir]

set -e  # Exit on any error

# Check if log directory is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <log_dir> [output_base_dir]"
    echo ""
    echo "Arguments:"
    echo "  log_dir         - Input log directory containing run results"
    echo "  output_base_dir - Base output directory (default: analysis_results)"
    echo ""
    echo "Example:"
    echo "  $0 data/run_log/crawler/A40_enhanced_schedulers_v2"
    echo "  $0 data/run_log/crawler/A40_enhanced_schedulers_v2 my_analysis"
    exit 1
fi

LOG_DIR="$1"
OUTPUT_BASE_DIR="${2:-analysis_results}"

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory '$LOG_DIR' does not exist"
    exit 1
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

echo "Generating all plots for log directory: $LOG_DIR"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo ""

# Extract log directory name for output subdirectory
LOG_DIR_NAME=$(basename "$LOG_DIR")
OUTPUT_DIR="$OUTPUT_BASE_DIR/$LOG_DIR_NAME"

echo "=== 1. Generating TTFT CDF Plots ==="
python scripts/crawler/plotter_utils/plot_ttft_cdf_percentiles.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/ttft_cdf_percentiles" \
    --grid-cols 5
echo "✓ TTFT CDF plots saved to: $OUTPUT_DIR/ttft_cdf_percentiles"
echo ""

echo "=== 2. Generating TTFT CCDF Tail Comparison Plots ==="
python scripts/crawler/plotter_utils/plot_ttft_ccdf_tail_comparison.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/ccdf_tail_comparison"
echo "✓ TTFT CCDF tail comparison plots saved to: $OUTPUT_DIR/ccdf_tail_comparison"
echo ""

echo "=== 3. Generating Combined TTFT CCDF Plots (All Schedulers Per QPS) ==="
python scripts/crawler/plotter_utils/plot_ttft_ccdf_combined.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/ccdf_combined"
echo "✓ Combined CCDF plots saved to: $OUTPUT_DIR/ccdf_combined"
echo ""

echo "=== 4. Generating Scheduler Comparison TTFT CDF Plots ==="
python scripts/crawler/plotter_utils/plot_ttft_cdf_scheduler_comparison.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/scheduler_comparison"
echo "✓ Scheduler comparison plots saved to: $OUTPUT_DIR/scheduler_comparison"
echo ""

echo "=== 5. Generating Trace Completion Time vs QPS Plots ==="
python scripts/crawler/plotter_utils/plot_trace_completion_scheduler_comparison.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/trace_completion_comparison"
echo "✓ Trace completion plots saved to: $OUTPUT_DIR/trace_completion_comparison"
echo ""

echo "=== 6. Generating Robust Statistical Scheduler Comparison Plots ==="
python scripts/crawler/plotter_utils/plot_scheduler_grid_robust_comparison.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/robust_statistical_analysis"
echo "✓ Robust statistical analysis plots saved to: $OUTPUT_DIR/robust_statistical_analysis"
echo ""

echo "=== 7. Generating Engine Event Duration Plots ==="
python scripts/crawler/plotter_utils/plot_engine_event_durations_by_scheduler.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/engine_event_durations"
echo "✓ Engine event duration plots saved to: $OUTPUT_DIR/engine_event_durations"
echo ""

echo "=== 8. Generating Preemption Analysis Plots ==="
python scripts/crawler/plotter_utils/plot_preemptions_by_scheduler.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/preemption_analysis"
echo "✓ Preemption analysis plots saved to: $OUTPUT_DIR/preemption_analysis"
echo ""

echo "=== 9. Generating TTFT vs QPS Comparison Plots ==="
python scripts/crawler/plotter_utils/plot_ttft_qps_comparison.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/ttft_qps_comparison"
echo "✓ TTFT vs QPS comparison plots saved to: $OUTPUT_DIR/ttft_qps_comparison"
echo ""

echo "=== 10. Running Completion Time Analysis ==="
python experiments/crawler/analyze_completion_times.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR/completion_time_analysis"
echo "✓ Completion time analysis saved to: $OUTPUT_DIR/completion_time_analysis"
echo ""

echo "========================================="
echo "All plots generated successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated plot categories:"
echo "  - TTFT CDF percentiles"
echo "  - TTFT CCDF tail comparison"
echo "  - Combined CCDF plots (all schedulers per QPS)"
echo "  - Scheduler comparison TTFT CDFs"
echo "  - Trace completion time comparisons"
echo "  - Robust statistical analysis"
echo "  - Engine event duration analysis"
echo "  - Preemption analysis"
echo "  - TTFT vs QPS comparison"
echo "  - Completion time analysis"
echo ""
echo "To view results:"
echo "  ls -la $OUTPUT_DIR/"