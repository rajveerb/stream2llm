#!/usr/bin/env python3
"""
Plot TTFT metrics (average and P95) vs QPS for ANN experiments.
Creates a single figure with subplots for different delay multipliers.

Usage:
    python driver/anns/plotter_utils/plot_ttft_qps_comparison.py \
        --log-dir driver/anns/test_run_log \
        --output-dir driver/anns/analysis_results/ttft_qps
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    from .aggregate import extract_replay_rate, extract_delay_multiplier, extract_scheduler
except ImportError:
    from aggregate import extract_replay_rate, extract_delay_multiplier, extract_scheduler


def simplify_scheduler_name(scheduler: str) -> str:
    """Simplify scheduler name for display in plots."""
    name_mapping = {
        'default_vllm': 'Default vLLM',
        'fcfs_lru': 'FCFS',
        'lcas_cplusp': 'LCAS',
        'mcps_lce': 'MCPS',
    }
    return name_mapping.get(scheduler, scheduler)


def extract_ttft_metrics(
        metrics_file: str,
        percentile: float = 95) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Extract TTFT metrics (mean and custom percentile) for streaming and non-streaming."""
    df = pd.read_csv(metrics_file, low_memory=False)
    ttft_df = df[df['event_type'] == 'query_ttft']

    streaming_ttft = ttft_df[ttft_df['stream'] == True]['duration_secs'].values
    non_streaming_ttft = ttft_df[ttft_df['stream'] ==
                                 False]['duration_secs'].values

    streaming_metrics = {}
    non_streaming_metrics = {}

    if len(streaming_ttft) > 0:
        streaming_metrics = {
            'mean': np.mean(streaming_ttft),
            f'p{int(percentile)}': np.percentile(streaming_ttft, percentile)
        }

    if len(non_streaming_ttft) > 0:
        non_streaming_metrics = {
            'mean': np.mean(non_streaming_ttft),
            f'p{int(percentile)}': np.percentile(non_streaming_ttft,
                                                 percentile)
        }

    return streaming_metrics, non_streaming_metrics


def collect_data_by_delay(
    metrics_files: List[str],
    percentile: float = 95,
    min_rate: float = 0,
    max_rate: float = float('inf')
) -> Tuple[Dict[Tuple[float, str], Dict], Dict[Tuple[float, str], List[str]]]:
    """Collect TTFT metrics data grouped by (delay_multiplier, scheduler), then by QPS.

    Returns:
        Tuple of (data_by_delay_sched, run_dirs_by_delay_sched)
    """
    data_by_delay_sched = defaultdict(lambda: defaultdict(lambda: {
        'streaming': {},
        'non_streaming': {}
    }))
    run_dirs_by_delay_sched = defaultdict(list)

    for metrics_file in metrics_files:
        run_dir = os.path.dirname(metrics_file)
        try:
            scheduler = extract_scheduler(run_dir)
            qps = extract_replay_rate(run_dir)
            delay_mult = extract_delay_multiplier(run_dir)

            if min_rate <= qps <= max_rate:
                streaming_metrics, non_streaming_metrics = extract_ttft_metrics(
                    metrics_file, percentile)

                key = (delay_mult, scheduler)

                if streaming_metrics:
                    data_by_delay_sched[key][qps][
                        'streaming'] = streaming_metrics
                if non_streaming_metrics:
                    data_by_delay_sched[key][qps][
                        'non_streaming'] = non_streaming_metrics

                # Track run directories by (delay_multiplier, scheduler)
                run_name = os.path.basename(run_dir)
                if run_name not in run_dirs_by_delay_sched[key]:
                    run_dirs_by_delay_sched[key].append(run_name)
        except Exception as e:
            print(f"Warning: Could not process {metrics_file}: {e}")
            continue

    return dict(data_by_delay_sched), dict(run_dirs_by_delay_sched)


def plot_ttft_qps_comparison_consolidated(data_by_delay_sched: Dict,
                                          run_dirs_by_delay_sched: Dict,
                                          output_dir: str,
                                          percentile: float = 95,
                                          min_rate: float = 0,
                                          max_rate: float = float('inf'),
                                          output_prefix: str = "ttft_qps_comparison_anns"):
    """Create subplots for each delay multiplier showing TTFT vs QPS with separate lines per scheduler."""

    if not data_by_delay_sched:
        print("No data available for plotting.")
        return

    # Group keys by delay multiplier to get unique delays and schedulers
    delays_to_schedulers = defaultdict(set)
    for (delay_mult, scheduler) in data_by_delay_sched.keys():
        delays_to_schedulers[delay_mult].add(scheduler)

    sorted_delays = sorted(delays_to_schedulers.keys())
    n_delays = len(sorted_delays)

    # Determine grid layout
    n_cols = min(2, n_delays)
    n_rows = (n_delays + n_cols - 1) // n_cols

    # Create figure with subplots - each row has 2 plots (mean and percentile) for one delay
    fig = plt.figure(figsize=(14 * n_cols, 6 * n_rows))

    percentile_key = f'p{int(percentile)}'

    # Define markers for different schedulers (use default colors)
    scheduler_markers = {
        'default_vllm': 'o',
        'fcfs_lru': 's',
        'lcas_cplusp': '^',
        'mcps_lce': 'D',
    }

    for delay_idx, delay_mult in enumerate(sorted_delays):
        schedulers = sorted(delays_to_schedulers[delay_mult])

        # Create two subplots for this delay multiplier
        ax1 = plt.subplot(n_rows, n_cols * 2, delay_idx * 2 + 1)
        ax2 = plt.subplot(n_rows, n_cols * 2, delay_idx * 2 + 2)

        # Plot data for each scheduler
        for scheduler in schedulers:
            key = (delay_mult, scheduler)
            data = data_by_delay_sched.get(key, {})

            if not data:
                continue

            # Get marker for this scheduler (fallback to default if not defined)
            marker = scheduler_markers.get(scheduler, 'x')

            # Separate data for streaming and non-streaming
            streaming_qps = []
            streaming_mean = []
            streaming_percentile = []

            non_streaming_qps = []
            non_streaming_mean = []
            non_streaming_percentile = []

            for qps in sorted(data.keys()):
                if 'streaming' in data[qps] and data[qps]['streaming']:
                    streaming_qps.append(qps)
                    streaming_mean.append(data[qps]['streaming']['mean'])
                    streaming_percentile.append(
                        data[qps]['streaming'][percentile_key])

                if 'non_streaming' in data[qps] and data[qps]['non_streaming']:
                    non_streaming_qps.append(qps)
                    non_streaming_mean.append(
                        data[qps]['non_streaming']['mean'])
                    non_streaming_percentile.append(
                        data[qps]['non_streaming'][percentile_key])

            # Simplify scheduler name for legend
            sched_name = simplify_scheduler_name(scheduler)

            # Plot 1: Average TTFT vs QPS
            if streaming_qps and streaming_mean:
                ax1.plot(streaming_qps,
                         streaming_mean,
                         marker=marker,
                         linestyle='-',
                         linewidth=3.5,
                         markersize=8,
                         label=f'{sched_name} (Streaming)')

            if non_streaming_qps and non_streaming_mean:
                ax1.plot(non_streaming_qps,
                         non_streaming_mean,
                         marker=marker,
                         linestyle='--',
                         linewidth=3.5,
                         markersize=8,
                         label=f'{sched_name} (Non-Streaming)',
                         alpha=0.6)

            # Plot 2: Custom percentile TTFT vs QPS
            if streaming_qps and streaming_percentile:
                ax2.plot(streaming_qps,
                         streaming_percentile,
                         marker=marker,
                         linestyle='-',
                         linewidth=3.5,
                         markersize=8,
                         label=f'{sched_name} (Streaming)')

            if non_streaming_qps and non_streaming_percentile:
                ax2.plot(non_streaming_qps,
                         non_streaming_percentile,
                         marker=marker,
                         linestyle='--',
                         linewidth=3.5,
                         markersize=8,
                         label=f'{sched_name} (Non-Streaming)',
                         alpha=0.6)

        # Customize Plot 1 (Average TTFT)
        ax1.set_xlabel('QPS (Queries Per Second)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average TTFT (seconds)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # Customize Plot 2 (Custom percentile TTFT)
        ax2.set_xlabel('QPS (Queries Per Second)', fontsize=14, fontweight='bold')
        ax2.set_ylabel(f'P{int(percentile)} TTFT (seconds)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=12)

    # Main title
    fig.suptitle('TTFT Performance Comparison - ANNS Experiments',
                 fontsize=20,
                 fontweight='bold')

    # Create a consolidated legend from the first subplot
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(4, len(handles)), fontsize=14, frameon=True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_prefix}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"TTFT vs QPS comparison plot saved to {output_path}")


def main():
    """Main function to create TTFT vs QPS comparison plots."""
    parser = argparse.ArgumentParser(
        description=
        "Create consolidated TTFT vs QPS comparison plots for ANN experiments")
    parser.add_argument("--log-dir",
                        type=str,
                        required=True,
                        help="Directory containing run logs")
    parser.add_argument("--output-dir",
                        type=str,
                        required=True,
                        help="Output directory")
    parser.add_argument("--min-rate",
                        type=float,
                        default=0,
                        help="Minimum QPS to include")
    parser.add_argument("--max-rate",
                        type=float,
                        default=float('inf'),
                        help="Maximum QPS to include")
    parser.add_argument(
        "-p",
        "--percentile",
        type=float,
        default=95,
        help="Percentile to use for the second plot (default: 95)")
    parser.add_argument("--output-prefix",
                        type=str,
                        default="ttft_qps_comparison_anns",
                        help="Prefix for output filename")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all metrics files
    metrics_files = glob.glob(os.path.join(args.log_dir, "**/run_metrics.csv"),
                              recursive=True)
    if not metrics_files:
        print("No metrics files found. Exiting.")
        return

    print(f"Found {len(metrics_files)} metrics files")

    # Collect and plot data
    data_by_delay, run_dirs_by_delay = collect_data_by_delay(
        metrics_files, args.percentile, args.min_rate, args.max_rate)

    if not data_by_delay:
        print("No data collected")
        return

    print(
        f"Creating plots for {len(data_by_delay)} delay multiplier configurations"
    )
    plot_ttft_qps_comparison_consolidated(data_by_delay, run_dirs_by_delay,
                                          args.output_dir, args.percentile,
                                          args.min_rate, args.max_rate,
                                          output_prefix=args.output_prefix)

    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
