#!/usr/bin/env python3
"""
Create CDF plots of Time-To-First-Token (TTFT) latency for ANN experiments.
Creates a single figure with multiple subplots - one per run configuration.

This script is adapted from the crawler version but simplified for ANN experiments,
which have no scheduler dimension – only streaming vs non-streaming modes and replay rate (QPS).

Usage:
    python driver/anns/plotter_utils/plot_ttft_cdf_percentiles.py \
        --log-dir driver/anns/test_run_log \
        --output-dir driver/anns/analysis_results/ttft_cdf
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from typing import List, Dict, Tuple
from collections import defaultdict
from pathlib import Path

try:
    from .aggregate import extract_replay_rate, extract_delay_multiplier, extract_scheduler
except ImportError:
    from aggregate import extract_replay_rate, extract_delay_multiplier, extract_scheduler


def extract_ttft_times(metrics_file: str) -> tuple:
    """Extract Time To First Token (TTFT) metrics from a metrics file."""
    df = pd.read_csv(metrics_file, low_memory=False)
    ttft_df = df[df['event_type'] == 'query_ttft']
    streaming_ttft = ttft_df[ttft_df['stream'] == True]['duration_secs'].values
    non_streaming_ttft = ttft_df[ttft_df['stream'] ==
                                 False]['duration_secs'].values
    return streaming_ttft, non_streaming_ttft


def create_improvement_table(streaming_p: dict,
                             non_streaming_p: dict,
                             baseline_streaming_p: dict = None,
                             baseline_non_streaming_p: dict = None,
                             scheduler_name: str = None) -> str:
    """Create improvement ratio table text.

    For default_vllm: shows streaming vs non-streaming comparison
    For other schedulers: shows comparison vs both default_vllm baselines
    """
    if scheduler_name == 'default_vllm' or baseline_streaming_p is None:
        # Original behavior for default_vllm
        improvements = {}
        for perc in ['P25', 'P50', 'P75', 'P95']:
            improvements[perc] = non_streaming_p[perc] / streaming_p[
                perc] if streaming_p[perc] > 0 else float('inf')

        table_lines = [
            'TTFT Improvement Ratio:', '─' * 22, ' P25   P50   P75   P95',
            f'{improvements["P25"]:5.2f} {improvements["P50"]:5.2f} {improvements["P75"]:5.2f} {improvements["P95"]:5.2f}'
        ]
    else:
        # Dual baseline comparison for other schedulers
        # Compare streaming to both baselines
        improvements_vs_baseline_s = {}
        improvements_vs_baseline_ns = {}

        for perc in ['P25', 'P50', 'P75', 'P95']:
            # This scheduler's streaming vs default_vllm streaming
            improvements_vs_baseline_s[perc] = baseline_streaming_p[perc] / streaming_p[perc] \
                if streaming_p[perc] > 0 else float('inf')
            # This scheduler's streaming vs default_vllm non-streaming
            improvements_vs_baseline_ns[perc] = baseline_non_streaming_p[perc] / streaming_p[perc] \
                if streaming_p[perc] > 0 else float('inf')

        table_lines = [
            'Improvement vs Baseline:', '─' * 28,
            '           P25   P50   P75   P95',
            f'vs stream  {improvements_vs_baseline_s["P25"]:5.2f} '
            f'{improvements_vs_baseline_s["P50"]:5.2f} '
            f'{improvements_vs_baseline_s["P75"]:5.2f} '
            f'{improvements_vs_baseline_s["P95"]:5.2f}',
            f'vs non_stream {improvements_vs_baseline_ns["P25"]:5.2f} '
            f'{improvements_vs_baseline_ns["P50"]:5.2f} '
            f'{improvements_vs_baseline_ns["P75"]:5.2f} '
            f'{improvements_vs_baseline_ns["P95"]:5.2f}'
        ]

    return '\n'.join(table_lines)


def plot_single_cdf(ax, ttft_data: np.ndarray, mode: str, color: str,
                    style: str):
    """Plot a single CDF and return percentiles."""
    if len(ttft_data) == 0:
        return None

    sorted_data = np.sort(ttft_data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(
        sorted_data,
        cdf_values,
        style,
        color=color,
        linewidth=3,  # Increased from 2 to 3
        label=mode)

    return {
        'P25': np.percentile(ttft_data, 25),
        'P50': np.percentile(ttft_data, 50),
        'P75': np.percentile(ttft_data, 75),
        'P95': np.percentile(ttft_data, 95)
    }


def collect_data(
    metrics_files: List[str],
    min_rate: float = 0,
    max_rate: float = float('inf')
) -> Dict[Tuple[str, float, float], Dict]:
    """Collect TTFT data grouped by (scheduler, QPS, delay_multiplier)."""
    data = defaultdict(lambda: {
        'streaming': [],
        'non_streaming': [],
        'run_dirs': []
    })

    for metrics_file in metrics_files:
        run_dir = os.path.dirname(metrics_file)
        try:
            scheduler = extract_scheduler(run_dir)
            qps = extract_replay_rate(run_dir)
            delay_mult = extract_delay_multiplier(run_dir)

            if min_rate <= qps <= max_rate:
                streaming_ttft, non_streaming_ttft = extract_ttft_times(
                    metrics_file)

                if len(streaming_ttft) > 0 or len(non_streaming_ttft) > 0:
                    key = (scheduler, qps, delay_mult)
                    data[key]['streaming'].extend(streaming_ttft)
                    data[key]['non_streaming'].extend(non_streaming_ttft)
                    # Track run directory name
                    run_name = os.path.basename(run_dir)
                    if run_name not in data[key]['run_dirs']:
                        data[key]['run_dirs'].append(run_name)
        except Exception as e:
            print(f"Warning: Could not process {metrics_file}: {e}")
            continue

    return dict(data)


def plot_ttft_cdf_consolidated(metrics_files: List[str],
                               output_dir: str,
                               min_rate: float = 0,
                               max_rate: float = float('inf')):
    """Create separate figures per QPS, with subplots for each scheduler."""

    # Collect data
    data = collect_data(metrics_files, min_rate, max_rate)

    if not data:
        print(f"No data found for QPS between {min_rate} and {max_rate}")
        return

    print(f"Creating TTFT CDF plots with {len(data)} run configurations")

    # Group data by (QPS, delay_mult) to create one figure per QPS/delay combination
    qps_grouped = defaultdict(dict)
    for (scheduler, qps, delay_mult), run_data in data.items():
        qps_key = (qps, delay_mult)
        qps_grouped[qps_key][scheduler] = run_data

    # Create one figure for each QPS/delay combination
    os.makedirs(output_dir, exist_ok=True)

    for (qps, delay_mult), scheduler_data in sorted(qps_grouped.items()):
        # Extract default_vllm baseline data if available
        baseline_streaming = None
        baseline_non_streaming = None
        if 'default_vllm' in scheduler_data:
            baseline_streaming = np.array(
                scheduler_data['default_vllm']['streaming'])
            baseline_non_streaming = np.array(
                scheduler_data['default_vllm']['non_streaming'])

        n_schedulers = len(scheduler_data)

        # Determine grid layout
        n_cols = min(2, n_schedulers)
        n_rows = int(np.ceil(n_schedulers / n_cols))

        # Create figure with subplots - reduced size for better readability
        fig, axes = plt.subplots(n_rows,
                                 n_cols,
                                 figsize=(7 * n_cols, 5 * n_rows),
                                 squeeze=False)

        # Flatten axes for easier iteration
        axes = axes.flatten()

        # Plot each scheduler
        for idx, (scheduler,
                  run_data) in enumerate(sorted(scheduler_data.items())):
            ax = axes[idx]

            streaming_data = np.array(run_data['streaming'])
            non_streaming_data = np.array(run_data['non_streaming'])
            run_dirs = run_data['run_dirs']

            # Plot baseline default_vllm data first (if not plotting default_vllm itself)
            if scheduler != 'default_vllm' and baseline_streaming is not None:
                plot_single_cdf(ax, baseline_streaming,
                                'default_vllm (Streaming)', 'red', '-')
            if scheduler != 'default_vllm' and baseline_non_streaming is not None:
                plot_single_cdf(ax, baseline_non_streaming,
                                'default_vllm (Non-Streaming)', 'red', ':')

            # Plot this scheduler's CDFs
            streaming_p = plot_single_cdf(ax, streaming_data,
                                          f'{scheduler} Streaming', 'green',
                                          '-')
            non_streaming_p = plot_single_cdf(ax, non_streaming_data,
                                              f'{scheduler} Non-Streaming',
                                              'red', '--')

            # Setup subplot
            ax.set_xlabel('TTFT (seconds)', fontsize=16)
            ax.set_ylabel('CDF', fontsize=16)
            title = f'{scheduler}'
            ax.set_title(title, fontsize=18, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

            # Add run directory info as subtitle
            run_info = ', '.join(run_dirs) if len(
                run_dirs) <= 2 else f"{run_dirs[0]} (+{len(run_dirs)-1} more)"
            ax.text(0.5,
                    0.88,
                    f'Run: {run_info}',
                    transform=ax.transAxes,
                    fontsize=11,
                    ha='center',
                    va='top',
                    style='italic',
                    color='gray')

            # Add vertical line at 1s
            vertical_line_x = 1
            ax.axvline(
                x=vertical_line_x,
                color='black',
                linestyle=':',
                linewidth=2,  # Increased from 1.5 to 2
                alpha=0.7)

            # Calculate percentage of requests below 1s
            if len(streaming_data) > 0:
                streaming_below_x = np.sum(streaming_data <= vertical_line_x
                                           ) / len(streaming_data) * 100

            if len(non_streaming_data) > 0:
                non_streaming_below_x = np.sum(
                    non_streaming_data <= vertical_line_x) / len(
                        non_streaming_data) * 100

            ax.tick_params(axis='both', which='major',
                           labelsize=13)  # Increased from 10 to 13

            # Calculate baseline percentiles
            baseline_streaming_p = None
            baseline_non_streaming_p = None
            if scheduler != 'default_vllm' and baseline_streaming is not None and len(
                    baseline_streaming) > 0:
                baseline_streaming_p = {
                    'P25': np.percentile(baseline_streaming, 25),
                    'P50': np.percentile(baseline_streaming, 50),
                    'P75': np.percentile(baseline_streaming, 75),
                    'P95': np.percentile(baseline_streaming, 95)
                }
            if scheduler != 'default_vllm' and baseline_non_streaming is not None and len(
                    baseline_non_streaming) > 0:
                baseline_non_streaming_p = {
                    'P25': np.percentile(baseline_non_streaming, 25),
                    'P50': np.percentile(baseline_non_streaming, 50),
                    'P75': np.percentile(baseline_non_streaming, 75),
                    'P95': np.percentile(baseline_non_streaming, 95)
                }

            # Add improvement table
            # For default_vllm: need both streaming and non-streaming data
            # For other schedulers: need streaming data (compare to baselines)
            should_show_table = False
            if scheduler == 'default_vllm':
                should_show_table = streaming_p and non_streaming_p
            else:
                should_show_table = streaming_p and baseline_streaming_p and baseline_non_streaming_p

            if should_show_table:
                table_text = create_improvement_table(
                    streaming_p,
                    non_streaming_p if non_streaming_p else streaming_p,
                    baseline_streaming_p, baseline_non_streaming_p, scheduler)
                ax.text(
                    0.98,
                    0.02,
                    table_text,
                    transform=ax.transAxes,
                    fontsize=11,  # Increased from 8 to 11
                    verticalalignment='bottom',
                    fontweight='bold',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='wheat',
                              alpha=0.9,
                              edgecolor='gray'))

            # Add legend to each subplot for clarity - placed outside the plot
            ax.legend(loc='upper left',
                      fontsize=12,
                      bbox_to_anchor=(0, 1.15, 1, 0.2),
                      ncol=2,
                      mode="expand",
                      borderaxespad=0)

        # Hide unused subplots
        for idx in range(n_schedulers, len(axes)):
            axes[idx].axis('off')

        # Add main title
        fig.suptitle(
            f'TTFT Latency CDFs - QPS: {qps:.3f}, Delay: {delay_mult}x',
            fontsize=22,  # Increased from 18 to 22
            fontweight='bold')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot with QPS in filename
        filename = f'ttft_cdf_qps_{qps:.3f}_delay_{delay_mult}x.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"TTFT CDF plot saved to {filepath}")


def main():
    """Main function to create TTFT CDF plots."""
    parser = argparse.ArgumentParser(
        description="Create consolidated TTFT CDF plot for ANN experiments")
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
                        help="Minimum arrival rate (QPS)")
    parser.add_argument("--max-rate",
                        type=float,
                        default=float('inf'),
                        help="Maximum arrival rate (QPS)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_files = glob.glob(os.path.join(args.log_dir, "**/run_metrics.csv"),
                              recursive=True)
    if not metrics_files:
        print("No metrics files found. Exiting.")
        return

    print(f"Found {len(metrics_files)} metrics files")
    plot_ttft_cdf_consolidated(metrics_files, args.output_dir, args.min_rate,
                               args.max_rate)
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
