#!/usr/bin/env python3
"""
Plot TTFT CDF for streaming and non-streaming requests across different schedulers on the same figure.
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


def extract_replay_rate(run_dir: str) -> float:
    """Extract the replay rate from the config file in the run directory."""
    config_files = glob.glob(os.path.join(run_dir, "config_*.yaml"))
    if not config_files:
        raise ValueError(f"No config file found in {run_dir}")

    config_file = config_files[0]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return 1.0 / config['replay']['poisson_avg_arrival_time']


def extract_scheduler_name(run_dir: str) -> str:
    """Extract scheduler name from run directory path."""
    # Handle different directory structures
    parts = run_dir.split('/')

    # Look for scheduler names in common positions
    scheduler_names = [
        'recomp', 'swap', 'recomp_and_swap', 'default_vllm', 'fcfs_lru',
        'lcas_lifo', 'mcps_lce', 'oeda_pbas', 'stream_based_v1', 'lcas_cplusp'
    ]

    for part in reversed(parts):
        if part in scheduler_names:
            return part

    # If not found in path, try to extract from config
    try:
        config_files = glob.glob(os.path.join(run_dir, "config_*.yaml"))
        if config_files:
            with open(config_files[0], 'r') as f:
                config = yaml.safe_load(f)
            # Try to extract scheduler info from config if available
            if 'scheduler' in config:
                return config['scheduler'].get('type', 'unknown')
    except:
        pass

    return 'unknown'


def extract_ttft_times(metrics_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Time To First Token (TTFT) metrics from a metrics file."""
    df = pd.read_csv(metrics_file, low_memory=False)
    ttft_df = df[df['event_type'] == 'query_ttft']
    streaming_ttft = ttft_df[ttft_df['stream'] == True]['duration_secs'].values
    non_streaming_ttft = ttft_df[ttft_df['stream'] ==
                                 False]['duration_secs'].values
    return streaming_ttft, non_streaming_ttft


def plot_single_cdf(ax, ttft_data: np.ndarray, label: str, color: str,
                    linestyle: str):
    """Plot a single CDF and return percentiles."""
    if len(ttft_data) == 0:
        return None

    sorted_data = np.sort(ttft_data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data,
            cdf_values,
            linestyle,
            color=color,
            linewidth=2,
            label=label,
            alpha=0.8)

    return {
        'P25': np.percentile(ttft_data, 25),
        'P50': np.percentile(ttft_data, 50),
        'P75': np.percentile(ttft_data, 75),
        'P95': np.percentile(ttft_data, 95)
    }


def create_improvement_table(streaming_p: dict, non_streaming_p: dict) -> str:
    """Create improvement ratio table text."""
    improvements = {}
    for perc in ['P25', 'P50', 'P75', 'P95']:
        improvements[perc] = non_streaming_p[perc] / streaming_p[
            perc] if streaming_p[perc] > 0 else float('inf')

    table_lines = [
        'TTFT Improvement Ratio:', '─' * 22, ' P25   P50   P75   P95',
        f'{improvements["P25"]:5.2f} {improvements["P50"]:5.2f} {improvements["P75"]:5.2f} {improvements["P95"]:5.2f}'
    ]
    return '\n'.join(table_lines)


def organize_data_by_scheduler_and_rate(metrics_files: List[str]) -> Dict:
    """Organize data by scheduler type and replay rate."""
    data = defaultdict(lambda: defaultdict(lambda: {
        'streaming': [],
        'non_streaming': []
    }))

    for metrics_file in metrics_files:
        run_dir = os.path.dirname(metrics_file)
        try:
            replay_rate = extract_replay_rate(run_dir)
            scheduler_name = extract_scheduler_name(run_dir)
            streaming_ttft, non_streaming_ttft = extract_ttft_times(
                metrics_file)

            data[scheduler_name][replay_rate]['streaming'].extend(
                streaming_ttft)
            data[scheduler_name][replay_rate]['non_streaming'].extend(
                non_streaming_ttft)
        except Exception as e:
            print(f"Warning: Skipping {metrics_file} due to error: {e}")
            continue

    return data


def plot_ttft_cdf_scheduler_comparison(metrics_files: List[str],
                                       output_dir: str,
                                       target_rate: float = None):
    """Create TTFT CDF comparison plots across different schedulers."""

    # Organize data
    data = organize_data_by_scheduler_and_rate(metrics_files)

    if not data:
        print("No data found to plot.")
        return

    # Get all available rates if target_rate not specified
    if target_rate is None:
        all_rates = set()
        for scheduler_data in data.values():
            all_rates.update(scheduler_data.keys())
        rates_to_plot = sorted(all_rates)
    else:
        # Find the closest rate to target_rate
        all_rates = set()
        for scheduler_data in data.values():
            all_rates.update(scheduler_data.keys())

        if all_rates:
            closest_rate = min(all_rates, key=lambda x: abs(x - target_rate))
            rates_to_plot = [closest_rate]
        else:
            rates_to_plot = []

    if not rates_to_plot:
        print(f"No data found for target rate {target_rate}")
        return

    # Color and style mappings
    scheduler_colors = {
        'recomp': '#1f77b4',
        'swap': '#ff7f0e',
        'recomp_and_swap': '#2ca02c',
        'default_vllm': '#d62728',
        'fcfs_lru': '#9467bd',
        'lcas_lifo': '#8c564b',
        'mcps_lce': '#e377c2',
        'oeda_pbas': '#7f7f7f',
        'stream_based_v1': '#bcbd22',
        'lcas_cplusp': '#17becf',
    }

    # Plot for each replay rate
    for rate in rates_to_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        streaming_plotted = False
        non_streaming_plotted = False

        for scheduler_name in sorted(data.keys()):
            if rate in data[scheduler_name]:
                color = scheduler_colors.get(scheduler_name, '#000000')

                # Plot streaming data
                streaming_data = np.array(
                    data[scheduler_name][rate]['streaming'])
                if len(streaming_data) > 0:
                    plot_single_cdf(ax1, streaming_data, scheduler_name, color,
                                    '-')
                    streaming_plotted = True

                # Plot non-streaming data
                non_streaming_data = np.array(
                    data[scheduler_name][rate]['non_streaming'])
                if len(non_streaming_data) > 0:
                    plot_single_cdf(ax2, non_streaming_data, scheduler_name,
                                    color, '--')
                    non_streaming_plotted = True

        # Setup streaming subplot
        if streaming_plotted:
            ax1.set_xlabel('TTFT (seconds)', fontsize=12)
            ax1.set_ylabel('CDF', fontsize=12)
            ax1.set_title(f'Streaming TTFT CDF ({rate:.3f} QPS)',
                          fontsize=14,
                          fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Setup non-streaming subplot
        if non_streaming_plotted:
            ax2.set_xlabel('TTFT (seconds)', fontsize=12)
            ax2.set_ylabel('CDF', fontsize=12)
            ax2.set_title(f'Non-Streaming TTFT CDF ({rate:.3f} QPS)',
                          fontsize=14,
                          fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Overall title
        fig.suptitle(f'TTFT CDF Comparison Across Schedulers - {rate:.3f} QPS',
                     fontsize=16,
                     fontweight='bold')

        # Save plot
        plt.tight_layout()
        filename = f'ttft_cdf_scheduler_comparison_{rate:.3f}_qps.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Scheduler comparison plot saved: {filepath}")


def plot_ttft_cdf_combined_comparison(metrics_files: List[str],
                                      output_dir: str,
                                      target_rate: float = None):
    """Create subplots for each scheduler with streaming/non-streaming on same subplot."""

    # Organize data
    data = organize_data_by_scheduler_and_rate(metrics_files)

    if not data:
        print("No data found to plot.")
        return

    # Get rates to plot
    if target_rate is None:
        all_rates = set()
        for scheduler_data in data.values():
            all_rates.update(scheduler_data.keys())
        rates_to_plot = sorted(all_rates)
    else:
        # Find the closest rate to target_rate
        all_rates = set()
        for scheduler_data in data.values():
            all_rates.update(scheduler_data.keys())

        if all_rates:
            closest_rate = min(all_rates, key=lambda x: abs(x - target_rate))
            rates_to_plot = [closest_rate]
        else:
            rates_to_plot = []

    if not rates_to_plot:
        print(f"No data found for target rate {target_rate}")
        return

    # Plot for each replay rate
    for rate in rates_to_plot:
        # Get schedulers with data for this rate
        schedulers_with_data = [
            s for s in sorted(data.keys()) if rate in data[s]
        ]
        n_schedulers = len(schedulers_with_data)

        if n_schedulers == 0:
            continue

        # Calculate subplot grid (prefer wider layout)
        if n_schedulers <= 3:
            rows, cols = 1, n_schedulers
        elif n_schedulers <= 6:
            rows, cols = 2, 3
        else:
            rows = (n_schedulers + 2) // 3
            cols = 3

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_schedulers == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()

        for idx, scheduler_name in enumerate(schedulers_with_data):
            ax = axes[idx]

            # Plot streaming data (solid line)
            streaming_data = np.array(data[scheduler_name][rate]['streaming'])
            streaming_p = None
            if len(streaming_data) > 0:
                streaming_p = plot_single_cdf(ax, streaming_data, 'Streaming',
                                              '#1f77b4', '-')

            # Plot non-streaming data (dashed line)
            non_streaming_data = np.array(
                data[scheduler_name][rate]['non_streaming'])
            non_streaming_p = None
            if len(non_streaming_data) > 0:
                non_streaming_p = plot_single_cdf(ax, non_streaming_data,
                                                  'Non-Streaming', '#ff7f0e',
                                                  '--')

            # Setup subplot
            ax.set_xlabel('TTFT (seconds)', fontsize=12)
            ax.set_ylabel('CDF', fontsize=12)
            ax.set_title(f'{scheduler_name.replace("_", " ").title()}',
                         fontsize=14,
                         fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Add vertical reference line at 1 second
            vertical_line_x = 1
            ax.axvline(x=vertical_line_x,
                       color='black',
                       linestyle=':',
                       linewidth=1.5,
                       alpha=0.7)

            # Calculate and display percentage below threshold
            if len(streaming_data) > 0:
                streaming_below_x = np.sum(streaming_data <= vertical_line_x
                                           ) / len(streaming_data) * 100
                ax.text(0.3,
                        0.7,
                        f'{streaming_below_x:.1f}%',
                        rotation=0,
                        fontsize=9,
                        color='#1f77b4',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white',
                                  alpha=0.8))

            if len(non_streaming_data) > 0:
                non_streaming_below_x = np.sum(
                    non_streaming_data <= vertical_line_x) / len(
                        non_streaming_data) * 100
                ax.text(0.3,
                        0.5,
                        f'{non_streaming_below_x:.1f}%',
                        rotation=0,
                        fontsize=9,
                        color='#ff7f0e',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white',
                                  alpha=0.8))

            # Add improvement table if both modes have data
            if streaming_p and non_streaming_p:
                table_text = create_improvement_table(streaming_p,
                                                      non_streaming_p)
                ax.text(0.97,
                        0.08,
                        table_text,
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment='bottom',
                        fontweight='bold',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='white',
                                  alpha=0.9,
                                  edgecolor='gray'))

            # Add legend to each subplot
            if len(streaming_data) > 0 or len(non_streaming_data) > 0:
                ax.legend(fontsize=10)

        # Hide unused subplots
        for idx in range(n_schedulers, len(axes)):
            axes[idx].set_visible(False)

        # Add overall title
        fig.suptitle(f'TTFT CDF Comparison by Scheduler ({rate:.3f} QPS)',
                     fontsize=18,
                     fontweight='bold',
                     y=0.98)

        # Save plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        filename = f'ttft_cdf_subplots_comparison_{rate:.3f}_qps.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Subplots comparison saved: {filepath}")


def main():
    """Main function to create TTFT CDF scheduler comparison plots."""
    parser = argparse.ArgumentParser(
        description=
        "Create TTFT CDF comparison plots across different schedulers")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing run logs with scheduler subdirectories")
    parser.add_argument("--output-dir",
                        type=str,
                        default="ttft_scheduler_comparison",
                        help="Output directory")
    parser.add_argument(
        "--target-rate",
        type=float,
        help=
        "Target replay rate (QPS) to plot. If not specified, plots all rates.")
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help=
        "Only create combined plots (streaming and non-streaming on same figure)"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all metrics files recursively
    metrics_files = glob.glob(os.path.join(args.log_dir, "**/run_metrics.csv"),
                              recursive=True)
    if not metrics_files:
        print("No metrics files found. Exiting.")
        return

    print(f"Found {len(metrics_files)} metrics files")

    if args.combined_only:
        plot_ttft_cdf_combined_comparison(metrics_files, args.output_dir,
                                          args.target_rate)
    else:
        # Create both separated and combined plots
        plot_ttft_cdf_scheduler_comparison(metrics_files, args.output_dir,
                                           args.target_rate)
        plot_ttft_cdf_combined_comparison(metrics_files, args.output_dir,
                                          args.target_rate)

    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
