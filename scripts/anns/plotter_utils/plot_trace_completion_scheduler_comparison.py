#!/usr/bin/env python3
"""
Plot trace completion times across different schedulers for comparative analysis.
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


def extract_trace_completion_times(metrics_file: str) -> Tuple[float, float]:
    """Extract trace completion times (time from replay_start to replay_end)."""
    df = pd.read_csv(metrics_file, low_memory=False)

    # Find replay_start and replay_end events
    end_events = df[df['event_type'] == 'replay_end']

    trace_times = {}

    for _, end_row in end_events.iterrows():
        stream_value = end_row['stream']
        duration = end_row['duration_secs']
        trace_times[stream_value] = duration

    return trace_times.get(True), trace_times.get(False)


def organize_data_by_scheduler_and_rate(metrics_files: List[str]) -> Dict:
    """Organize trace completion time data by scheduler type and replay rate."""
    data = defaultdict(lambda: defaultdict(lambda: {
        'streaming': [],
        'non_streaming': []
    }))

    for metrics_file in metrics_files:
        run_dir = os.path.dirname(metrics_file)
        try:
            replay_rate = extract_replay_rate(run_dir)
            scheduler_name = extract_scheduler_name(run_dir)
            streaming_time, non_streaming_time = extract_trace_completion_times(
                metrics_file)

            if streaming_time is not None:
                data[scheduler_name][replay_rate]['streaming'].append(
                    streaming_time)
            if non_streaming_time is not None:
                data[scheduler_name][replay_rate]['non_streaming'].append(
                    non_streaming_time)
        except Exception as e:
            print(f"Warning: Skipping {metrics_file} due to error: {e}")
            continue

    return data


def plot_trace_completion_combined(metrics_files: List[str], output_dir: str):
    """Create a single plot with both streaming and non-streaming lines for each scheduler."""

    # Organize data
    data = organize_data_by_scheduler_and_rate(metrics_files)

    if not data:
        print("No data found to plot.")
        return

    # Color mappings for schedulers
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
        'unknown': '#17becf'
    }

    # Get all unique rates and create equally spaced positions
    all_rates = set()
    for scheduler_data in data.values():
        all_rates.update(scheduler_data.keys())
    sorted_rates = sorted(all_rates)
    rate_positions = {rate: i for i, rate in enumerate(sorted_rates)}

    # Create single combined plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot both streaming and non-streaming for each scheduler
    for scheduler_name in sorted(data.keys()):
        x_positions = []
        streaming_times = []
        non_streaming_times = []
        streaming_errors = []
        non_streaming_errors = []

        for rate in sorted(data[scheduler_name].keys()):
            stream_times = data[scheduler_name][rate]['streaming']
            non_stream_times = data[scheduler_name][rate]['non_streaming']

            if stream_times or non_stream_times:
                x_positions.append(rate_positions[rate])

                if stream_times:
                    streaming_times.append(np.mean(stream_times))
                    streaming_errors.append(
                        np.std(stream_times) if len(stream_times) > 1 else 0)
                else:
                    streaming_times.append(np.nan)
                    streaming_errors.append(0)

                if non_stream_times:
                    non_streaming_times.append(np.mean(non_stream_times))
                    non_streaming_errors.append(
                        np.std(non_stream_times) if len(non_stream_times) >
                        1 else 0)
                else:
                    non_streaming_times.append(np.nan)
                    non_streaming_errors.append(0)

        if x_positions:
            color = scheduler_colors.get(scheduler_name, '#000000')
            scheduler_label = scheduler_name.replace('_', ' ').title()

            # Plot streaming (solid line, circles)
            ax.errorbar(x_positions,
                        streaming_times,
                        yerr=streaming_errors,
                        marker='o',
                        linestyle='-',
                        linewidth=2,
                        markersize=6,
                        color=color,
                        label=f'{scheduler_label} (Streaming)',
                        capsize=4,
                        alpha=0.8)

            # Plot non-streaming (dashed line, squares)
            ax.errorbar(x_positions,
                        non_streaming_times,
                        yerr=non_streaming_errors,
                        marker='s',
                        linestyle='--',
                        linewidth=2,
                        markersize=6,
                        color=color,
                        label=f'{scheduler_label} (Non-Streaming)',
                        capsize=4,
                        alpha=0.8)

    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Trace Completion Time (seconds)', fontsize=12)
    ax.set_title(
        'Trace Completion Time vs QPS: Streaming vs Non-Streaming by Scheduler',
        fontsize=14,
        fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set equally spaced x-axis ticks with QPS labels
    ax.set_xticks(range(len(sorted_rates)))
    ax.set_xticklabels([f'{rate:.3f}' for rate in sorted_rates], rotation=45)

    # Save plot
    plt.tight_layout()
    filename = 'trace_completion_vs_qps_combined.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Combined trace completion vs QPS plot saved: {filepath}")


def plot_trace_completion_speedup(metrics_files: List[str], output_dir: str):
    """Create speedup line plot showing streaming vs non-streaming trace completion time ratios vs QPS."""

    # Organize data
    data = organize_data_by_scheduler_and_rate(metrics_files)

    if not data:
        print("No data found to plot.")
        return

    # Color mappings for schedulers
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
        'unknown': '#17becf'
    }

    # Get all unique rates and create equally spaced positions
    all_rates = set()
    for scheduler_data in data.values():
        all_rates.update(scheduler_data.keys())
    sorted_rates = sorted(all_rates)
    rate_positions = {rate: i for i, rate in enumerate(sorted_rates)}

    # Create speedup vs QPS plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot speedup for each scheduler
    for scheduler_name in sorted(data.keys()):
        x_positions = []
        speedups = []

        for rate in sorted(data[scheduler_name].keys()):
            stream_times = data[scheduler_name][rate]['streaming']
            non_stream_times = data[scheduler_name][rate]['non_streaming']

            if stream_times and non_stream_times:
                stream_mean = np.mean(stream_times)
                non_stream_mean = np.mean(non_stream_times)
                if stream_mean > 0:
                    speedup = non_stream_mean / stream_mean
                    x_positions.append(rate_positions[rate])
                    speedups.append(speedup)

        if x_positions and speedups:
            color = scheduler_colors.get(scheduler_name, '#000000')
            ax.plot(x_positions,
                    speedups,
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    color=color,
                    label=scheduler_name.replace('_', ' ').title(),
                    alpha=0.8)

    # Add horizontal line at y=1 (no speedup)
    ax.axhline(y=1,
               color='red',
               linestyle='--',
               alpha=0.7,
               label='No Speedup (1.0x)')

    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Speedup (Non-Streaming / Streaming)', fontsize=12)
    ax.set_title('Trace Completion Time Speedup vs QPS by Scheduler',
                 fontsize=14,
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set equally spaced x-axis ticks with QPS labels
    ax.set_xticks(range(len(sorted_rates)))
    ax.set_xticklabels([f'{rate:.3f}' for rate in sorted_rates], rotation=45)

    # Save plot
    plt.tight_layout()
    filename = 'trace_completion_speedup_vs_qps.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Trace completion speedup vs QPS plot saved: {filepath}")


def main():
    """Main function to create trace completion time scheduler comparison plots."""
    parser = argparse.ArgumentParser(
        description=
        "Create trace completion time vs QPS plots across different schedulers"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing run logs with scheduler subdirectories")
    parser.add_argument("--output-dir",
                        type=str,
                        default="trace_completion_scheduler_comparison",
                        help="Output directory")
    parser.add_argument(
        "--speedup-only",
        action="store_true",
        help="Only create speedup plots (streaming vs non-streaming ratios)")
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help=
        "Only create combined plot (streaming and non-streaming on same figure)"
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

    if args.speedup_only:
        plot_trace_completion_speedup(metrics_files, args.output_dir)
    elif args.combined_only:
        plot_trace_completion_combined(metrics_files, args.output_dir)
    else:
        # Create all plot types
        plot_trace_completion_combined(metrics_files, args.output_dir)
        plot_trace_completion_speedup(metrics_files, args.output_dir)

    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
