#!/usr/bin/env python3
"""
Plot TTFT metrics (average and P95) vs QPS for different scheduling policies.
Creates two plots on a single figure: Average TTFT vs QPS and P95 TTFT vs QPS.
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


def simplify_scheduler_name(scheduler: str) -> str:
    """Simplify scheduler name for display in plots."""
    name_mapping = {
        'default_vllm': 'Default vLLM',
        'fcfs_lru': 'FCFS',
        'lcas_cplusp': 'LCAS',
        'mcps_lce': 'MCPS',
    }
    return name_mapping.get(scheduler, scheduler)


def extract_replay_rate(run_dir: str) -> float:
    """Extract the replay rate from the config file in the run directory."""
    config_file = os.path.join(run_dir, f"config_{os.path.basename(run_dir)}.yaml")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return 1.0 / config['replay']['poisson_avg_arrival_time']


def extract_scheduler_name(metrics_file: str) -> str:
    """Extract scheduler name from metrics file path."""
    path_parts = metrics_file.split(os.sep)
    # Find the scheduler name (should be 3rd from the end: scheduler/timestamp/run_metrics.csv)
    for i, part in enumerate(path_parts):
        if part == 'run_metrics.csv' and i >= 2:
            return path_parts[i - 2]
    return 'unknown'


def extract_ttft_metrics(metrics_file: str, percentile: float = 95) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Extract TTFT metrics (mean and custom percentile) for streaming and non-streaming."""
    df = pd.read_csv(metrics_file, low_memory=False)
    ttft_df = df[df['event_type'] == 'query_ttft']
    
    streaming_ttft = ttft_df[ttft_df['stream'] == True]['duration_secs'].values
    non_streaming_ttft = ttft_df[ttft_df['stream'] == False]['duration_secs'].values
    
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
            f'p{int(percentile)}': np.percentile(non_streaming_ttft, percentile)
        }
    
    return streaming_metrics, non_streaming_metrics


def collect_data(metrics_files: List[str], percentile: float = 95, min_rate: float = 0, max_rate: float = float('inf'), exclude_schedulers: List[str] = None) -> Dict:
    """Collect TTFT metrics data grouped by scheduler and QPS."""
    if exclude_schedulers is None:
        exclude_schedulers = []

    data = defaultdict(lambda: defaultdict(lambda: {'streaming': {}, 'non_streaming': {}}))

    for metrics_file in metrics_files:
        run_dir = os.path.dirname(metrics_file)
        try:
            replay_rate = extract_replay_rate(run_dir)
            scheduler = extract_scheduler_name(metrics_file)

            # Skip excluded schedulers
            if scheduler in exclude_schedulers:
                continue

            if min_rate <= replay_rate <= max_rate:
                streaming_metrics, non_streaming_metrics = extract_ttft_metrics(metrics_file, percentile)
                
                if streaming_metrics:
                    data[scheduler][replay_rate]['streaming'] = streaming_metrics
                if non_streaming_metrics:
                    data[scheduler][replay_rate]['non_streaming'] = non_streaming_metrics
        except Exception as e:
            print(f"Warning: Could not process {metrics_file}: {e}")
            continue
    
    return dict(data)


def plot_ttft_qps_comparison(data: Dict, output_dir: str, percentile: float = 95, min_rate: float = 0, max_rate: float = float('inf'), output_prefix: str = "ttft_qps_comparison_crawler"):
    """Create two plots: Average TTFT vs QPS and custom percentile TTFT vs QPS for different schedulers."""
    if not data:
        print("No data available for plotting.")
        return
    
    # Define colors and markers for different schedulers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scheduler_names = sorted(data.keys())
    color_map = {scheduler: colors[i % len(colors)] for i, scheduler in enumerate(scheduler_names)}
    marker_map = {scheduler: markers[i % len(markers)] for i, scheduler in enumerate(scheduler_names)}
    
    # Get percentile key
    percentile_key = f'p{int(percentile)}'
    
    # Plot for each scheduler
    for scheduler in scheduler_names:
        scheduler_data = data[scheduler]
        
        # Separate data for streaming and non-streaming
        streaming_qps = []
        streaming_mean = []
        streaming_percentile = []
        
        non_streaming_qps = []
        non_streaming_mean = []
        non_streaming_percentile = []
        
        for qps in sorted(scheduler_data.keys()):
            if 'streaming' in scheduler_data[qps] and scheduler_data[qps]['streaming']:
                streaming_qps.append(qps)
                streaming_mean.append(scheduler_data[qps]['streaming']['mean'])
                streaming_percentile.append(scheduler_data[qps]['streaming'][percentile_key])
            
            if 'non_streaming' in scheduler_data[qps] and scheduler_data[qps]['non_streaming']:
                non_streaming_qps.append(qps)
                non_streaming_mean.append(scheduler_data[qps]['non_streaming']['mean'])
                non_streaming_percentile.append(scheduler_data[qps]['non_streaming'][percentile_key])
        
        # Simplify scheduler name for legend
        sched_name = simplify_scheduler_name(scheduler)

        # Plot 1: Average TTFT vs QPS
        if streaming_qps and streaming_mean:
            ax1.plot(streaming_qps, streaming_mean,
                    color=color_map[scheduler], marker=marker_map[scheduler],
                    linestyle='-', linewidth=3.5, markersize=8,
                    label=f'{sched_name} (Streaming)')

        if non_streaming_qps and non_streaming_mean:
            ax1.plot(non_streaming_qps, non_streaming_mean,
                    color=color_map[scheduler], marker=marker_map[scheduler],
                    linestyle='--', linewidth=3.5, markersize=8, alpha=0.7,
                    label=f'{sched_name} (Non-Streaming)')

        # Plot 2: Custom percentile TTFT vs QPS
        if streaming_qps and streaming_percentile:
            ax2.plot(streaming_qps, streaming_percentile,
                    color=color_map[scheduler], marker=marker_map[scheduler],
                    linestyle='-', linewidth=3.5, markersize=8,
                    label=f'{sched_name} (Streaming)')

        if non_streaming_qps and non_streaming_percentile:
            ax2.plot(non_streaming_qps, non_streaming_percentile,
                    color=color_map[scheduler], marker=marker_map[scheduler],
                    linestyle='--', linewidth=3.5, markersize=8, alpha=0.7,
                    label=f'{sched_name} (Non-Streaming)')
    
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
    rate_info = ""
    if min_rate > 0 or max_rate != float('inf'):
        if max_rate == float('inf'):
            rate_info = f" (QPS ≥ {min_rate})"
        elif min_rate == 0:
            rate_info = f" (QPS ≤ {max_rate})"
        else:
            rate_info = f" (QPS: {min_rate}-{max_rate})"
    
    fig.suptitle(f'TTFT Performance Comparison Across Schedulers{rate_info}',
                 fontsize=20, fontweight='bold', y=0.98)

    # Create a consolidated legend from the first subplot
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=min(5, len(handles)), fontsize=14, frameon=True)

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15)
    
    # Generate filename
    if min_rate > 0 or max_rate != float('inf'):
        if max_rate == float('inf'):
            filename = f'{output_prefix}_{min_rate}plus_qps.png'
        elif min_rate == 0:
            filename = f'{output_prefix}_0_{max_rate}_qps.png'
        else:
            filename = f'{output_prefix}_{min_rate}_{max_rate}_qps.png'
    else:
        filename = f'{output_prefix}.png'
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"TTFT vs QPS comparison plot saved to {output_path}")


def main():
    """Main function to create TTFT vs QPS comparison plots."""
    parser = argparse.ArgumentParser(
        description="Create TTFT vs QPS comparison plots for different schedulers"
    )
    parser.add_argument("--log-dir", 
                       type=str, 
                       required=True,
                       help="Directory containing run logs")
    parser.add_argument("--output-dir", 
                       type=str, 
                       default="ttft_qps_analysis",
                       help="Output directory")
    parser.add_argument("--min-rate", 
                       type=float, 
                       default=0,
                       help="Minimum QPS to include")
    parser.add_argument("--max-rate", 
                       type=float, 
                       default=float('inf'),
                       help="Maximum QPS to include")
    parser.add_argument("-p", "--percentile",
                       type=float,
                       default=95,
                       help="Percentile to use for the second plot (default: 95)")
    parser.add_argument("--exclude-schedulers",
                       type=str,
                       nargs='+',
                       default=[],
                       help="List of scheduler names to exclude from the plot")
    parser.add_argument("--output-prefix",
                       type=str,
                       default="ttft_qps_comparison_crawler",
                       help="Prefix for output filename")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all metrics files
    metrics_files = glob.glob(os.path.join(args.log_dir, "**/run_metrics.csv"), recursive=True)
    if not metrics_files:
        print("No metrics files found. Exiting.")
        return

    print(f"Found {len(metrics_files)} metrics files")
    if args.exclude_schedulers:
        print(f"Excluding schedulers: {args.exclude_schedulers}")

    # Collect and plot data
    data = collect_data(metrics_files, args.percentile, args.min_rate, args.max_rate, args.exclude_schedulers)
    plot_ttft_qps_comparison(data, args.output_dir, args.percentile, args.min_rate, args.max_rate, output_prefix=args.output_prefix)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()