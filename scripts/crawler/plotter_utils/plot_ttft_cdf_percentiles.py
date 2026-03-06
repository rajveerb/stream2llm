#!/usr/bin/env python3
"""
Standalone script to create CDF plots of TTFT latency with improvement ratios.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from typing import List


def extract_replay_rate(run_dir: str) -> float:
    """Extract the replay rate from the config file in the run directory."""
    config_file = os.path.join(run_dir,
                               f"config_{os.path.basename(run_dir)}.yaml")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return 1.0 / config['replay']['poisson_avg_arrival_time']


def extract_ttft_times(metrics_file: str) -> tuple:
    """Extract Time To First Token (TTFT) metrics from a metrics file."""
    df = pd.read_csv(metrics_file, low_memory=False)
    ttft_df = df[df['event_type'] == 'query_ttft']
    streaming_ttft = ttft_df[ttft_df['stream'] == True]['duration_secs'].values
    non_streaming_ttft = ttft_df[ttft_df['stream'] ==
                                 False]['duration_secs'].values
    return streaming_ttft, non_streaming_ttft


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


def plot_single_cdf(ax, ttft_data: np.ndarray, mode: str, color: str,
                    style: str):
    """Plot a single CDF and return percentiles."""
    if len(ttft_data) == 0:
        return None

    sorted_data = np.sort(ttft_data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data,
            cdf_values,
            style,
            color=color,
            linewidth=2,
            label=mode)

    return {
        'P25': np.percentile(ttft_data, 25),
        'P50': np.percentile(ttft_data, 50),
        'P75': np.percentile(ttft_data, 75),
        'P95': np.percentile(ttft_data, 95)
    }


def setup_subplot(ax, rate: float, streaming_p: dict, non_streaming_p: dict,
                  streaming_data: np.ndarray, non_streaming_data: np.ndarray):
    """Setup individual subplot with styling and improvement table."""
    ax.set_xlabel('TTFT (seconds)', fontsize=14)
    ax.set_ylabel('CDF', fontsize=14)
    ax.set_title(f'{rate:.2f} QPS', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    vertical_line_x = 1

    # Add vertical line at x ms
    ax.axvline(x=vertical_line_x,
               color='black',
               linestyle=':',
               linewidth=1.5,
               alpha=0.7,
               label='200ms')

    # Calculate percentage of requests below 200ms for each curve
    if len(streaming_data) > 0:
        streaming_below_x = np.sum(
            streaming_data <= vertical_line_x) / len(streaming_data) * 100
        # Add text annotation for streaming
        ax.text(0.3,
                0.7,
                f'{streaming_below_x:.1f}%',
                rotation=0,
                fontsize=9,
                color='green',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white',
                          alpha=0.8))

    if len(non_streaming_data) > 0:
        non_streaming_below_x = np.sum(non_streaming_data <= vertical_line_x
                                       ) / len(non_streaming_data) * 100
        # Add text annotation for non-streaming
        ax.text(0.3,
                0.5,
                f'{non_streaming_below_x:.1f}%',
                rotation=0,
                fontsize=9,
                color='red',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white',
                          alpha=0.8))

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add improvement table if both modes have data
    if streaming_p and non_streaming_p:
        table_text = create_improvement_table(streaming_p, non_streaming_p)
        ax.text(0.97,
                0.08,
                table_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='bottom',
                fontweight='bold',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white',
                          alpha=0.9,
                          edgecolor='gray'))


def generate_filename(min_rate: float, max_rate: float,
                      sorted_rates: List[float]) -> str:
    """Generate filename based on actual or specified rate ranges."""
    if not sorted_rates:
        return f'ttft_cdf_percentiles_{min_rate}_{max_rate}_qps.png'

    actual_min, actual_max = min(sorted_rates), max(sorted_rates)
    filename_min = actual_min if min_rate == 0 else min_rate
    filename_max = actual_max if max_rate == float('inf') else max_rate

    if filename_max == float('inf'):
        filename_max = actual_max

    return f'ttft_cdf_percentiles_{filename_min:.2f}_{filename_max:.2f}_qps.png'


def extract_scheduler_name(metrics_file: str) -> str:
    """Extract scheduler name from metrics file path."""
    path_parts = metrics_file.split(os.sep)
    # Find the scheduler name (should be 3rd from the end: scheduler/timestamp/run_metrics.csv)
    for i, part in enumerate(path_parts):
        if part == 'run_metrics.csv' and i >= 2:
            return path_parts[i - 2]
    return 'unknown'


def plot_ttft_cdf_with_percentiles(metrics_files: List[str],
                                   output_dir: str,
                                   min_rate: float = 0,
                                   max_rate: float = float('inf'),
                                   grid_rows: int = None,
                                   grid_cols: int = None):
    """Create CDF plots of TTFT latency with improvement ratios for each scheduler separately."""
    # Group data by scheduler first, then by replay rate
    scheduler_data = {}
    for metrics_file in metrics_files:
        run_dir = os.path.dirname(metrics_file)
        replay_rate = extract_replay_rate(run_dir)
        scheduler = extract_scheduler_name(metrics_file)

        if min_rate <= replay_rate <= max_rate:
            streaming_ttft, non_streaming_ttft = extract_ttft_times(
                metrics_file)
            if len(streaming_ttft) > 0 or len(non_streaming_ttft) > 0:
                if scheduler not in scheduler_data:
                    scheduler_data[scheduler] = {}

                if replay_rate not in scheduler_data[scheduler]:
                    scheduler_data[scheduler][replay_rate] = {
                        'streaming': [],
                        'non_streaming': []
                    }

                scheduler_data[scheduler][replay_rate]['streaming'].extend(
                    streaming_ttft)
                scheduler_data[scheduler][replay_rate]['non_streaming'].extend(
                    non_streaming_ttft)

    if not scheduler_data:
        print(
            f"No data found for arrival rates between {min_rate} and {max_rate} QPS"
        )
        return

    # Create plots for each scheduler separately
    for scheduler, rate_data in scheduler_data.items():
        print(f"Creating TTFT CDF plot for scheduler: {scheduler}")

        sorted_rates = sorted(rate_data.keys())
        n_rates = len(sorted_rates)

        # Determine grid layout
        if grid_rows is None and grid_cols is None:
            grid_rows, grid_cols = 1, n_rates
        elif grid_rows is None:
            grid_rows = (n_rates + grid_cols - 1) // grid_cols
        elif grid_cols is None:
            grid_cols = (n_rates + grid_rows - 1) // grid_rows

        # Create subplot layout
        fig, axes = plt.subplots(grid_rows,
                                 grid_cols,
                                 figsize=(5 * grid_cols, 4 * grid_rows))
        if n_rates == 1:
            axes = [
                axes
            ] if grid_rows == 1 and grid_cols == 1 else axes.flatten()[:1]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Plot each rate
        for idx, rate in enumerate(sorted_rates):
            ax = axes[idx]
            data = rate_data[rate]

            # Convert lists to numpy arrays
            streaming_data = np.array(data['streaming'])
            non_streaming_data = np.array(data['non_streaming'])

            streaming_p = plot_single_cdf(ax, streaming_data, 'Streaming',
                                          'green', '-')
            non_streaming_p = plot_single_cdf(ax, non_streaming_data,
                                              'Non-Streaming', 'red', '--')
            setup_subplot(ax, rate, streaming_p, non_streaming_p,
                          streaming_data, non_streaming_data)

        # Hide unused subplots
        for i in range(n_rates, len(axes)):
            axes[i].set_visible(False)

        # Add title and legend
        fig.suptitle(
            f'TTFT Latency CDFs for {scheduler.replace("_", " ").title()} - Queries per Second (QPS)',
            fontsize=18,
            fontweight='bold',
            y=0.98)

        from matplotlib.lines import Line2D
        line_legend = [
            Line2D([0], [0],
                   color='red',
                   linewidth=2,
                   linestyle='--',
                   label='Non-Streaming'),
            Line2D([0], [0],
                   color='green',
                   linewidth=2,
                   linestyle='-',
                   label='Streaming')
        ]
        fig.legend(handles=line_legend,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 0.92),
                   ncol=2,
                   fontsize=14)

        # Adjust layout and save
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(top=0.70, hspace=0.5, wspace=0.3)

        # Generate filename with scheduler name
        scheduler_filename = f'ttft_cdf_percentiles_{scheduler}_{generate_filename(min_rate, max_rate, sorted_rates)}'
        plt.savefig(os.path.join(output_dir, scheduler_filename),
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

        print(
            f"TTFT CDF plot for {scheduler} saved to {output_dir}/{scheduler_filename}"
        )


def main():
    """Main function to create TTFT CDF plots."""
    parser = argparse.ArgumentParser(
        description="Create TTFT CDF plots with improvement ratios")
    parser.add_argument("--log-dir",
                        type=str,
                        required=True,
                        help="Directory containing run logs")
    parser.add_argument("--output-dir",
                        type=str,
                        default="ttft_cdf_analysis",
                        help="Output directory")
    parser.add_argument("--min-rate",
                        type=float,
                        default=0,
                        help="Minimum arrival rate (QPS)")
    parser.add_argument("--max-rate",
                        type=float,
                        default=float('inf'),
                        help="Maximum arrival rate (QPS)")
    parser.add_argument("--grid-rows",
                        type=int,
                        help="Number of rows in subplot grid")
    parser.add_argument("--grid-cols",
                        type=int,
                        help="Number of columns in subplot grid")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_files = glob.glob(os.path.join(args.log_dir, "**/run_metrics.csv"),
                              recursive=True)
    if not metrics_files:
        print("No metrics files found. Exiting.")
        return

    print(f"Found {len(metrics_files)} metrics files")
    plot_ttft_cdf_with_percentiles(metrics_files, args.output_dir,
                                   args.min_rate, args.max_rate,
                                   args.grid_rows, args.grid_cols)
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
