#!/usr/bin/env python3
"""
Plot box plot of queuing duration across all runs on a single figure.
Shows the distribution of queuing times per query grouped by QPS and delay multiplier.
Displays streaming vs non-streaming as side-by-side box plots.
"""
import argparse
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from vllm.v1.engine import EngineCoreEventType

try:
    from .aggregate import extract_replay_rate, extract_delay_multiplier, extract_scheduler
except ImportError:
    from aggregate import extract_replay_rate, extract_delay_multiplier, extract_scheduler

# Set global font sizes for better readability
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def convert_event_type(x):
    """Convert event_type strings to EngineCoreEventType enum."""
    if pd.isna(x):
        return None
    try:
        # Handle string format "EngineCoreEventType.EVENT_NAME"
        if isinstance(x, str) and x.startswith('EngineCoreEventType.'):
            event_name = x.split('.')[-1]
            return EngineCoreEventType[event_name]
        # Handle direct enum values
        elif isinstance(x, str):
            return EngineCoreEventType[x]
        # Handle already converted enum values
        elif isinstance(x, EngineCoreEventType):
            return x
        else:
            return x
    except (KeyError, AttributeError):
        # Return the original string for custom events
        return x


def extract_queue_durations(csv_file: Path) -> Dict[str, List[float]]:
    """Extract total queuing duration per query, separated by streaming status."""
    df = pd.read_csv(csv_file, low_memory=False)

    # Convert event types
    df['event_type'] = df['event_type'].apply(convert_event_type)

    # Filter for queued events
    queued_events = df[df['event_type'] == EngineCoreEventType.QUEUED]

    if queued_events.empty:
        return {'streaming': [], 'non_streaming': []}

    # Group by query_id and streaming status, then sum queuing durations
    streaming_queries = queued_events[queued_events['stream'] == True].groupby(
        'query_id')['duration_secs'].sum().tolist()
    non_streaming_queries = queued_events[
        queued_events['stream'] == False].groupby(
            'query_id')['duration_secs'].sum().tolist()

    return {
        'streaming': streaming_queries,
        'non_streaming': non_streaming_queries
    }


def collect_data(
        log_dir: Path
) -> Dict[Tuple[str, float, float], Dict[str, List[float]]]:
    """
    Collect all queue duration data from log directory.
    Groups by (scheduler, QPS, delay_multiplier) tuple.

    Returns:
        Dict mapping (scheduler, QPS, delay) -> {'streaming': [...], 'non_streaming': [...], 'run_dirs': [...]}
    """
    data = defaultdict(lambda: {
        'streaming': [],
        'non_streaming': [],
        'run_dirs': []
    })

    csv_files = list(log_dir.rglob("run_metrics.csv"))

    for csv_file in csv_files:
        try:
            scheduler = extract_scheduler(str(csv_file.parent))
            qps_val = extract_replay_rate(str(csv_file.parent))
            delay_mult = extract_delay_multiplier(str(csv_file.parent))
            key = (scheduler, qps_val, delay_mult)

            queue_durations = extract_queue_durations(csv_file)

            data[key]['streaming'].extend(queue_durations['streaming'])
            data[key]['non_streaming'].extend(queue_durations['non_streaming'])

            # Track run directory name
            run_name = csv_file.parent.name
            if run_name not in data[key]['run_dirs']:
                data[key]['run_dirs'].append(run_name)

        except Exception as e:
            print(f"[warn] Error processing {csv_file}: {e}")

    return data


def plot_queue_boxplot_consolidated(data: Dict, output_dir: Path):
    """
    Create box plots grouped by QPS showing different schedulers.
    Streaming vs non-streaming shown as side-by-side boxes with default_vllm baseline.
    """

    if not data:
        print("No queue data found")
        return

    # Group data by (QPS, delay_mult)
    qps_grouped = defaultdict(dict)
    for (scheduler, qps, delay_mult), queue_data in data.items():
        qps_key = (qps, delay_mult)
        qps_grouped[qps_key][scheduler] = queue_data

    # Create one plot for each QPS/delay combination
    output_dir.mkdir(parents=True, exist_ok=True)

    for (qps, delay_mult), scheduler_data in sorted(qps_grouped.items()):
        schedulers = sorted(scheduler_data.keys())

        # Collect data for box plots
        streaming_box_data = []
        non_streaming_box_data = []
        scheduler_labels = []
        colors_streaming = []
        colors_non_streaming = []

        for scheduler in schedulers:
            streaming_durations = scheduler_data[scheduler].get(
                'streaming', [])
            non_streaming_durations = scheduler_data[scheduler].get(
                'non_streaming', [])

            # Only add if we have data for at least one type
            if streaming_durations or non_streaming_durations:
                streaming_box_data.append(
                    streaming_durations if streaming_durations else [0])
                non_streaming_box_data.append(non_streaming_durations if
                                              non_streaming_durations else [0])
                scheduler_labels.append(scheduler)

                # Use red for default_vllm, green/yellow for others
                if scheduler == 'default_vllm':
                    colors_streaming.append('lightcoral')
                    colors_non_streaming.append('lightcoral')
                else:
                    colors_streaming.append('lightgreen')
                    colors_non_streaming.append('lightyellow')

        if not scheduler_labels:
            print(f"No queue data to plot for QPS {qps}")
            continue

        # Create side-by-side box plots
        fig, ax = plt.subplots(figsize=(max(10, len(scheduler_labels) * 2), 6))
        box_width = 0.35
        x = np.arange(len(scheduler_labels))

        # Boxplot for non-streaming (left side - baseline)
        if any(non_streaming_box_data):
            bp1 = ax.boxplot(non_streaming_box_data,
                             positions=x - box_width / 2,
                             widths=box_width,
                             patch_artist=True,
                             whiskerprops=dict(color='black'),
                             capprops=dict(color='black'),
                             flierprops=dict(marker='x',
                                             color='gray',
                                             markersize=6,
                                             markeredgecolor='gray'),
                             showfliers=True)
            # Color boxes individually
            for patch, color in zip(bp1['boxes'], colors_non_streaming):
                patch.set_facecolor(color)
                patch.set_edgecolor('black')

        # Boxplot for streaming (right side)
        if any(streaming_box_data):
            bp2 = ax.boxplot(streaming_box_data,
                             positions=x + box_width / 2,
                             widths=box_width,
                             patch_artist=True,
                             whiskerprops=dict(color='black'),
                             capprops=dict(color='black'),
                             flierprops=dict(marker='x',
                                             color='gray',
                                             markersize=6,
                                             markeredgecolor='gray'),
                             showfliers=True)
            # Color boxes individually
            for patch, color in zip(bp2['boxes'], colors_streaming):
                patch.set_facecolor(color)
                patch.set_edgecolor('black')

        # Customize the plot
        ax.set_xticks(x)
        ax.set_xticklabels(scheduler_labels, rotation=45, ha='right')
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('Total Queuing Duration per Query (seconds)')
        ax.set_title(
            f'Queuing Duration - QPS: {qps:.3f}, Delay: {delay_mult}x')

        # Add legend
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor='lightyellow',
                  edgecolor='black',
                  label='Non-Streaming'),
            Patch(facecolor='lightgreen', edgecolor='black',
                  label='Streaming'),
            Patch(facecolor='lightcoral',
                  edgecolor='black',
                  label='default_vllm (baseline)')
        ]
        ax.legend(handles=legend_handles, loc='upper right')

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Save plot
        output_file = output_dir / f'queue_boxplot_qps_{qps:.3f}_delay_{delay_mult}x.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved queue boxplot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description=
        'Plot consolidated box plot of queuing duration across all ANNS runs')
    parser.add_argument('--log-dir',
                        type=Path,
                        required=True,
                        help='Path to log directory containing run results')
    parser.add_argument('--output-dir',
                        type=Path,
                        required=True,
                        help='Output directory for plots')

    args = parser.parse_args()

    if not args.log_dir.exists():
        raise ValueError(f"Log directory does not exist: {args.log_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting queue duration data from {args.log_dir}")
    data = collect_data(args.log_dir)

    if not data:
        print("No data collected")
        return

    print(f"Found data for {len(data)} scheduler/QPS/delay configurations")
    plot_queue_boxplot_consolidated(data, args.output_dir)
    print(f"Plot saved to {args.output_dir}")


if __name__ == "__main__":
    main()
