#!/usr/bin/env python3
"""
Plot preemption counts (SWAP and RECOMPUTE) by QPS and delay multiplier for ANNS experiments.
Creates stacked bar charts showing preemption counts for streaming and non-streaming requests.
Adapted from crawler version - removes scheduler dimension.
"""
import argparse
import os
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
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

PREEMPTION_TYPES = [
    EngineCoreEventType.PREEMPTED_SWAP, EngineCoreEventType.PREEMPTED_RECOMPUTE
]


def convert_event_type(x):
    """Convert event_type strings to EngineCoreEventType enum."""
    if isinstance(x, str):
        if x.startswith('EngineCoreEventType.'):
            event_name = x.split('.')[-1]
            return EngineCoreEventType[event_name]
        else:
            return x
    else:
        raise ValueError(f"Invalid event type: {x} type: {type(x)}")


def extract_preemption_counts(
        csv_file: Path) -> Dict[EngineCoreEventType, Dict[str, int]]:
    """Extract preemption counts from metrics CSV, separated by streaming status."""
    df = pd.read_csv(csv_file, low_memory=False)

    # Filter for preemption events
    preemption_events = df[df['event_type'].str.startswith(
        'EngineCoreEventType.PREEMPTED_', na=False)]

    if preemption_events.empty:
        return {}

    # Convert event types
    preemption_events = preemption_events.copy()
    preemption_events['event_type'] = preemption_events['event_type'].apply(
        convert_event_type)

    # Count preemptions by type and streaming status
    preemption_counts = {}
    for preemption_type in PREEMPTION_TYPES:
        event_data = preemption_events[preemption_events['event_type'] ==
                                       preemption_type]
        if not event_data.empty:
            streaming_count = len(event_data[event_data['stream'] == True])
            non_streaming_count = len(
                event_data[event_data['stream'] == False])

            preemption_counts[preemption_type] = {
                'streaming': streaming_count,
                'non_streaming': non_streaming_count
            }

    return preemption_counts


def collect_preemption_data(
    log_dir: Path
) -> Tuple[Dict[Tuple[str, float, float], Dict[EngineCoreEventType, Dict[
        str, int]]], Dict[Tuple[str, float, float], List[str]]]:
    """
    Collect all preemption data from log directory for ANNS experiments.
    Groups by (scheduler, QPS, delay_multiplier) tuple.

    Returns:
        Tuple of (data, run_dirs_by_key)
    """
    data = defaultdict(lambda: defaultdict(lambda: {
        'streaming': 0,
        'non_streaming': 0
    }))
    run_dirs_by_key = defaultdict(list)

    csv_files = list(log_dir.rglob("run_metrics.csv"))

    for csv_file in csv_files:
        try:
            scheduler = extract_scheduler(str(csv_file.parent))
            qps_val = extract_replay_rate(str(csv_file.parent))
            delay_mult = extract_delay_multiplier(str(csv_file.parent))
            key = (scheduler, qps_val, delay_mult)

            preemption_counts = extract_preemption_counts(csv_file)

            for preemption_type, counts in preemption_counts.items():
                data[key][preemption_type]['streaming'] += counts['streaming']
                data[key][preemption_type]['non_streaming'] += counts[
                    'non_streaming']

            # Track run directory name
            run_name = csv_file.parent.name
            if run_name not in run_dirs_by_key[key]:
                run_dirs_by_key[key].append(run_name)

        except Exception as e:
            print(f"[warn] Error processing {csv_file}: {e}")

    return data, dict(run_dirs_by_key)


def plot_preemptions_by_qps(data: Dict, run_dirs_by_key: Dict,
                            output_dir: Path):
    """Create stacked bar plots grouped by QPS showing different schedulers."""

    # Get all preemption types that have data
    all_preemption_types = set()
    for qps_data in data.values():
        all_preemption_types.update(qps_data.keys())

    if not all_preemption_types:
        print("No preemption data found")
        return

    # Group data by (QPS, delay_mult) for each preemption type
    for preemption_type in sorted(all_preemption_types, key=lambda x: x.name):
        print(f"Creating plots for {preemption_type.name}")

        # Group by (QPS, delay_mult)
        qps_grouped = defaultdict(dict)
        for (scheduler, qps, delay_mult), preempt_data in data.items():
            if preemption_type in preempt_data:
                qps_key = (qps, delay_mult)
                qps_grouped[qps_key][scheduler] = preempt_data[preemption_type]

        if not qps_grouped:
            print(f"No data found for {preemption_type.name}")
            continue

        # Create one plot for each QPS/delay combination
        for (qps, delay_mult), scheduler_data in sorted(qps_grouped.items()):
            schedulers = sorted(scheduler_data.keys())

            scheduler_labels = []
            streaming_counts = []
            non_streaming_counts = []
            bar_colors_streaming = []
            bar_colors_non_streaming = []

            for scheduler in schedulers:
                streaming_count = scheduler_data[scheduler].get('streaming', 0)
                non_streaming_count = scheduler_data[scheduler].get(
                    'non_streaming', 0)

                if streaming_count > 0 or non_streaming_count > 0:
                    scheduler_labels.append(scheduler)
                    streaming_counts.append(streaming_count)
                    non_streaming_counts.append(non_streaming_count)

                    # Use red for default_vllm, other colors for rest
                    if scheduler == 'default_vllm':
                        bar_colors_streaming.append('lightcoral')
                        bar_colors_non_streaming.append('lightcoral')
                    else:
                        bar_colors_streaming.append('lightgreen')
                        bar_colors_non_streaming.append('lightyellow')

            if not scheduler_labels:
                continue

            # Create stacked bar chart
            fig, ax = plt.subplots(figsize=(max(10,
                                                len(scheduler_labels) * 1.5),
                                            6))
            x = np.arange(len(scheduler_labels))
            width = 0.6

            # Create bars for non-streaming (bottom) - with individual colors
            if any(non_streaming_counts):
                bars1 = ax.bar(x,
                               non_streaming_counts,
                               width,
                               color=bar_colors_non_streaming,
                               edgecolor='black',
                               label='Non-Streaming')

            # Create bars for streaming (top) - with individual colors
            if any(streaming_counts):
                bars2 = ax.bar(x,
                               streaming_counts,
                               width,
                               bottom=non_streaming_counts,
                               color=bar_colors_streaming,
                               edgecolor='black',
                               label='Streaming')

            # Customize the plot
            ax.set_xlabel('Scheduler')
            ax.set_ylabel('Number of Preemptions')
            ax.set_title(
                f'{preemption_type.name} - QPS: {qps:.3f}, Delay: {delay_mult}x'
            )
            ax.set_xticks(x)
            ax.set_xticklabels(scheduler_labels, rotation=45, ha='right')

            # Add legend
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor='lightyellow',
                      edgecolor='black',
                      label='Non-Streaming'),
                Patch(facecolor='lightgreen',
                      edgecolor='black',
                      label='Streaming'),
                Patch(facecolor='lightcoral',
                      edgecolor='black',
                      label='default_vllm (baseline)')
            ]
            ax.legend(handles=legend_handles, loc='upper right')

            ax.grid(True, alpha=0.3, axis='y')
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Add value labels on bars if counts are not too small
            for i, (streaming, non_streaming) in enumerate(
                    zip(streaming_counts, non_streaming_counts)):
                total = streaming + non_streaming
                if total > 0:
                    if non_streaming > 0:
                        ax.text(i,
                                non_streaming / 2,
                                str(non_streaming),
                                ha='center',
                                va='center',
                                fontweight='bold')
                    if streaming > 0:
                        ax.text(i,
                                non_streaming + streaming / 2,
                                str(streaming),
                                ha='center',
                                va='center',
                                fontweight='bold')

            fig.tight_layout()

            # Save plot
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'preemption_{preemption_type.name}_qps_{qps:.3f}_delay_{delay_mult}x.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved plot to {output_file}")

        print(f"Saved plot to {output_file}")


def plot_combined_preemptions(data: Dict, run_dirs_by_key: Dict,
                              output_dir: Path):
    """Create combined stacked bar plots showing both SWAP and RECOMPUTE preemptions for ANNS."""

    # Sort data by QPS then delay_multiplier
    sorted_keys = sorted(data.keys())

    if not sorted_keys:
        print("No data found")
        return

    qps_labels = []
    swap_streaming = []
    swap_non_streaming = []
    recompute_streaming = []
    recompute_non_streaming = []

    for key in sorted_keys:
        qps, delay_mult = key
        # Add run info to label
        run_dirs = run_dirs_by_key.get(key, [])
        run_info = run_dirs[0] if len(
            run_dirs) == 1 else f"{run_dirs[0][:10]}..."
        qps_labels.append(f"{qps:.3f}\n({delay_mult}x)\n{run_info}")

        # SWAP counts
        if EngineCoreEventType.PREEMPTED_SWAP in data[key]:
            swap_streaming.append(
                data[key][EngineCoreEventType.PREEMPTED_SWAP]['streaming'])
            swap_non_streaming.append(
                data[key][EngineCoreEventType.PREEMPTED_SWAP]['non_streaming'])
        else:
            swap_streaming.append(0)
            swap_non_streaming.append(0)

        # RECOMPUTE counts
        if EngineCoreEventType.PREEMPTED_RECOMPUTE in data[key]:
            recompute_streaming.append(data[key][
                EngineCoreEventType.PREEMPTED_RECOMPUTE]['streaming'])
            recompute_non_streaming.append(data[key][
                EngineCoreEventType.PREEMPTED_RECOMPUTE]['non_streaming'])
        else:
            recompute_streaming.append(0)
            recompute_non_streaming.append(0)

    if not qps_labels:
        print("No data to plot")
        return

    # Create stacked bar chart with 4 categories
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(qps_labels))
    width = 0.6

    # Stack order: SWAP non-streaming, SWAP streaming, RECOMPUTE non-streaming, RECOMPUTE streaming
    legend_handles = []

    if any(swap_non_streaming):
        bars1 = ax.bar(x,
                       swap_non_streaming,
                       width,
                       label='SWAP - Non-Streaming',
                       color='darkblue',
                       edgecolor='black')
        legend_handles.append(bars1)
    else:
        bars1 = ax.bar(x,
                       swap_non_streaming,
                       width,
                       color='darkblue',
                       edgecolor='black')

    if any(swap_streaming):
        bars2 = ax.bar(x,
                       swap_streaming,
                       width,
                       bottom=swap_non_streaming,
                       label='SWAP - Streaming',
                       color='lightblue',
                       edgecolor='black')
        legend_handles.append(bars2)
    else:
        bars2 = ax.bar(x,
                       swap_streaming,
                       width,
                       bottom=swap_non_streaming,
                       color='lightblue',
                       edgecolor='black')

    bottom_recompute = np.array(swap_non_streaming) + np.array(swap_streaming)

    if any(recompute_non_streaming):
        bars3 = ax.bar(x,
                       recompute_non_streaming,
                       width,
                       bottom=bottom_recompute,
                       label='RECOMPUTE - Non-Streaming',
                       color='darkred',
                       edgecolor='black')
        legend_handles.append(bars3)
    else:
        bars3 = ax.bar(x,
                       recompute_non_streaming,
                       width,
                       bottom=bottom_recompute,
                       color='darkred',
                       edgecolor='black')

    if any(recompute_streaming):
        bars4 = ax.bar(x,
                       recompute_streaming,
                       width,
                       bottom=bottom_recompute +
                       np.array(recompute_non_streaming),
                       label='RECOMPUTE - Streaming',
                       color='lightpink',
                       edgecolor='black')
        legend_handles.append(bars4)
    else:
        bars4 = ax.bar(x,
                       recompute_streaming,
                       width,
                       bottom=bottom_recompute +
                       np.array(recompute_non_streaming),
                       color='lightpink',
                       edgecolor='black')

    # Customize the plot
    ax.set_xlabel('Load (QPS)\n(delay multiplier)')
    ax.set_ylabel('Number of Preemptions')
    ax.set_title('Preemption Counts (SWAP & RECOMPUTE) vs Load (ANNS)')
    ax.set_xticks(x)
    ax.set_xticklabels(qps_labels, rotation=45, ha='right')
    if legend_handles:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'preemption_counts_combined_stacked_anns.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved combined plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot preemption counts by QPS for ANNS experiments")
    parser.add_argument("--log-dir",
                        required=True,
                        type=Path,
                        help="Directory containing ANNS run logs")
    parser.add_argument("--output-dir",
                        required=True,
                        type=Path,
                        help="Directory to save plots")

    args = parser.parse_args()

    if not args.log_dir.exists():
        raise ValueError(f"Log directory does not exist: {args.log_dir}")

    print(f"Collecting preemption data from {args.log_dir}")
    data, run_dirs_by_key = collect_preemption_data(args.log_dir)

    if not data:
        print("No preemption data collected")
        return

    print(f"Found data for {len(data)} QPS/delay configurations")
    plot_preemptions_by_qps(data, run_dirs_by_key, args.output_dir)
    plot_combined_preemptions(data, run_dirs_by_key, args.output_dir)
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
