#!/usr/bin/env python3
"""
Plot preemption counts (SWAP and RECOMPUTE) by scheduler and QPS.
Creates stacked bar charts showing preemption counts for streaming and non-streaming requests.
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

# Set global font sizes for better readability
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

SCHEDULERS = [
    "default_vllm", "fcfs_lru", "lcas_lifo", "mcps_lce", "oeda_pbas",
    "stream_based_v1", "lcas_cplusp"
]

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


def _find_config(dir_: Path) -> Path:
    """Find config file in directory."""
    cfgs = list(dir_.glob("config_*.yaml"))
    if not cfgs:
        raise FileNotFoundError(dir_)
    return cfgs[0]


def _qps(dir_: Path) -> float:
    """Extract QPS from config file."""
    cfg = yaml.safe_load(_find_config(dir_).read_text())
    return 1.0 / float(cfg["replay"]["poisson_avg_arrival_time"])


def _sched(dir_: Path) -> str:
    """Extract scheduler name from directory path."""
    for part in dir_.parts[::-1]:
        if part in SCHEDULERS:
            return part
    try:
        cfg = yaml.safe_load(_find_config(dir_).read_text())
        return cfg.get("scheduler", {}).get("type", "unknown")
    except Exception:
        return "unknown"


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
) -> Dict[str, Dict[float, Dict[EngineCoreEventType, Dict[str, int]]]]:
    """Collect all preemption data from log directory."""
    data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {
            'streaming': 0,
            'non_streaming': 0
        })))

    csv_files = list(log_dir.rglob("run_metrics.csv"))

    for csv_file in csv_files:
        try:
            qps_val = _qps(csv_file.parent)
            sched = _sched(csv_file.parent)

            if sched == "unknown":
                continue

            preemption_counts = extract_preemption_counts(csv_file)

            for preemption_type, counts in preemption_counts.items():
                data[sched][qps_val][preemption_type]['streaming'] += counts[
                    'streaming']
                data[sched][qps_val][preemption_type][
                    'non_streaming'] += counts['non_streaming']

        except Exception as e:
            print(f"[warn] Error processing {csv_file}: {e}")

    return data


def plot_preemptions_by_scheduler(data: Dict, output_dir: Path):
    """Create stacked bar plots for preemption counts by scheduler."""

    # Get all preemption types that have data
    all_preemption_types = set()
    for sched_data in data.values():
        for qps_data in sched_data.values():
            all_preemption_types.update(qps_data.keys())

    if not all_preemption_types:
        print("No preemption data found")
        return

    for preemption_type in sorted(all_preemption_types, key=lambda x: x.name):
        print(f"Creating plot for {preemption_type.name}")

        # Determine grid layout
        available_schedulers = [s for s in SCHEDULERS if s in data]
        n_schedulers = len(available_schedulers)

        if n_schedulers == 0:
            continue

        cols = min(3, n_schedulers)
        rows = math.ceil(n_schedulers / cols)

        fig, axes = plt.subplots(rows,
                                 cols,
                                 figsize=(5 * cols, 4 * rows),
                                 squeeze=False)
        axes = axes.flatten()

        for idx, scheduler in enumerate(available_schedulers):
            ax = axes[idx]

            if scheduler not in data:
                ax.set_visible(False)
                continue

            # Collect QPS values and preemption counts
            sorted_qps = sorted(data[scheduler].keys())
            qps_labels = []
            streaming_counts = []
            non_streaming_counts = []

            for qps in sorted_qps:
                if preemption_type in data[scheduler][qps]:
                    qps_labels.append(f"{qps:.3f}")
                    streaming_counts.append(
                        data[scheduler][qps][preemption_type]['streaming'])
                    non_streaming_counts.append(
                        data[scheduler][qps][preemption_type]['non_streaming'])

            if not qps_labels:
                ax.text(0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{scheduler.replace('_', ' ').title()}")
                continue

            # Create stacked bar chart
            x = np.arange(len(qps_labels))
            width = 0.6

            # Create bars for non-streaming (bottom) and streaming (top)
            legend_handles = []
            if any(non_streaming_counts):
                bars1 = ax.bar(x,
                               non_streaming_counts,
                               width,
                               label='Non-Streaming',
                               color='lightcoral',
                               edgecolor='black')
                legend_handles.append(bars1)
            else:
                # Create empty bars if no non-streaming data
                bars1 = ax.bar(x,
                               non_streaming_counts,
                               width,
                               color='lightcoral',
                               edgecolor='black')

            if any(streaming_counts):
                bars2 = ax.bar(x,
                               streaming_counts,
                               width,
                               bottom=non_streaming_counts,
                               label='Streaming',
                               color='lightgreen',
                               edgecolor='black')
                legend_handles.append(bars2)
            else:
                # Create empty bars if no streaming data
                bars2 = ax.bar(x,
                               streaming_counts,
                               width,
                               bottom=non_streaming_counts,
                               color='lightgreen',
                               edgecolor='black')

            # Customize the plot
            ax.set_xlabel('Load (QPS)')
            ax.set_ylabel('Number of Preemptions')
            ax.set_title(f"{scheduler.replace('_', ' ').title()}")
            ax.set_xticks(x)
            ax.set_xticklabels(qps_labels, rotation=45, ha='right')
            if legend_handles:
                ax.legend()
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

        # Hide unused subplots
        for idx in range(n_schedulers, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'{preemption_type.name} Counts vs Load by Scheduler',
                     fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'preemption_counts_stacked_{preemption_type.name}.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved plot to {output_file}")


def plot_combined_preemptions_by_scheduler(data: Dict, output_dir: Path):
    """Create combined stacked bar plots showing both SWAP and RECOMPUTE preemptions."""

    available_schedulers = [s for s in SCHEDULERS if s in data]
    n_schedulers = len(available_schedulers)

    if n_schedulers == 0:
        print("No scheduler data found")
        return

    cols = min(3, n_schedulers)
    rows = math.ceil(n_schedulers / cols)

    fig, axes = plt.subplots(rows,
                             cols,
                             figsize=(5 * cols, 4 * rows),
                             squeeze=False)
    axes = axes.flatten()

    for idx, scheduler in enumerate(available_schedulers):
        ax = axes[idx]

        if scheduler not in data:
            ax.set_visible(False)
            continue

        # Collect QPS values and preemption counts for both types
        sorted_qps = sorted(data[scheduler].keys())
        qps_labels = []

        # Initialize counts for both preemption types
        swap_streaming = []
        swap_non_streaming = []
        recompute_streaming = []
        recompute_non_streaming = []

        for qps in sorted_qps:
            qps_labels.append(f"{qps:.3f}")

            # SWAP counts
            if EngineCoreEventType.PREEMPTED_SWAP in data[scheduler][qps]:
                swap_streaming.append(data[scheduler][qps][
                    EngineCoreEventType.PREEMPTED_SWAP]['streaming'])
                swap_non_streaming.append(data[scheduler][qps][
                    EngineCoreEventType.PREEMPTED_SWAP]['non_streaming'])
            else:
                swap_streaming.append(0)
                swap_non_streaming.append(0)

            # RECOMPUTE counts
            if EngineCoreEventType.PREEMPTED_RECOMPUTE in data[scheduler][qps]:
                recompute_streaming.append(data[scheduler][qps][
                    EngineCoreEventType.PREEMPTED_RECOMPUTE]['streaming'])
                recompute_non_streaming.append(data[scheduler][qps][
                    EngineCoreEventType.PREEMPTED_RECOMPUTE]['non_streaming'])
            else:
                recompute_streaming.append(0)
                recompute_non_streaming.append(0)

        if not qps_labels:
            ax.text(0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{scheduler.replace('_', ' ').title()}")
            continue

        # Create stacked bar chart with 4 categories
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

        bottom_recompute = np.array(swap_non_streaming) + np.array(
            swap_streaming)

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
        ax.set_xlabel('Load (QPS)')
        ax.set_ylabel('Number of Preemptions')
        ax.set_title(f"{scheduler.replace('_', ' ').title()}")
        ax.set_xticks(x)
        ax.set_xticklabels(qps_labels, rotation=45, ha='right')
        if legend_handles:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Hide unused subplots
    for idx in range(n_schedulers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Preemption Counts (SWAP & RECOMPUTE) vs Load by Scheduler',
                 fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'preemption_counts_combined_stacked.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved combined plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot preemption counts by scheduler and QPS")
    parser.add_argument("--log-dir",
                        required=True,
                        type=Path,
                        help="Directory containing scheduler run logs")
    parser.add_argument("--output-dir",
                        required=True,
                        type=Path,
                        help="Directory to save plots")

    args = parser.parse_args()

    if not args.log_dir.exists():
        raise ValueError(f"Log directory does not exist: {args.log_dir}")

    print(f"Collecting preemption data from {args.log_dir}")
    data = collect_preemption_data(args.log_dir)

    if not data:
        print("No preemption data collected")
        return

    print(f"Found data for schedulers: {list(data.keys())}")
    plot_preemptions_by_scheduler(data, args.output_dir)
    plot_combined_preemptions_by_scheduler(data, args.output_dir)
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
