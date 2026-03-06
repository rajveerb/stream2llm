#!/usr/bin/env python3
"""
Plot EngineCoreEventType durations by scheduler and QPS.
Creates subplots for each scheduler showing QPS vs duration for each event type.
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

# Color mapping for event types
EVENT_COLORS = {
    EngineCoreEventType.QUEUED: '#1f77b4',
    EngineCoreEventType.SCHEDULED: '#ff7f0e',
    EngineCoreEventType.KV_ON_GPU: '#2ca02c',
    EngineCoreEventType.PREEMPTED_SWAP: '#d62728',
    EngineCoreEventType.PREEMPTED_RECOMPUTE: '#9467bd',
}


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


def extract_engine_events(
        csv_file: Path) -> Dict[EngineCoreEventType, Dict[str, List[float]]]:
    """Extract EngineCoreEventType durations from metrics CSV, separated by streaming status."""
    df = pd.read_csv(csv_file, low_memory=False)

    # Filter for EngineCoreEventType events
    engine_events = df[df['event_type'].str.startswith('EngineCoreEventType.',
                                                       na=False)]

    if engine_events.empty:
        return {}

    # Convert event types
    engine_events = engine_events.copy()
    engine_events['event_type'] = engine_events['event_type'].apply(
        convert_event_type)

    # Group by event type and streaming status, then extract durations
    event_durations = {}
    for event_type in EngineCoreEventType:
        event_data = engine_events[engine_events['event_type'] == event_type]
        if not event_data.empty:
            streaming_data = event_data[event_data['stream'] ==
                                        True]['duration_secs'].tolist()
            non_streaming_data = event_data[event_data['stream'] ==
                                            False]['duration_secs'].tolist()

            event_durations[event_type] = {
                'streaming': streaming_data,
                'non_streaming': non_streaming_data
            }

    return event_durations


def collect_data(
    log_dir: Path
) -> Dict[str, Dict[float, Dict[EngineCoreEventType, Dict[str, List[float]]]]]:
    """Collect all data from log directory."""
    data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {
            'streaming': [],
            'non_streaming': []
        })))

    csv_files = list(log_dir.rglob("run_metrics.csv"))

    for csv_file in csv_files:
        try:
            qps_val = _qps(csv_file.parent)
            sched = _sched(csv_file.parent)

            if sched == "unknown":
                continue

            event_durations = extract_engine_events(csv_file)

            for event_type, streaming_data in event_durations.items():
                data[sched][qps_val][event_type]['streaming'].extend(
                    streaming_data['streaming'])
                data[sched][qps_val][event_type]['non_streaming'].extend(
                    streaming_data['non_streaming'])

        except Exception as e:
            print(f"[warn] Error processing {csv_file}: {e}")

    return data


def plot_engine_event_durations_by_scheduler(data: Dict, output_dir: Path):
    """Create box plots for each EngineCoreEventType showing scheduler subplots with streaming/non-streaming separation."""

    # Get all event types that have data
    all_event_types = set()
    for sched_data in data.values():
        for qps_data in sched_data.values():
            all_event_types.update(qps_data.keys())

    if not all_event_types:
        print("No engine event data found")
        return

    for event_type in sorted(all_event_types, key=lambda x: x.name):
        print(f"Creating plot for {event_type.name}")

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

            # Collect QPS values and duration data for box plots
            sorted_qps = sorted(data[scheduler].keys())
            streaming_box_data = []
            non_streaming_box_data = []
            qps_labels = []

            for qps in sorted_qps:
                if event_type in data[scheduler][qps]:
                    streaming_durations = data[scheduler][qps][event_type].get(
                        'streaming', [])
                    non_streaming_durations = data[scheduler][qps][
                        event_type].get('non_streaming', [])

                    # Only add if we have data for at least one type
                    if streaming_durations or non_streaming_durations:
                        streaming_box_data.append(streaming_durations)
                        non_streaming_box_data.append(non_streaming_durations)
                        qps_labels.append(f"{qps:.3f}")

            if not qps_labels:
                ax.text(0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{scheduler.replace('_', ' ').title()}")
                continue

            # Create side-by-side box plots similar to event_durations.py
            box_width = 0.35
            x = np.arange(len(qps_labels))

            # Boxplot for non-streaming events (left side - baseline)
            if any(non_streaming_box_data):
                bp1 = ax.boxplot(non_streaming_box_data,
                                 positions=x - box_width / 2,
                                 widths=box_width,
                                 patch_artist=True,
                                 boxprops=dict(facecolor='lightcoral',
                                               color='black'),
                                 medianprops=dict(color='red'),
                                 whiskerprops=dict(color='black'),
                                 capprops=dict(color='black'),
                                 flierprops=dict(marker='x',
                                                 color='gray',
                                                 markersize=6,
                                                 markeredgecolor='gray'),
                                 showfliers=True)

            # Boxplot for streaming events (right side)
            if any(streaming_box_data):
                bp2 = ax.boxplot(streaming_box_data,
                                 positions=x + box_width / 2,
                                 widths=box_width,
                                 patch_artist=True,
                                 boxprops=dict(facecolor='lightgreen',
                                               color='black'),
                                 medianprops=dict(color='green'),
                                 whiskerprops=dict(color='black'),
                                 capprops=dict(color='black'),
                                 flierprops=dict(marker='x',
                                                 color='gray',
                                                 markersize=6,
                                                 markeredgecolor='gray'),
                                 showfliers=True)

            # Customize the plot
            ax.set_xticks(x)
            ax.set_xticklabels(qps_labels, rotation=45, ha='right')
            ax.set_xlabel('Load (QPS)')
            ax.set_ylabel('Duration (seconds)')
            ax.set_title(f"{scheduler.replace('_', ' ').title()}")

            # Add legend
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor='lightcoral',
                      edgecolor='black',
                      label='Non-Streaming'),
                Patch(facecolor='lightgreen',
                      edgecolor='black',
                      label='Streaming')
            ]
            ax.legend(handles=legend_handles, loc='upper right')

            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_schedulers, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'{event_type.name} Duration vs Load by Scheduler',
                     fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'engine_event_duration_boxplot_sidebyside_{event_type.name}.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot EngineCoreEventType durations by scheduler and QPS")
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

    print(f"Collecting data from {args.log_dir}")
    data = collect_data(args.log_dir)

    if not data:
        print("No data collected")
        return

    print(f"Found data for schedulers: {list(data.keys())}")
    plot_engine_event_durations_by_scheduler(data, args.output_dir)
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
