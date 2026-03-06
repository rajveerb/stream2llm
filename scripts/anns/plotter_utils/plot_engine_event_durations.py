#!/usr/bin/env python3
"""
Plot EngineCoreEventType durations by QPS and delay multiplier for ANNS experiments.
Creates plots showing QPS vs duration for each event type with streaming/non-streaming separation.
Adapted from crawler version - removes scheduler dimension.
"""
import argparse
import os
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
) -> Tuple[Dict[Tuple[str, float, float], Dict[EngineCoreEventType, Dict[
        str, List[float]]]], Dict[Tuple[str, float, float], List[str]]]:
    """
    Collect all data from log directory for ANNS experiments.
    Groups by (scheduler, QPS, delay_multiplier) tuple.

    Returns:
        Tuple of (data, run_dirs_by_key)
    """
    data = defaultdict(lambda: defaultdict(lambda: {
        'streaming': [],
        'non_streaming': []
    }))
    run_dirs_by_key = defaultdict(list)

    csv_files = list(log_dir.rglob("run_metrics.csv"))

    for csv_file in csv_files:
        try:
            scheduler = extract_scheduler(str(csv_file.parent))
            qps_val = extract_replay_rate(str(csv_file.parent))
            delay_mult = extract_delay_multiplier(str(csv_file.parent))
            key = (scheduler, qps_val, delay_mult)

            event_durations = extract_engine_events(csv_file)

            for event_type, streaming_data in event_durations.items():
                data[key][event_type]['streaming'].extend(
                    streaming_data['streaming'])
                data[key][event_type]['non_streaming'].extend(
                    streaming_data['non_streaming'])

            # Track run directory name
            run_name = csv_file.parent.name
            if run_name not in run_dirs_by_key[key]:
                run_dirs_by_key[key].append(run_name)

        except Exception as e:
            print(f"[warn] Error processing {csv_file}: {e}")

    return data, dict(run_dirs_by_key)


def plot_engine_event_durations(data: Dict, run_dirs_by_key: Dict,
                                output_dir: Path):
    """
    Create box plots for each EngineCoreEventType.
    One plot per (event_type, QPS, delay) showing different schedulers with default_vllm baseline.
    """

    # Get all event types that have data
    all_event_types = set()
    for qps_data in data.values():
        all_event_types.update(qps_data.keys())

    if not all_event_types:
        print("No engine event data found")
        return

    # Group data by (QPS, delay_mult) for each event type
    for event_type in sorted(all_event_types, key=lambda x: x.name):
        print(f"Creating plots for {event_type.name}")

        # Group by (QPS, delay_mult)
        qps_grouped = defaultdict(dict)
        for (scheduler, qps, delay_mult), event_data in data.items():
            if event_type in event_data:
                qps_key = (qps, delay_mult)
                qps_grouped[qps_key][scheduler] = event_data[event_type]

        if not qps_grouped:
            print(f"No data found for {event_type.name}")
            continue

        # Create one plot for each QPS/delay combination
        for (qps, delay_mult), scheduler_data in sorted(qps_grouped.items()):
            # Extract default_vllm baseline if available
            baseline_streaming = None
            baseline_non_streaming = None
            if 'default_vllm' in scheduler_data:
                baseline_streaming = scheduler_data['default_vllm'].get(
                    'streaming', [])
                baseline_non_streaming = scheduler_data['default_vllm'].get(
                    'non_streaming', [])

            schedulers = sorted(scheduler_data.keys())

            # Prepare data for box plots
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

                if streaming_durations or non_streaming_durations:
                    streaming_box_data.append(streaming_durations)
                    non_streaming_box_data.append(non_streaming_durations)
                    scheduler_labels.append(scheduler)

                    # Use red for default_vllm, green/coral for others
                    if scheduler == 'default_vllm':
                        colors_streaming.append('lightcoral')
                        colors_non_streaming.append('lightcoral')
                    else:
                        colors_streaming.append('lightgreen')
                        colors_non_streaming.append('lightyellow')

            if not scheduler_labels:
                continue

            # Create side-by-side box plots
            fig, ax = plt.subplots(figsize=(max(10,
                                                len(scheduler_labels) * 2), 6))
            box_width = 0.35
            x = np.arange(len(scheduler_labels))

            # Boxplot for non-streaming events (left side - baseline)
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

            # Boxplot for streaming events (right side)
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
            ax.set_ylabel('Duration (seconds)')
            ax.set_title(
                f'{event_type.name} - QPS: {qps:.3f}, Delay: {delay_mult}x')

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

            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            # Save plot
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'engine_event_{event_type.name}_qps_{qps:.3f}_delay_{delay_mult}x.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Plot EngineCoreEventType durations by QPS for ANNS experiments")
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

    print(f"Collecting data from {args.log_dir}")
    data, run_dirs_by_key = collect_data(args.log_dir)

    if not data:
        print("No data collected")
        return

    print(f"Found data for {len(data)} scheduler/QPS/delay configurations")
    plot_engine_event_durations(data, run_dirs_by_key, args.output_dir)
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
