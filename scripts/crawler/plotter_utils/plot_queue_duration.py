#!/usr/bin/env python3
"""plot_queue_duration.py
Generate queue duration analysis plots showing time requests spend waiting in queue.

Creates box plots showing queue duration distributions for each scheduler at different QPS levels.

Usage:
    python driver/crawler/plotter_utils/plot_queue_duration.py \
        --log-dir driver/crawler/run_log/H200_enhanced_schedulers_v1_full \
        --output-dir driver/crawler/analysis_results/H200_enhanced_schedulers_v1_full \
        --min-qps 0.5 --max-qps 4.0 --title-suffix " (H200 Crawler)"
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SCHEDULERS: Tuple[str,
                  ...] = ("recomp", "swap", "recomp_and_swap", "default_vllm",
                          "fcfs_lru", "lcas_lifo", "mcps_lce",
                          "stream_based_v1", "lcas_cplusp")

# Color palette for schedulers
SCHEDULER_COLORS: dict[str, str] = {
    "default_vllm": "#1f77b4",
    "fcfs_lru": "#ff7f0e",
    "lcas_lifo": "#2ca02c",
    "mcps_lce": "#9467bd",
    "stream_based_v1": "#e377c2",
    "recomp": "#7f7f7f",
    "swap": "#bcbd22",
    "recomp_and_swap": "#17becf",
    "lcas_cplusp": "#d62728",
}


def _find_config(dir_: Path) -> Path:
    cfgs = list(dir_.glob("config_*.yaml"))
    if not cfgs:
        raise FileNotFoundError(dir_)
    return cfgs[0]


def _qps(dir_: Path) -> float:
    cfg = yaml.safe_load(_find_config(dir_).read_text())
    return 1.0 / float(cfg["replay"]["poisson_avg_arrival_time"])


def _sched(dir_: Path) -> str:
    for part in dir_.parts[::-1]:
        if part in SCHEDULERS:
            return part
    try:
        cfg = yaml.safe_load(_find_config(dir_).read_text())
        scheduler = cfg.get("scheduler", "unknown")
        if isinstance(scheduler, str):
            return scheduler
        elif isinstance(scheduler, dict):
            return scheduler.get("type", "unknown")
        return "unknown"
    except Exception:
        return "unknown"


def _extract_queue_durations(csv_file: Path) -> List[float]:
    """Extract queue durations from run_metrics.csv QUEUED events (streaming mode only)."""
    try:
        df = pd.read_csv(csv_file, low_memory=False)

        # Filter for QUEUED events in streaming mode only (stream == True)
        queued_events = df[(df['event_type'] == 'EngineCoreEventType.QUEUED') & (df['stream'] == True)]

        if queued_events.empty:
            return []

        # duration_secs contains the queue duration for QUEUED events
        queue_durations = []
        for _, row in queued_events.iterrows():
            duration = row.get('duration_secs')
            if pd.notna(duration) and duration > 0:
                queue_durations.append(float(duration))

        return queue_durations
    except Exception as err:
        print(f"[warn] Error reading {csv_file}: {err}")
        return []


def _dataset(log_dir: Path, min_qps: float = 0, max_qps: float = float('inf')):
    """Extract dataset from run_metrics.csv files with QPS filtering."""
    data: MutableMapping[str,
                         MutableMapping[float, List[float]]] = (
                             defaultdict(lambda: defaultdict(list)))

    # Find all run_metrics.csv files
    csv_files = list(log_dir.rglob("run_metrics.csv"))

    if not csv_files:
        print("[warn] No run_metrics.csv files found")
        return data

    print(f"[info] Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        try:
            run_dir = csv_file.parent
            qps_val = _qps(run_dir)

            # Skip if outside QPS range
            if not (min_qps <= qps_val <= max_qps):
                continue

            sched = _sched(run_dir)

            # Skip oeda_pbas
            if sched == "oeda_pbas":
                continue

            queue_durations = _extract_queue_durations(csv_file)
            if queue_durations:
                data[sched][qps_val].extend(queue_durations)
                print(f"[loaded] {sched} QPS {qps_val}: {len(queue_durations)} queue events")

        except Exception as err:
            print(f"[warn] Error processing {csv_file}: {err}")

    return data


def _get_display_name(sched: str) -> str:
    """Convert scheduler name to display name."""
    display_names = {
        "default_vllm": "Default vLLM",
        "fcfs_lru": "FCFS",
        "lcas_lifo": "LCAS",
        "lcas_cplusp": "LCAS",
        "mcps_lce": "MCPS",
    }
    return display_names.get(sched, sched.replace('_', ' ').title())


def plot_queue_duration_boxplots(data, output_dir: Path, qps_level: float,
                                  title_suffix: str = ""):
    """Create box plots showing queue duration by scheduler at a specific QPS level."""

    if qps_level not in [q for sched in data.values() for q in sched]:
        print(f"[warn] QPS {qps_level} not found in data")
        return

    # Collect queue durations for each scheduler at this QPS
    queue_data = []
    scheduler_labels = []
    colors = []

    for sched in sorted(data.keys()):
        if qps_level in data[sched] and len(data[sched][qps_level]) > 0:
            queue_data.append(data[sched][qps_level])
            scheduler_labels.append(_get_display_name(sched))
            colors.append(SCHEDULER_COLORS.get(sched, "#000000"))

    if not queue_data:
        print(f"[warn] No data available for QPS {qps_level}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create box plot
    bp = ax.boxplot(queue_data, labels=scheduler_labels, patch_artist=True,
                    widths=0.6)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize the plot
    ax.set_xlabel('Scheduler', fontsize=13, fontweight='bold')
    ax.set_ylabel('Queue Duration (seconds)', fontsize=13, fontweight='bold')
    ax.set_title(f'Queue Duration Distribution at QPS {qps_level}{title_suffix}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x labels if needed
    plt.xticks(rotation=45, ha='right')

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"queue_duration_qps_{qps_level:.2f}.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {output_dir / filename}")


def plot_queue_duration_by_qps(data, output_dir: Path, title_suffix: str = ""):
    """Create scatter/line plots showing median queue duration vs QPS by scheduler."""

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})

    if not all_qps:
        print("[error] No QPS data found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot for each scheduler
    for sched in sorted(data.keys()):
        qps_values = sorted(data[sched].keys())
        median_durations = []

        for qps_val in qps_values:
            queue_durations = data[sched][qps_val]
            if queue_durations:
                median_durations.append(np.median(queue_durations))
            else:
                median_durations.append(np.nan)

        display_name = _get_display_name(sched)
        color = SCHEDULER_COLORS.get(sched, "#000000")

        ax.plot(qps_values, median_durations,
               marker='o',
               label=display_name,
               linewidth=2.5,
               markersize=8,
               color=color,
               alpha=0.85)

    ax.set_xlabel('Load (QPS)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Median Queue Duration (seconds)', fontsize=13, fontweight='bold')
    ax.set_title(f'Queue Duration vs Load by Scheduler{title_suffix}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=11)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "queue_duration_vs_qps.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {output_dir / filename}")


def plot_queue_duration_p95_by_qps(data, output_dir: Path, title_suffix: str = ""):
    """Create scatter/line plots showing P95 queue duration vs QPS by scheduler."""

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})

    if not all_qps:
        print("[error] No QPS data found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot for each scheduler
    for sched in sorted(data.keys()):
        qps_values = sorted(data[sched].keys())
        p95_durations = []

        for qps_val in qps_values:
            queue_durations = data[sched][qps_val]
            if queue_durations:
                p95_durations.append(np.percentile(queue_durations, 95))
            else:
                p95_durations.append(np.nan)

        display_name = _get_display_name(sched)
        color = SCHEDULER_COLORS.get(sched, "#000000")

        ax.plot(qps_values, p95_durations,
               marker='o',
               label=display_name,
               linewidth=2.5,
               markersize=8,
               color=color,
               alpha=0.85)

    ax.set_xlabel('Load (QPS)', fontsize=13, fontweight='bold')
    ax.set_ylabel('P95 Queue Duration (seconds)', fontsize=13, fontweight='bold')
    ax.set_title(f'P95 Queue Duration vs Load by Scheduler{title_suffix}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=11)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "queue_duration_p95_vs_qps.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {output_dir / filename}")


def plot_queue_duration_p99_by_qps(data, output_dir: Path, title_suffix: str = ""):
    """Create scatter/line plots showing P99 queue duration vs QPS by scheduler."""

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})

    if not all_qps:
        print("[error] No QPS data found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot for each scheduler
    for sched in sorted(data.keys()):
        qps_values = sorted(data[sched].keys())
        p99_durations = []

        for qps_val in qps_values:
            queue_durations = data[sched][qps_val]
            if queue_durations:
                p99_durations.append(np.percentile(queue_durations, 99))
            else:
                p99_durations.append(np.nan)

        display_name = _get_display_name(sched)
        color = SCHEDULER_COLORS.get(sched, "#000000")

        ax.plot(qps_values, p99_durations,
               marker='o',
               label=display_name,
               linewidth=2.5,
               markersize=8,
               color=color,
               alpha=0.85)

    ax.set_xlabel('Load (QPS)', fontsize=13, fontweight='bold')
    ax.set_ylabel('P99 Queue Duration (seconds)', fontsize=13, fontweight='bold')
    ax.set_title(f'P99 Queue Duration vs Load by Scheduler{title_suffix}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=11)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "queue_duration_p99_vs_qps.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {output_dir / filename}")


def calculate_queue_duration_efficiency_ratios(data, output_dir: Path, title_suffix: str = ""):
    """Calculate queue duration efficiency ratios and save to file."""

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})
    if not all_qps:
        return

    output_file = output_dir / "queue_duration_efficiency_ratios.txt"
    with open(output_file, 'w') as f:
        f.write(f"Queue Duration Efficiency Ratios vs Baseline (Default vLLM){title_suffix}\n")
        f.write("=" * 80 + "\n")
        f.write("Lower ratio = better scheduling efficiency (less time waiting in queue)\n")
        f.write("=" * 80 + "\n\n")

        for qps_val in sorted(all_qps):
            f.write(f"\nQPS {qps_val}:\n")
            f.write("-" * 60 + "\n")

            # Calculate percentiles for all schedulers
            percentile_data = {}
            for sched in sorted(data.keys()):
                if qps_val not in data[sched]:
                    continue

                queue_durations = np.array(data[sched][qps_val])
                if len(queue_durations) == 0:
                    continue

                percentile_data[sched] = {
                    'p50': np.percentile(queue_durations, 50),
                    'p95': np.percentile(queue_durations, 95),
                    'mean': np.mean(queue_durations),
                }

            # Calculate percentages vs baseline
            if 'default_vllm' in percentile_data:
                baseline = percentile_data['default_vllm']

                for sched in sorted(percentile_data.keys()):
                    if sched != 'default_vllm':
                        p50_ratio = percentile_data[sched]['p50'] / baseline['p50'] if baseline['p50'] > 0 else float('inf')
                        p95_ratio = percentile_data[sched]['p95'] / baseline['p95'] if baseline['p95'] > 0 else float('inf')
                        mean_ratio = percentile_data[sched]['mean'] / baseline['mean'] if baseline['mean'] > 0 else float('inf')

                        # Convert to percentage change (negative = less time in queue, better)
                        p50_pct = (p50_ratio - 1) * 100 if p50_ratio != float('inf') else float('inf')
                        p95_pct = (p95_ratio - 1) * 100 if p95_ratio != float('inf') else float('inf')
                        mean_pct = (mean_ratio - 1) * 100 if mean_ratio != float('inf') else float('inf')

                        f.write(f"  {_get_display_name(sched):15s}: "
                               f"P50={p50_pct:+.1f}%  "
                               f"P95={p95_pct:+.1f}%  "
                               f"Mean={mean_pct:+.1f}%\n")

                # Also print baseline values for reference
                f.write(f"\n  (Baseline P50={baseline['p50']:.4f}s, P95={baseline['p95']:.4f}s, Mean={baseline['mean']:.4f}s)\n")

    print(f"[saved] {output_file}")


def plot_queue_duration_p50_p95_sidebyside(data, output_dir: Path, title_suffix: str = ""):
    """Create side-by-side plots showing P50 and P95 queue duration vs QPS."""

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})

    if not all_qps:
        print("[error] No QPS data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot P50 (left) and P95 (right)
    for percentile, ax, ylabel in [(50, ax1, 'P50 Queue Duration (seconds)'),
                                     (95, ax2, 'P95 Queue Duration (seconds)')]:
        # Plot for each scheduler
        for sched in sorted(data.keys()):
            qps_values = sorted(data[sched].keys())
            percentile_durations = []

            for qps_val in qps_values:
                queue_durations = data[sched][qps_val]
                if queue_durations:
                    percentile_durations.append(np.percentile(queue_durations, percentile))
                else:
                    percentile_durations.append(np.nan)

            display_name = _get_display_name(sched)
            color = SCHEDULER_COLORS.get(sched, "#000000")

            ax.plot(qps_values, percentile_durations,
                   marker='o',
                   label=display_name,
                   linewidth=2.5,
                   markersize=8,
                   color=color,
                   alpha=0.85)

        ax.set_xlabel('Load (QPS)', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(f'P{percentile} Queue Duration vs Load{title_suffix}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)

    # Shared legend
    ax2.legend(fontsize=11, loc='best')

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "queue_duration_p50_p95_sidebyside.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {output_dir / filename}")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate queue duration analysis plots")
    parser.add_argument("--log-dir", required=True, type=Path,
                        help="Directory containing run logs")
    parser.add_argument("--output-dir",
                        default=Path("."),
                        type=Path,
                        help="Output directory for plots")
    parser.add_argument("--title-suffix",
                        type=str,
                        default="",
                        help="Additional text to append to the title")
    parser.add_argument("--min-qps",
                        type=float,
                        default=0,
                        help="Minimum QPS to include (default: 0)")
    parser.add_argument("--max-qps",
                        type=float,
                        default=float('inf'),
                        help="Maximum QPS to include (default: inf)")
    args = parser.parse_args(argv)

    data = _dataset(args.log_dir, args.min_qps, args.max_qps)
    if not data:
        raise SystemExit("no data discovered")

    print(f"\n[info] Filtering QPS range: {args.min_qps} - {args.max_qps}")
    print(f"[info] Found {len(data)} schedulers with data\n")

    # Generate queue duration plots
    # 1. Calculate efficiency ratios
    calculate_queue_duration_efficiency_ratios(data, args.output_dir, args.title_suffix)

    # 2. Plot P50 and P95 side-by-side
    plot_queue_duration_p50_p95_sidebyside(data, args.output_dir, args.title_suffix)

    # 3. For key QPS levels, create box plots
    all_qps = sorted({q for sched in data.values() for q in sched})
    for qps_val in all_qps:
        plot_queue_duration_boxplots(data, args.output_dir, qps_val, args.title_suffix)


if __name__ == "__main__":
    main()
