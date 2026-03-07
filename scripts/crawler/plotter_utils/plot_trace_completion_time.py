#!/usr/bin/env python3
"""plot_trace_completion_time.py
Generate plot of trace completion time vs QPS for different schedulers.

Creates a plot showing how trace completion time scales with load (QPS)
for all schedulers, comparing streaming vs non-streaming variants.

Usage:
    # Crawler workload (QPS 0.5-4.0)
    python driver/crawler/plotter_utils/plot_trace_completion_time.py \
        --log-dir driver/crawler/run_log/H200_enhanced_schedulers_v1_full \
        --output-dir driver/crawler/analysis_results/H200_enhanced_schedulers_v1_full \
        --min-qps 0.5 --max-qps 4.0 --title-suffix " (H200 Crawler)"

    # ANNS workload (QPS 0.25-2.0)
    python driver/anns/plotter_utils/plot_trace_completion_time.py \
        --log-dir driver/anns/run_log/H200_enhanced_schedulers_v1_full \
        --output-dir driver/anns/analysis_results/H200_enhanced_schedulers_v1_full \
        --min-qps 0.25 --max-qps 2.0 --title-suffix " (H200 ANNS)"
"""
from __future__ import annotations

import argparse
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


def _extract_trace_completion_time(csv_file: Path) -> Tuple[float, float] | None:
    """Extract trace completion time for streaming and non-streaming."""
    df = pd.read_csv(csv_file, low_memory=False)

    # Get replay_end events which contain the trace completion time
    replay_end_df = df[df['event_type'] == 'replay_end']

    if len(replay_end_df) == 0:
        return None

    # Split by streaming flag
    streaming_times = []
    non_streaming_times = []

    for _, row in replay_end_df.iterrows():
        time = row['duration_secs']
        if pd.isna(time):
            continue
        if row['stream']:
            streaming_times.append(time)
        else:
            non_streaming_times.append(time)

    # Average the times if multiple entries
    streaming_time = np.mean(streaming_times) if streaming_times else None
    non_streaming_time = np.mean(non_streaming_times) if non_streaming_times else None

    return (streaming_time, non_streaming_time)


def _dataset(csv_files: Sequence[Path], min_qps: float = 0, max_qps: float = float('inf')):
    """Extract dataset with QPS filtering."""
    data: MutableMapping[str,
                         MutableMapping[float, Dict[str, float]]] = (
                             defaultdict(lambda: defaultdict(lambda: {
                                 "streaming": None,
                                 "non_streaming": None
                             })))
    for f in csv_files:
        try:
            qps_val = _qps(f.parent)
            # Skip if outside QPS range
            if not (min_qps <= qps_val <= max_qps):
                continue
            sched = _sched(f.parent)
            # Skip oeda_pbas
            if sched == "oeda_pbas":
                continue

            result = _extract_trace_completion_time(f)
            if result:
                streaming_time, non_streaming_time = result
                if streaming_time:
                    data[sched][qps_val]["streaming"] = streaming_time
                if non_streaming_time:
                    data[sched][qps_val]["non_streaming"] = non_streaming_time
        except Exception as err:
            print("[warn]", err)
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


def plot_trace_completion_time(data, output_dir: Path, title_suffix: str = "", output_prefix: str = "trace_completion_time_crawler"):
    """Create trace completion time vs QPS plot."""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})

    if not all_qps:
        print("[error] No QPS data found")
        return

    # Prepare data for plotting
    scheduler_data = {}
    for sched in sorted(data.keys()):
        qps_values = []
        trace_times_streaming = []
        trace_times_non_streaming = []

        for qps in all_qps:
            if qps not in data[sched]:
                continue

            streaming_time = data[sched][qps]["streaming"]
            non_streaming_time = data[sched][qps]["non_streaming"]

            # Skip if no streaming data (streaming is required)
            if streaming_time is None:
                continue

            qps_values.append(qps)
            trace_times_streaming.append(streaming_time)

            # Only add non-streaming if available
            if non_streaming_time is not None:
                trace_times_non_streaming.append(non_streaming_time)
            else:
                trace_times_non_streaming.append(None)

        scheduler_data[sched] = {
            "qps": qps_values,
            "trace_times_streaming": trace_times_streaming,
            "trace_times_non_streaming": trace_times_non_streaming,
        }

    # Plot trace completion time vs QPS
    for sched in sorted(scheduler_data.keys()):
        display_name = _get_display_name(sched)
        color = SCHEDULER_COLORS.get(sched, "#000000")
        sched_data = scheduler_data[sched]

        # Plot streaming
        ax.plot(sched_data["qps"],
                sched_data["trace_times_streaming"],
                marker='o',
                linestyle='-',
                linewidth=4.5,
                markersize=14,
                markeredgewidth=2,
                markeredgecolor='white',
                label=f"{display_name} (Streaming)",
                color=color,
                alpha=0.85)

        # Only plot non-streaming if we have data
        non_streaming_data = sched_data["trace_times_non_streaming"]
        if any(v is not None for v in non_streaming_data):
            # Filter out None values for plotting
            qps_ns = [q for q, v in zip(sched_data["qps"], non_streaming_data) if v is not None]
            trace_ns = [v for v in non_streaming_data if v is not None]
            if qps_ns:
                ax.plot(qps_ns,
                        trace_ns,
                        marker='s',
                        linestyle='--',
                        linewidth=4,
                        markersize=12,
                        markeredgewidth=1.5,
                        markeredgecolor='white',
                        label=f"{display_name} (Non-Streaming)",
                        color=color,
                        alpha=0.6)

    ax.set_xlabel("QPS", fontsize=36, fontweight='bold')
    ax.set_ylabel("Trace Completion Time (s)", fontsize=36, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=28)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{output_prefix}.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {output_dir / filename}")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate trace completion time vs QPS plots")
    parser.add_argument("--log-dir", required=True, type=Path,
                        help="Directory containing run logs")
    parser.add_argument("--output-dir",
                        default=Path("."),
                        type=Path,
                        help="Output directory for plots")
    parser.add_argument("--output-prefix",
                        default="trace_completion_time_crawler",
                        type=str,
                        help="Prefix for output filename")
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

    csv_files = list(args.log_dir.rglob("run_metrics.csv"))
    if not csv_files:
        raise SystemExit("no run_metrics.csv found")

    data = _dataset(csv_files, args.min_qps, args.max_qps)
    if not data:
        raise SystemExit("no data discovered")

    print(f"[info] Filtering QPS range: {args.min_qps} - {args.max_qps}")
    print(f"[info] Found {len(data)} schedulers with data")

    # Generate trace completion time plot
    plot_trace_completion_time(data, args.output_dir, args.title_suffix, output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
