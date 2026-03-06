#!/usr/bin/env python3
"""plot_trace_completion_time_simple.py
Generate simple trace completion time plots with all scheduler variants.

Creates a single plot showing trace completion time vs QPS for:
- Default vLLM (streaming and non-streaming)
- FCFS, LCAS, MCPS (streaming only)

Usage:
    python driver/crawler/plotter_utils/plot_trace_completion_time_simple.py \
        --log-dir driver/crawler/test_run_log \
        --output-dir driver/crawler/analysis_results/H200_enhanced_schedulers_v1_full \
        --min-qps 0.5 --max-qps 4.0 --title-suffix " (H200 Crawler)"
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SCHEDULERS: Tuple[str,
                  ...] = ("recomp", "swap", "recomp_and_swap", "default_vllm",
                          "fcfs_lru", "lcas_lifo", "mcps_lce",
                          "stream_based_v1", "lcas_cplusp", "oeda_pbas")

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
    "oeda_pbas": "#ff9896",
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


def _get_trace_completion_time(csv_file: Path) -> Tuple[float, float]:
    """Extract trace completion times for streaming and non-streaming.

    Returns:
        Tuple of (streaming_time, non_streaming_time) in seconds (relative to first event)
    """
    df = pd.read_csv(csv_file, low_memory=False)
    if "event_type" not in df.columns:
        raise ValueError("missing event_type column")

    # Get first timestamp to calculate relative times
    min_ts = df["event_timestamp"].min()

    # Get the maximum timestamp for each group
    streaming_df = df[(df["event_type"] == "query_ttft") & (df["stream"] == True)]
    non_streaming_df = df[(df["event_type"] == "query_ttft") & (df["stream"] == False)]

    streaming_time = (streaming_df["event_timestamp"].max() - min_ts) if len(streaming_df) > 0 else np.nan
    non_streaming_time = (non_streaming_df["event_timestamp"].max() - min_ts) if len(non_streaming_df) > 0 else np.nan

    return float(streaming_time), float(non_streaming_time)


def _dataset(csv_files: Sequence[Path], min_qps: float = 0, max_qps: float = float('inf')):
    """Extract dataset with QPS filtering."""
    data = defaultdict(lambda: defaultdict(lambda: {
        "streaming": np.nan,
        "non_streaming": np.nan
    }))

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

            streaming_time, non_streaming_time = _get_trace_completion_time(f)
            data[sched][qps_val]["streaming"] = streaming_time
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


def plot_trace_completion_time(data, output_dir: Path, title_suffix: str = ""):
    """Create simple trace completion time plot with all variants."""

    fig, ax = plt.subplots(figsize=(12, 7))

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})

    if not all_qps:
        print("[error] No QPS data found")
        return

    # Prepare data for plotting
    scheduler_data = {}
    for sched in sorted(data.keys()):
        qps_values = []
        streaming_times = []
        non_streaming_times = []

        for qps in all_qps:
            if qps not in data[sched]:
                continue

            streaming = data[sched][qps]["streaming"]
            non_streaming = data[sched][qps]["non_streaming"]

            # Add QPS value
            qps_values.append(qps)
            streaming_times.append(streaming if not np.isnan(streaming) else None)
            non_streaming_times.append(non_streaming if not np.isnan(non_streaming) else None)

        scheduler_data[sched] = {
            "qps": qps_values,
            "streaming": streaming_times,
            "non_streaming": non_streaming_times,
        }

    # Define line styles and markers for better distinction
    line_styles = {
        "default_vllm": ("-", "o"),
        "fcfs_lru": ("-", "s"),
        "lcas_cplusp": ("-", "^"),
        "mcps_lce": ("-", "D"),
    }

    # Plot trace completion time vs QPS
    for sched in sorted(scheduler_data.keys()):
        display_name = _get_display_name(sched)
        color = SCHEDULER_COLORS.get(sched, "#000000")
        sched_data = scheduler_data[sched]

        # Get line style and marker for this scheduler
        linestyle, marker = line_styles.get(sched, ("-", "o"))

        # Plot streaming
        streaming_vals = sched_data["streaming"]
        if any(v is not None for v in streaming_vals):
            qps_s = [q for q, v in zip(sched_data["qps"], streaming_vals) if v is not None]
            times_s = [v for v in streaming_vals if v is not None]
            if qps_s:
                ax.plot(qps_s,
                       times_s,
                       marker=marker,
                       linestyle=linestyle,
                       linewidth=2.5,
                       markersize=8,
                       label=f"{display_name} (Streaming)",
                       color=color,
                       alpha=0.85)

        # Plot non-streaming if available (only for Default vLLM)
        non_streaming_vals = sched_data["non_streaming"]
        if any(v is not None for v in non_streaming_vals):
            qps_ns = [q for q, v in zip(sched_data["qps"], non_streaming_vals) if v is not None]
            times_ns = [v for v in non_streaming_vals if v is not None]
            if qps_ns:
                ax.plot(qps_ns,
                       times_ns,
                       marker="s",
                       linestyle="--",
                       linewidth=2,
                       markersize=7,
                       label=f"{display_name} (Non-Streaming)",
                       color=color,
                       alpha=0.6)

    ax.set_xlabel("QPS", fontsize=16, fontweight='bold')
    ax.set_ylabel("Trace Completion Time (seconds)", fontsize=16, fontweight='bold')
    ax.set_title(f"Trace Completion Time vs QPS{title_suffix}",
                 fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=11, loc='best')

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "trace_completion_time.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {output_dir / filename}")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate simple trace completion time plots")
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

    csv_files = list(args.log_dir.rglob("run_metrics.csv"))
    if not csv_files:
        raise SystemExit("no run_metrics.csv found")

    data = _dataset(csv_files, args.min_qps, args.max_qps)
    if not data:
        raise SystemExit("no data discovered")

    print(f"[info] Filtering QPS range: {args.min_qps} - {args.max_qps}")
    print(f"[info] Found {len(data)} schedulers with data")

    # Generate plots
    plot_trace_completion_time(data, args.output_dir, args.title_suffix)


if __name__ == "__main__":
    main()
