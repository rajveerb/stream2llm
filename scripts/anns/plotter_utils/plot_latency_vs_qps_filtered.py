#!/usr/bin/env python3
"""plot_latency_vs_qps_filtered.py
Generate latency vs QPS plots for P50 and P99 percentiles with QPS range filtering.

Creates side-by-side plots showing how P50 and P99 latencies scale with QPS
for all schedulers, comparing streaming vs non-streaming variants.
Only includes specified QPS range (e.g., crawler: 0.5-4.0, ANNS: 0.25-2.0).

Usage:
    # Crawler workload (QPS 0.5-4.0)
    python driver/crawler/plotter_utils/plot_latency_vs_qps_filtered.py \
        --log-dir driver/crawler/test_run_log \
        --output-dir driver/crawler/analysis_results/H200_enhanced_schedulers_v1_full \
        --min-qps 0.5 --max-qps 4.0 --title-suffix " (H200 Crawler)"

    # ANNS workload (QPS 0.25-2.0)
    python driver/anns/plotter_utils/plot_latency_vs_qps_filtered.py \
        --log-dir driver/anns/test_run_log \
        --output-dir driver/anns/analysis_results/H200_enhanced_schedulers_v1_full \
        --min-qps 0.25 --max-qps 2.0 --title-suffix " (H200 ANNS)"
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

TTFTArray = np.ndarray


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


def _extract(csv_file: Path) -> Tuple[TTFTArray, TTFTArray]:
    df = pd.read_csv(csv_file, low_memory=False)
    if {"event_type", "duration_secs", "stream"}.issubset(df.columns):
        df = df[df["event_type"] == "query_ttft"]
        stream_mask = df["stream"].astype(bool)
        return (
            df[stream_mask]["duration_secs"].to_numpy(float),
            df[~stream_mask]["duration_secs"].to_numpy(float),
        )
    raise ValueError("missing columns")


def _dataset(csv_files: Sequence[Path], min_qps: float = 0, max_qps: float = float('inf')):
    """Extract dataset with QPS filtering."""
    data: MutableMapping[str,
                         MutableMapping[float, Dict[str, List[float]]]] = (
                             defaultdict(lambda: defaultdict(lambda: {
                                 "streaming": [],
                                 "non_streaming": []
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
            s, ns = _extract(f)
            data[sched][qps_val]["streaming"].extend(s.tolist())
            data[sched][qps_val]["non_streaming"].extend(ns.tolist())
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


def plot_latency_vs_qps(data, output_dir: Path, title_suffix: str = ""):
    """Create side-by-side P50 and P99 latency vs QPS plots."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})

    if not all_qps:
        print("[error] No QPS data found")
        return

    # Prepare data for plotting
    scheduler_data = {}
    for sched in sorted(data.keys()):

        qps_values = []
        p50_streaming = []
        p50_non_streaming = []
        p99_streaming = []
        p99_non_streaming = []

        for qps in all_qps:
            if qps not in data[sched]:
                continue

            streaming = np.array(data[sched][qps]["streaming"])
            non_streaming = np.array(data[sched][qps]["non_streaming"])

            # Skip if no streaming data (streaming is required)
            if len(streaming) == 0:
                continue

            qps_values.append(qps)
            p50_streaming.append(np.percentile(streaming, 50))
            p99_streaming.append(np.percentile(streaming, 99))

            # Only add non-streaming if available
            if len(non_streaming) > 0:
                p50_non_streaming.append(np.percentile(non_streaming, 50))
                p99_non_streaming.append(np.percentile(non_streaming, 99))
            else:
                # Use None if non-streaming data is not available
                p50_non_streaming.append(None)
                p99_non_streaming.append(None)

        scheduler_data[sched] = {
            "qps": qps_values,
            "p50_streaming": p50_streaming,
            "p50_non_streaming": p50_non_streaming,
            "p99_streaming": p99_streaming,
            "p99_non_streaming": p99_non_streaming,
        }

    # Plot P50 latency vs QPS
    ax_p50 = axes[0]
    for sched in sorted(scheduler_data.keys()):
        display_name = _get_display_name(sched)
        color = SCHEDULER_COLORS.get(sched, "#000000")
        sched_data = scheduler_data[sched]

        ax_p50.plot(sched_data["qps"],
                    sched_data["p50_streaming"],
                    marker='o',
                    linestyle='-',
                    linewidth=2.5,
                    markersize=8,
                    label=f"{display_name} (Streaming)",
                    color=color,
                    alpha=0.85)

        # Only plot non-streaming if we have data
        non_streaming_data = sched_data["p50_non_streaming"]
        if any(v is not None for v in non_streaming_data):
            # Filter out None values for plotting
            qps_ns = [q for q, v in zip(sched_data["qps"], non_streaming_data) if v is not None]
            p50_ns = [v for v in non_streaming_data if v is not None]
            if qps_ns:
                ax_p50.plot(qps_ns,
                            p50_ns,
                            marker='s',
                            linestyle='--',
                            linewidth=2,
                            markersize=7,
                            label=f"{display_name} (Non-Streaming)",
                            color=color,
                            alpha=0.6)

    ax_p50.set_xlabel("QPS", fontsize=16, fontweight='bold')
    ax_p50.set_ylabel("P50 Latency (s)", fontsize=16, fontweight='bold')
    ax_p50.set_title("P50 Latency vs QPS", fontsize=18, fontweight='bold')
    ax_p50.grid(True, alpha=0.3)
    ax_p50.tick_params(axis='both', which='major', labelsize=14)
    ax_p50.legend(fontsize=11, loc='upper left', ncol=2)

    # Plot P99 latency vs QPS
    ax_p99 = axes[1]
    for sched in sorted(scheduler_data.keys()):
        display_name = _get_display_name(sched)
        color = SCHEDULER_COLORS.get(sched, "#000000")
        sched_data = scheduler_data[sched]

        ax_p99.plot(sched_data["qps"],
                    sched_data["p99_streaming"],
                    marker='o',
                    linestyle='-',
                    linewidth=2.5,
                    markersize=8,
                    label=f"{display_name} (Streaming)",
                    color=color,
                    alpha=0.85)

        # Only plot non-streaming if we have data
        non_streaming_data = sched_data["p99_non_streaming"]
        if any(v is not None for v in non_streaming_data):
            # Filter out None values for plotting
            qps_ns = [q for q, v in zip(sched_data["qps"], non_streaming_data) if v is not None]
            p99_ns = [v for v in non_streaming_data if v is not None]
            if qps_ns:
                ax_p99.plot(qps_ns,
                            p99_ns,
                            marker='s',
                            linestyle='--',
                            linewidth=2,
                            markersize=7,
                            label=f"{display_name} (Non-Streaming)",
                            color=color,
                            alpha=0.6)

    ax_p99.set_xlabel("QPS", fontsize=16, fontweight='bold')
    ax_p99.set_ylabel("P99 Latency (s)", fontsize=16, fontweight='bold')
    ax_p99.set_title("P99 Latency vs QPS", fontsize=18, fontweight='bold')
    ax_p99.grid(True, alpha=0.3)
    ax_p99.tick_params(axis='both', which='major', labelsize=14)
    ax_p99.legend(fontsize=11, loc='upper left', ncol=2)

    # Overall title
    fig.suptitle(f"Latency Scaling vs QPS{title_suffix}",
                 fontsize=20, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "latency_vs_qps.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {output_dir / filename}")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate latency vs QPS plots for P50 and P99 with QPS range filtering")
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

    # Generate latency vs QPS plots
    plot_latency_vs_qps(data, args.output_dir, args.title_suffix)


if __name__ == "__main__":
    main()
