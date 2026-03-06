#!/usr/bin/env python3
"""plot_ttft_ccdf_combined.py
Generate combined CCDF plots with all schedulers together per QPS value.

Creates one plot per QPS value showing all schedulers on the same axes,
using inverted log-scale CCDF visualization for tail latency comparison.
Each scheduler gets a unique color with solid lines for streaming and
dashed lines for non-streaming.
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
                          "fcfs_lru", "lcas_lifo", "mcps_lce", "oeda_pbas",
                          "stream_based_v1", "lcas_cplusp")

# Color palette for schedulers
SCHEDULER_COLORS: dict[str, str] = {
    "default_vllm": "#1f77b4",
    "fcfs_lru": "#ff7f0e",
    "lcas_lifo": "#2ca02c",
    "mcps_lce": "#9467bd",
    "oeda_pbas": "#8c564b",
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
        return cfg.get("scheduler", {}).get("type", "unknown")
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


def _dataset(csv_files: Sequence[Path]):
    data: MutableMapping[str,
                         MutableMapping[float, Dict[str, List[float]]]] = (
                             defaultdict(lambda: defaultdict(lambda: {
                                 "streaming": [],
                                 "non_streaming": []
                             })))
    for f in csv_files:
        try:
            qps_val = _qps(f.parent)
            sched = _sched(f.parent)
            s, ns = _extract(f)
            data[sched][qps_val]["streaming"].extend(s.tolist())
            data[sched][qps_val]["non_streaming"].extend(ns.tolist())
        except Exception as err:
            print("[warn]", err)
    return data


def _plot_ccdf(ax, arr: TTFTArray, label: str, **style):
    """Plot CCDF curve."""
    if arr.size == 0:
        return
    srt = np.sort(arr)
    cdf = np.arange(1, srt.size + 1) / srt.size
    ccdf = 1 - cdf
    ax.plot(srt, ccdf, label=label, **style)


def _combined_plot_per_qps(data, qps: float, out_dir: Path):
    """Create a combined plot with all schedulers for a specific QPS value."""

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each scheduler at this QPS
    legend_entries = []
    for sched in sorted(data.keys()):
        if qps not in data[sched]:
            continue

        color = SCHEDULER_COLORS.get(sched, "#000000")

        # Plot streaming data
        streaming_data = np.asarray(data[sched][qps]["streaming"])
        if streaming_data.size > 0:
            label = f"{sched.replace('_', ' ').title()} (Streaming)"
            _plot_ccdf(ax,
                       streaming_data,
                       label,
                       color=color,
                       linestyle="-",
                       linewidth=2.5,
                       alpha=0.85)
            legend_entries.append(label)

        # Plot non-streaming data
        non_streaming_data = np.asarray(data[sched][qps]["non_streaming"])
        if non_streaming_data.size > 0:
            label = f"{sched.replace('_', ' ').title()} (Non-Streaming)"
            _plot_ccdf(ax,
                       non_streaming_data,
                       label,
                       color=color,
                       linestyle="--",
                       linewidth=2,
                       alpha=0.7)
            legend_entries.append(label)

    # Set up axes
    ax.set_xlabel("TTFT (s)", fontsize=14)
    ax.set_ylabel("P(TTFT > x) [%]", fontsize=14)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.invert_yaxis()

    # Format y-axis as percentages
    from matplotlib.ticker import FuncFormatter

    def percent_formatter(y, pos):
        pct = y * 100
        if pct >= 10:
            return f'{pct:.0f}%'
        elif pct >= 1:
            return f'{pct:.1f}%'
        elif pct >= 0.1:
            return f'{pct:.2f}%'
        else:
            return f'{pct:.3f}%'

    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    ax.grid(alpha=0.3, which='both')

    # Title
    title = f"Combined TTFT CCDF — All Schedulers @ QPS {qps:.3f}"
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Legend inside plot
    ax.legend(loc='lower right',
              fontsize=10,
              ncol=2 if len(legend_entries) > 12 else 1,
              framealpha=0.95)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"ttft_ccdf_combined_{qps:.3f}_qps.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_dir / filename}")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=
        "Generate combined CCDF plots showing all schedulers per QPS value")
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-dir",
                        default=Path("ttft_ccdf_combined"),
                        type=Path)
    parser.add_argument("--qps",
                        type=float,
                        help="Plot specific QPS only (optional)")
    args = parser.parse_args(argv)

    csv_files = list(args.log_dir.rglob("run_metrics.csv"))
    if not csv_files:
        raise SystemExit("no run_metrics.csv found")

    data = _dataset(csv_files)
    if not data:
        raise SystemExit("no data discovered")

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})

    if args.qps is not None:
        # Filter to closest QPS if specified
        all_qps = [min(all_qps, key=lambda x: abs(x - args.qps))]

    # Generate one plot per QPS with all schedulers
    for qps in all_qps:
        _combined_plot_per_qps(data, qps, args.output_dir)


if __name__ == "__main__":
    main()
