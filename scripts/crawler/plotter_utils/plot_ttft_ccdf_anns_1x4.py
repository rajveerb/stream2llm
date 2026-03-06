#!/usr/bin/env python3
"""plot_ttft_ccdf_anns_1x4.py
Generate a combined figure with 4 CCDF subplots (QPS 0.25, 0.5, 1.0, 2.0) for ANNS data.

Creates a 1x4 subplot layout showing all schedulers at different QPS values,
using inverted log-scale CCDF visualization for tail latency comparison.
Each scheduler gets a unique color with solid lines for streaming and
dashed lines for non-streaming. Excludes oeda_pbas scheduler.
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
        # Handle both string and dict formats
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
            # Skip oeda_pbas
            if sched == "oeda_pbas":
                continue
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


def _combined_plot_1x4(data, out_dir: Path, title_suffix: str = ""):
    """Create a 1x4 subplot figure with QPS 0.25, 0.5, 1.0, 2.0."""

    # Target QPS values
    target_qps = [0.25, 0.5, 1.0, 2.0]

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    for idx, target_qps_val in enumerate(target_qps):
        ax = axes[idx]

        # Find closest QPS to target
        all_qps = {q for sched in data.values() for q in sched}
        if not all_qps:
            continue
        qps = min(all_qps, key=lambda x: abs(x - target_qps_val))

        # Plot each scheduler at this QPS
        legend_entries = []
        for sched in sorted(data.keys()):
            if qps not in data[sched]:
                continue

            color = SCHEDULER_COLORS.get(sched, "#000000")
            display_name = _get_display_name(sched)

            # Plot streaming data
            streaming_data = np.asarray(data[sched][qps]["streaming"])
            if streaming_data.size > 0:
                label = f"{display_name} (Streaming)"
                _plot_ccdf(ax,
                           streaming_data,
                           label,
                           color=color,
                           linestyle="-",
                           linewidth=5,
                           alpha=0.85)
                legend_entries.append(label)

            # Plot non-streaming data
            non_streaming_data = np.asarray(data[sched][qps]["non_streaming"])
            if non_streaming_data.size > 0:
                label = f"{display_name} (Non-Streaming)"
                _plot_ccdf(ax,
                           non_streaming_data,
                           label,
                           color=color,
                           linestyle="--",
                           linewidth=4.5,
                           alpha=0.7)
                legend_entries.append(label)

        # Set up axes
        ax.set_xlabel("TTFT (s)", fontsize=22, fontweight='bold')
        ax.set_ylabel("P(TTFT > x) [%]", fontsize=22, fontweight='bold')
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

        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.grid(alpha=0.3, which='both')

        # Subplot title
        ax.set_title(f"QPS {qps:.3f}", fontsize=24, fontweight='bold')

    # Add shared legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=5, fontsize=18, framealpha=0.95)

    # Overall title
    fig.suptitle(f"Combined TTFT CCDF — All Schedulers (QPS 0.25 to 2.0){title_suffix}",
                 fontsize=28, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = "ttft_ccdf_combined_1x4.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_dir / filename}")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=
        "Generate a 1x4 combined CCDF plot showing QPS 0.25, 0.5, 1.0, 2.0 for ANNS data (except oeda_pbas)")
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-dir",
                        default=Path("ttft_ccdf_combined"),
                        type=Path)
    parser.add_argument("--title-suffix",
                        type=str,
                        default="",
                        help="Additional text to append to the title")
    args = parser.parse_args(argv)

    csv_files = list(args.log_dir.rglob("run_metrics.csv"))
    if not csv_files:
        raise SystemExit("no run_metrics.csv found")

    data = _dataset(csv_files)
    if not data:
        raise SystemExit("no data discovered")

    # Generate 1x4 subplot figure
    _combined_plot_1x4(data, args.output_dir, args.title_suffix)


if __name__ == "__main__":
    main()
