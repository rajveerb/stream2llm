#!/usr/bin/env python3
"""plot_ttft_ccdf_stacked_2x4.py
Generate a 2x4 stacked figure with Crawler (top row) and ANNS (bottom row).

Each row shows 4 CCDF subplots at different QPS values. The workload name
is displayed on the right side of each row. No main title.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yaml

SCHEDULERS: Tuple[str, ...] = (
    "recomp", "swap", "recomp_and_swap", "default_vllm",
    "fcfs", "mcps", "lcas",
)

SCHEDULER_COLORS: dict[str, str] = {
    "default_vllm": "#1f77b4",
    "fcfs": "#ff7f0e",
    "lcas": "#2ca02c",
    "mcps": "#d62728",
    "recomp": "#7f7f7f",
    "swap": "#bcbd22",
    "recomp_and_swap": "#17becf",
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


def _dataset(csv_files: Sequence[Path]):
    data: MutableMapping[str,
                         MutableMapping[float, Dict[str, List[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: {
            "streaming": [], "non_streaming": [],
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
    if arr.size == 0:
        return
    srt = np.sort(arr)
    cdf = np.arange(1, srt.size + 1) / srt.size
    ccdf = 1 - cdf
    ax.plot(srt, ccdf, label=label, **style)


def _get_display_name(sched: str) -> str:
    display_names = {
        "default_vllm": "Default vLLM",
        "fcfs": "FCFS",
        "lcas": "LCAS",
        "mcps": "MCPS",
    }
    return display_names.get(sched, sched.replace('_', ' ').title())


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


def _plot_row(axes, data, target_qps, workload_name, is_top_row):
    """Plot one row of 4 CCDF subplots."""
    for idx, target_qps_val in enumerate(target_qps):
        ax = axes[idx]

        all_qps = {q for sched in data.values() for q in sched}
        if not all_qps:
            continue
        qps = min(all_qps, key=lambda x: abs(x - target_qps_val))

        for sched in sorted(data.keys()):
            if qps not in data[sched]:
                continue

            color = SCHEDULER_COLORS.get(sched, "#000000")
            display_name = _get_display_name(sched)

            streaming_data = np.asarray(data[sched][qps]["streaming"])
            if streaming_data.size > 0:
                _plot_ccdf(ax, streaming_data,
                           f"{display_name} (Streaming)",
                           color=color, linestyle="-",
                           linewidth=5, alpha=0.85)

            non_streaming_data = np.asarray(data[sched][qps]["non_streaming"])
            if non_streaming_data.size > 0:
                _plot_ccdf(ax, non_streaming_data,
                           f"{display_name} (Non-Streaming)",
                           color=color, linestyle="--",
                           linewidth=4.5, alpha=0.7)

        # Axes setup
        ax.set_xlabel("TTFT (seconds)", fontsize=26, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("P(TTFT > x)", fontsize=26, fontweight='bold')
        else:
            ax.set_ylabel("")

        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.invert_yaxis()
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.grid(alpha=0.3, which='both')

        # Subplot title (QPS)
        ax.set_title(f"QPS {qps:.3f}", fontsize=28, fontweight='bold')

    # Workload label on the right side of the row
    axes[-1].annotate(
        workload_name, xy=(1.08, 0.5), xycoords='axes fraction',
        fontsize=30, fontweight='bold', rotation=-90,
        ha='center', va='center',
    )


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate a stacked 2x4 CCDF plot (Crawler + ANNS)")
    parser.add_argument("--crawler-log-dir", required=True, type=Path)
    parser.add_argument("--anns-log-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("figures"), type=Path)
    parser.add_argument("--output-filename", default="ttft_ccdf_stacked_2x4.png",
                        type=str)
    args = parser.parse_args(argv)

    # Load data
    crawler_csvs = list(args.crawler_log_dir.rglob("run_metrics.csv"))
    anns_csvs = list(args.anns_log_dir.rglob("run_metrics.csv"))
    if not crawler_csvs:
        raise SystemExit("no crawler run_metrics.csv found")
    if not anns_csvs:
        raise SystemExit("no ANNS run_metrics.csv found")

    crawler_data = _dataset(crawler_csvs)
    anns_data = _dataset(anns_csvs)

    # Create 2x4 figure
    fig, axes = plt.subplots(2, 4, figsize=(28, 10))

    _plot_row(axes[0], crawler_data,
              target_qps=[0.5, 1.0, 2.0, 4.0],
              workload_name="Crawler", is_top_row=True)
    _plot_row(axes[1], anns_data,
              target_qps=[0.25, 0.5, 1.0, 2.0],
              workload_name="ANNS", is_top_row=False)

    # Shared legend from first subplot
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.48, -0.04),
               ncol=len(handles), fontsize=24, framealpha=0.95)

    fig.tight_layout(rect=[0, 0.04, 0.96, 1.0])
    fig.subplots_adjust(wspace=0.35, hspace=0.55)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
