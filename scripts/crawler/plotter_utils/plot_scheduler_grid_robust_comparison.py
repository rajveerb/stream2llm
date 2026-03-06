#!/usr/bin/env python3
"""robust_scheduler_grid_no_verdict.py
Generate TTFT comparison dashboards for vLLM schedulers with quantitative
metrics only (W₁, Cliff’s Δ, p50×, p95×).
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
# Set global font sizes for clearer readability
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})
import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml

SCHEDULERS: Tuple[str,
                  ...] = ("recomp", "swap", "recomp_and_swap", "default_vllm",
                          "fcfs_lru", "lcas_lifo", "mcps_lce", "oeda_pbas",
                          "stream_based_v1", "lcas_cplusp")

COLORS: Mapping[str, str] = {
    "baseline": "#d62728",
    "default_vllm_streaming": "#1f77b4",
    "fcfs_lru": "#ff7f0e",
    "lcas_lifo": "#2ca02c",
    "mcps_lce": "#9467bd",
    "oeda_pbas": "#8c564b",
    "stream_based_v1": "#e377c2",
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
    data: MutableMapping[str, MutableMapping[float, Dict[
        str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: {
            "streaming": [],
            "non_streaming": []
        }))
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


def _cliffs_delta(x: TTFTArray, y: TTFTArray) -> float:
    gt = np.sum(x[:, None] > y)
    lt = np.sum(x[:, None] < y)
    return (gt - lt) / (x.size * y.size)


def _best_performance_ratio(b: TTFTArray, c: TTFTArray):
    """Find the percentile where baseline/comparison ratio is maximum."""
    if b.size == 0 or c.size == 0:
        return np.nan, np.nan

    max_ratio = 0.0  # Start with minimum ratio
    best_percentile = np.nan

    # Test CDF levels from 5 to 95 to avoid extreme edge cases
    for percentile in range(5, 96):
        # Find the x-values (TTFT) at this percentile for both distributions
        b_value = np.percentile(b, percentile)
        c_value = np.percentile(c, percentile)

        # Skip if either value is 0 or very close to 0 to avoid division issues
        if c_value > 1e-6 and b_value > 1e-6:
            # Calculate baseline/comparison ratio - higher means comparison is slower
            ratio = b_value / c_value

            # Keep track of the maximum ratio and its percentile
            if ratio > max_ratio:
                max_ratio = ratio
                best_percentile = percentile

    return (max_ratio if max_ratio > 0 else np.nan, best_percentile)


def _metrics(b: TTFTArray, c: TTFTArray):
    best_ratio, best_percentile = _best_performance_ratio(b, c)
    return {
        "w1": stats.wasserstein_distance(b, c),
        "delta": _cliffs_delta(b, c),
        "best": best_ratio,
        "best_percentile": best_percentile,
        "p50": np.percentile(b, 50) / np.percentile(c, 50),
        "p95": np.percentile(b, 95) / np.percentile(c, 95),
    }


def _cdf(ax, arr: TTFTArray, label: str, **style):
    if arr.size == 0:
        return
    srt = np.sort(arr)
    cdf = np.arange(1, srt.size + 1) / srt.size
    ax.plot(srt, cdf, label=label, **style)


def _cdf_adaptive_color(ax, baseline_arr: TTFTArray, comp_arr: TTFTArray,
                        label: str, **style):
    """Plot comparison CDF with adaptive coloring: green where better, red where worse."""
    if comp_arr.size == 0 or baseline_arr.size == 0:
        return

    # Create sorted comparison data
    comp_srt = np.sort(comp_arr)
    comp_cdf = np.arange(1, comp_srt.size + 1) / comp_srt.size

    # For each x-value in comparison, find corresponding baseline CDF value
    baseline_cdf_interp = np.interp(
        comp_srt, np.sort(baseline_arr),
        np.arange(1, baseline_arr.size + 1) / baseline_arr.size)

    # Determine color for each point: green if comp_cdf > baseline_cdf, red otherwise
    colors = [
        'green' if comp_y > baseline_y else 'red'
        for comp_y, baseline_y in zip(comp_cdf, baseline_cdf_interp)
    ]

    # Plot line segments with appropriate colors
    for i in range(len(comp_srt) - 1):
        ax.plot([comp_srt[i], comp_srt[i + 1]], [comp_cdf[i], comp_cdf[i + 1]],
                color=colors[i],
                linewidth=style.get('lw', 2.5),
                alpha=style.get('alpha', 1.0))

    # Add a single legend entry (use the most common color)
    legend_color = 'green' if colors.count('green') > colors.count(
        'red') else 'red'
    ax.plot([], [],
            color=legend_color,
            linewidth=style.get('lw', 2.5),
            label=label)


def _grid(data, qps_val: float, out_dir: Path):
    base = np.asarray(data["default_vllm"][qps_val]["non_streaming"])
    if base.size == 0:
        print("[skip] baseline", qps_val)
        return

    comps: List[Tuple[str, str, str]] = []
    if data["default_vllm"][qps_val]["streaming"]:
        comps.append(("default_vllm", "streaming", "Streaming Default vLLM"))
    for s in sorted(data):
        if s == "default_vllm" or qps_val not in data[s]:
            continue
        for mode in ("streaming", "non_streaming"):
            if data[s][qps_val][mode]:
                title = f"{'Streaming - ' if mode=='streaming' else 'Non-Streaming'} {s.replace('_',' ').title()}"
                comps.append((s, mode, title))

    if not comps:
        print("[skip] no comps", qps_val)
        return

    rows = math.ceil(len(comps) / 3)
    cols = min(3, len(comps))
    fig, axes = plt.subplots(rows,
                             cols,
                             figsize=(5 * cols, 4 * rows),
                             squeeze=False)
    axes = axes.flatten()

    for idx, (sched, mode, title) in enumerate(comps):
        ax = axes[idx]
        # Always use blue for baseline
        _cdf(
            ax,
            base,
            "Non-Streaming - Default vLLM",
            color="#1f77b4",  # Blue
            ls=":",
            lw=3,
            alpha=0.8)
        comp_arr = np.asarray(data[sched][qps_val][mode])
        if comp_arr.size == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        # Use adaptive coloring for comparison curve
        _cdf_adaptive_color(ax, base, comp_arr, title, lw=2.5)

        # Calculate metrics for display
        m = _metrics(base, comp_arr)
        # Use wider plus/minus signs for clarity in Δ
        # Format Δ: show plus sign for positive values; if rounded value is 0.00 or negative,
        # display without any sign (no negative sign even for negative numbers as requested).
        rounded_delta = round(m['delta'], 2)
        if rounded_delta > 0:
            delta_str = f"+{rounded_delta:.2f}"
        else:
            # Use a wider Unicode minus sign (U+2212) so that negative sign is clearly visible in the plot text.
            unicode_minus = "\u2212"  # "−"
            delta_str = f"{unicode_minus}{abs(rounded_delta):.2f}"
        # Format the best performance display with percentile
        if not np.isnan(m['best_percentile']):
            best_str = f"Best×={m['best']:.2f}@P{m['best_percentile']:.0f}"
        else:
            best_str = f"Best×={m['best']:.2f}"

        ax.text(
            0.97,
            0.58,
            f"W₁={m['w1']:.3f}s\nΔ={delta_str}\n{best_str}\np50×={m['p50']:.2f}\np95×={m['p95']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.8", fc="white", alpha=0.85),
        )
        ax.set_title(title)
        ax.set_xlabel("TTFT (s)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11, loc="lower right")

    for ax in axes[len(comps):]:
        ax.set_visible(False)

    fig.suptitle(f"TTFT CDF Grid — QPS {qps_val:.3f}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"ttft_grid_{qps_val:.3f}_qps.png", dpi=300)
    plt.close(fig)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("ttft_grids"), type=Path)
    parser.add_argument("--qps", type=float)
    args = parser.parse_args(argv)

    csv_files = list(args.log_dir.rglob("run_metrics.csv"))
    if not csv_files:
        raise SystemExit("no run_metrics.csv found")

    data = _dataset(csv_files)
    all_qps = {q for sched in data.values() for q in sched}
    if not all_qps:
        raise SystemExit("no qps values discovered")

    if args.qps is not None:
        all_qps = {min(all_qps, key=lambda x: abs(x - args.qps))}

    for q in sorted(all_qps):
        _grid(data, q, args.output_dir)


if __name__ == "__main__":
    main()
