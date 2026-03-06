#!/usr/bin/env python3
"""plot_tokens_invalidated_combined.py
Generate combined token invalidation CDF plots showing all schedulers.

Creates plots showing how many tokens are invalidated (need recomputation)
per request across all schedulers at representative QPS levels.

Usage:
    # ANNS workload token invalidation
    python driver/anns/plotter_utils/plot_tokens_invalidated_combined.py \
        --log-dir driver/anns/run_log/H200_enhanced_schedulers_v1_full \
        --output-dir driver/anns/analysis_results/H200_enhanced_schedulers_v1_full \
        --min-qps 0.25 --max-qps 2.0 --title-suffix " (H200 ANNS)"
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
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


def _extract_tokens_invalidated(json_file: Path) -> List[int]:
    """Extract tokens invalidated per request from collected_outputs_streaming.json."""
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        invalidation_counts = []
        for item in json_data:
            num_invalidated = item.get('num_tokens_invalidated')
            if num_invalidated is not None:
                invalidation_counts.append(int(num_invalidated))

        return invalidation_counts
    except Exception as err:
        print(f"[warn] Error reading {json_file}: {err}")
        return []


def _dataset(log_dir: Path, min_qps: float = 0, max_qps: float = float('inf')):
    """Extract dataset from collected_outputs_streaming.json files with QPS filtering."""
    data: MutableMapping[str,
                         MutableMapping[float, List[int]]] = (
                             defaultdict(lambda: defaultdict(list)))

    # Find all collected_outputs_streaming.json files
    json_files = list(log_dir.rglob("collected_outputs_streaming.json"))

    if not json_files:
        print("[warn] No collected_outputs_streaming.json files found")
        return data

    print(f"[info] Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            run_dir = json_file.parent
            qps_val = _qps(run_dir)

            # Skip if outside QPS range
            if not (min_qps <= qps_val <= max_qps):
                continue

            sched = _sched(run_dir)

            # Skip oeda_pbas
            if sched == "oeda_pbas":
                continue

            tokens = _extract_tokens_invalidated(json_file)
            if tokens:
                data[sched][qps_val].extend(tokens)
                print(f"[loaded] {sched} QPS {qps_val}: {len(tokens)} queries")

        except Exception as err:
            print(f"[warn] Error processing {json_file}: {err}")

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


def plot_tokens_invalidated_combined(data, output_dir: Path, title_suffix: str = ""):
    """Create combined token invalidation CDF plots for each QPS level."""

    # Get all QPS values and select representative ones for plotting
    all_qps = sorted({q for sched in data.values() for q in sched})

    if not all_qps:
        print("[error] No QPS data found")
        return

    # Use only the QPS levels we have data for (0.25, 0.5, 1.0, 2.0 for ANNS)
    qps_to_plot = [q for q in all_qps if q in [0.25, 0.5, 1.0, 2.0, 4.0]]
    if not qps_to_plot:
        qps_to_plot = all_qps[:4]  # Fallback to first 4 QPS values

    # Create subplots for each QPS level (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, qps_val in enumerate(qps_to_plot):
        ax = axes[idx]

        # Prepare data for this QPS level
        for sched in sorted(data.keys()):
            if qps_val not in data[sched]:
                continue

            tokens = np.array(data[sched][qps_val])
            if len(tokens) == 0:
                continue

            # Sort for CDF
            tokens_sorted = np.sort(tokens)
            y = np.arange(1, len(tokens_sorted) + 1) / len(tokens_sorted)

            display_name = _get_display_name(sched)
            color = SCHEDULER_COLORS.get(sched, "#000000")

            ax.plot(tokens_sorted, y,
                   label=display_name,
                   linewidth=2.5,
                   color=color,
                   alpha=0.85)

        ax.set_xlabel("Tokens Invalidated per Request", fontsize=13, fontweight='bold')
        ax.set_ylabel("CDF", fontsize=13, fontweight='bold')
        ax.set_title(f"QPS {qps_val}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)

        if idx == 0:  # Only put legend on first plot to avoid clutter
            ax.legend(fontsize=10, loc='lower right')

    # Overall title
    fig.suptitle(f"Token Invalidation Distribution Across Schedulers{title_suffix}",
                 fontsize=16, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "tokens_invalidated_combined.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {output_dir / filename}")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate combined token invalidation CDF plots")
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

    # Generate combined token invalidation plot
    plot_tokens_invalidated_combined(data, args.output_dir, args.title_suffix)


if __name__ == "__main__":
    main()
