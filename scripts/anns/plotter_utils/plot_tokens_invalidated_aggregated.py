#!/usr/bin/env python3
"""plot_tokens_invalidated_aggregated.py
Generate aggregated token invalidation plots showing scheduler efficiency ratios.

Similar to TTFT CCDF plots, shows P(Tokens Invalidated > x) with percentile markers
and calculates efficiency ratios comparing schedulers to baseline.

Usage:
    python driver/anns/plotter_utils/plot_tokens_invalidated_aggregated.py \
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
import matplotlib.ticker as ticker
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


def plot_tokens_invalidated_ccdf_1x4(data, output_dir: Path, title_suffix: str = ""):
    """Create 1x4 CCDF plots similar to TTFT style with percentile annotations."""

    # Get all QPS values and select representative ones for plotting
    all_qps = sorted({q for sched in data.values() for q in sched})

    if not all_qps:
        print("[error] No QPS data found")
        return

    # Use only the QPS levels we have data for (0.25, 0.5, 1.0, 2.0 for ANNS)
    qps_to_plot = [q for q in all_qps if q in [0.25, 0.5, 1.0, 2.0, 4.0]]
    if not qps_to_plot:
        qps_to_plot = all_qps[:4]  # Fallback to first 4 QPS values

    # Create 1x4 subplot layout
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, qps_val in enumerate(qps_to_plot):
        ax = axes[idx]

        # Store percentile data for annotation
        percentile_data = {}

        # Prepare data for this QPS level
        for sched in sorted(data.keys()):
            if qps_val not in data[sched]:
                continue

            tokens = np.array(data[sched][qps_val])
            if len(tokens) == 0:
                continue

            # Sort for CCDF (P(X > x))
            tokens_sorted = np.sort(tokens)
            # CCDF: 1 - CDF
            ccdf = (len(tokens_sorted) - np.arange(len(tokens_sorted))) / len(tokens_sorted)

            # Convert to percentage
            ccdf_pct = ccdf * 100

            display_name = _get_display_name(sched)
            color = SCHEDULER_COLORS.get(sched, "#000000")

            ax.plot(tokens_sorted, ccdf_pct,
                   label=display_name,
                   linewidth=2.5,
                   color=color,
                   alpha=0.85)

            # Store P50, P95, P99 values
            p50_idx = np.searchsorted(ccdf, 0.50, side='right')
            p95_idx = np.searchsorted(ccdf, 0.05, side='right')
            p99_idx = np.searchsorted(ccdf, 0.01, side='right')

            percentile_data[sched] = {
                'p50': tokens_sorted[min(p50_idx, len(tokens_sorted)-1)] if p50_idx < len(tokens_sorted) else tokens_sorted[-1],
                'p95': tokens_sorted[min(p95_idx, len(tokens_sorted)-1)] if p95_idx < len(tokens_sorted) else tokens_sorted[-1],
                'p99': tokens_sorted[min(p99_idx, len(tokens_sorted)-1)] if p99_idx < len(tokens_sorted) else tokens_sorted[-1],
            }

        ax.set_xlabel("Tokens Invalidated", fontsize=13, fontweight='bold')
        ax.set_ylabel("P(Tokens Invalidated > x) [%]", fontsize=13, fontweight='bold')
        ax.set_title(f"QPS {qps_val}", fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(axis='both', which='major', labelsize=11)

        # Set y-axis to show percentages
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.1f}%'))
        ax.set_ylim([0.1, 100])

        if idx == 0:  # Only put legend on first plot
            ax.legend(fontsize=10, loc='upper right')

        # Print efficiency ratios (lower is better for token invalidation)
        if 'default_vllm' in percentile_data:
            baseline_p50 = percentile_data['default_vllm']['p50']
            baseline_p95 = percentile_data['default_vllm']['p95']

            print(f"\n[QPS {qps_val}] Token Invalidation Efficiency vs Baseline:")
            for sched in sorted(percentile_data.keys()):
                if sched != 'default_vllm':
                    p50_ratio = percentile_data[sched]['p50'] / baseline_p50 if baseline_p50 > 0 else float('inf')
                    p95_ratio = percentile_data[sched]['p95'] / baseline_p95 if baseline_p95 > 0 else float('inf')
                    print(f"  {_get_display_name(sched)}: P50 ratio={p50_ratio:.2f}x, P95 ratio={p95_ratio:.2f}x (lower is better)")

    # Overall title
    fig.suptitle(f"Token Invalidation CCDF Across Load Levels{title_suffix}",
                 fontsize=16, fontweight='bold', y=1.02)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "tokens_invalidated_ccdf_1x4.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[saved] {output_dir / filename}")


def plot_tokens_invalidated_efficiency_table(data, output_dir: Path, title_suffix: str = ""):
    """Generate a table showing token invalidation efficiency ratios."""

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})
    qps_to_plot = [q for q in all_qps if q in [0.25, 0.5, 1.0, 2.0, 4.0]]
    if not qps_to_plot:
        qps_to_plot = all_qps[:4]

    results = []

    for qps_val in qps_to_plot:
        # Calculate percentiles for all schedulers
        percentile_data = {}

        for sched in sorted(data.keys()):
            if qps_val not in data[sched]:
                continue

            tokens = np.array(data[sched][qps_val])
            if len(tokens) == 0:
                continue

            percentile_data[sched] = {
                'p50': np.percentile(tokens, 50),
                'p95': np.percentile(tokens, 95),
                'p99': np.percentile(tokens, 99),
                'mean': np.mean(tokens),
            }

        # Calculate ratios vs baseline
        if 'default_vllm' in percentile_data:
            baseline = percentile_data['default_vllm']

            for sched in sorted(percentile_data.keys()):
                if sched != 'default_vllm':
                    p50_ratio = percentile_data[sched]['p50'] / baseline['p50'] if baseline['p50'] > 0 else float('inf')
                    p95_ratio = percentile_data[sched]['p95'] / baseline['p95'] if baseline['p95'] > 0 else float('inf')
                    mean_ratio = percentile_data[sched]['mean'] / baseline['mean'] if baseline['mean'] > 0 else float('inf')

                    results.append({
                        'QPS': qps_val,
                        'Scheduler': _get_display_name(sched),
                        'P50_Ratio': p50_ratio,
                        'P95_Ratio': p95_ratio,
                        'Mean_Ratio': mean_ratio,
                    })

    # Save results to text file
    output_file = output_dir / "tokens_invalidated_efficiency_ratios.txt"
    with open(output_file, 'w') as f:
        f.write(f"Token Invalidation Efficiency Ratios vs Baseline (Default vLLM){title_suffix}\n")
        f.write("=" * 80 + "\n")
        f.write("Lower ratio = better cache efficiency (fewer tokens invalidated)\n")
        f.write("=" * 80 + "\n\n")

        for qps_val in qps_to_plot:
            f.write(f"\nQPS {qps_val}:\n")
            f.write("-" * 60 + "\n")
            for result in results:
                if result['QPS'] == qps_val:
                    f.write(f"  {result['Scheduler']:15s}: "
                           f"P50={result['P50_Ratio']:.3f}x  "
                           f"P95={result['P95_Ratio']:.3f}x  "
                           f"Mean={result['Mean_Ratio']:.3f}x\n")

    print(f"[saved] {output_file}")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate aggregated token invalidation efficiency plots")
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

    # Generate CCDF plot with efficiency annotations
    plot_tokens_invalidated_ccdf_1x4(data, args.output_dir, args.title_suffix)

    # Generate efficiency ratio table
    plot_tokens_invalidated_efficiency_table(data, args.output_dir, args.title_suffix)


if __name__ == "__main__":
    main()
