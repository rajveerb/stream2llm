#!/usr/bin/env python3
"""
Plot CDF and histogram of total tokens invalidated for ANNS streaming queries.
Creates consolidated plots showing all runs on a single figure.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from .aggregate import extract_replay_rate, extract_delay_multiplier, extract_scheduler
except ImportError:
    from aggregate import extract_replay_rate, extract_delay_multiplier, extract_scheduler

# Set global font sizes for better readability
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def collect_invalidation_data(
        log_dir: Path) -> Dict[Tuple[str, float, float], Dict]:
    """
    Collect token invalidation data from all runs.
    Groups by (scheduler, QPS, delay_multiplier) tuple.

    Returns:
        Dict mapping (scheduler, QPS, delay) -> {'data': List[int], 'run_dirs': List[str]}
    """
    data = defaultdict(lambda: {'data': [], 'run_dirs': []})

    # Find all collected_outputs_streaming.json files recursively
    json_files = list(log_dir.rglob("collected_outputs_streaming.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No collected_outputs_streaming.json files found in {log_dir}")

    print(f"Found {len(json_files)} streaming output files\n")

    for json_file in json_files:
        try:
            # Extract scheduler, QPS and delay multiplier from run directory
            run_dir = json_file.parent
            scheduler = extract_scheduler(str(run_dir))
            qps_val = extract_replay_rate(str(run_dir))
            delay_mult = extract_delay_multiplier(str(run_dir))
            key = (scheduler, qps_val, delay_mult)

            # Load invalidation counts from JSON
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            invalidation_counts = []
            for item in json_data:
                num_invalidated = item.get('num_tokens_invalidated')
                if num_invalidated is not None:
                    invalidation_counts.append(num_invalidated)

            data[key]['data'].extend(invalidation_counts)

            # Track run directory name
            run_name = run_dir.name
            if run_name not in data[key]['run_dirs']:
                data[key]['run_dirs'].append(run_name)

            # Print statistics for this file
            print(
                f"Loaded: {run_name} ({scheduler}, QPS: {qps_val:.3f}, Delay: {delay_mult}x)"
            )
            if invalidation_counts:
                invalidation_array = np.array(invalidation_counts)
                print(f"  Queries: {len(invalidation_counts)}, "
                      f"Mean: {invalidation_array.mean():.2f}, "
                      f"Median: {np.median(invalidation_array):.2f}")

        except Exception as e:
            print(f"[warn] Error processing {json_file}: {e}")

    print()
    return dict(data)


def plot_separate_cdfs(data: Dict[Tuple[str, float, float], Dict],
                       output_dir: Path):
    """Plot CDFs grouped by QPS, one subplot per scheduler with default_vllm baseline."""

    if not data:
        print("No invalidation data to plot")
        return

    # Group data by (QPS, delay_mult)
    qps_grouped = defaultdict(dict)
    for (scheduler, qps, delay_mult), inv_data in data.items():
        qps_key = (qps, delay_mult)
        qps_grouped[qps_key][scheduler] = inv_data

    # Create one figure for each QPS/delay combination
    output_dir.mkdir(parents=True, exist_ok=True)

    for (qps, delay_mult), scheduler_data in sorted(qps_grouped.items()):
        # Extract default_vllm baseline if available
        baseline_data = None
        if 'default_vllm' in scheduler_data:
            baseline_data = scheduler_data['default_vllm']['data']

        schedulers = sorted(scheduler_data.keys())
        n_schedulers = len(schedulers)

        # Determine grid layout
        n_cols = min(2, n_schedulers)
        n_rows = int(np.ceil(n_schedulers / n_cols))

        # Create figure with subplots
        fig, axes = plt.subplots(n_rows,
                                 n_cols,
                                 figsize=(10 * n_cols, 8 * n_rows),
                                 squeeze=False)
        axes = axes.flatten()

        for idx, scheduler in enumerate(schedulers):
            inv_data = scheduler_data[scheduler]
            invalidation_data = inv_data['data']
            run_dirs = inv_data['run_dirs']

            if not invalidation_data:
                continue

            ax = axes[idx]

            # Plot baseline first (if not default_vllm itself)
            if scheduler != 'default_vllm' and baseline_data and len(
                    baseline_data) > 0:
                sorted_baseline = np.sort(baseline_data)
                cdf_baseline = np.arange(
                    1,
                    len(sorted_baseline) + 1) / len(sorted_baseline)
                ax.plot(sorted_baseline,
                        cdf_baseline,
                        linewidth=2,
                        color='red',
                        linestyle='-',
                        label='default_vllm (baseline)',
                        alpha=0.7)

            # Compute and plot this scheduler's CDF
            sorted_data = np.sort(invalidation_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            # Color based on scheduler type
            if scheduler == 'default_vllm':
                color = 'red'
                label = scheduler
            else:
                color = 'blue'
                label = scheduler

            ax.plot(sorted_data, cdf, linewidth=2, color=color, label=label)

            ax.set_xlabel('Number of Tokens Invalidated', fontsize=12)
            ax.set_ylabel('CDF', fontsize=12)
            ax.set_title(f'{scheduler}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=9)

            # Add run directory info
            run_info = ', '.join(run_dirs) if len(run_dirs) <= 2 \
                else f"{run_dirs[0]} (+{len(run_dirs)-1} more)"
            ax.text(0.5,
                    0.98,
                    f'Run: {run_info}',
                    transform=ax.transAxes,
                    fontsize=9,
                    ha='center',
                    va='top',
                    style='italic',
                    color='gray')

            # Add statistics
            mean_val = np.mean(invalidation_data)
            median_val = np.median(invalidation_data)
            p95_val = np.percentile(invalidation_data, 95)

            stats_text = (f'N={len(invalidation_data)}\n'
                          f'Mean={mean_val:.1f}\n'
                          f'Median={median_val:.1f}\n'
                          f'P95={p95_val:.1f}')
            ax.text(0.98,
                    0.02,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    ha='right',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='wheat',
                              alpha=0.9,
                              edgecolor='gray'))

        # Hide unused subplots
        for idx in range(n_schedulers, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(
            f'CDF of Tokens Invalidated - QPS: {qps:.3f}, Delay: {delay_mult}x',
            fontsize=18,
            fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot with QPS in filename
        output_file = output_dir / f'tokens_invalidated_cdf_qps_{qps:.3f}_delay_{delay_mult}x.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved CDF plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description=
        'Plot consolidated CDF and histogram of tokens invalidated from ANNS driver logs.'
    )
    parser.add_argument(
        '--log_dir',
        type=Path,
        required=True,
        help='Path to test_run_log directory (e.g., driver/anns/test_run_log)')
    parser.add_argument('--output_dir',
                        type=Path,
                        required=True,
                        help='Directory to save plots')

    args = parser.parse_args()

    if not args.log_dir.exists():
        raise ValueError(f"Log directory does not exist: {args.log_dir}")

    # Collect data
    print(f"Loading invalidation data from: {args.log_dir}\n")
    data = collect_invalidation_data(args.log_dir)

    if not data:
        print("No invalidation data found!")
        return

    print(f"Found data for {len(data)} QPS/delay configurations")

    # Create consolidated plots
    print("\nGenerating consolidated plots...")
    plot_separate_cdfs(data, args.output_dir)

    print(f"\nPlots saved to {args.output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
