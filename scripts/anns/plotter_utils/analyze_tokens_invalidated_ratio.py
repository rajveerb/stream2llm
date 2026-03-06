#!/usr/bin/env python3
"""Analyze tokens invalidated ratios based on total tokens (not median).

This script calculates efficiency ratios for token invalidation across schedulers,
where the ratio is defined as:
    Ratio = (Scheduler Total Tokens Invalidated) / (Default vLLM Total Tokens Invalidated)

Lower ratio indicates better cache efficiency (fewer total tokens invalidated).

Usage:
    python driver/anns/plotter_utils/analyze_tokens_invalidated_ratio.py \
        --log-dir driver/anns/run_log/H200_enhanced_schedulers_v1_full
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import yaml


def get_qps(run_dir: Path) -> Optional[float]:
    """Extract QPS from config file."""
    config_files = list(run_dir.glob("config_*.yaml"))
    if not config_files:
        return None
    try:
        cfg = yaml.safe_load(config_files[0].read_text())
        return 1.0 / float(cfg["replay"]["poisson_avg_arrival_time"])
    except Exception:
        return None


def extract_tokens_invalidated(json_file: Path) -> Tuple[int, int]:
    """Extract total tokens invalidated and query count from json file.

    Returns:
        Tuple of (total_tokens_invalidated, query_count)
    """
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        total_invalidated = 0
        count = 0
        for item in json_data:
            num_invalidated = item.get('num_tokens_invalidated')
            if num_invalidated is not None:
                total_invalidated += int(num_invalidated)
                count += 1
        return total_invalidated, count
    except Exception as e:
        print(f"[warn] Error reading {json_file}: {e}")
        return 0, 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze token invalidation efficiency ratios by total tokens")
    parser.add_argument("--log-dir",
                       required=True,
                       type=Path,
                       help="Directory containing run logs")
    parser.add_argument("--output-file",
                       default=None,
                       type=Path,
                       help="Output file for results (default: stdout)")
    args = parser.parse_args(argv)

    # Aggregate data by scheduler and QPS
    data = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'count': 0}))

    # Collect data from all run directories
    json_files = list(args.log_dir.rglob("collected_outputs_streaming.json"))
    if not json_files:
        raise SystemExit(f"No collected_outputs_streaming.json files found in {args.log_dir}")

    for json_file in json_files:
        run_dir = json_file.parent
        scheduler = run_dir.parent.name
        qps = get_qps(run_dir)

        # Skip invalid schedulers or missing QPS
        if scheduler == "oeda_pbas" or qps is None:
            continue

        total, count = extract_tokens_invalidated(json_file)
        data[scheduler][qps]['total'] += total
        data[scheduler][qps]['count'] += count

    # Generate report
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("Total Tokens Invalidated Efficiency Ratios (normalized to default_vllm)")
    output_lines.append("=" * 100)
    output_lines.append("Ratio = (Scheduler Total Tokens Invalidated) / (Default vLLM Total Tokens Invalidated)")
    output_lines.append("Lower ratio = better efficiency (fewer total tokens invalidated)")
    output_lines.append("=" * 100)

    for qps in sorted(set(q for sched in data.values() for q in sched)):
        output_lines.append(f"\nQPS: {qps:.2f}")
        output_lines.append("-" * 100)

        if 'default_vllm' not in data or qps not in data['default_vllm']:
            output_lines.append("  No default_vllm data")
            continue

        baseline_total = data['default_vllm'][qps]['total']
        baseline_count = data['default_vllm'][qps]['count']

        output_lines.append(
            f"  Default vLLM: Total={baseline_total:,} tokens, Count={baseline_count} queries, "
            f"Avg={baseline_total/baseline_count:.1f} tokens/query")

        for scheduler in sorted(data.keys()):
            if scheduler == 'default_vllm' or qps not in data[scheduler]:
                continue

            sched_total = data[scheduler][qps]['total']
            sched_count = data[scheduler][qps]['count']
            ratio = sched_total / baseline_total if baseline_total > 0 else 1.0
            improvement_pct = (ratio - 1) * 100

            output_lines.append(
                f"  {scheduler:15s}: Total={sched_total:,} tokens, Count={sched_count} queries, "
                f"Avg={sched_total/sched_count:.1f} tokens/query, Ratio={ratio:.4f}x ({improvement_pct:+.1f}%)")

    output_lines.append("\n" + "=" * 100)

    # Output results
    output_text = "\n".join(output_lines)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            f.write(output_text)
        print(f"[saved] {args.output_file}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
