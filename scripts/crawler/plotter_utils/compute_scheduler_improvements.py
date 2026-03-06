#!/usr/bin/env python3
"""compute_scheduler_improvements.py
Compute performance improvements of schedulers compared to Default vLLM baselines.

Generates tables showing percentage improvements at various percentiles
(p50, p95, p99, p99.9) for each scheduler compared to both:
- Default vLLM Streaming
- Default vLLM Non-Streaming
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

SCHEDULERS: Tuple[str,
                  ...] = ("recomp", "swap", "recomp_and_swap", "default_vllm",
                          "fcfs_lru", "lcas_lifo", "mcps_lce",
                          "stream_based_v1", "lcas_cplusp")

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


def _compute_percentiles(data: TTFTArray) -> Dict[str, float]:
    """Compute percentiles for latency data."""
    if len(data) == 0:
        return {"p50": np.nan, "p95": np.nan, "p99": np.nan, "p99.9": np.nan}
    return {
        "p50": np.percentile(data, 50),
        "p95": np.percentile(data, 95),
        "p99": np.percentile(data, 99),
        "p99.9": np.percentile(data, 99.9),
    }


def _compute_improvement(baseline: float, current: float) -> float:
    """Compute improvement percentage (negative is worse)."""
    if baseline == 0 or np.isnan(baseline) or np.isnan(current):
        return np.nan
    return ((baseline - current) / baseline) * 100


def _compute_ratio(baseline: float, current: float) -> float:
    """Compute speedup ratio (baseline / current). >1 is better."""
    if current == 0 or np.isnan(baseline) or np.isnan(current):
        return np.nan
    return baseline / current


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Compute scheduler performance improvements vs Default vLLM")
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-dir",
                        default=Path("."),
                        type=Path)
    parser.add_argument("--dataset-name",
                        type=str,
                        default="",
                        help="Name of dataset for output file")
    args = parser.parse_args(argv)

    csv_files = list(args.log_dir.rglob("run_metrics.csv"))
    if not csv_files:
        raise SystemExit("no run_metrics.csv found")

    data = _dataset(csv_files)
    if not data:
        raise SystemExit("no data discovered")

    # Get all QPS values
    all_qps = sorted({q for sched in data.values() for q in sched})

    # Prepare output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"scheduler_improvements{f'_{args.dataset_name}' if args.dataset_name else ''}.txt"

    with open(output_file, 'w') as f:
        for qps in all_qps:
            f.write(f"\n{'='*100}\n")
            f.write(f"QPS: {qps:.3f}\n")
            f.write(f"{'='*100}\n\n")

            # Get baseline data
            baseline_streaming = np.array(
                data["default_vllm"][qps]["streaming"]) if qps in data.get(
                    "default_vllm", {}) else np.array([])
            baseline_non_streaming = np.array(
                data["default_vllm"][qps]["non_streaming"]) if qps in data.get(
                    "default_vllm", {}) else np.array([])

            baseline_streaming_percentiles = _compute_percentiles(
                baseline_streaming)
            baseline_non_streaming_percentiles = _compute_percentiles(
                baseline_non_streaming)

            # Print baseline
            f.write("Baseline - Default vLLM (Streaming):\n")
            f.write(
                f"  P50: {baseline_streaming_percentiles['p50']:.4f}s, "
                f"P95: {baseline_streaming_percentiles['p95']:.4f}s, "
                f"P99: {baseline_streaming_percentiles['p99']:.4f}s, "
                f"P99.9: {baseline_streaming_percentiles['p99.9']:.4f}s\n\n"
            )

            f.write("Baseline - Default vLLM (Non-Streaming):\n")
            f.write(
                f"  P50: {baseline_non_streaming_percentiles['p50']:.4f}s, "
                f"P95: {baseline_non_streaming_percentiles['p95']:.4f}s, "
                f"P99: {baseline_non_streaming_percentiles['p99']:.4f}s, "
                f"P99.9: {baseline_non_streaming_percentiles['p99.9']:.4f}s\n\n"
            )

            # Speedup ratio vs streaming baseline
            f.write("Speedup Ratio vs Default vLLM Streaming (baseline / scheduler):\n")
            f.write(
                f"{'Scheduler':<20} {'P50':<12} {'P95':<12} {'P99':<12} {'P99.9':<12}\n"
            )
            f.write("-" * 68 + "\n")

            for sched in sorted(data.keys()):
                if sched == "default_vllm" or qps not in data[sched]:
                    continue

                display_name = _get_display_name(sched)

                # Consider streaming data for streaming vs streaming comparison
                sched_streaming = np.array(data[sched][qps]["streaming"])
                sched_streaming_percentiles = _compute_percentiles(
                    sched_streaming)

                ratios = {
                    "p50":
                    _compute_ratio(baseline_streaming_percentiles["p50"],
                                   sched_streaming_percentiles["p50"]),
                    "p95":
                    _compute_ratio(baseline_streaming_percentiles["p95"],
                                   sched_streaming_percentiles["p95"]),
                    "p99":
                    _compute_ratio(baseline_streaming_percentiles["p99"],
                                   sched_streaming_percentiles["p99"]),
                    "p99.9":
                    _compute_ratio(baseline_streaming_percentiles["p99.9"],
                                   sched_streaming_percentiles["p99.9"]),
                }

                f.write(
                    f"{display_name:<20} "
                    f"{ratios['p50']:>10.3f}x "
                    f"{ratios['p95']:>10.3f}x "
                    f"{ratios['p99']:>10.3f}x "
                    f"{ratios['p99.9']:>10.3f}x\n"
                )

            # Speedup ratio vs non-streaming baseline
            f.write("\n\nSpeedup Ratio vs Default vLLM Non-Streaming (baseline / scheduler):\n")
            f.write(
                f"{'Scheduler':<20} {'P50':<12} {'P95':<12} {'P99':<12} {'P99.9':<12}\n"
            )
            f.write("-" * 68 + "\n")

            for sched in sorted(data.keys()):
                if sched == "default_vllm" or qps not in data[sched]:
                    continue

                display_name = _get_display_name(sched)

                # Consider streaming data for comparison with non-streaming baseline
                sched_streaming = np.array(data[sched][qps]["streaming"])
                sched_streaming_percentiles = _compute_percentiles(
                    sched_streaming)

                ratios = {
                    "p50":
                    _compute_ratio(baseline_non_streaming_percentiles["p50"],
                                   sched_streaming_percentiles["p50"]),
                    "p95":
                    _compute_ratio(baseline_non_streaming_percentiles["p95"],
                                   sched_streaming_percentiles["p95"]),
                    "p99":
                    _compute_ratio(baseline_non_streaming_percentiles["p99"],
                                   sched_streaming_percentiles["p99"]),
                    "p99.9":
                    _compute_ratio(baseline_non_streaming_percentiles["p99.9"],
                                   sched_streaming_percentiles["p99.9"]),
                }

                f.write(
                    f"{display_name:<20} "
                    f"{ratios['p50']:>10.3f}x "
                    f"{ratios['p95']:>10.3f}x "
                    f"{ratios['p99']:>10.3f}x "
                    f"{ratios['p99.9']:>10.3f}x\n"
                )

    print(f"[saved] {output_file}")
    print(f"\nContent preview:\n")
    with open(output_file, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    main()
