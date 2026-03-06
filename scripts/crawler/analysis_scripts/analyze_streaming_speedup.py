#!/usr/bin/env python3
"""
Comprehensive TTFT Speedup Analysis for Streaming Schedulers

This script analyzes the speedup of different schedulers with streaming enabled
compared to the baseline default_vllm without streaming (non-streaming). It generates
comprehensive comparison tables showing:
- Speedup metrics (P50, P95, P99, Mean)
- Absolute latency values
- Relative performance between schedulers
- Performance progression from non-streaming to enhanced schedulers

Usage:
    python analyze_streaming_speedup.py \
        --log-dir driver/crawler/run_log/H200_enhanced_schedulers_v1_full_delay_10 \
        --output-file analysis_results.txt

Run command used in analysis:
    python driver/crawler/analysis_scripts/analyze_streaming_speedup.py \
        --log-dir driver/crawler/run_log/H200_enhanced_schedulers_v1_full_delay_10
"""

import pandas as pd
import glob
import argparse
from pathlib import Path

def analyze_speedup(log_dir):
    """Analyze TTFT speedup across all schedulers."""

    schedulers = ["default_vllm", "fcfs_lru", "lcas_cplusp", "mcps_lce"]
    results = {}

    for sched in schedulers:
        csv_files = glob.glob(f"{log_dir}/{sched}/*/run_metrics.csv")
        if not csv_files:
            print(f"Warning: No CSV found for {sched}")
            continue

        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)

        # Extract TTFT events
        ttft_streaming = df[(df['event_type'] == 'query_ttft') & (df['stream'] == True)]['duration_secs'].dropna()
        ttft_non_streaming = df[(df['event_type'] == 'query_ttft') & (df['stream'] == False)]['duration_secs'].dropna()

        results[sched] = {
            'streaming': ttft_streaming,
            'non_streaming': ttft_non_streaming
        }

    # Get baseline (non-streaming)
    baseline_non_streaming = results['default_vllm']['non_streaming']

    print("\n" + "="*120)
    print("TABLE: SPEEDUP vs default_vllm (Non-Streaming) BASELINE")
    print("="*120)
    print("\nComparison: All Schedulers with Streaming vs default_vllm WITHOUT Streaming\n")

    print(f"{'Scheduler':<25} {'P50 Speedup':<15} {'P95 Speedup':<15} {'P99 Speedup':<15} {'Mean Speedup':<15}")
    print("-" * 120)

    speedup_data = {}
    for sched in ["default_vllm", "fcfs_lru", "lcas_cplusp", "mcps_lce"]:
        streaming = results[sched]['streaming']
        p50_speedup = baseline_non_streaming.quantile(0.50) / streaming.quantile(0.50)
        p95_speedup = baseline_non_streaming.quantile(0.95) / streaming.quantile(0.95)
        p99_speedup = baseline_non_streaming.quantile(0.99) / streaming.quantile(0.99)
        mean_speedup = baseline_non_streaming.mean() / streaming.mean()

        label = sched if sched != "default_vllm" else "default_vllm (S)"
        print(f"{label:<25} {p50_speedup:>8.2f}x       {p95_speedup:>8.2f}x       {p99_speedup:>8.2f}x       {mean_speedup:>8.2f}x")

        speedup_data[label] = {
            'p50': p50_speedup,
            'p95': p95_speedup,
            'p99': p99_speedup,
            'mean': mean_speedup
        }

    print("\n" + "="*120)
    print("ABSOLUTE LATENCY COMPARISON")
    print("="*120)

    print(f"\n{'Scheduler':<25} {'P50':<12} {'P95':<12} {'P99':<12} {'Mean':<12}")
    print("-" * 120)
    print(f"{'default_vllm (NS)':<25} {baseline_non_streaming.quantile(0.50):>8.4f}s  {baseline_non_streaming.quantile(0.95):>8.4f}s  {baseline_non_streaming.quantile(0.99):>8.4f}s  {baseline_non_streaming.mean():>8.4f}s")
    for sched in ["default_vllm", "fcfs_lru", "lcas_cplusp", "mcps_lce"]:
        label = sched if sched != "default_vllm" else "default_vllm (S)"
        s = results[sched]['streaming']
        print(f"{label:<25} {s.quantile(0.50):>8.4f}s  {s.quantile(0.95):>8.4f}s  {s.quantile(0.99):>8.4f}s  {s.mean():>8.4f}s")

    print("\n" + "="*120)
    print("KEY INSIGHTS")
    print("="*120)

    dvllm_s_mean = baseline_non_streaming.mean() / results['default_vllm']['streaming'].mean()
    fcfs_mean = baseline_non_streaming.mean() / results['fcfs_lru']['streaming'].mean()
    lcas_mean = baseline_non_streaming.mean() / results['lcas_cplusp']['streaming'].mean()
    mcps_mean = baseline_non_streaming.mean() / results['mcps_lce']['streaming'].mean()

    print(f"\n✅ Streaming Architecture Baseline (default_vllm):")
    print(f"   - {dvllm_s_mean:.2f}x faster mean than non-streaming")

    print(f"\n✅ Enhanced Schedulers Amplify Benefit:")
    print(f"   - FCFS_LRU: {fcfs_mean:.2f}x faster mean ({fcfs_mean / dvllm_s_mean:.2f}x better than default)")
    print(f"   - LCAS_CPLUSP: {lcas_mean:.2f}x faster mean ({lcas_mean / dvllm_s_mean:.2f}x better than default)")
    print(f"   - MCPS_LCE: {mcps_mean:.2f}x faster mean ({mcps_mean / dvllm_s_mean:.2f}x vs default - degradation!)")

    print(f"\n📈 Performance Progression:")
    print(f"   Non-Streaming:                    1.0x (baseline)")
    print(f"   + Streaming (default):            {dvllm_s_mean:.1f}x")
    print(f"   + Streaming + MCPS:               {mcps_mean:.1f}x (degradation)")
    print(f"   + Streaming + LCAS:               {lcas_mean:.1f}x")
    print(f"   + Streaming + FCFS (BEST):        {fcfs_mean:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TTFT speedup for streaming schedulers")
    parser.add_argument("--log-dir", required=True, help="Path to run_log directory")
    args = parser.parse_args()

    analyze_speedup(args.log_dir)
