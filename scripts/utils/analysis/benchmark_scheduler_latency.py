#!/usr/bin/env python3
"""benchmark_scheduler_latency.py
Benchmark the computational overhead (wall-clock time) of each scheduling
policy's sorting + budget-allocation logic using realistic request populations
derived from existing run log data.

Outputs a table to tables/scheduler_sorting_latency_{dataset_name}.txt
"""
from __future__ import annotations

import argparse
import gc
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# MockRequest: lightweight stand-in for vllm.v1.request.Request
# ---------------------------------------------------------------------------
class MockRequest:
    __slots__ = (
        "arrival_time",
        "last_chunk_arrival_time",
        "num_computed_tokens",
        "num_tokens_with_spec",
        "_is_streaming_prompt",
        "_is_streaming_prompt_finished",
    )

    def __init__(
        self,
        arrival_time: float,
        last_chunk_arrival_time: float,
        num_computed_tokens: int,
        num_tokens_with_spec: int,
        is_streaming_prompt: bool,
        is_streaming_prompt_finished: bool,
    ):
        self.arrival_time = arrival_time
        self.last_chunk_arrival_time = last_chunk_arrival_time
        self.num_computed_tokens = num_computed_tokens
        self.num_tokens_with_spec = num_tokens_with_spec
        self._is_streaming_prompt = is_streaming_prompt
        self._is_streaming_prompt_finished = is_streaming_prompt_finished

    def check_is_streaming_prompt(self) -> bool:
        return self._is_streaming_prompt

    def check_is_streaming_prompt_finished(self) -> bool:
        return self._is_streaming_prompt_finished

    def is_finished(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Sorting functions (extracted from scheduler.py)
# ---------------------------------------------------------------------------
def sort_default_vllm(waiting: List[MockRequest],
                      running: List[MockRequest]) -> List[MockRequest]:
    """Default vLLM: no custom sort, just concatenation."""
    return list(waiting) + running


def sort_fcfs(waiting: List[MockRequest],
              running: List[MockRequest]) -> List[MockRequest]:
    """FCFS: full requests by arrival_time, then partial by arrival_time."""
    full_requests = []
    partial_requests = []
    unfinished_reqs = [
        req for req in (list(waiting) + running) if not req.is_finished()
    ]
    for req in unfinished_reqs:
        if req.check_is_streaming_prompt():
            if req.check_is_streaming_prompt_finished():
                full_requests.append(req)
            else:
                partial_requests.append(req)
        else:
            full_requests.append(req)
    full_requests.sort(key=lambda req: req.arrival_time)
    tokens_to_compute_cond = (
        lambda req: req.num_tokens_with_spec - req.num_computed_tokens > 0)
    partial_requests.sort(key=lambda req: req.arrival_time
                          if tokens_to_compute_cond(req) else float('inf'))
    return full_requests + partial_requests


def sort_lcas(waiting: List[MockRequest],
              running: List[MockRequest]) -> List[MockRequest]:
    """LCAS: full requests by last_chunk_arrival_time, then partial."""
    full_requests = []
    partial_requests = []
    unfinished_reqs = [
        req for req in (list(waiting) + running) if not req.is_finished()
    ]
    for req in unfinished_reqs:
        if req.check_is_streaming_prompt():
            if req.check_is_streaming_prompt_finished():
                full_requests.append(req)
            else:
                partial_requests.append(req)
        else:
            full_requests.append(req)
    full_requests.sort(key=lambda req: req.last_chunk_arrival_time)
    tokens_to_compute_cond = (
        lambda req: req.num_tokens_with_spec - req.num_computed_tokens > 0)
    partial_requests.sort(key=lambda req: req.last_chunk_arrival_time
                          if tokens_to_compute_cond(req) else float('inf'))
    return full_requests + partial_requests


def sort_mcps(waiting: List[MockRequest],
              running: List[MockRequest]) -> List[MockRequest]:
    """MCPS: all requests sorted by (-num_computed_tokens, arrival_time)."""
    all_requests = [
        req for req in (list(waiting) + running) if not req.is_finished()
    ]

    def get_sort_key(req):
        if req.num_tokens_with_spec - req.num_computed_tokens > 0:
            return (-req.num_computed_tokens, req.arrival_time)
        else:
            return (float('inf'), float('inf'))

    all_requests.sort(key=get_sort_key)
    return all_requests


# ---------------------------------------------------------------------------
# Budget allocation loop (extracted from scheduler.py lines 423-459)
# ---------------------------------------------------------------------------
def budget_allocation(ordered_reqs: List[MockRequest], token_budget: int,
                      total_gpu_blocks: int,
                      block_size: int) -> Tuple[List[MockRequest], int]:
    """Simulate the budget allocation loop. Returns (scheduled, not_scheduled count)."""
    available_token_budget = token_budget
    available_gpu_blocks = total_gpu_blocks
    num_preallocate_blocks = 1
    scheduled = []
    not_scheduled = 0
    for req in ordered_reqs:
        if available_token_budget <= 0:
            not_scheduled += 1
            continue
        max_req_new_tokens = req.num_tokens_with_spec - req.num_computed_tokens
        if max_req_new_tokens == 0:
            not_scheduled += 1
            continue
        new_tokens = min(max_req_new_tokens, available_token_budget)
        gpu_blocks_needed = (math.ceil(
            (new_tokens + req.num_computed_tokens) / block_size) +
                             num_preallocate_blocks)
        if available_gpu_blocks < gpu_blocks_needed or new_tokens == 0:
            not_scheduled += 1
            continue
        scheduled.append(req)
        available_token_budget -= new_tokens
        available_gpu_blocks -= gpu_blocks_needed
    return scheduled, not_scheduled


# ---------------------------------------------------------------------------
# Generate realistic mock request populations from run logs
# ---------------------------------------------------------------------------
def extract_distributions(
        log_dir: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract concurrency, request_size distributions and streaming ratio."""
    concurrencies = []
    request_sizes = []
    stream_true = 0
    stream_total = 0

    for csv_file in log_dir.rglob("run_metrics.csv"):
        try:
            df = pd.read_csv(csv_file, low_memory=False)
        except Exception:
            continue
        if "concurrent_requests" in df.columns:
            cr = df["concurrent_requests"].dropna()
            concurrencies.extend(cr.tolist())
        if "request_size" in df.columns:
            rs = df["request_size"].dropna()
            request_sizes.extend(rs.tolist())
        if "stream" in df.columns:
            s = df["stream"].dropna()
            stream_true += (s == True).sum() + (s == "True").sum()  # noqa: E712
            stream_total += len(s)

    concurrencies = np.array(concurrencies) if concurrencies else np.array(
        [5.0])
    request_sizes = np.array(request_sizes) if request_sizes else np.array(
        [10000.0])
    streaming_ratio = (stream_true /
                       stream_total) if stream_total > 0 else 0.5

    return concurrencies, request_sizes, streaming_ratio


def generate_mock_requests(
    n: int,
    request_sizes: np.ndarray,
    streaming_ratio: float,
    rng: random.Random,
) -> Tuple[List[MockRequest], List[MockRequest]]:
    """Generate n mock requests split into waiting/running lists."""
    base_time = 1_000_000_000.0
    requests = []
    for i in range(n):
        size = int(rng.choice(request_sizes))
        is_streaming = rng.random() < streaming_ratio
        # ~60% of requests have some computed tokens (running), rest are waiting
        has_computed = rng.random() < 0.6
        computed = int(size * rng.uniform(0.1, 0.9)) if has_computed else 0
        arrival = base_time + rng.uniform(0, 100)
        last_chunk = arrival + rng.uniform(0, 50) if is_streaming else arrival
        finished = rng.random() < 0.7 if is_streaming else False
        requests.append(
            MockRequest(
                arrival_time=arrival,
                last_chunk_arrival_time=last_chunk,
                num_computed_tokens=computed,
                num_tokens_with_spec=size,
                is_streaming_prompt=is_streaming,
                is_streaming_prompt_finished=finished,
            ))
    # Split into waiting and running
    split = max(1, n // 3)
    waiting = requests[:split]
    running = requests[split:]
    return waiting, running


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
ALGORITHMS = {
    "default_vllm": sort_default_vllm,
    "fcfs": sort_fcfs,
    "lcas": sort_lcas,
    "mcps": sort_mcps,
}

QUEUE_SIZES = [10, 25, 50, 100, 200, 500]

WARMUP_ITERS = 100
BENCH_ITERS = 1000


def benchmark_one(
    algo_fn,
    waiting: List[MockRequest],
    running: List[MockRequest],
    token_budget: int,
    total_gpu_blocks: int,
    block_size: int,
) -> np.ndarray:
    """Run benchmark and return array of latencies in nanoseconds."""
    # Warmup
    for _ in range(WARMUP_ITERS):
        ordered = algo_fn(waiting, running)
        budget_allocation(ordered, token_budget, total_gpu_blocks, block_size)

    # Timed runs
    latencies = np.empty(BENCH_ITERS, dtype=np.int64)
    gc.disable()
    try:
        for i in range(BENCH_ITERS):
            start = time.perf_counter_ns()
            ordered = algo_fn(waiting, running)
            budget_allocation(ordered, token_budget, total_gpu_blocks,
                              block_size)
            latencies[i] = time.perf_counter_ns() - start
    finally:
        gc.enable()
    return latencies


def run_benchmarks(
    request_sizes: np.ndarray,
    streaming_ratio: float,
) -> List[Dict]:
    """Run all benchmarks and return list of result dicts."""
    rng = random.Random(42)
    token_budget = 8192
    total_gpu_blocks = 30000
    block_size = 16

    results = []
    for queue_size in QUEUE_SIZES:
        waiting, running = generate_mock_requests(queue_size, request_sizes,
                                                  streaming_ratio, rng)
        for algo_name, algo_fn in ALGORITHMS.items():
            latencies = benchmark_one(algo_fn, waiting, running, token_budget,
                                      total_gpu_blocks, block_size)
            latencies_us = latencies / 1000.0  # ns -> us
            results.append({
                "scheduler": algo_name,
                "queue_size": queue_size,
                "mean_us": np.mean(latencies_us),
                "p50_us": np.percentile(latencies_us, 50),
                "p95_us": np.percentile(latencies_us, 95),
                "p99_us": np.percentile(latencies_us, 99),
            })
            print(
                f"  {algo_name:15s} | n={queue_size:4d} | "
                f"mean={results[-1]['mean_us']:8.2f} us | "
                f"p50={results[-1]['p50_us']:8.2f} us | "
                f"p95={results[-1]['p95_us']:8.2f} us | "
                f"p99={results[-1]['p99_us']:8.2f} us"
            )
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_table(results: List[Dict], output_dir: Path,
                dataset_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"scheduler_sorting_latency_{dataset_name}.txt"

    lines = []
    header = (f"{'Scheduler':15s} | {'Queue Size':>10s} | "
              f"{'Mean (us)':>10s} | {'P50 (us)':>10s} | "
              f"{'P95 (us)':>10s} | {'P99 (us)':>10s}")
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)
    for r in results:
        lines.append(
            f"{r['scheduler']:15s} | {r['queue_size']:10d} | "
            f"{r['mean_us']:10.2f} | {r['p50_us']:10.2f} | "
            f"{r['p95_us']:10.2f} | {r['p99_us']:10.2f}")
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark scheduler sorting + budget allocation latency")
    parser.add_argument("--log-dir",
                        type=Path,
                        required=True,
                        help="Run log directory to extract distributions from")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("tables"),
                        help="Output directory for result tables")
    parser.add_argument("--dataset-name",
                        type=str,
                        required=True,
                        help="Dataset name for output file naming")
    args = parser.parse_args()

    print(f"Extracting distributions from {args.log_dir} ...")
    concurrencies, request_sizes, streaming_ratio = extract_distributions(
        args.log_dir)
    print(f"  Concurrency: mean={concurrencies.mean():.1f}, "
          f"max={concurrencies.max():.0f}")
    print(f"  Request sizes: mean={request_sizes.mean():.0f}, "
          f"p95={np.percentile(request_sizes, 95):.0f}")
    print(f"  Streaming ratio: {streaming_ratio:.2%}")
    print()

    print("Running benchmarks ...")
    results = run_benchmarks(request_sizes, streaming_ratio)
    print()

    out_path = write_table(results, args.output_dir, args.dataset_name)
    print(f"Table written to {out_path}")


if __name__ == "__main__":
    main()
