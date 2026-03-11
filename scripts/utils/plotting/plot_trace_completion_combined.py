#!/usr/bin/env python3
"""plot_trace_completion_combined.py
Generate a combined 1x2 figure with Crawler (left) and ANNS (right) trace
completion time vs QPS plots. Single shared y-axis label, workload name
as subplot title.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
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
    "mcps": "#9467bd",
    "recomp": "#7f7f7f",
    "swap": "#bcbd22",
    "recomp_and_swap": "#17becf",
    "lcas": "#d62728",
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


def _extract_trace_completion_time(csv_file: Path) -> Tuple[float, float] | None:
    df = pd.read_csv(csv_file, low_memory=False)
    replay_end_df = df[df['event_type'] == 'replay_end']
    if len(replay_end_df) == 0:
        return None

    streaming_times = []
    non_streaming_times = []
    for _, row in replay_end_df.iterrows():
        time = row['duration_secs']
        if pd.isna(time):
            continue
        if row['stream']:
            streaming_times.append(time)
        else:
            non_streaming_times.append(time)

    streaming_time = np.mean(streaming_times) if streaming_times else None
    non_streaming_time = np.mean(non_streaming_times) if non_streaming_times else None
    return (streaming_time, non_streaming_time)


def _dataset(csv_files: Sequence[Path], min_qps: float = 0, max_qps: float = float('inf')):
    data: MutableMapping[str, MutableMapping[float, Dict[str, float]]] = (
        defaultdict(lambda: defaultdict(lambda: {
            "streaming": None, "non_streaming": None,
        })))
    for f in csv_files:
        try:
            qps_val = _qps(f.parent)
            if not (min_qps <= qps_val <= max_qps):
                continue
            sched = _sched(f.parent)
            result = _extract_trace_completion_time(f)
            if result:
                streaming_time, non_streaming_time = result
                if streaming_time:
                    data[sched][qps_val]["streaming"] = streaming_time
                if non_streaming_time:
                    data[sched][qps_val]["non_streaming"] = non_streaming_time
        except Exception as err:
            print("[warn]", err)
    return data


def _get_display_name(sched: str) -> str:
    display_names = {
        "default_vllm": "Default vLLM",
        "fcfs": "FCFS",
        "lcas": "LCAS",
        "mcps": "MCPS",
    }
    return display_names.get(sched, sched.replace('_', ' ').title())


def _plot_subplot(ax, data, workload_name, show_ylabel):
    all_qps = sorted({q for sched in data.values() for q in sched})
    if not all_qps:
        print(f"[error] No QPS data found for {workload_name}")
        return

    scheduler_data = {}
    for sched in sorted(data.keys()):
        qps_values = []
        trace_times_streaming = []
        trace_times_non_streaming = []

        for qps in all_qps:
            if qps not in data[sched]:
                continue
            streaming_time = data[sched][qps]["streaming"]
            non_streaming_time = data[sched][qps]["non_streaming"]
            if streaming_time is None:
                continue
            qps_values.append(qps)
            trace_times_streaming.append(streaming_time)
            trace_times_non_streaming.append(non_streaming_time)

        scheduler_data[sched] = {
            "qps": qps_values,
            "trace_times_streaming": trace_times_streaming,
            "trace_times_non_streaming": trace_times_non_streaming,
        }

    for sched in sorted(scheduler_data.keys()):
        display_name = _get_display_name(sched)
        color = SCHEDULER_COLORS.get(sched, "#000000")
        sched_data = scheduler_data[sched]

        ax.plot(sched_data["qps"],
                sched_data["trace_times_streaming"],
                marker='o', linestyle='-', linewidth=4.5, markersize=14,
                markeredgewidth=2, markeredgecolor='white',
                label=f"{display_name} (Streaming)",
                color=color, alpha=0.85)

        non_streaming_data = sched_data["trace_times_non_streaming"]
        if any(v is not None for v in non_streaming_data):
            qps_ns = [q for q, v in zip(sched_data["qps"], non_streaming_data) if v is not None]
            trace_ns = [v for v in non_streaming_data if v is not None]
            if qps_ns:
                ax.plot(qps_ns, trace_ns,
                        marker='s', linestyle='--', linewidth=4, markersize=12,
                        markeredgewidth=1.5, markeredgecolor='white',
                        label=f"{display_name} (Non-Streaming)",
                        color=color, alpha=0.6)

    ax.set_xlabel("QPS", fontsize=26, fontweight='bold', labelpad=12)
    if show_ylabel:
        ax.set_ylabel("Trace Completion Time\n(seconds)", fontsize=26, fontweight='bold', labelpad=12)
    else:
        ax.set_ylabel("")
    ax.set_title(workload_name, fontsize=28, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=22, pad=8)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate combined trace completion time plot (Crawler + ANNS)")
    parser.add_argument("--crawler-log-dir", required=True, type=Path)
    parser.add_argument("--anns-log-dir", required=True, type=Path)
    parser.add_argument("--crawler-max-qps", type=float, default=4.0)
    parser.add_argument("--anns-max-qps", type=float, default=2.0)
    parser.add_argument("--output-dir", default=Path("figures"), type=Path)
    parser.add_argument("--output-filename", default="trace_completion_time_combined.png", type=str)
    args = parser.parse_args(argv)

    crawler_csvs = list(args.crawler_log_dir.rglob("run_metrics.csv"))
    anns_csvs = list(args.anns_log_dir.rglob("run_metrics.csv"))
    if not crawler_csvs:
        raise SystemExit("no crawler run_metrics.csv found")
    if not anns_csvs:
        raise SystemExit("no ANNS run_metrics.csv found")

    crawler_data = _dataset(crawler_csvs, max_qps=args.crawler_max_qps)
    anns_data = _dataset(anns_csvs, max_qps=args.anns_max_qps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    _plot_subplot(ax1, crawler_data, "Crawler", show_ylabel=True)
    _plot_subplot(ax2, anns_data, "ANNS", show_ylabel=False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
