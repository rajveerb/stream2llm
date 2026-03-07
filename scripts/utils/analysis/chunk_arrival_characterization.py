#!/usr/bin/env python3
"""Chunk arrival time characterization for ANNS and Crawler workloads.

Generates figures and a summary table of inter-chunk arrival patterns.
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})


def load_anns_inter_chunk_times(data_dir):
    """Load ANNS pipeline traces and compute inter-chunk arrival times (ms)."""
    files = glob.glob(os.path.join(data_dir, "*_pipeline_trace.csv"))
    per_query_arrivals = []
    chunk_counts = []
    query_durations = []

    for f in files:
        df = pd.read_csv(f)
        if len(df) < 2:
            chunk_counts.append(len(df))
            if len(df) == 1:
                query_durations.append((df["EndTime_us"].iloc[-1] - df["StartTime_us"].iloc[0]) / 1000.0)
            continue
        starts = df["StartTime_us"].values
        inter_chunk = np.diff(starts) / 1000.0  # us -> ms
        per_query_arrivals.append(inter_chunk)
        chunk_counts.append(len(df))
        query_durations.append((df["EndTime_us"].iloc[-1] - df["StartTime_us"].iloc[0]) / 1000.0)

    all_inter_chunk = np.concatenate(per_query_arrivals) if per_query_arrivals else np.array([])
    return all_inter_chunk, chunk_counts, query_durations


def load_crawler_inter_chunk_times(data_dir):
    """Load Crawler traces and compute inter-chunk arrival times (ms)."""
    files = glob.glob(os.path.join(data_dir, "query_*.csv"))
    per_query_arrivals = []
    chunk_counts = []
    query_durations = []

    for f in files:
        df = pd.read_csv(f)
        # Each row is a chunk event (tavily_search or page_scrape)
        # Use endTime as chunk arrival (when data becomes available)
        ends = df["endTime"].dropna().sort_values().values
        if len(ends) < 2:
            chunk_counts.append(len(ends))
            if len(ends) >= 1:
                query_durations.append((ends[-1] - df["startTime"].min()) * 1000.0)
            continue
        inter_chunk = np.diff(ends) * 1000.0  # s -> ms
        per_query_arrivals.append(inter_chunk)
        chunk_counts.append(len(ends))
        query_durations.append((ends[-1] - df["startTime"].min()) * 1000.0)

    all_inter_chunk = np.concatenate(per_query_arrivals) if per_query_arrivals else np.array([])
    return all_inter_chunk, chunk_counts, query_durations


def compute_stats(values, label):
    """Return a dict of summary statistics."""
    return {
        "Workload": label,
        "Count": len(values),
        "Mean (ms)": np.mean(values),
        "Median (ms)": np.median(values),
        "Std (ms)": np.std(values),
        "P5 (ms)": np.percentile(values, 5),
        "P25 (ms)": np.percentile(values, 25),
        "P75 (ms)": np.percentile(values, 75),
        "P95 (ms)": np.percentile(values, 95),
        "P99 (ms)": np.percentile(values, 99),
        "Min (ms)": np.min(values),
        "Max (ms)": np.max(values),
    }


def plot_cdf(ax, values, label, color):
    sorted_v = np.sort(values)
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax.plot(sorted_v, cdf, label=label, color=color, linewidth=2)


def main():
    parser = argparse.ArgumentParser(description="Chunk arrival time characterization")
    parser.add_argument("--anns-dir", default="data/anns/res",
                        help="Path to ANNS pipeline trace directory")
    parser.add_argument("--crawler-dir", default="data/crawl/traces/simpleQA_ALL",
                        help="Path to Crawler trace directory")
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures")
    parser.add_argument("--table-dir", default="tables", help="Output directory for table")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    print("Loading ANNS data...")
    anns_inter, anns_chunks, anns_durations = load_anns_inter_chunk_times(args.anns_dir)
    print(f"  {len(anns_chunks)} queries, {len(anns_inter)} inter-chunk intervals")

    print("Loading Crawler data...")
    crawl_inter, crawl_chunks, crawl_durations = load_crawler_inter_chunk_times(args.crawler_dir)
    print(f"  {len(crawl_chunks)} queries, {len(crawl_inter)} inter-chunk intervals")

    # --- Summary statistics table ---
    stats_rows = []
    if len(anns_inter) > 0:
        stats_rows.append(compute_stats(anns_inter, "ANNS"))
    if len(crawl_inter) > 0:
        stats_rows.append(compute_stats(crawl_inter, "Crawler"))

    # Add chunk count stats
    chunk_stats = []
    for name, chunks in [("ANNS", anns_chunks), ("Crawler", crawl_chunks)]:
        if chunks:
            arr = np.array(chunks)
            chunk_stats.append({
                "Workload": name,
                "Queries": len(arr),
                "Mean Chunks/Query": np.mean(arr),
                "Median Chunks/Query": np.median(arr),
                "Min Chunks/Query": np.min(arr),
                "Max Chunks/Query": np.max(arr),
            })

    # Add query duration stats
    dur_stats = []
    for name, durs in [("ANNS", anns_durations), ("Crawler", crawl_durations)]:
        if durs:
            arr = np.array(durs)
            dur_stats.append({
                "Workload": name,
                "Mean Duration (ms)": np.mean(arr),
                "Median Duration (ms)": np.median(arr),
                "P95 Duration (ms)": np.percentile(arr, 95),
                "Max Duration (ms)": np.max(arr),
            })

    table_path = os.path.join(args.table_dir, "chunk_arrival_characterization.txt")
    with open(table_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Chunk Arrival Time Characterization\n")
        f.write("=" * 80 + "\n\n")

        f.write("--- Inter-Chunk Arrival Time Statistics ---\n")
        if stats_rows:
            df_stats = pd.DataFrame(stats_rows).set_index("Workload")
            f.write(df_stats.to_string(float_format="%.2f") + "\n\n")

        f.write("--- Chunks Per Query Statistics ---\n")
        if chunk_stats:
            df_chunks = pd.DataFrame(chunk_stats).set_index("Workload")
            f.write(df_chunks.to_string(float_format="%.2f") + "\n\n")

        f.write("--- Query Duration Statistics ---\n")
        if dur_stats:
            df_dur = pd.DataFrame(dur_stats).set_index("Workload")
            f.write(df_dur.to_string(float_format="%.2f") + "\n\n")

    print(f"Table written to {table_path}")

    # --- Figure 1: CDF of inter-chunk arrival times ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"ANNS": "#1f77b4", "Crawler": "#d62728"}
    if len(anns_inter) > 0:
        plot_cdf(ax, anns_inter, "ANNS", colors["ANNS"])
    if len(crawl_inter) > 0:
        plot_cdf(ax, crawl_inter, "Crawler", colors["Crawler"])
    ax.set_xscale("log")
    ax.set_xlabel("Inter-Chunk Arrival Time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Inter-Chunk Arrival Times (from Traces)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    cdf_path = os.path.join(args.output_dir, "chunk_arrival_cdf.png")
    fig.savefig(cdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"CDF figure saved to {cdf_path}")

    # --- Figure 2: Histogram of inter-chunk arrival times (side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, data, color) in zip(axes, [
        ("ANNS", anns_inter, colors["ANNS"]),
        ("Crawler", crawl_inter, colors["Crawler"]),
    ]):
        if len(data) == 0:
            continue
        # Use log-spaced bins
        bins = np.logspace(np.log10(max(data.min(), 0.01)), np.log10(data.max()), 50)
        ax.hist(data, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Inter-Chunk Arrival Time (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}")
        ax.grid(True, alpha=0.3, axis="y")
        med = np.median(data)
        ax.axvline(med, color="black", linestyle="--", linewidth=1.5, label=f"Median={med:.1f} ms")
        ax.legend()
    fig.suptitle("Distribution of Inter-Chunk Arrival Times (from Traces)", y=1.02)
    fig.tight_layout()
    hist_path = os.path.join(args.output_dir, "chunk_arrival_histogram.png")
    fig.savefig(hist_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram figure saved to {hist_path}")

    # --- Figure 3: Chunks per query distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, chunks, color) in zip(axes, [
        ("ANNS", anns_chunks, colors["ANNS"]),
        ("Crawler", crawl_chunks, colors["Crawler"]),
    ]):
        if not chunks:
            continue
        arr = np.array(chunks)
        unique, counts = np.unique(arr, return_counts=True)
        ax.bar(unique, counts, color=color, alpha=0.75, edgecolor="white")
        ax.set_xlabel("Chunks per Query")
        ax.set_ylabel("Number of Queries")
        ax.set_title(f"{name}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.suptitle("Distribution of Chunks per Query (from Traces)", y=1.02)
    fig.tight_layout()
    cpq_path = os.path.join(args.output_dir, "chunks_per_query.png")
    fig.savefig(cpq_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Chunks per query figure saved to {cpq_path}")

    # Print table to stdout as well
    print("\n" + "=" * 80)
    with open(table_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
