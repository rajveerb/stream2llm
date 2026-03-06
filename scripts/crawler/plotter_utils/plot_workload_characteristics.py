"""
Plotting utility to generate workload characteristics histograms for crawler workload.
Creates 1x3 figure showing: Query Total Tokens, Page Scrape Duration, Total Collection Time
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import logging
import warnings
import multiprocessing as mp
from functools import partial


def process_single_csv_file(filepath, tokenizer_model=None):
    """Process a single CSV file and return workload statistics"""
    result = {
        'page_scrape_durations': [],
        'total_collection_time': 0.0,
        'query_total_tokens': 0,
        'filepath': filepath,
        'success': False
    }

    # Initialize tokenizer if provided
    tokenizer = None
    if tokenizer_model:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model,
                                                      local_files_only=True)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            except Exception:
                pass

    try:
        df = pd.read_csv(filepath)

        # Validate required columns
        required_cols = ["startTime", "endTime", "type"]
        if not all(col in df.columns for col in required_cols):
            return result

        if df.empty:
            return result

        # Extract page scrape durations
        page_scrapes = df[df["type"] == "page_scrape"]
        for _, row in page_scrapes.iterrows():
            duration = row["endTime"] - row["startTime"]
            result['page_scrape_durations'].append(duration)

        # Calculate total collection time (from start to end of all events)
        result['total_collection_time'] = df['endTime'].max()

        # Calculate total tokens if tokenizer is available and content exists
        if tokenizer and "content" in df.columns:
            total_tokens = 0
            for _, row in page_scrapes.iterrows():
                if pd.notna(row["content"]):
                    content = str(row["content"])
                    try:
                        tokens = tokenizer.encode(content,
                                                  truncation=False,
                                                  add_special_tokens=True)
                        total_tokens += len(tokens)
                    except Exception:
                        pass
            result['query_total_tokens'] = total_tokens

        result['success'] = True
        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def process_csv_files(directory, tokenizer_model=None, num_cores=None):
    """Process CSV files in parallel"""
    page_scrape_durations = []
    total_collection_times = []
    query_total_tokens = []

    csv_files = sorted(glob.glob(os.path.join(directory, "query_*.csv")))

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return {}, [], [], []

    print(f"Found {len(csv_files)} CSV files to process")

    # Determine number of cores
    if num_cores is None:
        num_cores = mp.cpu_count()
    num_cores = min(num_cores, len(csv_files))
    print(f"Using {num_cores} cores for processing")

    # Create worker function
    worker_func = partial(process_single_csv_file, tokenizer_model=tokenizer_model)

    # Process files in parallel
    if num_cores == 1:
        results = []
        for filepath in tqdm(csv_files, desc="Processing files"):
            results.append(worker_func(filepath))
    else:
        with mp.Pool(num_cores) as pool:
            results = list(
                tqdm(pool.imap(worker_func, csv_files),
                     total=len(csv_files),
                     desc="Processing files"))

    # Aggregate results
    processed_count = 0
    error_count = 0

    for result in results:
        if result['success']:
            processed_count += 1
            page_scrape_durations.extend(result['page_scrape_durations'])
            total_collection_times.append(result['total_collection_time'])
            if result['query_total_tokens'] > 0:
                query_total_tokens.append(result['query_total_tokens'])
        else:
            error_count += 1

    print(f"Successfully processed: {processed_count} queries")
    if error_count > 0:
        print(f"Errors/Skipped: {error_count} files")

    return {
        'page_scrape_durations': page_scrape_durations,
        'total_collection_times': total_collection_times,
        'query_total_tokens': query_total_tokens
    }


def create_workload_characteristics_figure(data, output_dir):
    """Create 1x2 histogram figure for workload characteristics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Color palette
    colors = ['#1f77b4', '#ff7f0e']

    # Plot 1: Query Total Tokens
    if data['query_total_tokens']:
        values = np.array(data['query_total_tokens'])
        ax1.hist(values,
                 bins=50,
                 alpha=0.7,
                 edgecolor='black',
                 linewidth=0.5,
                 color=colors[0])
        ax1.set_xlabel('Total Tokens per Query', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Frequency (count)', fontsize=16, fontweight='bold')
        ax1.set_title('Query Total Tokens',
                      fontsize=18,
                      fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=13)
    else:
        ax1.text(0.5,
                 0.5,
                 'No data available',
                 transform=ax1.transAxes,
                 ha='center',
                 va='center',
                 fontsize=12)
        ax1.set_title('Query Total Tokens', fontsize=14, fontweight='bold')

    # Plot 2: Total Collection Time
    if data['total_collection_times']:
        values = np.array(data['total_collection_times'])
        ax2.hist(values,
                 bins=50,
                 alpha=0.7,
                 edgecolor='black',
                 linewidth=0.5,
                 color=colors[1])
        ax2.set_xlabel('Total Collection Time (seconds)',
                       fontsize=16,
                       fontweight='bold')
        ax2.set_ylabel('Frequency (count)', fontsize=16, fontweight='bold')
        ax2.set_title('Total Collection Time per Query',
                      fontsize=18,
                      fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=13)
    else:
        ax2.text(0.5,
                 0.5,
                 'No data available',
                 transform=ax2.transAxes,
                 ha='center',
                 va='center',
                 fontsize=12)
        ax2.set_title('Total Collection Time per Query', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'crawler_workload_characteristics.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Workload characteristics figure saved to {output_path}")


def print_statistics(data):
    """Print detailed statistics"""
    print("\n" + "=" * 70)
    print("CRAWLER WORKLOAD CHARACTERISTICS - STATISTICS")
    print("=" * 70)

    if data['query_total_tokens']:
        values = np.array(data['query_total_tokens'])
        print(f"\nQUERY TOTAL TOKENS (n={len(values)} queries)")
        print(f"  Mean: {values.mean():.0f} tokens")
        print(f"  P24: {np.percentile(values, 24):.0f} tokens")
        print(f"  P50: {np.percentile(values, 50):.0f} tokens")
        print(f"  P75: {np.percentile(values, 75):.0f} tokens")
        print(f"  P95: {np.percentile(values, 95):.0f} tokens")

    if data['page_scrape_durations']:
        values = np.array(data['page_scrape_durations'])
        print(f"\nPAGE SCRAPE DURATION (n={len(values)} page scrapes)")
        print(f"  Mean: {values.mean():.3f} seconds")
        print(f"  P24: {np.percentile(values, 24):.3f} seconds")
        print(f"  P50: {np.percentile(values, 50):.3f} seconds")
        print(f"  P75: {np.percentile(values, 75):.3f} seconds")
        print(f"  P95: {np.percentile(values, 95):.3f} seconds")

    if data['total_collection_times']:
        values = np.array(data['total_collection_times'])
        print(f"\nTOTAL COLLECTION TIME PER QUERY (n={len(values)} queries)")
        print(f"  Mean: {values.mean():.3f} seconds")
        print(f"  P24: {np.percentile(values, 24):.3f} seconds")
        print(f"  P50: {np.percentile(values, 50):.3f} seconds")
        print(f"  P75: {np.percentile(values, 75):.3f} seconds")
        print(f"  P95: {np.percentile(values, 95):.3f} seconds")

    print("=" * 70 + "\n")


def main():
    # Suppress warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    parser = argparse.ArgumentParser(
        description="Generate workload characteristics figure for crawler workload")
    parser.add_argument("--input-dir",
                        "-i",
                        required=True,
                        help="Directory containing crawler trace CSV files")
    parser.add_argument("--output-dir",
                        "-o",
                        help="Output directory for figure (default: same as input)")
    parser.add_argument("--tokenizer-model",
                        "-t",
                        default="meta-llama/Llama-2-7b-hf",
                        help="HuggingFace tokenizer model name")
    parser.add_argument("--cores",
                        "-c",
                        type=int,
                        default=None,
                        help="Number of CPU cores to use (default: all)")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing CSV files from: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Process files
    data = process_csv_files(input_dir, args.tokenizer_model, args.cores)

    if not any(data.values()):
        print("No valid data found to analyze")
        return

    # Print statistics
    print_statistics(data)

    # Generate figure
    print("Generating workload characteristics figure...")
    create_workload_characteristics_figure(data, output_dir)

    print(f"Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
