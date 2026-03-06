"""
Plotting utility to generate workload characteristics histograms for ANNS workload.
Creates 1x3 figure showing: Total Tokens per Query, Document Token Count, Query Duration
"""

import os
import json
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
from typing import Dict, List, Tuple


def load_corpus_content(corpus_parts_prefix: str) -> Dict[str, str]:
    """Load and merge corpus content from multiple part files"""
    print("Loading corpus content from part files...")
    corpus_content = {}
    part_num = 0

    while True:
        part_file = f"{corpus_parts_prefix}.{part_num}.json"
        if not os.path.exists(part_file):
            break

        print(f"  Loading {part_file}...")
        with open(part_file, 'r') as f:
            part_data = json.load(f)
            corpus_content.update(part_data)
        part_num += 1

    if part_num == 0:
        raise FileNotFoundError(
            f"No corpus part files found with prefix {corpus_parts_prefix}")

    print(f"Loaded {len(corpus_content)} documents from {part_num} part files")
    return corpus_content


def load_query_trace_map(map_file: str) -> Dict[str, Dict]:
    """Load the query to trace file mapping"""
    with open(map_file, 'r') as f:
        return json.load(f)


def parse_pipeline_pool(pool_str: str) -> List[str]:
    """Parse pipeline pool string to extract document IDs"""
    pool_str = pool_str.strip('()')
    if not pool_str:
        return []
    return [doc_id.strip() for doc_id in pool_str.split(',')]


def process_single_query(query_id: str,
                         query_info: Dict,
                         trace_dir: str,
                         corpus_content: Dict[str, str],
                         tokenizer_model: str = None) -> Dict:
    """Process a single query trace and return statistics"""
    result = {
        'query_id': query_id,
        'query_text': query_info['query'],
        'trace_file': query_info['trace_file'],
        'query_tokens': 0,
        'doc_token_counts': [],
        'total_query_tokens': 0,
        'total_doc_tokens': 0,
        'query_duration_secs': 0,
        'success': False,
        'error': None
    }

    # Initialize tokenizer if provided
    tokenizer = None
    if tokenizer_model:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model,
                                                      local_files_only=True)
        except Exception as e1:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            except Exception as e2:
                result['error'] = f"Failed to load tokenizer: {str(e2)}"
                return result

    try:
        # Read the trace CSV file
        trace_path = os.path.join(trace_dir, query_info['trace_file'])
        if not os.path.exists(trace_path):
            result['error'] = f"Trace file not found: {trace_path}"
            return result

        df = pd.read_csv(trace_path)

        if df.empty:
            result['error'] = "Empty trace file"
            return result

        # Extract timing information (microseconds to seconds)
        start_time_us = df['StartTime_us'].iloc[0]
        end_time_us = df['EndTime_us'].iloc[-1]
        result['query_duration_secs'] = (end_time_us - start_time_us) / 1e6

        # Get the final pipeline pool
        final_row = df.iloc[-1]
        pipeline_pool_str = str(final_row['PipelinePool'])
        doc_ids = parse_pipeline_pool(pipeline_pool_str)

        if not tokenizer:
            result['error'] = "No tokenizer provided"
            return result

        # Tokenize the query
        try:
            query_tokens = tokenizer.encode(query_info['query'],
                                            truncation=False,
                                            add_special_tokens=True)
            result['query_tokens'] = len(query_tokens)
        except Exception as e:
            result['error'] = f"Failed to tokenize query: {str(e)}"
            return result

        # Tokenize each document
        total_doc_tokens = 0
        for doc_id in doc_ids:
            if doc_id not in corpus_content:
                continue

            doc_text = corpus_content[doc_id]

            try:
                doc_tokens = tokenizer.encode(doc_text,
                                              truncation=False,
                                              add_special_tokens=True)
                token_count = len(doc_tokens)
                result['doc_token_counts'].append(token_count)
                total_doc_tokens += token_count
            except Exception as e:
                continue

        result['total_doc_tokens'] = total_doc_tokens
        result['total_query_tokens'] = result['query_tokens'] + total_doc_tokens
        result['success'] = True

    except Exception as e:
        result['error'] = f"Processing error: {str(e)}"

    return result


def process_all_queries(query_trace_map: Dict,
                        trace_dir: str,
                        corpus_content: Dict[str, str],
                        tokenizer_model: str = None,
                        num_cores: int = None,
                        max_queries: int = None) -> List[Dict]:
    """Process all queries in parallel"""

    # Limit number of queries if specified
    query_items = list(query_trace_map.items())
    if max_queries:
        query_items = query_items[:max_queries]

    print(f"Processing {len(query_items)} queries...")

    # Determine number of cores
    if num_cores is None:
        num_cores = mp.cpu_count()
    num_cores = min(num_cores, len(query_items))

    print(f"Using {num_cores} cores for processing")

    # Create worker function
    worker_func = partial(process_single_query,
                          trace_dir=trace_dir,
                          corpus_content=corpus_content,
                          tokenizer_model=tokenizer_model)

    # Process queries
    if num_cores == 1:
        results = []
        for query_id, query_info in tqdm(query_items, desc="Processing queries"):
            results.append(worker_func(query_id, query_info))
    else:
        with mp.Pool(num_cores) as pool:
            results = list(
                tqdm(pool.starmap(worker_func, query_items),
                     total=len(query_items),
                     desc="Processing queries"))

    return results


def create_workload_characteristics_figure(stats: Dict, output_dir: str):
    """Create 1x2 histogram figure for workload characteristics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Color palette
    colors = ['#1f77b4', '#ff7f0e']

    # Plot 1: Total Tokens per Query
    if stats['total_query_tokens']:
        values = np.array(stats['total_query_tokens'])
        ax1.hist(values,
                 bins=50,
                 alpha=0.7,
                 edgecolor='black',
                 linewidth=0.5,
                 color=colors[0])
        ax1.set_xlabel('Total Tokens per Query', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Frequency (count)', fontsize=16, fontweight='bold')
        ax1.set_title('Total Tokens per Query',
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
        ax1.set_title('Total Tokens per Query', fontsize=14, fontweight='bold')

    # Plot 2: Query Duration
    if stats['query_durations']:
        values = np.array(stats['query_durations'])
        ax2.hist(values,
                 bins=50,
                 alpha=0.7,
                 edgecolor='black',
                 linewidth=0.5,
                 color=colors[1])
        ax2.set_xlabel('Duration (seconds)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Frequency (count)', fontsize=16, fontweight='bold')
        ax2.set_title('Query Duration',
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
        ax2.set_title('Query Duration', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'anns_workload_characteristics.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Workload characteristics figure saved to {output_path}")


def print_statistics(results: List[Dict]):
    """Print detailed statistics"""
    # Aggregate statistics
    stats = {
        'total_query_tokens': [],
        'doc_tokens': [],
        'query_durations': [],
        'success_count': 0,
        'error_count': 0
    }

    for result in results:
        if result['success']:
            stats['success_count'] += 1
            stats['total_query_tokens'].append(result['total_query_tokens'])
            stats['doc_tokens'].extend(result['doc_token_counts'])
            stats['query_durations'].append(result['query_duration_secs'])
        else:
            stats['error_count'] += 1

    print("\n" + "=" * 70)
    print("ANNS WORKLOAD CHARACTERISTICS - STATISTICS")
    print("=" * 70)

    print(f"\nProcessed: {stats['success_count']} queries successfully")
    if stats['error_count'] > 0:
        print(f"Errors: {stats['error_count']} queries failed")

    if stats['total_query_tokens']:
        values = np.array(stats['total_query_tokens'])
        print(f"\nTOTAL TOKENS PER QUERY (n={len(values)} queries)")
        print(f"  Mean: {values.mean():.0f} tokens")
        print(f"  P24: {np.percentile(values, 24):.0f} tokens")
        print(f"  P50: {np.percentile(values, 50):.0f} tokens")
        print(f"  P75: {np.percentile(values, 75):.0f} tokens")
        print(f"  P95: {np.percentile(values, 95):.0f} tokens")

    if stats['doc_tokens']:
        values = np.array(stats['doc_tokens'])
        print(f"\nDOCUMENT TOKEN COUNT (n={len(values)} documents)")
        print(f"  Mean: {values.mean():.0f} tokens")
        print(f"  P24: {np.percentile(values, 24):.0f} tokens")
        print(f"  P50: {np.percentile(values, 50):.0f} tokens")
        print(f"  P75: {np.percentile(values, 75):.0f} tokens")
        print(f"  P95: {np.percentile(values, 95):.0f} tokens")

    if stats['query_durations']:
        values = np.array(stats['query_durations'])
        print(f"\nQUERY DURATION (n={len(values)} queries)")
        print(f"  Mean: {values.mean():.3f} seconds")
        print(f"  P24: {np.percentile(values, 24):.3f} seconds")
        print(f"  P50: {np.percentile(values, 50):.3f} seconds")
        print(f"  P75: {np.percentile(values, 75):.3f} seconds")
        print(f"  P95: {np.percentile(values, 95):.3f} seconds")

    print("=" * 70 + "\n")

    return stats


def main():
    # Suppress warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    parser = argparse.ArgumentParser(
        description="Generate workload characteristics figure for ANNS workload")
    parser.add_argument("--trace-dir",
                        "-d",
                        required=True,
                        help="Directory containing trace CSV files")
    parser.add_argument("--query-map",
                        "-q",
                        required=True,
                        help="Path to query trace map JSON file")
    parser.add_argument("--corpus-prefix",
                        "-c",
                        required=True,
                        help="Prefix for corpus content part files")
    parser.add_argument("--output-dir",
                        "-o",
                        help="Output directory for figure (default: same as trace-dir)")
    parser.add_argument("--tokenizer-model",
                        "-t",
                        default="meta-llama/Llama-2-7b-hf",
                        help="HuggingFace tokenizer model name")
    parser.add_argument("--cores",
                        type=int,
                        default=None,
                        help="Number of CPU cores to use (default: all)")
    parser.add_argument("--max-queries",
                        type=int,
                        default=None,
                        help="Maximum number of queries to process")

    args = parser.parse_args()

    output_dir = args.output_dir or args.trace_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ANNS WORKLOAD CHARACTERISTICS ANALYZER")
    print("=" * 70)
    print(f"Trace directory: {args.trace_dir}")
    print(f"Query map: {args.query_map}")
    print(f"Corpus prefix: {args.corpus_prefix}")
    print(f"Output directory: {output_dir}")

    # Load corpus content
    corpus_content = load_corpus_content(args.corpus_prefix)

    # Load query trace map
    print("\nLoading query trace map...")
    query_trace_map = load_query_trace_map(args.query_map)
    print(f"Found {len(query_trace_map)} queries in map")

    # Process queries
    results = process_all_queries(query_trace_map, args.trace_dir, corpus_content,
                                  args.tokenizer_model, args.cores, args.max_queries)

    # Aggregate and print statistics
    print("\nAggregating statistics...")
    stats = print_statistics(results)

    # Generate figure
    print("Generating workload characteristics figure...")
    create_workload_characteristics_figure(stats, output_dir)

    print(f"Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
