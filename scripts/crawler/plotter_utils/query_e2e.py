import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
from .extract_metrics import create_metrics
import pandas as pd


def extract_query_e2e_latencies(
        metrics_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract query e2e completion times from the metrics_file file."""
    df = pd.read_csv(metrics_file, low_memory=False)

    # Filter for query_e2e_latency events (from replay_query_crawls.py)
    e2e_df = df[df['event_type'] == 'query_e2e_latency']

    # Group by streaming flag
    streaming_e2e_times = e2e_df[e2e_df['stream'] ==
                                 True]['duration_secs'].values
    non_streaming_e2e_times = e2e_df[e2e_df['stream'] ==
                                     False]['duration_secs'].values

    return streaming_e2e_times, non_streaming_e2e_times


def plot_query_e2e_times_histogram(streaming_e2e_times: np.ndarray,
                                   non_streaming_e2e_times: np.ndarray,
                                   output_path: str,
                                   title_suffix: str) -> None:
    """Plot histograms of query completion times."""
    # Create a figure with two subplots - one for the plot and one for metrics text
    _ = plt.figure(figsize=(15, 8))

    # Main plot in the top subplot (80% of height)
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)

    # Create bins
    # get max over streaming and non streaming e2e times
    max_values = []
    min_values = []

    if len(streaming_e2e_times) > 0:
        max_values.append(np.max(streaming_e2e_times))
        min_values.append(np.min(streaming_e2e_times))

    if len(non_streaming_e2e_times) > 0:
        max_values.append(np.max(non_streaming_e2e_times))
        min_values.append(np.min(non_streaming_e2e_times))

    # Use the available data for range
    max_time = max(max_values)
    min_time = min(min_values)
    bins = np.linspace(min_time, max_time, 30)

    if len(streaming_e2e_times) > 0:
        ax1.hist(streaming_e2e_times,
                 bins=bins,
                 alpha=0.5,
                 label='Streaming',
                 density=True)
    if len(non_streaming_e2e_times) > 0:
        ax1.hist(non_streaming_e2e_times,
                 bins=bins,
                 alpha=0.5,
                 label='Non-Streaming',
                 density=True)

    metrics_text, speedup_text = create_metrics(streaming_e2e_times,
                                                non_streaming_e2e_times)

    # Add labels and title
    ax1.set_xlabel('Query E2E Completion Time (seconds)')
    ax1.set_ylabel('Density')
    title = 'Distribution of Query E2E Completion Times'
    if title_suffix:
        title += f' {title_suffix}'
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Text area for metrics in the bottom subplot (20% of height)
    ax2 = plt.subplot2grid((5, 1), (4, 0))
    ax2.axis('off')  # Hide axes
    if speedup_text:
        ax2.text(0.5,
                 0.8,
                 speedup_text,
                 ha='center',
                 va='center',
                 fontsize=12,
                 fontweight='bold',
                 wrap=True,
                 bbox=dict(facecolor='white',
                           alpha=0.8,
                           boxstyle='round,pad=0.5'))
        ax2.text(0.5,
                 0.2,
                 metrics_text,
                 ha='center',
                 va='center',
                 fontsize=12,
                 wrap=True,
                 bbox=dict(facecolor='white',
                           alpha=0.8,
                           boxstyle='round,pad=0.5'))
    else:
        ax2.text(0.5,
                 0.5,
                 metrics_text,
                 ha='center',
                 va='center',
                 fontsize=12,
                 wrap=True,
                 bbox=dict(facecolor='white',
                           alpha=0.8,
                           boxstyle='round,pad=0.5'))

    # Create filename based on title suffix
    filename = 'query_e2e_times_histogram'
    if title_suffix:
        title_suffix_clean = title_suffix.replace(" ", "_").replace(
            ":", "").replace(".", "_").lower()
        filename += f'_{title_suffix_clean}'
    filename += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def plot_query_e2e_times_cdf(streaming_e2e_times,
                             non_streaming_e2e_times,
                             output_path,
                             title_suffix=""):
    """Plot CDF of query completion times."""
    # Create a figure with two subplots - one for the plot and one for metrics text
    _ = plt.figure(figsize=(15, 8))

    # Main plot in the top subplot (80% of height)
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)

    # Sort the data
    streaming_sorted = np.sort(streaming_e2e_times)
    non_streaming_sorted = np.sort(non_streaming_e2e_times)

    # Calculate the CDF
    streaming_cdf = np.arange(0, len(streaming_sorted)) / len(streaming_sorted)
    non_streaming_cdf = np.arange(
        0, len(non_streaming_sorted)) / len(non_streaming_sorted)

    # Plot the CDF
    if len(streaming_sorted) > 0:
        ax1.plot(streaming_sorted, streaming_cdf, label='Streaming')
    if len(non_streaming_sorted) > 0:
        ax1.plot(non_streaming_sorted,
                 non_streaming_cdf,
                 label='Non-Streaming')

    # Create metrics text
    metrics_text, speedup_text = create_metrics(streaming_e2e_times,
                                                non_streaming_e2e_times)

    ax1.set_xlabel('Query E2E Completion Time (seconds)')
    ax1.set_ylabel('CDF')
    title = 'CDF of Query E2E Completion Times'
    if title_suffix:
        title += f' {title_suffix}'
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Text area for metrics in the bottom subplot (20% of height)
    ax2 = plt.subplot2grid((5, 1), (4, 0))
    ax2.axis('off')  # Hide axes
    if speedup_text:
        ax2.text(0.5,
                 0.8,
                 speedup_text,
                 ha='center',
                 va='center',
                 fontsize=12,
                 fontweight='bold',
                 wrap=True,
                 bbox=dict(facecolor='white',
                           alpha=0.8,
                           boxstyle='round,pad=0.5'))
        ax2.text(0.5,
                 0.2,
                 metrics_text,
                 ha='center',
                 va='center',
                 fontsize=12,
                 wrap=True,
                 bbox=dict(facecolor='white',
                           alpha=0.8,
                           boxstyle='round,pad=0.5'))
    else:
        ax2.text(0.5,
                 0.5,
                 metrics_text,
                 ha='center',
                 va='center',
                 fontsize=12,
                 wrap=True,
                 bbox=dict(facecolor='white',
                           alpha=0.8,
                           boxstyle='round,pad=0.5'))

    # Create filename based on title suffix
    filename = 'query_e2e_completion_time_cdf'
    if title_suffix:
        title_suffix_clean = title_suffix.replace(" ", "_").replace(
            ":", "").replace(".", "_").lower()
        filename += f'_{title_suffix_clean}'
    filename += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
