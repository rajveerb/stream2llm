from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from .extract_metrics import create_metrics


def extract_ttft_times(metrics_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Time To First Token (TTFT) metrics from a metrics file."""
    df = pd.read_csv(metrics_file, low_memory=False)

    # Filter for query_ttft events
    ttft_df = df[df['event_type'] == 'query_ttft']

    # Group by streaming flag
    streaming_ttft = ttft_df[ttft_df['stream'] == True]['duration_secs'].values
    non_streaming_ttft = ttft_df[ttft_df['stream'] ==
                                 False]['duration_secs'].values

    return streaming_ttft, non_streaming_ttft


def plot_ttft_histogram(streaming_ttft,
                        non_streaming_ttft,
                        output_path,
                        title_suffix=""):
    """Plot histograms of Time To First Token (TTFT)."""
    # Create a figure with two subplots - one for the plot and one for metrics text
    fig = plt.figure(figsize=(15, 8))

    # Main plot in the top subplot (80% of height)
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)

    # Create bins
    max_time = max(
        np.max(streaming_ttft) if len(streaming_ttft) > 0 else 0,
        np.max(non_streaming_ttft) if len(non_streaming_ttft) > 0 else 0)
    bins = np.linspace(0, max_time, 30)

    ax1.hist(streaming_ttft,
             bins=bins,
             alpha=0.5,
             label='Streaming',
             density=True)
    ax1.hist(non_streaming_ttft,
             bins=bins,
             alpha=0.5,
             label='Non-Streaming',
             density=True)

    # Calculate statistics for the metrics text area
    metrics_text, speedup_text = create_metrics(streaming_ttft,
                                                non_streaming_ttft)

    # Add labels and title
    ax1.set_xlabel('Time To First Token (seconds)')
    ax1.set_ylabel('Density')
    title = 'Distribution of Time To First Token (TTFT)'
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
    filename = 'ttft_histogram'
    if title_suffix:
        title_suffix_clean = title_suffix.replace(" ", "_").replace(
            ":", "").replace(".", "_").lower()
        filename += f'_{title_suffix_clean}'
    filename += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def plot_ttft_cdf(streaming_ttft,
                  non_streaming_ttft,
                  output_path,
                  title_suffix=""):
    """Plot CDF of Time To First Token (TTFT)."""
    # Create a figure with two subplots - one for the plot and one for metrics text
    _ = plt.figure(figsize=(15, 8))

    # Main plot in the top subplot (80% of height)
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)

    # Sort the data
    streaming_sorted = np.sort(streaming_ttft)
    non_streaming_sorted = np.sort(non_streaming_ttft)

    # Calculate the CDF
    streaming_cdf = np.arange(1,
                              len(streaming_sorted) +
                              1) / len(streaming_sorted) if len(
                                  streaming_sorted) > 0 else []
    non_streaming_cdf = np.arange(1,
                                  len(non_streaming_sorted) +
                                  1) / len(non_streaming_sorted) if len(
                                      non_streaming_sorted) > 0 else []

    # Plot the CDF
    if len(streaming_sorted) > 0:
        ax1.plot(streaming_sorted, streaming_cdf, label='Streaming')
    if len(non_streaming_sorted) > 0:
        ax1.plot(non_streaming_sorted,
                 non_streaming_cdf,
                 label='Non-Streaming')

    # Create metrics text
    metrics_text, speedup_text = create_metrics(streaming_ttft,
                                                non_streaming_ttft)

    ax1.set_xlabel('Time To First Token (seconds)')
    ax1.set_ylabel('CDF')
    title = 'CDF of Time To First Token (TTFT)'
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
    filename = 'ttft_cdf'
    if title_suffix:
        title_suffix_clean = title_suffix.replace(" ", "_").replace(
            ":", "").replace(".", "_").lower()
        filename += f'_{title_suffix_clean}'
    filename += '.png'

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def plot_ttft_boxplot_vs_replay_rate(data: Dict[float, Dict], output_dir: str):
    """Create TTFT boxplots vs replay rate with split view for low/high rates."""
    # Split data into two groups
    low_rates = [(rate, data[rate]) for rate in sorted(data.keys())
                 if rate <= 0.25]
    high_rates = [(rate, data[rate]) for rate in sorted(data.keys())
                  if rate > 0.25]

    if not low_rates and not high_rates:
        print("No data to plot.")
        return

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    box_width = 0.35

    # Helper function to create boxplots
    def create_boxplots(ax, filtered_data, title):
        if not filtered_data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title)
            return

        sorted_rates = [rate for rate, _ in filtered_data]
        streaming_ttft_data = [d['streaming_ttft'] for _, d in filtered_data]
        non_streaming_ttft_data = [
            d['non_streaming_ttft'] for _, d in filtered_data
        ]
        x = np.arange(len(sorted_rates))

        # Boxplot for streaming TTFT
        bp1 = ax.boxplot(streaming_ttft_data,
                         positions=x - box_width / 2,
                         widths=box_width,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightgreen', color='black'),
                         medianprops=dict(color='green'),
                         whiskerprops=dict(color='black'),
                         capprops=dict(color='black'),
                         flierprops=dict(marker='x',
                                         color='gray',
                                         markersize=6,
                                         markeredgecolor='gray'),
                         showfliers=True)
        # Boxplot for non-streaming TTFT
        bp2 = ax.boxplot(non_streaming_ttft_data,
                         positions=x + box_width / 2,
                         widths=box_width,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightcoral', color='black'),
                         medianprops=dict(color='red'),
                         whiskerprops=dict(color='black'),
                         capprops=dict(color='black'),
                         flierprops=dict(marker='x',
                                         color='gray',
                                         markersize=6,
                                         markeredgecolor='gray'),
                         showfliers=True)

        # Set x-ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.2f}" for r in sorted_rates])
        ax.set_xlabel('Replay Rate (QPS)')
        ax.set_ylabel('TTFT time (seconds)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return bp1, bp2

    # Create the two subplots
    create_boxplots(ax1, low_rates, 'TTFT vs Replay Rate (QPS <= 0.25)')
    create_boxplots(ax2, high_rates, 'TTFT vs Replay Rate (QPS > 0.25)')

    # Add common legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='lightgreen',
              edgecolor='black',
              label='Streaming TTFT'),
        Patch(facecolor='lightcoral',
              edgecolor='black',
              label='Non-Streaming TTFT')
    ]
    fig.legend(handles=legend_handles,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.05),
               ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttft_boxplot_vs_replay_rate.png'),
                bbox_inches='tight')
    plt.close()
    print(
        f"TTFT-only plot saved to {output_dir}/ttft_boxplot_vs_replay_rate.png"
    )


def plot_ttft_trace_combined(data: Dict[float, Dict], output_dir: str):
    """Create combined TTFT boxplots and trace completion times vs replay rate."""
    # Split data into two groups
    low_rates = [(rate, data[rate]) for rate in sorted(data.keys())
                 if rate <= 0.25]
    high_rates = [(rate, data[rate]) for rate in sorted(data.keys())
                  if rate > 0.25]

    if not low_rates and not high_rates:
        print("No data to plot.")
        return

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    box_width = 0.35

    # Helper function to create boxplots and trace time points
    def create_combined_plot(ax, filtered_data, title):
        if not filtered_data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title)
            return

        sorted_rates = [rate for rate, _ in filtered_data]
        streaming_ttft_data = [d['streaming_ttft'] for _, d in filtered_data]
        non_streaming_ttft_data = [
            d['non_streaming_ttft'] for _, d in filtered_data
        ]
        streaming_trace_data = [
            np.mean(d['streaming_trace']) if d['streaming_trace'] else np.nan
            for _, d in filtered_data
        ]
        non_streaming_trace_data = [
            np.mean(d['non_streaming_trace'])
            if d['non_streaming_trace'] else np.nan for _, d in filtered_data
        ]
        x = np.arange(len(sorted_rates))

        # Create secondary axis for trace times
        ax2 = ax.twinx()

        # Boxplot for streaming TTFT
        bp1 = ax.boxplot(streaming_ttft_data,
                         positions=x - box_width / 2,
                         widths=box_width,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightgreen', color='black'),
                         medianprops=dict(color='green'),
                         whiskerprops=dict(color='black'),
                         capprops=dict(color='black'),
                         flierprops=dict(marker='x',
                                         color='gray',
                                         markersize=6,
                                         markeredgecolor='gray'),
                         showfliers=True)
        # Boxplot for non-streaming TTFT
        bp2 = ax.boxplot(non_streaming_ttft_data,
                         positions=x + box_width / 2,
                         widths=box_width,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightcoral', color='black'),
                         medianprops=dict(color='red'),
                         whiskerprops=dict(color='black'),
                         capprops=dict(color='black'),
                         flierprops=dict(marker='x',
                                         color='gray',
                                         markersize=6,
                                         markeredgecolor='gray'),
                         showfliers=True)

        # Overlay trace completion times as points on secondary axis
        trace1 = ax2.plot(x - box_width / 2,
                          streaming_trace_data,
                          'go',
                          label='Streaming Trace Completion Time',
                          markersize=10)[0]
        trace2 = ax2.plot(x + box_width / 2,
                          non_streaming_trace_data,
                          'ro',
                          label='Non-Streaming Trace Completion Time',
                          markersize=10)[0]

        # Set x-ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.2f}" for r in sorted_rates])
        ax.set_xlabel('Replay Rate (QPS)')
        ax.set_ylabel('TTFT (seconds)')
        ax2.set_ylabel('Trace Completion Time (seconds)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return bp1, bp2, trace1, trace2

    # Create the two subplots
    plots1 = create_combined_plot(
        ax1, low_rates, 'TTFT and Trace Time vs Replay Rate (QPS <= 0.25)')
    plots2 = create_combined_plot(
        ax2, high_rates, 'TTFT and Trace Time vs Replay Rate (QPS > 0.25)')

    # Add common legend
    if plots1 and plots2:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor='lightgreen',
                  edgecolor='black',
                  label='Streaming TTFT'),
            Patch(facecolor='lightcoral',
                  edgecolor='black',
                  label='Non-Streaming TTFT'),
            plt.Line2D([0], [0],
                       marker='o',
                       color='g',
                       label='Streaming Trace Time',
                       markersize=10,
                       linestyle='None'),
            plt.Line2D([0], [0],
                       marker='o',
                       color='r',
                       label='Non-Streaming Trace Time',
                       markersize=10,
                       linestyle='None')
        ]
        fig.legend(handles=legend_handles,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.05),
                   ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             'ttft_trace_boxplot_vs_replay_rate.png'),
                bbox_inches='tight')
    plt.close()
    print(
        f"Combined plot saved to {output_dir}/ttft_trace_boxplot_vs_replay_rate.png"
    )


def aggregate_ttft_data(metrics_files: List[str],
                        replay_rates: List[float]) -> Dict[float, Dict]:
    """Aggregate TTFT data by replay rate for boxplot analysis."""
    data = {}

    for metrics_file, replay_rate in zip(metrics_files, replay_rates):
        if replay_rate not in data:
            data[replay_rate] = {
                'streaming_ttft': [],
                'non_streaming_ttft': [],
                'streaming_trace': [],
                'non_streaming_trace': []
            }

        # Extract TTFT times
        streaming_ttft, non_streaming_ttft = extract_ttft_times(metrics_file)
        data[replay_rate]['streaming_ttft'].extend(streaming_ttft)
        data[replay_rate]['non_streaming_ttft'].extend(non_streaming_ttft)

        # Extract trace completion times
        from .trace_e2e import extract_trace_completion_times
        streaming_trace, non_streaming_trace = extract_trace_completion_times(
            metrics_file)
        if streaming_trace is not None:
            data[replay_rate]['streaming_trace'].append(streaming_trace)
        if non_streaming_trace is not None:
            data[replay_rate]['non_streaming_trace'].append(
                non_streaming_trace)

    return data
