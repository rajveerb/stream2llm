import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from collections import defaultdict
import pandas as pd
from matplotlib.ticker import MaxNLocator


def generate_metric_time_stats(streaming_times, non_streaming_times,
                               trace_time_streaming, trace_time_non_streaming,
                               streaming_ttft, non_streaming_ttft):
    """
    Generate statistics about the query metrics.
    Returns a JSON-serializable dictionary with the statistics.
    """

    return {
        'Streaming': {
            'count': len(streaming_times),
            'mean': float(np.mean(streaming_times)),
            'median': float(np.median(streaming_times)),
            'min': float(np.min(streaming_times)),
            'max': float(np.max(streaming_times)),
            'std': float(np.std(streaming_times)),
            'trace_time': float(trace_time_streaming),
            'ttft_mean': float(np.mean(streaming_ttft)),
            'ttft_median': float(np.median(streaming_ttft)),
            'ttft_min': float(np.min(streaming_ttft)),
            'ttft_max': float(np.max(streaming_ttft)),
            'ttft_std': float(np.std(streaming_ttft)),
        },
        'Non-Streaming': {
            'count': len(non_streaming_times),
            'mean': float(np.mean(non_streaming_times)),
            'median': float(np.median(non_streaming_times)),
            'min': float(np.min(non_streaming_times)),
            'max': float(np.max(non_streaming_times)),
            'std': float(np.std(non_streaming_times)),
            'trace_time': float(trace_time_non_streaming),
            'ttft_mean': float(np.mean(non_streaming_ttft)),
            'ttft_median': float(np.median(non_streaming_ttft)),
            'ttft_min': float(np.min(non_streaming_ttft)),
            'ttft_max': float(np.max(non_streaming_ttft)),
            'ttft_std': float(np.std(non_streaming_ttft)),
        },
        'Speedup': float(trace_time_non_streaming / trace_time_streaming)
    }


def extract_replay_rate(run_dir) -> float:
    """Extract the replay rate from the config file in the run directory."""
    config_file = os.path.join(run_dir,
                               f"config_{os.path.basename(run_dir)}.yaml")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract the poisson average arrival time (smaller value = higher rate)
    if 'replay' in config and 'poisson_avg_arrival_time' in config['replay']:
        # Convert to replay rate (queries per second)
        return 1.0 / config['replay']['poisson_avg_arrival_time']
    else:
        raise Exception(
            f"Error reading poisson_avg_arrival_time from config file {config_file}"
        )


def process_trace_time_data(run_dirs, streaming_times, non_streaming_times):
    """
    Process trace time data for plotting.
    Returns a tuple of (data, replay_rates, streaming_values, non_streaming_values, speedups, run_labels)
    where data is a list of tuples (replay_rate, streaming_time, non_streaming_time, speedup, run_dir).
    """
    data = []

    # Group data by replay rate
    replay_rate_to_data = defaultdict(lambda: {
        'streaming': [],
        'non_streaming': [],
        'run_dirs': []
    })

    for i, run_dir in enumerate(run_dirs):
        # Skip if we don't have both streaming and non-streaming data
        if streaming_times[i] is None or non_streaming_times[i] is None:
            continue

        # Extract replay rate
        replay_rate = extract_replay_rate(run_dir)

        # Calculate speedup (non-streaming / streaming)
        speedup = non_streaming_times[i] / streaming_times[i]

        replay_rate_to_data[replay_rate]['streaming'].append(
            streaming_times[i])
        replay_rate_to_data[replay_rate]['non_streaming'].append(
            non_streaming_times[i])
        replay_rate_to_data[replay_rate]['run_dirs'].append(run_dir)

        data.append((replay_rate, streaming_times[i], non_streaming_times[i],
                     speedup, run_dir))

    if not data:
        return None

    # Sort by replay rate
    data.sort(key=lambda x: x[0])

    # Extract data for plotting
    replay_rates = [d[0] for d in data]
    streaming_values = [d[1] for d in data]
    non_streaming_values = [d[2] for d in data]
    speedups = [d[3] for d in data]
    run_labels = [os.path.basename(d[4]) for d in data]

    return replay_rates, streaming_values, non_streaming_values, speedups, run_labels


def plot_speedup_vs_replay_rate(replay_rates, speedups, run_labels,
                                output_dir):
    """Plot the trace completion time ratio (speedup) against replay rate."""

    if not replay_rates or not speedups or not run_labels:
        print("No data available for speedup vs replay rate plot")
        return

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot individual data points
    plt.scatter(replay_rates,
                speedups,
                s=100,
                alpha=0.7,
                label='Individual Runs')
    plt.plot(replay_rates, speedups, '-', alpha=0.6)

    # Add labels to each point
    for i, label in enumerate(run_labels):
        plt.annotate(label, (replay_rates[i], speedups[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    # Add a horizontal line at y=1 (no speedup)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)

    # Add labels and title
    plt.xlabel('Replay Rate (queries per second)')
    plt.ylabel('Speedup Ratio (Non-streaming / Streaming)')
    plt.title('Trace Completion Time Speedup vs Replay Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_vs_replay_rate.png'))
    plt.close()

    print(
        f"Speedup vs Replay Rate plot saved to {output_dir}/speedup_vs_replay_rate.png"
    )


def plot_raw_trace_times(replay_rates, streaming_values, non_streaming_values,
                         run_labels, output_dir):
    """Plot the raw streaming and non-streaming trace end times for each replay rate."""

    if not replay_rates or not streaming_values or not non_streaming_values or not run_labels:
        print("No data available for raw trace times plot")
        return

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot streaming times in green
    plt.scatter(replay_rates,
                streaming_values,
                color='green',
                s=100,
                alpha=0.7,
                label='Streaming')
    plt.plot(replay_rates, streaming_values, 'g-', alpha=0.6)

    # Plot non-streaming times in red
    plt.scatter(replay_rates,
                non_streaming_values,
                color='red',
                s=100,
                alpha=0.7,
                label='Non-Streaming')
    plt.plot(replay_rates, non_streaming_values, 'r-', alpha=0.6)

    # Add labels to each point
    for i, label in enumerate(run_labels):
        plt.annotate(label, (replay_rates[i], streaming_values[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     color='green')
        plt.annotate(label, (replay_rates[i], non_streaming_values[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     color='red')

    # Add labels and title
    plt.xlabel('Replay Rate (queries per second)')
    plt.ylabel('Trace Completion Time (seconds)')
    plt.title('Raw Trace Completion Times vs Replay Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw_trace_times_vs_replay_rate.png'))
    plt.close()

    print(
        f"Raw Trace Times vs Replay Rate plot saved to {output_dir}/raw_trace_times_vs_replay_rate.png"
    )


def plot_preemptions_vs_replay_rate(preemption_data, output_dir):
    """Plot preemption counts vs replay rate.
    
    Args:
        preemption_data: List of dictionaries containing preemption counts for each replay rate
        output_dir: Directory to save the plot
    """
    if not preemption_data:
        print("No preemption data available for plotting")
        return

    # Convert to DataFrame
    df = pd.DataFrame(preemption_data)

    # Sort by replay rate
    df = df.sort_values('replay_rate')

    # Create custom x-axis labels: replay_rate + newline + run_name + streaming info
    df['streaming_label'] = df['stream'].apply(lambda x: 'Streaming'
                                               if x else 'Non-Streaming')
    df['x_label'] = (df['replay_rate'].astype(str) + '\n(' + df['run_name'] +
                     ' - ' + df['streaming_label'] + ')')

    # Create stacked bar plot
    plt.figure(figsize=(14, 6))  # Increased width to accommodate longer labels
    ax = df.plot(x='x_label',
                 y=['PREEMPTED_SWAP', 'PREEMPTED_RECOMPUTE'],
                 kind='bar',
                 stacked=True)

    # Set y-axis to show only integers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title('Preemption Counts vs Replay Rate')
    plt.xlabel('Replay Rate - QPS\n(log run name - streaming status)')
    plt.ylabel('Number of Preemptions')
    plt.legend(title='Preemption Type')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(output_dir, 'preemptions_vs_replay_rate.png'),
                bbox_inches='tight')
    plt.close()

    print(
        f"Preemptions vs Replay Rate plot saved to {output_dir}/preemptions_vs_replay_rate.png"
    )
