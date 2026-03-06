import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vllm.v1.engine import EngineCoreEventType
from typing import Dict, List


# Convert event_type strings to EngineCoreEventType enum
def convert_event_type(x):
    if isinstance(x, str):
        # Handle different string formats
        if x.startswith('EngineCoreEventType.'):
            event_name = x.split('.')[-1]
            return EngineCoreEventType[event_name]
        else:
            return x
    else:
        raise ValueError(f"Invalid event type: {x} type: {type(x)}")


def plot_event_durations_vs_replay_rate(data_files: List[str],
                                        output_dir: str):
    """Create box plots for EngineCoreEventType durations vs replay rate."""
    from .aggregate import extract_replay_rate

    # Initialize data structure to store event durations by replay rate
    data = {
    }  # replay_rate -> event_type -> {streaming: [], non_streaming: []}

    # Process each metrics file
    for metrics_file in data_files:
        run_dir = os.path.dirname(metrics_file)
        replay_rate = extract_replay_rate(run_dir)

        if replay_rate not in data:
            data[replay_rate] = {
                evt: {
                    'streaming': [],
                    'non_streaming': []
                }
                for evt in EngineCoreEventType
            }

        df = pd.read_csv(metrics_file, low_memory=False)

        df['event_type'] = df['event_type'].apply(convert_event_type)

        # Filter for engine events only
        engine_events = df[df['event_type'].apply(
            lambda x: isinstance(x, EngineCoreEventType))]

        # Group events by type and streaming status
        for evt_type in EngineCoreEventType:
            streaming_data = engine_events[
                (engine_events['event_type'] == evt_type)
                & (engine_events['stream'] == True)]['duration_secs']
            non_streaming_data = engine_events[
                (engine_events['event_type'] == evt_type)
                & (engine_events['stream'] == False)]['duration_secs']

            data[replay_rate][evt_type]['streaming'].extend(
                streaming_data.tolist())
            data[replay_rate][evt_type]['non_streaming'].extend(
                non_streaming_data.tolist())

    # Create a plot for each event type that has data
    for evt_type in EngineCoreEventType:
        # Check if we have any data for this event type
        has_data = False
        for rate_data in data.values():
            if rate_data[evt_type]['streaming'] or rate_data[evt_type][
                    'non_streaming']:
                has_data = True
                break

        if not has_data:
            continue

        plt.figure(figsize=(15, 8))
        box_width = 0.35
        sorted_rates = sorted(data.keys())
        x = np.arange(len(sorted_rates))

        # Prepare data for plotting
        streaming_data = [
            data[rate][evt_type]['streaming'] for rate in sorted_rates
        ]
        non_streaming_data = [
            data[rate][evt_type]['non_streaming'] for rate in sorted_rates
        ]

        # Boxplot for non-streaming events (left side - baseline)
        bp1 = plt.boxplot(non_streaming_data,
                          positions=x - box_width / 2,
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

        # Boxplot for streaming events (right side)
        bp2 = plt.boxplot(streaming_data,
                          positions=x + box_width / 2,
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

        # Customize the plot
        plt.xticks(x, [f"{r:.2f}" for r in sorted_rates],
                   rotation=45,
                   ha='right')
        plt.xlabel('Replay Rate (QPS)')
        plt.ylabel('Duration (seconds)')
        plt.title(f'Event Duration vs Replay Rate - {evt_type.name}')

        # Add legend
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor='lightcoral',
                  edgecolor='black',
                  label='Non-Streaming'),
            Patch(facecolor='lightgreen', edgecolor='black', label='Streaming')
        ]
        plt.legend(handles=legend_handles)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'event_duration_vs_replay_rate_{evt_type.name}.png'),
                    bbox_inches='tight')
        plt.close()
        print(f"Plot saved for event type {evt_type.name}")


def plot_aggregated_event_durations(data_files: List[str], output_dir: str):
    """Create aggregated box plots for all EngineCoreEventType durations."""
    # Initialize data structures to store event durations
    streaming_events = {evt: [] for evt in EngineCoreEventType}
    non_streaming_events = {evt: [] for evt in EngineCoreEventType}

    # Process each metrics file
    for metrics_file in data_files:
        df = pd.read_csv(metrics_file, low_memory=False)

        df['event_type'] = df['event_type'].apply(convert_event_type)

        # Filter for engine events only
        engine_events = df[df['event_type'].apply(
            lambda x: isinstance(x, EngineCoreEventType))]

        # Group events by type and streaming status
        for evt_type in EngineCoreEventType:
            streaming_data = engine_events[
                (engine_events['event_type'] == evt_type)
                & (engine_events['stream'] == True)]['duration_secs']
            non_streaming_data = engine_events[
                (engine_events['event_type'] == evt_type)
                & (engine_events['stream'] == False)]['duration_secs']

            streaming_events[evt_type].extend(streaming_data.tolist())
            non_streaming_events[evt_type].extend(non_streaming_data.tolist())

    # Prepare data for plotting
    events_to_plot = [
        evt for evt in EngineCoreEventType
        if streaming_events[evt] or non_streaming_events[evt]
    ]

    if not events_to_plot:
        print("No event data to plot")
        return

    # Create the plot
    plt.figure(figsize=(15, 8))
    box_width = 0.35
    x = np.arange(len(events_to_plot))

    # Boxplot for non-streaming events (left side - baseline)
    bp1 = plt.boxplot([non_streaming_events[evt] for evt in events_to_plot],
                      positions=x - box_width / 2,
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

    # Boxplot for streaming events (right side)
    bp2 = plt.boxplot([streaming_events[evt] for evt in events_to_plot],
                      positions=x + box_width / 2,
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

    # Customize the plot
    plt.xticks(x, [evt.name for evt in events_to_plot],
               rotation=45,
               ha='right')
    plt.xlabel('Event Type')
    plt.ylabel('Duration (seconds)')
    plt.title('Engine Core Event Durations')

    # Add legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='lightcoral', edgecolor='black',
              label='Non-Streaming'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Streaming')
    ]
    plt.legend(handles=legend_handles)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'event_durations_boxplot.png'),
                bbox_inches='tight')
    plt.close()
    print(
        f"Event durations plot saved to {output_dir}/event_durations_boxplot.png"
    )
