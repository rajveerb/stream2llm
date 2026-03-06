import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from vllm.v1.engine import EngineCoreEventType


class EventAnalyzer:

    def __init__(self, metrics_file: str, run_name: str):
        """Initialize the event analyzer with a metrics file."""
        self.df = pd.read_csv(metrics_file, low_memory=False)
        self.time_window = 0.1  # 100ms window
        self.run_name = run_name

        # Convert event_type to EngineCoreEventType only for vLLM engine events
        def convert_event_type(x):
            if pd.isna(x):
                return None
            try:
                # Handle string format "EngineCoreEventType.EVENT_NAME"
                if isinstance(x, str) and x.startswith('EngineCoreEventType.'):
                    event_name = x.split('.')[-1]
                    return EngineCoreEventType[event_name]
                # Handle direct enum values
                elif isinstance(x, str):
                    return EngineCoreEventType[x]
                # Handle already converted enum values
                elif isinstance(x, EngineCoreEventType):
                    return x
                else:
                    return x
            except (KeyError, AttributeError):
                # Return the original string for custom events
                return x

        self.df['event_type'] = self.df['event_type'].apply(convert_event_type)

    def analyze_event_durations(self) -> pd.DataFrame:
        """Analyze time spent in each event type per request."""
        # Filter out custom events for duration analysis
        engine_events = self.df[self.df['event_type'].apply(
            lambda x: isinstance(x, EngineCoreEventType))]
        if engine_events.empty:
            print("No engine events found in the data")
            return pd.DataFrame()  # Return empty DataFrame if no events found

        event_durations = engine_events.groupby(['request_id', 'event_type'
                                                 ])['duration_secs'].sum()
        return event_durations.reset_index()

    def plot_event_timeline(self, output_dir: str) -> None:
        """
        Create stacked-area charts of EngineCoreEventType counts
        over time, split into streaming / non-streaming traces.

        Saves     <output_dir>/event_timeline_stream.png
            and <output_dir>/event_timeline_non_streaming.png
        """
        # ------------------------------------------------------------------ #
        # 0.  House-keeping / pre-processing
        # ------------------------------------------------------------------ #
        os.makedirs(output_dir, exist_ok=True)

        if 'is_engine_evt' not in self.df.columns:
            self.df['is_engine_evt'] = self.df['event_type'].apply(
                lambda v: isinstance(v, EngineCoreEventType))

        # Event-type → printable name (build once, vectorised `.map`)
        evt_name_map = {
            evt: evt.name if isinstance(evt, EngineCoreEventType) else str(evt)
            for evt in self.df['event_type'].unique()
        }
        self.df['event_name'] = self.df['event_type'].map(evt_name_map)

        # Sensible default if caller forgot to set it
        time_window = getattr(self, "time_window", 1.0) or 1.0

        # ------------------------------------------------------------------ #
        # 1.  Two passes: streaming = True / False
        # ------------------------------------------------------------------ #
        for is_streaming in (True, False):
            # Vectorised filter
            mask = (self.df['stream']
                    == is_streaming) & self.df['is_engine_evt']
            stream_data = self.df.loc[mask]

            if stream_data.empty:
                continue  # nothing to plot for this slice

            # ------------------------------------------------------------------ #
            # 2.  Work in seconds since slice start, regardless of dtype
            # ------------------------------------------------------------------ #
            t0 = stream_data['event_timestamp'].min()
            ts_delta = stream_data['event_timestamp'] - t0

            # If dtype is timedelta64 → convert to seconds; else assume float/ints
            if np.issubdtype(ts_delta.dtype, np.timedelta64):
                ts_sec = ts_delta.dt.total_seconds().to_numpy()
            else:
                ts_sec = ts_delta.astype(float).to_numpy()

            time_range = ts_sec.max()
            if time_range == 0:
                # All events at same instant → skip or draw a single-bar chart
                continue

            # ------------------------------------------------------------------ #
            # 3.  Bin the events
            # ------------------------------------------------------------------ #
            num_bins = max(1, int(time_range / time_window))
            bin_edges = np.linspace(0.0, time_range,
                                    num_bins + 1)  # +1 → num_bins bins
            bin_idxs = np.digitize(ts_sec, bin_edges, right=False)  # 1-based

            # DataFrame: rows = bins, cols = event names, values = counts
            event_counts = (pd.DataFrame({
                'bin':
                bin_idxs,
                'event_name':
                stream_data['event_name']
            }).groupby(['bin', 'event_name']).size().unstack(fill_value=0))

            if event_counts.empty:
                continue

            # Replace bin index with mid-point time (seconds)
            bin_size = time_range / num_bins
            event_counts.index = (event_counts.index - 0.5) * bin_size

            # ------------------------------------------------------------------ #
            # 4.  Plot
            # ------------------------------------------------------------------ #
            fig, ax = plt.subplots(figsize=(20, 6))
            event_counts.plot(kind='area', stacked=True, alpha=0.7,
                              ax=ax)  # draw legend manually

            ax.set_title(f'Engine Event Distribution Over Time '
                         f'(Streaming: {is_streaming})')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Number of Events')

            # place legend outside to avoid overlap
            ax.legend(title='Event Type',
                      loc='upper left',
                      bbox_to_anchor=(1.02, 1),
                      borderaxespad=0.)

            plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave room for legend

            # ------------------------------------------------------------------ #
            # 5.  Save
            # ------------------------------------------------------------------ #
            suffix = "stream" if is_streaming else "non_streaming"
            fig.savefig(os.path.join(output_dir,
                                     f'event_timeline_{suffix}.png'),
                        bbox_inches='tight')
            plt.close(fig)

        # ------------------------------------------------------------------ #
        # 6.  Simple textual summary
        # ------------------------------------------------------------------ #
        engine_events = self.df.loc[self.df['is_engine_evt'], 'event_name']
        print("\nEngine Event Type Summary:")
        print("--------------------------")
        print(engine_events.value_counts())

    def plot_event_cdf(self, output_dir: str):
        """Plot CDF for each event type duration."""
        # Only plot CDFs for engine events
        engine_events = self.df[self.df['event_type'].apply(
            lambda x: isinstance(x, EngineCoreEventType))]

        for event_type in EngineCoreEventType:
            # Check if we have any data for this event type
            event_data = engine_events[engine_events['event_type'] ==
                                       event_type]
            if event_data.empty:
                continue  # Skip if no data for this event type

            plt.figure(figsize=(10, 6))
            has_data = False  # Track if we actually plot anything

            for is_streaming in [True, False]:
                data = engine_events[
                    (engine_events['event_type'] == event_type) &
                    (engine_events['stream'] == is_streaming)]['duration_secs']

                if not data.empty:
                    sns.ecdfplot(data=data,
                                 label=f'Streaming'
                                 if is_streaming else 'Non-Streaming')

            plt.title(f'CDF of requests with status={event_type.name}')
            plt.xlabel('Duration (s)')
            plt.ylabel('Cumulative Probability')
            plt.legend()
            plt.savefig(f'{output_dir}/cdf_{event_type.name}.png',
                        bbox_inches='tight')
            plt.close()

    def plot_preemption_vs_load(self, output_dir: str):
        """Plot preemption events against system load for both streaming and non-streaming cases."""
        # Filter for preemption events
        preemption_data = self.df[(self.df['event_type'].isin([
            EngineCoreEventType.PREEMPTED_SWAP,
            EngineCoreEventType.PREEMPTED_RECOMPUTE
        ]))]

        if preemption_data.empty:
            print("No preemption events found in the data")
            return

        # Create a copy and convert event_type to string names for plotting
        plot_data = preemption_data.copy()
        plot_data['event_type_str'] = plot_data['event_type'].apply(
            lambda x: x.name if isinstance(x, EngineCoreEventType) else str(x))

        # Process both streaming and non-streaming cases
        for is_streaming in [True, False]:
            stream_suffix = "streaming" if is_streaming else "non_streaming"
            stream_data = plot_data[plot_data['stream'] == is_streaming]

            if stream_data.empty:
                print(f"No preemption events found for {stream_suffix} case")
                continue

            # Plot duration vs load for this streaming case
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=stream_data,
                x='concurrent_requests',
                y='duration_secs',
                hue='event_type_str',  # Use string version
                alpha=0.5)
            plt.title(
                f'Preemption Duration vs Concurrent Requests ({stream_suffix})'
            )
            plt.xlabel('Number of Concurrent Requests')
            plt.ylabel('Preemption Duration (s)')
            plt.legend(title='Preemption Type')
            plt.savefig(f'{output_dir}/preemption_vs_load_{stream_suffix}.png',
                        bbox_inches='tight')
            plt.close()

            # Plot preemption frequency by type for this streaming case
            plt.figure(figsize=(10, 6))
            preemption_counts = stream_data.groupby(
                ['concurrent_requests',
                 'event_type_str']).size().unstack(fill_value=0)
            if not preemption_counts.empty:
                preemption_counts.plot(kind='bar', stacked=True)
                plt.title(
                    f'Preemption Frequency vs Concurrent Requests ({stream_suffix})'
                )
                plt.xlabel('Number of Concurrent Requests')
                plt.ylabel('Number of Preemptions')
                plt.legend(title='Preemption Type')
                # Set y-axis to use integer ticks
                plt.gca().yaxis.set_major_locator(
                    plt.MaxNLocator(integer=True))
                plt.savefig(
                    f'{output_dir}/preemption_frequency_{stream_suffix}.png',
                    bbox_inches='tight')
            plt.close()

            # Print summary statistics for this streaming case
            print(f"\nPreemption Statistics ({stream_suffix}):")
            print("-" * (30 + len(stream_suffix)))
            print(f"Total preemptions: {len(stream_data)}")
            print("\nBy type:")
            print(stream_data['event_type_str'].value_counts())
            print("\nAverage duration by type:")
            print(
                stream_data.groupby('event_type_str')['duration_secs'].mean())

        # Overall summary statistics
        print("\nOverall Preemption Statistics:")
        print("------------------------------")
        print(f"Total preemptions: {len(plot_data)}")
        print("\nBy streaming state:")
        print(plot_data['stream'].value_counts())
        print("\nBy type:")
        print(plot_data['event_type_str'].value_counts())
        print("\nAverage duration by type:")
        print(plot_data.groupby('event_type_str')['duration_secs'].mean())

    def extract_preemptions_vs_replay_rate(self, replay_rate: float):
        """Extract preemption events against replay rate for both streaming and non-streaming cases.
        
        Args:
            replay_rate: Current replay rate in QPS
        """
        # Filter for preemption events (both streaming and non-streaming)
        preemption_data = self.df[(self.df['event_type'].isin([
            EngineCoreEventType.PREEMPTED_SWAP,
            EngineCoreEventType.PREEMPTED_RECOMPUTE
        ]))]

        if preemption_data.empty:
            print(
                f"\tNo preemption events found for replay rate {replay_rate}")
            return None

        results = []

        # Process both streaming and non-streaming cases
        for stream in [True, False]:
            stream_data = preemption_data[preemption_data['stream'] == stream]

            if not stream_data.empty:
                # Count preemptions by type
                preemption_counts = stream_data['event_type'].value_counts()

                # Create a result dictionary for this streaming case
                result = {
                    'run_name':
                    self.run_name,
                    'replay_rate':
                    replay_rate,
                    'stream':
                    stream,
                    'PREEMPTED_SWAP':
                    preemption_counts.get(EngineCoreEventType.PREEMPTED_SWAP,
                                          0),
                    'PREEMPTED_RECOMPUTE':
                    preemption_counts.get(
                        EngineCoreEventType.PREEMPTED_RECOMPUTE, 0)
                }
                results.append(result)

        return results

    def plot_queue_length_over_time(self, output_dir: str):
        """Plot queue length over time."""
        # Filter for queued events
        queued_events = self.df[self.df['event_type'] ==
                                EngineCoreEventType.QUEUED]

        if queued_events.empty:
            print("No queued events found in the data")
            return

        # Normalize timestamps to start at 0
        t0 = queued_events['event_timestamp'].min()
        queued_events = queued_events.copy()
        queued_events[
            'normalized_timestamp'] = queued_events['event_timestamp'] - t0

        # Group by normalized timestamp and count queue length
        queue_length = queued_events.groupby('normalized_timestamp').size()

        plt.figure(figsize=(12, 6))
        queue_length.plot()
        plt.title('Queue Length Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Number of Requests in Queue')
        plt.savefig(f'{output_dir}/queue_length.png')
        plt.close()

    def generate_summary_stats(self) -> pd.DataFrame:
        """Generate summary statistics for each event type."""
        # Filter for engine events only
        engine_events = self.df[self.df['event_type'].apply(
            lambda x: isinstance(x, EngineCoreEventType))]

        stats = []
        for event_type in EngineCoreEventType:
            event_data = engine_events[engine_events['event_type'] ==
                                       event_type]

            if not event_data.empty:
                # Calculate statistics for each streaming state
                for is_streaming in [True, False]:
                    stream_data = event_data[event_data['stream'] ==
                                             is_streaming]

                    if not stream_data.empty:
                        # Calculate statistics, handling NaN values
                        event_stats = {
                            'event_type':
                            event_type.name,
                            'stream':
                            is_streaming,
                            'count':
                            len(stream_data),
                            'mean_duration':
                            stream_data['duration_secs'].mean(),
                            'std_duration':
                            stream_data['duration_secs'].std(),
                            'min_duration':
                            stream_data['duration_secs'].min(),
                            'max_duration':
                            stream_data['duration_secs'].max(),
                            'mean_concurrent_requests':
                            stream_data['concurrent_requests'].mean(),
                            'mean_request_size':
                            stream_data['request_size'].mean(),
                            'mean_replay_rate':
                            stream_data['replay_rate'].mean()
                        }

                        # Replace NaN values with 0 or appropriate default
                        for key, value in event_stats.items():
                            if pd.isna(value):
                                if key == 'count':
                                    event_stats[key] = 0
                                else:
                                    event_stats[key] = 0.0

                        stats.append(event_stats)

        # Create DataFrame and sort by event type and streaming state
        stats_df = pd.DataFrame(stats)
        if not stats_df.empty:
            stats_df = stats_df.sort_values(['event_type', 'stream'])

        return stats_df
