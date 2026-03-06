import pandas as pd


def extract_trace_completion_times(metrics_file):
    """Extract ANNS trace completion times (time from replay_start to replay_end)."""
    df = pd.read_csv(metrics_file, low_memory=False)

    # Find replay_start and replay_end events
    end_events = df[df['event_type'] == 'replay_end']

    trace_times = {}

    for _, end_row in end_events.iterrows():
        stream_value = end_row['stream']
        duration = end_row['duration_secs']
        trace_times[stream_value] = duration

    return trace_times.get(True), trace_times.get(False)
