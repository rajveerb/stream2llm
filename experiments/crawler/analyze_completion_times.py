#!/usr/bin/env python3
import os, glob, json
import pandas as pd
import matplotlib.pyplot as plt
from plotter_utils.trace_e2e import extract_trace_completion_times
from plotter_utils.ttft import extract_ttft_times, plot_ttft_histogram, plot_ttft_cdf, plot_ttft_boxplot_vs_replay_rate, plot_ttft_trace_combined, aggregate_ttft_data
from plotter_utils.query_e2e import extract_query_e2e_latencies, plot_query_e2e_times_cdf, plot_query_e2e_times_histogram
from plotter_utils.aggregate import extract_replay_rate, generate_metric_time_stats, plot_speedup_vs_replay_rate, plot_raw_trace_times, process_trace_time_data, plot_preemptions_vs_replay_rate
from plotter_utils.event_analysis import EventAnalyzer
from plotter_utils.event_durations import plot_event_durations_vs_replay_rate, plot_aggregated_event_durations


def load_metrics_files(log_dir):
    """Load all run_metrics.csv files from the log directory."""
    metrics_files = glob.glob(os.path.join(log_dir, "*/run_metrics.csv"))
    return metrics_files


def detect_scheduler_structure(log_dir):
    """Detect if log_dir contains multiple schedulers or single scheduler."""
    # Use recursive search to find all run_metrics.csv files
    metrics_files = glob.glob(os.path.join(log_dir, "**/run_metrics.csv"),
                              recursive=True)

    if not metrics_files:
        return "empty", []

    # Determine structure based on path depth
    scheduler_dirs = set()
    single_scheduler = False

    for metrics_file in metrics_files:
        rel_path = os.path.relpath(metrics_file, log_dir)
        path_parts = rel_path.split(os.sep)

        if len(path_parts) == 2:  # timestamp/run_metrics.csv
            single_scheduler = True
        elif len(path_parts) == 3:  # scheduler/timestamp/run_metrics.csv
            scheduler_name = path_parts[0]
            scheduler_path = os.path.join(log_dir, scheduler_name)
            scheduler_dirs.add(scheduler_path)

    if single_scheduler and not scheduler_dirs:
        return "single", [log_dir]
    elif scheduler_dirs and not single_scheduler:
        return "multi", sorted(list(scheduler_dirs))
    else:
        return "unknown", []


def process_single_scheduler(log_dir, output_dir):
    """Process a single scheduler directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load all metrics files
    metrics_files = load_metrics_files(log_dir)
    print(f"Found {len(metrics_files)} metrics files")

    # Initialize aggregated data for metrics
    run_dirs = []
    streaming_trace_times = []
    non_streaming_trace_times = []
    preemption_data = []  # List to store preemption data for each replay rate

    # Process each metrics file
    for metrics_file in metrics_files:
        print(f"Processing {metrics_file}...")

        # Get the run directory name for the title suffix
        run_dir = os.path.dirname(metrics_file)
        run_dirs.append(run_dir)
        run_name = os.path.basename(run_dir)

        # Extract replay rate
        replay_rate = extract_replay_rate(run_dir)
        title_suffix = f"for {run_name} (Request arrival rate: {replay_rate:.2f} qps)"

        # Create event analysis directory
        event_dir = os.path.join(
            output_dir, "event_analysis",
            f"{run_name}_request_arrival_rate_{replay_rate:.2f}")
        os.makedirs(event_dir, exist_ok=True)

        # Initialize event analyzer
        event_analyzer = EventAnalyzer(metrics_file, run_name)

        # Collect preemption data for this replay rate
        preemption_result = event_analyzer.extract_preemptions_vs_replay_rate(
            replay_rate)
        if preemption_result is not None:
            # preemption_result is now a list of dictionaries (one for each streaming state)
            preemption_data.extend(preemption_result)

        # Generate event analysis plots
        print("Generating event analysis plots...")
        event_analyzer.plot_event_timeline(event_dir)
        event_analyzer.plot_event_cdf(event_dir)
        event_analyzer.plot_preemption_vs_load(event_dir)
        event_analyzer.plot_queue_length_over_time(event_dir)

        # Generate and save summary statistics
        stats = event_analyzer.generate_summary_stats()
        # Convert DataFrame to dictionary format for JSON serialization
        stats_dict = stats.to_dict(orient='records')
        with open(os.path.join(event_dir, 'event_stats.json'), 'w') as f:
            json.dump(stats_dict, f, indent=2)

        # Extract query e2e times for this file
        streaming_e2e_times, non_streaming_e2e_times = extract_query_e2e_latencies(
            metrics_file)

        # Extract TTFT times for this file
        streaming_ttft, non_streaming_ttft = extract_ttft_times(metrics_file)

        # create cdf and histogram dir if it doesn't exist
        cdf_dir = os.path.join(output_dir, "cdf")
        os.makedirs(cdf_dir, exist_ok=True)
        histogram_dir = os.path.join(output_dir, "histogram")
        os.makedirs(histogram_dir, exist_ok=True)

        # Plot histograms for this file
        plot_query_e2e_times_histogram(streaming_e2e_times,
                                       non_streaming_e2e_times, histogram_dir,
                                       title_suffix)

        # Plot CDFs for this file
        plot_query_e2e_times_cdf(streaming_e2e_times, non_streaming_e2e_times,
                                 cdf_dir, title_suffix)

        # Plot TTFT histograms for this file
        plot_ttft_histogram(streaming_ttft, non_streaming_ttft, histogram_dir,
                            title_suffix)

        # Plot TTFT CDFs for this file
        plot_ttft_cdf(streaming_ttft, non_streaming_ttft, cdf_dir,
                      title_suffix)

        # Extract trace completion times for this file
        streaming_trace_time, non_streaming_trace_time = extract_trace_completion_times(
            metrics_file)
        streaming_trace_times.append(streaming_trace_time)
        non_streaming_trace_times.append(non_streaming_trace_time)

        # only generate if all fields are available
        if streaming_e2e_times is not None and non_streaming_e2e_times is not None and streaming_trace_time is not None and non_streaming_trace_time is not None and streaming_ttft is not None and non_streaming_ttft is not None:
            stats_dict = generate_metric_time_stats(streaming_e2e_times,
                                                    non_streaming_e2e_times,
                                                    streaming_trace_time,
                                                    non_streaming_trace_time,
                                                    streaming_ttft,
                                                    non_streaming_ttft)

            # create stats dir if it doesn't exist
            stats_dir = os.path.join(output_dir, "stats")
            os.makedirs(stats_dir, exist_ok=True)

            # Save statistics to a JSON file
            stats_json_file = os.path.join(
                stats_dir,
                f"completion_time_stats_{title_suffix.replace(' ', '_')}.json")
            with open(stats_json_file, 'w') as f:
                json.dump(stats_dict, f, indent=2)

            # print summary of stats
            print(f"\nStats for run: {run_dir}")
            print(
                f"\tStreaming trace time: {stats_dict['Streaming']['trace_time']:.2f}s"
            )
            print(
                f"\tNon-streaming trace time: {stats_dict['Non-Streaming']['trace_time']:.2f}s"
            )
            print(f"\tE2E trace time speedup: {stats_dict['Speedup']:.2f}x")
            # print data saved in stats dir
            print(f"\nDetailed stats saved in {stats_json_file}")

    # Process data for plots
    data_result = process_trace_time_data(run_dirs, streaming_trace_times,
                                          non_streaming_trace_times)
    if data_result is not None:
        replay_rates, streaming_values, non_streaming_values, speedups, run_labels = data_result

        # Plot speedup vs replay rate
        plot_speedup_vs_replay_rate(replay_rates, speedups, run_labels,
                                    output_dir)

        # Plot raw trace times
        plot_raw_trace_times(replay_rates, streaming_values,
                             non_streaming_values, run_labels, output_dir)
    else:
        print("No data available for trace time plots")

    # Plot preemptions vs replay rate
    plot_preemptions_vs_replay_rate(preemption_data, output_dir)

    # Generate TTFT boxplot analysis
    print("\nGenerating TTFT boxplot analysis...")
    if run_dirs:
        # Extract replay rates from run directories
        replay_rates = []
        for metrics_file in metrics_files:
            run_dir = os.path.dirname(metrics_file)
            replay_rate = extract_replay_rate(run_dir)
            replay_rates.append(replay_rate)

        # Aggregate TTFT data by replay rate
        ttft_data = aggregate_ttft_data(metrics_files, replay_rates)

        # Create TTFT boxplot analysis
        plot_ttft_boxplot_vs_replay_rate(ttft_data, output_dir)
        plot_ttft_trace_combined(ttft_data, output_dir)

        # Generate event duration analysis
        print("\nGenerating event duration analysis...")
        plot_aggregated_event_durations(metrics_files, output_dir)
        plot_event_durations_vs_replay_rate(metrics_files, output_dir)

    print(f"\nAnalysis complete. Results saved to {output_dir}")


def main():
    """Main function to analyze completion times from all metrics files."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze query completion times from metrics files")
    parser.add_argument("--log-dir",
                        type=str,
                        required=True,
                        help="Directory containing run logs")
    parser.add_argument("--output-dir",
                        type=str,
                        default="analysis_output",
                        help="Directory to save analysis results")
    args = parser.parse_args()

    log_dir = args.log_dir
    output_dir = args.output_dir

    # Detect directory structure
    structure_type, scheduler_dirs = detect_scheduler_structure(log_dir)

    if structure_type == "empty":
        print(f"No metrics files found in {log_dir}")
        return
    elif structure_type == "single":
        print(f"Detected single scheduler structure")
        process_single_scheduler(log_dir, output_dir)
    elif structure_type == "multi":
        print(
            f"Detected multi-scheduler structure with {len(scheduler_dirs)} schedulers"
        )
        for scheduler_dir in scheduler_dirs:
            scheduler_name = os.path.basename(scheduler_dir)
            scheduler_output_dir = os.path.join(output_dir, scheduler_name)
            print(f"Processing scheduler: {scheduler_name}")
            process_single_scheduler(scheduler_dir, scheduler_output_dir)
    else:
        print(f"Unknown directory structure in {log_dir}")


if __name__ == "__main__":
    main()
