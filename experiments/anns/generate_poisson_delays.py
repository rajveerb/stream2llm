#!/usr/bin/env python3
"""
Generate Poisson arrival delays for ANNS query replay experiments.
Creates delay files for different QPS loads with exactly 500 queries each.
"""

import argparse
import numpy as np
from datetime import datetime
import os


def generate_poisson_delays(avg_arrival_time: float,
                            num_queries: int = 500,
                            seed: int = 42) -> list:
    """
    Generate Poisson arrival delays.

    Args:
        avg_arrival_time: Average time between arrivals (seconds)
        num_queries: Number of queries to generate delays for
        seed: Random seed for reproducibility

    Returns:
        List of inter-arrival delays in seconds
    """
    np.random.seed(seed)
    delays = [
        np.random.exponential(avg_arrival_time) for _ in range(num_queries)
    ]
    return delays


def save_poisson_delays(delays: list, avg_arrival_time: float,
                        output_dir: str):
    """
    Save Poisson delays to file in crawler-compatible format.

    Args:
        delays: List of delay values
        avg_arrival_time: Average inter-arrival time in seconds
        output_dir: Directory to save file
    """
    # Create filename based on arrival time
    arrival_time_str = str(avg_arrival_time).replace('.', '_')
    filename = f"poisson_delays_{arrival_time_str}.txt"
    filepath = os.path.join(output_dir, filename)

    # Generate header comment
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    qps = 1.0 / avg_arrival_time

    header = f"""# Poisson Arrival Times Generated
# Generated on: {timestamp}
# Arrival time: {avg_arrival_time} seconds (avg inter-arrival time)
# QPS: {qps:.4f} queries per second
# Number of queries: {len(delays)}
# Random seed: 42
# Format: One delay value per line (seconds)
#"""

    # Save file
    with open(filepath, 'w') as f:
        f.write(header + '\n')
        for delay in delays:
            f.write(f"{delay:.6f}\n")

    print(
        f"Generated {filename} with {len(delays)} delays for arrival time {avg_arrival_time}s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate Poisson delays for ANNS experiments")
    parser.add_argument("--output-dir",
                        default="experiments/anns/configs",
                        help="Output directory for delay files")
    parser.add_argument("--num-queries",
                        type=int,
                        default=500,
                        help="Number of queries per delay file")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Arrival times matching crawler configuration
    arrival_times = [4, 2, 1, 0.5, 0.25, 0.125, 0.0625]  # seconds

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating Poisson delays for {len(arrival_times)} arrival times")
    print(f"Number of queries per file: {args.num_queries}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print()

    for avg_arrival_time in arrival_times:
        delays = generate_poisson_delays(avg_arrival_time, args.num_queries,
                                         args.seed)
        save_poisson_delays(delays, avg_arrival_time, args.output_dir)

    print(f"\nGenerated delay files for {len(arrival_times)} arrival times")


if __name__ == "__main__":
    main()
