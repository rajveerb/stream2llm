#!/usr/bin/env python3
"""
Generate scheduler-specific ANNS config files for different QPS loads.
"""

import os
import yaml
from typing import Dict, Any


def create_anns_config(scheduler: str,
                       arrival_time: float,
                       hardware: str = "H200",
                       query_range: int = 500) -> Dict[str, Any]:
    """
    Create ANNS config dictionary for a specific scheduler and arrival time.

    Args:
        scheduler: Scheduler type (default_vllm, fcfs, lcas, mcps)
        arrival_time: Average inter-arrival time in seconds
        hardware: Hardware type (H200, H100)
        query_range: Number of queries to process

    Returns:
        Configuration dictionary
    """
    arrival_time_str = str(arrival_time).replace('.', '_')

    config = {
        'data_dir': 'data/anns',
        'trace_dir': 'data/anns/res',
        'query_trace_map': 'data/anns/query_trace_map_5k.json',
        'config_for_run':
        f'data/run_log/anns/{hardware}_enhanced_schedulers_v1_full/{scheduler}',
        'use_saved_poisson_delays':
        f'experiments/anns/configs/{hardware}_enhanced_schedulers_v1_full/poisson_delays_{arrival_time_str}.txt',
        'prefix_caching_flag': False,
        'scheduler': scheduler,
        'log_stats': True,
        'query_range': query_range,
        'model': {
            'name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'max_token_budget': 8192,
            'gpu_memory_utilization': 0.8,
            'tensor_parallel_size': 2,
            'seed': 42
        },
        'replay': {
            'poisson_avg_arrival_time': arrival_time,
            'max_output_tokens': 1,
            'delay_multiplier': 1.0,
            'template':
            'Answer the question based only on the provided context. If the answer is not in the context, say "I don\'t know."\nContext:',
            'end_template': '\n\nQuestion:'
        },
        'logging': {
            'verbose_console': False,
            'suppress_vllm_logs': True
        },
        'experiments': [{
            'concurrency': True,
            'stream': True
        }]
    }

    # Add non-stream experiment only for default_vllm
    if scheduler == 'default_vllm':
        config['experiments'].append({'concurrency': True, 'stream': False})

    # Add scheduler-specific performance model configurations
    if scheduler != 'default_vllm':
        config.update({
            'use_recomputation_latency_predictor': 1,
            'use_saved_recomputation_data_path':
            'data/perf_model/recomputation/H200_tp2_recomputation_latency.json',
            'use_swap_latency_predictor': 1,
            'use_saved_swap_latency_data_path':
            'data/perf_model/swap/H200_tp2_swap_kernel_latency.json',
            'use_swap_kernel': 1
        })

    return config


def save_config(config: Dict[str, Any],
                scheduler: str,
                arrival_time: float,
                output_dir: str,
                hardware: str = "H200"):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        scheduler: Scheduler type
        arrival_time: Average inter-arrival time in seconds
        output_dir: Output directory
    """
    arrival_time_str = str(arrival_time).replace('.', '_')
    filename = f"config_replay_anns_replay_{arrival_time_str}.yaml"
    scheduler_dir = os.path.join(output_dir, scheduler)
    os.makedirs(scheduler_dir, exist_ok=True)

    filepath = os.path.join(scheduler_dir, filename)

    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created {scheduler}/{filename}")


def main():
    # Configuration parameters
    schedulers = ['default_vllm', 'fcfs', 'lcas', 'mcps']
    arrival_times = [4, 2, 1, 0.5, 0.25, 0.125,
                     0.0625]  # Inter-arrival times in seconds
    query_range = 500
    hardware_types = ['H200', 'H100']

    total_configs = 0

    for hardware in hardware_types:
        output_dir = f'experiments/anns/configs/{hardware}_enhanced_schedulers_v1_full'

        print(f"Generating ANNS scheduler configs for {hardware}")
        print(f"Schedulers: {schedulers}")
        print(f"Arrival times: {arrival_times}")
        print(f"Query range: {query_range}")
        print(f"Output directory: {output_dir}")
        print()

        for scheduler in schedulers:
            print(f"Generating {hardware} configs for scheduler: {scheduler}")
            for arrival_time in arrival_times:
                config = create_anns_config(scheduler, arrival_time, hardware,
                                            query_range)
                save_config(config, scheduler, arrival_time, output_dir,
                            hardware)
                total_configs += 1
            print()

    print(f"Generated {total_configs} configuration files total")


if __name__ == "__main__":
    main()
