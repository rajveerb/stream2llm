"""
Clean plotter for recomputation vs swap latency comparison.

This script creates publication-ready plots comparing recomputation and total swap
latencies with the following features:
- Large, readable fonts for all text elements
- Only shows recomputation and total swap curves (no separate swap in/out)
- No command text overlay
- Customizable x-axis spacing (linear until 10k tokens, log-compressed after)
- Enhanced y-axis tick marks showing all powers of 10
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os


def read_json_data(path):
    with open(path, "r") as f:
        return json.load(f)


def filter_data(data, x_min=None, x_max=None, y_min=None, y_max=None):
    recomp_tokens = data["recomp_tokens"]
    recomp_averages = data["recomp_averages"]
    swap_tokens = data["swap_tokens"]
    total_swap_latencies = data["total_swap_latencies"]

    if x_min is not None or x_max is not None or y_min is not None or y_max is not None:
        # Filter recomputation data
        filtered_recomp_tokens = []
        filtered_recomp_averages = []

        for token, avg in zip(recomp_tokens, recomp_averages):
            x_in_range = (x_min is None or token >= x_min) and (x_max is None or token <= x_max)
            y_in_range = (y_min is None or avg >= y_min) and (y_max is None or avg <= y_max)

            if x_in_range and y_in_range:
                filtered_recomp_tokens.append(token)
                filtered_recomp_averages.append(avg)

        # Filter swap data
        filtered_swap_tokens = []
        filtered_total_swap_latencies = []

        for token, total_lat in zip(swap_tokens, total_swap_latencies):
            x_in_range = (x_min is None or token >= x_min) and (x_max is None or token <= x_max)
            y_in_range = (y_min is None or total_lat >= y_min) and (y_max is None or total_lat <= y_max)

            if x_in_range and y_in_range:
                filtered_swap_tokens.append(token)
                filtered_total_swap_latencies.append(total_lat)

        # Use filtered data
        recomp_tokens = filtered_recomp_tokens
        recomp_averages = filtered_recomp_averages
        swap_tokens = filtered_swap_tokens
        total_swap_latencies = filtered_total_swap_latencies

    data["recomp_tokens"] = recomp_tokens
    data["recomp_averages"] = recomp_averages
    data["swap_tokens"] = swap_tokens
    data["total_swap_latencies"] = total_swap_latencies
    return data


def plot_dual_hardware_comparison(recomp_data_1, swap_data_1, title_1,
                                   recomp_data_2, swap_data_2, title_2,
                                   output_dir, output_prefix="",
                                   use_log_scale_y=True,
                                   use_log_scale_x=False,
                                   use_custom_x_spacing=False,
                                   x_min=None, x_max=None,
                                   y_min=None, y_max=None):
    """
    Create a side-by-side (1x2) comparison of recomputation vs swap latencies for two hardware configurations.

    Args:
        recomp_data_1, swap_data_1: Data for first hardware (e.g., H200)
        recomp_data_2, swap_data_2: Data for second hardware (e.g., A40)
        title_1, title_2: Titles for each subplot
        output_dir, output_prefix: Output directory and filename prefix
        use_log_scale_y, use_log_scale_x, use_custom_x_spacing: Scale options
        x_min, x_max, y_min, y_max: Axis limits
    """
    # Extract data for first hardware
    recomp_tokens_1 = sorted([int(k) for k in recomp_data_1["recomputation_latencies"].keys()])
    recomp_averages_1 = [np.mean(recomp_data_1["recomputation_latencies"][str(k)]) for k in recomp_tokens_1]

    swap_tokens_1 = sorted([int(k) for k in swap_data_1["swap_in_latency"].keys()])
    swap_in_latencies_1 = [np.mean(swap_data_1["swap_in_latency"][str(token)]) for token in swap_tokens_1]
    swap_out_latencies_1 = [np.mean(swap_data_1["swap_out_latency"][str(token)]) for token in swap_tokens_1]
    total_swap_latencies_1 = [in_lat + out_lat for in_lat, out_lat in zip(swap_in_latencies_1, swap_out_latencies_1)]

    data_1 = {
        "recomp_tokens": recomp_tokens_1,
        "recomp_averages": recomp_averages_1,
        "swap_tokens": swap_tokens_1,
        "total_swap_latencies": total_swap_latencies_1
    }
    data_1 = filter_data(data_1, x_min, x_max, y_min, y_max)

    # Extract data for second hardware
    recomp_tokens_2 = sorted([int(k) for k in recomp_data_2["recomputation_latencies"].keys()])
    recomp_averages_2 = [np.mean(recomp_data_2["recomputation_latencies"][str(k)]) for k in recomp_tokens_2]

    swap_tokens_2 = sorted([int(k) for k in swap_data_2["swap_in_latency"].keys()])
    swap_in_latencies_2 = [np.mean(swap_data_2["swap_in_latency"][str(token)]) for token in swap_tokens_2]
    swap_out_latencies_2 = [np.mean(swap_data_2["swap_out_latency"][str(token)]) for token in swap_tokens_2]
    total_swap_latencies_2 = [in_lat + out_lat for in_lat, out_lat in zip(swap_in_latencies_2, swap_out_latencies_2)]

    data_2 = {
        "recomp_tokens": recomp_tokens_2,
        "recomp_averages": recomp_averages_2,
        "swap_tokens": swap_tokens_2,
        "total_swap_latencies": total_swap_latencies_2
    }
    data_2 = filter_data(data_2, x_min, x_max, y_min, y_max)

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot first hardware
    ax1.plot(data_1["recomp_tokens"], data_1["recomp_averages"],
             marker='o', linestyle='-', linewidth=3, markersize=12,
             label='Recomputation', color='#ff7f0e')
    ax1.plot(data_1["swap_tokens"], data_1["total_swap_latencies"],
             marker='D', linestyle='-', linewidth=3, markersize=12,
             label='Total Swap', color='#0077b6')
    ax1.set_xlabel('Number of Tokens', fontsize=26, fontweight='bold', labelpad=12)
    ax1.set_ylabel('Latency (ms)', fontsize=26, fontweight='bold', labelpad=12)
    ax1.set_title(title_1, fontsize=28, fontweight='bold')
    ax1.legend(fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=22, pad=8)
    ax1.grid(True, linestyle='--', alpha=0.7)
    # Reduce x-axis tick density for better spacing
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Plot second hardware
    ax2.plot(data_2["recomp_tokens"], data_2["recomp_averages"],
             marker='o', linestyle='-', linewidth=3, markersize=12,
             label='Recomputation', color='#ff7f0e')
    ax2.plot(data_2["swap_tokens"], data_2["total_swap_latencies"],
             marker='D', linestyle='-', linewidth=3, markersize=12,
             label='Total Swap', color='#0077b6')
    ax2.set_xlabel('Number of Tokens', fontsize=26, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Latency (ms)', fontsize=26, fontweight='bold', labelpad=12)
    ax2.set_title(title_2, fontsize=28, fontweight='bold')
    ax2.legend(fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=22, pad=8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    # Reduce x-axis tick density for better spacing
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Set scales for both subplots
    if use_log_scale_y:
        from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator
        for ax in [ax1, ax2]:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
            ax.yaxis.set_minor_formatter(NullFormatter())

    if use_log_scale_x:
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    elif use_custom_x_spacing:
        ax1.set_xscale('symlog', linthresh=10000, linscale=0.5, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax2.set_xscale('symlog', linthresh=10000, linscale=0.5, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Set axis limits if specified
    if y_min is not None or y_max is not None:
        ax1.set_ylim(bottom=y_min, top=y_max)
        ax2.set_ylim(bottom=y_min, top=y_max)
    if x_min is not None or x_max is not None:
        ax1.set_xlim(left=x_min, right=x_max)
        ax2.set_xlim(left=x_min, right=x_max)

    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_prefix}_combined_latency_plot.png"
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Dual hardware comparison plot saved to: {output_path}")


def plot_combined_latencies(recomp_data,
                            swap_data,
                            output_dir,
                            output_prefix="",
                            title="",
                            use_log_scale_y=True,
                            use_log_scale_x=False,
                            use_custom_x_spacing=False,
                            x_min=None,
                            x_max=None,
                            y_min=None,
                            y_max=None):
    """
    Create a clean plot comparing recomputation vs total swap latencies.

    Args:
        recomp_data: Dict with recomputation latency data
        swap_data: Dict with swap latency data
        output_dir: Directory to save the plot
        output_prefix: Prefix for output filename
        title: Plot title
        use_log_scale_y: Use log scale for y-axis (default True)
        use_log_scale_x: Use log scale for x-axis (default False)
        use_custom_x_spacing: Use symlog x-axis (linear until 10k, log after)
        x_min: Minimum x value to display
        x_max: Maximum x value to display
        y_min: Minimum y value to display
        y_max: Maximum y value to display
    """
    # Extract data for recomputation
    recomp_tokens = sorted(
        [int(k) for k in recomp_data["recomputation_latencies"].keys()])
    recomp_averages = [
        np.mean(recomp_data["recomputation_latencies"][str(k)])
        for k in recomp_tokens
    ]

    # Extract data for swap
    swap_tokens = sorted([int(k) for k in swap_data["swap_in_latency"].keys()])
    swap_in_latencies = [
        np.mean(swap_data["swap_in_latency"][str(token)])
        for token in swap_tokens
    ]
    swap_out_latencies = [
        np.mean(swap_data["swap_out_latency"][str(token)])
        for token in swap_tokens
    ]
    total_swap_latencies = [
        in_lat + out_lat
        for in_lat, out_lat in zip(swap_in_latencies, swap_out_latencies)
    ]

    data = {
        "recomp_tokens": recomp_tokens,
        "recomp_averages": recomp_averages,
        "swap_tokens": swap_tokens,
        "total_swap_latencies": total_swap_latencies
    }

    # Filter data based on x-axis and y-axis ranges
    data = filter_data(data, x_min, x_max, y_min, y_max)

    recomp_tokens = data["recomp_tokens"]
    recomp_averages = data["recomp_averages"]
    swap_tokens = data["swap_tokens"]
    total_swap_latencies = data["total_swap_latencies"]

    # Create figure
    plt.figure(figsize=(14, 6))

    # Plot recomputation data
    plt.plot(recomp_tokens,
             recomp_averages,
             marker='o',
             linestyle='-',
             linewidth=3,
             markersize=12,
             label='Recomputation',
             color='#ff7f0e')

    # Plot total swap data only
    plt.plot(swap_tokens,
             total_swap_latencies,
             marker='D',
             linestyle='-',
             linewidth=3,
             markersize=12,
             label='Total Swap',
             color='#0077b6')

    # Add labels and title with larger font sizes
    plt.xlabel('Number of Tokens', fontsize=26, fontweight='bold', labelpad=12)
    plt.ylabel('Latency (ms)', fontsize=26, fontweight='bold', labelpad=12)
    plt.title(title, fontsize=28, fontweight='bold')
    plt.legend(fontsize=24)

    # Increase tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=22, pad=8)

    # Set axis scales based on parameters
    ax = plt.gca()

    # Reduce x-axis tick density for better spacing
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if use_log_scale_y:
        plt.yscale('log')
        # Add more y-axis ticks for better readability - show every power of 10
        from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator
        # Force display of all major ticks from 10^0 to 10^4
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if use_log_scale_x:
        plt.xscale('log')
    elif use_custom_x_spacing:
        # Use symlog scale with linear threshold at 10000
        # This makes the axis wider until 10000, then compressed after
        plt.xscale('symlog', linthresh=10000, linscale=0.5, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        # Set x-axis to start from 0 with proper spacing
        if x_min is None:
            x_min = 0

    # Set axis limits if specified
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)
    if x_min is not None or x_max is not None:
        plt.xlim(left=x_min, right=x_max)

    # Add more space by adjusting the x-axis view to show 0 but start curve further right
    if use_custom_x_spacing and x_min == 0:
        # Manually set better x limits to add space
        current_xlim = ax.get_xlim()
        ax.set_xlim(-1000, current_xlim[1])

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_prefix}_combined_latency_plot.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Combined plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create clean recomputation vs swap latency comparison plots"
    )
    parser.add_argument("--recomp_input",
                        type=str,
                        help="Path to the recomputation latency JSON file")
    parser.add_argument("--swap_input",
                        type=str,
                        help="Path to the swap latency JSON file")
    parser.add_argument("--recomp_input_2",
                        type=str,
                        help="Path to second hardware recomputation latency JSON file (for dual comparison)")
    parser.add_argument("--swap_input_2",
                        type=str,
                        help="Path to second hardware swap latency JSON file (for dual comparison)")
    parser.add_argument("--output_dir",
                        type=str,
                        default=".",
                        help="Directory to save the plots")
    parser.add_argument("--output_prefix",
                        type=str,
                        default="",
                        help="Prefix for output filenames")
    parser.add_argument("--title",
                        type=str,
                        default="",
                        help="Plot title (for single plot mode)")
    parser.add_argument("--title_1",
                        type=str,
                        help="Title for first hardware (dual comparison mode)")
    parser.add_argument("--title_2",
                        type=str,
                        help="Title for second hardware (dual comparison mode)")
    parser.add_argument(
        "--linear_yaxis_scale",
        action="store_true",
        help="Use linear scale for y-axis instead of log scale")
    parser.add_argument("--log_xaxis_scale",
                        action="store_true",
                        help="Use log scale for x-axis")
    parser.add_argument("--custom_x_spacing",
                        action="store_true",
                        help="Use custom x-axis spacing (wider until 10k tokens, compressed after)")
    parser.add_argument(
        "--x_min",
        type=int,
        help="Minimum x-axis value (number of tokens) to include in plot")
    parser.add_argument(
        "--x_max",
        type=int,
        help="Maximum x-axis value (number of tokens) to include in plot")
    parser.add_argument(
        "--y_min",
        type=float,
        help="Minimum y-axis value (latency in ms) to include in plot")
    parser.add_argument(
        "--y_max",
        type=float,
        help="Maximum y-axis value (latency in ms) to include in plot")
    args = parser.parse_args()

    # Check if running in dual hardware comparison mode
    if args.recomp_input_2 and args.swap_input_2:
        if not args.title_1 or not args.title_2:
            print("Error: --title_1 and --title_2 are required for dual hardware comparison mode")
            return

        recomp_data_1 = read_json_data(args.recomp_input)
        swap_data_1 = read_json_data(args.swap_input)
        recomp_data_2 = read_json_data(args.recomp_input_2)
        swap_data_2 = read_json_data(args.swap_input_2)

        plot_dual_hardware_comparison(recomp_data_1, swap_data_1, args.title_1,
                                      recomp_data_2, swap_data_2, args.title_2,
                                      args.output_dir,
                                      args.output_prefix,
                                      use_log_scale_y=not args.linear_yaxis_scale,
                                      use_log_scale_x=args.log_xaxis_scale,
                                      use_custom_x_spacing=args.custom_x_spacing,
                                      x_min=args.x_min,
                                      x_max=args.x_max,
                                      y_min=args.y_min,
                                      y_max=args.y_max)
    else:
        # Single hardware plot mode
        if not args.recomp_input or not args.swap_input:
            print("Error: --recomp_input and --swap_input are required for single plot mode")
            return

        recomp_data = read_json_data(args.recomp_input)
        swap_data = read_json_data(args.swap_input)

        plot_combined_latencies(recomp_data,
                                swap_data,
                                args.output_dir,
                                args.output_prefix,
                                args.title,
                                use_log_scale_y=not args.linear_yaxis_scale,
                                use_log_scale_x=args.log_xaxis_scale,
                                use_custom_x_spacing=args.custom_x_spacing,
                                x_min=args.x_min,
                                x_max=args.x_max,
                                y_min=args.y_min,
                                y_max=args.y_max)


if __name__ == "__main__":
    main()
