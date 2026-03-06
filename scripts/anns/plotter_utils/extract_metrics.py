import numpy as np


def create_metrics(streaming_times: np.ndarray,
                   non_streaming_times: np.ndarray) -> str:
    """Create metrics for ANNS query e2e completion times."""
    metrics_text = ""

    if len(streaming_times) > 0:
        streaming_mean = np.mean(streaming_times)
        streaming_median = np.median(streaming_times)
        streaming_p25 = np.percentile(streaming_times, 25)
        streaming_p75 = np.percentile(streaming_times, 75)
        streaming_p90 = np.percentile(streaming_times, 90)
        streaming_min = np.min(streaming_times)
        streaming_max = np.max(streaming_times)
        streaming_std = np.std(streaming_times)

        metrics_text += f"Streaming Metrics: "
        metrics_text += f"Mean: {streaming_mean:.2f}s | "
        metrics_text += f"Median: {streaming_median:.2f}s | "
        metrics_text += f"25th %: {streaming_p25:.2f}s | "
        metrics_text += f"75th %: {streaming_p75:.2f}s | "
        metrics_text += f"90th %: {streaming_p90:.2f}s | "
        metrics_text += f"Min: {streaming_min:.2f}s | "
        metrics_text += f"Max: {streaming_max:.2f}s | "
        metrics_text += f"Std: {streaming_std:.2f}s"

    if len(non_streaming_times) > 0:
        non_streaming_mean = np.mean(non_streaming_times)
        non_streaming_median = np.median(non_streaming_times)
        non_streaming_p25 = np.percentile(non_streaming_times, 25)
        non_streaming_p75 = np.percentile(non_streaming_times, 75)
        non_streaming_p90 = np.percentile(non_streaming_times, 90)
        non_streaming_min = np.min(non_streaming_times)
        non_streaming_max = np.max(non_streaming_times)
        non_streaming_std = np.std(non_streaming_times)

        metrics_text += f"\nNon-Streaming Metrics: "
        metrics_text += f"Mean: {non_streaming_mean:.2f}s | "
        metrics_text += f"Median: {non_streaming_median:.2f}s | "
        metrics_text += f"25th %: {non_streaming_p25:.2f}s | "
        metrics_text += f"75th %: {non_streaming_p75:.2f}s | "
        metrics_text += f"90th %: {non_streaming_p90:.2f}s | "
        metrics_text += f"Min: {non_streaming_min:.2f}s | "
        metrics_text += f"Max: {non_streaming_max:.2f}s | "
        metrics_text += f"Std: {non_streaming_std:.2f}s"

    speedup_text = None
    # Add speedup metrics if both datasets are available
    if len(streaming_times) > 0 and len(non_streaming_times) > 0:
        mean_speedup = np.mean(non_streaming_times) / np.mean(streaming_times)
        median_speedup = np.median(non_streaming_times) / np.median(
            streaming_times)
        p25_speedup = np.percentile(non_streaming_times, 25) / np.percentile(
            streaming_times, 25)
        p75_speedup = np.percentile(non_streaming_times, 75) / np.percentile(
            streaming_times, 75)
        p90_speedup = np.percentile(non_streaming_times, 90) / np.percentile(
            streaming_times, 90)

        speedup_text = f"Speedup (Non-streaming / Streaming): "
        speedup_text += f"Mean: {mean_speedup:.2f}x | "
        speedup_text += f"Median: {median_speedup:.2f}x | "
        speedup_text += f"25th %: {p25_speedup:.2f}x | "
        speedup_text += f"75th %: {p75_speedup:.2f}x | "
        speedup_text += f"90th %: {p90_speedup:.2f}x"
    return metrics_text, speedup_text
