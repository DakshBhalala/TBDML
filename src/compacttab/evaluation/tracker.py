"""Resource, latency, and memory profiling utilities."""

import os
import time
from typing import Any, Callable, Dict, Optional, Tuple
import psutil
import torch


class ResourceTracker:
    """Profiles computational efficiency, memory consumption, and inference speed."""

    @staticmethod
    def get_peak_ram_mb() -> float:
        """Get peak Resident Set Size (RSS) in megabytes for current process."""
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))

    @staticmethod
    def get_peak_vram_mb() -> float:
        """Get peak allocated CUDA memory in megabytes."""
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        return 0.0

    @staticmethod
    def reset_vram_peak() -> None:
        """Reset CUDA memory peak stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def measure_latency_and_throughput(
        predict_fn: Callable[[], Any],
        n_samples: int,
        warmup_runs: int = 2,
        benchmark_runs: int = 5,
    ) -> Tuple[float, float]:
        """Measure inference latency (ms/sample) and throughput (samples/sec)."""
        # Warmup
        for _ in range(warmup_runs):
            predict_fn()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(benchmark_runs):
            predict_fn()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time
        avg_batch_time = total_time / benchmark_runs

        latency_ms_per_sample = (avg_batch_time / max(1, n_samples)) * 1000.0
        throughput_samples_per_sec = n_samples / max(1e-6, avg_batch_time)

        return float(latency_ms_per_sample), float(throughput_samples_per_sec)
