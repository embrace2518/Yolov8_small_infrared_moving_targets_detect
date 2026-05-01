"""
Unified timing utilities for inference and training.

Usage:
    from utils.timing import CUDATimer
    timer = CUDATimer()
    with timer:
        model(images)
    print(f"Inference took {timer.elapsed_ms:.2f} ms")
"""

import time
import torch


class CUDATimer:
    """Context manager and manual timer with CUDA synchronization."""

    def __init__(self, cuda_sync: bool = True):
        self.cuda_sync = cuda_sync and torch.cuda.is_available()
        self._start: float = 0.0
        self._end: float = 0.0

    def start(self) -> None:
        if self.cuda_sync:
            torch.cuda.synchronize()
        self._start = time.perf_counter()

    def stop(self) -> None:
        if self.cuda_sync:
            torch.cuda.synchronize()
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return self._end - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class FPSMetrics:
    """Compute FPS and latency metrics from a list of per-batch timings."""

    def __init__(self, batch_times: list[float], total_images: int, batch_size: int):
        total_time = sum(batch_times)
        avg_latency_ms = (total_time / total_images * 1000.0) if total_images > 0 else 0.0
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

        self.total_images = total_images
        self.batch_size = batch_size
        self.total_inference_time_s = float(f"{total_time:.3f}")
        self.avg_latency_ms = float(f"{avg_latency_ms:.2f}")
        self.fps = float(f"{fps:.2f}")

    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "batch_size": self.batch_size,
            "total_inference_time_s": self.total_inference_time_s,
            "avg_latency_ms": self.avg_latency_ms,
            "fps": self.fps,
        }
