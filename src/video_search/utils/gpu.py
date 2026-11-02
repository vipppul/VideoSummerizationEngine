"""
GPU memory and device management utilities.
"""

from typing import Optional
import torch


def get_optimal_device(requested_device: Optional[str] = "cuda:0") -> torch.device:
    """
    Returns optimal device. If requested CUDA device is unavailable,
    falls back safely to CPU.
    """
    if requested_device and requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested_device)
        return torch.device("cpu")
    return torch.device(requested_device or "cpu")


def clear_gpu_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

