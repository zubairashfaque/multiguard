"""Device detection and VRAM monitoring utilities."""

from dataclasses import dataclass

import torch
from loguru import logger


@dataclass
class DeviceInfo:
    """Container for device information."""

    device: torch.device
    device_name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_available: bool


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def get_device_info() -> DeviceInfo:
    """Get detailed device information including VRAM stats."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        vram_free = (
            torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)
        ) / (1024**3)
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        vram_total = 0.0
        vram_free = 0.0

    return DeviceInfo(
        device=device,
        device_name=device_name,
        vram_total_gb=round(vram_total, 2),
        vram_free_gb=round(vram_free, 2),
        cuda_available=cuda_available,
    )


def log_vram_usage(tag: str = "") -> None:
    """Log current VRAM usage. Useful for monitoring during training."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    logger.info(
        f"[VRAM {tag}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Total: {total:.2f}GB"
    )
