"""Model loading and caching for serving."""

from pathlib import Path
from typing import Any

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

_model_cache: dict[str, Any] = {}


def load_model(checkpoint_path: str | Path, device: str = "cuda") -> Any:
    """Load a trained model for inference.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Target device (cuda/cpu).

    Returns:
        Loaded model ready for inference.
    """
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path in _model_cache:
        logger.info(f"Using cached model: {checkpoint_path}")
        return _model_cache[checkpoint_path]

    logger.info(f"Loading model from {checkpoint_path}")
    raise NotImplementedError("Implement after model architecture is finalized")


def clear_cache() -> None:
    """Clear the model cache."""
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared")
