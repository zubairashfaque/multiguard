"""Model loading and caching for serving."""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from src.models.model_factory import MultiGuardModel, build_model
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_model_cache: dict[str, Any] = {}


def load_model(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    config: DictConfig | None = None,
    device: str = "cuda",
) -> MultiGuardModel:
    """Load a trained model for inference.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file).
        config_path: Path to config YAML used during training.
        config: Pre-loaded config (alternative to config_path).
        device: Target device (cuda/cpu).

    Returns:
        Loaded model ready for inference.
    """
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path in _model_cache:
        logger.info(f"Using cached model: {checkpoint_path}")
        return _model_cache[checkpoint_path]

    logger.info(f"Loading model from {checkpoint_path}")

    # Load config
    if config is None and config_path is not None:
        config = load_config(config_path)
    if config is None:
        raise ValueError("Either config or config_path must be provided")

    # Build model architecture
    model = build_model(config)

    # Trigger lazy backbone loading with a dummy forward pass
    dummy_batch = {
        "pixel_values": torch.randn(1, 3, 224, 224),
        "input_ids": torch.randint(0, 30000, (1, 77)),
        "attention_mask": torch.ones(1, 77, dtype=torch.long),
    }
    with torch.no_grad():
        model(dummy_batch)

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()

    _model_cache[checkpoint_path] = model
    logger.info(f"Model loaded and cached on {target_device}")
    return model


def clear_cache() -> None:
    """Clear the model cache."""
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared")
