"""Data augmentation for images and text."""

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_image_transforms(config: dict[str, Any], split: str = "train") -> Any:
    """Build image augmentation pipeline from config.

    Args:
        config: Augmentation config dict.
        split: Dataset split (train/val/test).

    Returns:
        torchvision or albumentations transform pipeline.
    """
    raise NotImplementedError("Implement after torchvision/albumentations integration")


def build_text_augmentation(config: dict[str, Any], split: str = "train") -> Any:
    """Build text augmentation pipeline from config.

    Args:
        config: Augmentation config dict.
        split: Dataset split (train/val/test).

    Returns:
        Text augmentation callable.
    """
    raise NotImplementedError("Implement after NLP augmentation library integration")
