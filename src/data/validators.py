"""Data validation utilities for multimodal inputs."""

from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_sample(sample: dict[str, Any]) -> bool:
    """Validate that a multimodal sample has required fields.

    Args:
        sample: Dict with 'text', 'img' (path), and optionally 'label'.

    Returns:
        True if sample is valid.
    """
    required_fields = ["text", "img"]
    for field in required_fields:
        if field not in sample or sample[field] is None:
            logger.warning(f"Missing required field: {field}")
            return False

    if not isinstance(sample["text"], str) or len(sample["text"].strip()) == 0:
        logger.warning("Empty or invalid text field")
        return False

    img_path = Path(sample["img"])
    if not img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        logger.warning(f"Invalid image extension: {img_path.suffix}")
        return False

    return True


def validate_dataset(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter dataset to only valid samples."""
    valid = [s for s in samples if validate_sample(s)]
    removed = len(samples) - len(valid)
    if removed > 0:
        logger.info(f"Removed {removed} invalid samples from {len(samples)} total")
    return valid
