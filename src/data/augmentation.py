"""Data augmentation for images and text."""

from typing import Any

from torchvision import transforms

from src.utils.logging import get_logger

logger = get_logger(__name__)

# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_image_transforms(
    config: dict[str, Any] | None = None, split: str = "train"
) -> transforms.Compose:
    """Build image augmentation pipeline from config.

    Args:
        config: Augmentation config dict (optional, uses sensible defaults).
        split: Dataset split (train/val/test).

    Returns:
        torchvision transform pipeline.
    """
    config = config or {}
    image_size = config.get("image_size", 224)

    if split == "train":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=config.get("crop_scale", (0.8, 1.0)),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=config.get("brightness", 0.2),
                    contrast=config.get("contrast", 0.2),
                    saturation=config.get("saturation", 0.2),
                    hue=config.get("hue", 0.1),
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
                transforms.RandomErasing(p=config.get("erasing_p", 0.25)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    logger.info(f"Built image transforms for split={split}")
    return transform
