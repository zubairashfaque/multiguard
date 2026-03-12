"""Dataset loaders for multimodal content (text + image)."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MultimodalDataset(Dataset):
    """Dataset for paired text-image samples with labels.

    Loads samples from the Hateful Memes or similar multimodal datasets
    where each sample has an image path, text caption, and label.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_transform: Any | None = None,
        tokenizer: Any | None = None,
        max_text_length: int = 128,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.samples: list[dict[str, Any]] = []
        logger.info(f"Initialized MultimodalDataset: split={split}, dir={data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("Implement in subclass or after data ingestion pipeline")


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
