"""Dataset loaders for multimodal content (text + image)."""

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
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
        raise NotImplementedError("Use HatefulMemesDataset or implement in subclass")


class HatefulMemesDataset(MultimodalDataset):
    """Dataset loader for the Facebook Hateful Memes Challenge.

    Reads JSONL annotation files and loads paired image-text samples.
    Each line in the JSONL has: {"id", "img", "text", "label"}.
    """

    SPLIT_MAP = {
        "train": "train.jsonl",
        "val": "dev.jsonl",
        "dev": "dev.jsonl",
        "test": "test.jsonl",
    }

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_transform: Any | None = None,
        tokenizer: Any | None = None,
        max_text_length: int = 77,
    ) -> None:
        super().__init__(data_dir, split, image_transform, tokenizer, max_text_length)
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Load JSONL annotations for the specified split."""
        jsonl_name = self.SPLIT_MAP.get(self.split)
        if jsonl_name is None:
            raise ValueError(
                f"Unknown split: {self.split}. Available: {list(self.SPLIT_MAP.keys())}"
            )

        jsonl_path = self.data_dir / jsonl_name
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {jsonl_path}")

        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and return a single multimodal sample.

        Returns:
            Dict with pixel_values [C, H, W], input_ids [seq_len],
            attention_mask [seq_len], and label (int).
        """
        sample = self.samples[idx]

        # Load and transform image
        img_path = self.data_dir / sample["img"]
        image = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            pixel_values = self.image_transform(image)
        else:
            from torchvision import transforms

            pixel_values = transforms.ToTensor()(image)

        # Tokenize text
        text = sample["text"]
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_text_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_length, dtype=torch.long)

        result = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": sample["id"],
        }

        # Label may not be present in test split
        if "label" in sample:
            result["labels"] = torch.tensor(sample["label"], dtype=torch.long)

        return result


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
