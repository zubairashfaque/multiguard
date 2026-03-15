"""Hateful Memes Challenge benchmark evaluation (AUROC / F1)."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.augmentation import build_image_transforms
from src.data.loaders import HatefulMemesDataset, build_dataloader
from src.training.metrics import compute_classification_metrics
from src.utils.device import get_device
from src.utils.logging import get_logger
from src.utils.registry import BENCHMARK_REGISTRY

logger = get_logger(__name__)


@BENCHMARK_REGISTRY.register("hateful_memes")
class HatefulMemesBenchmark:
    """Facebook Hateful Memes Challenge benchmark.

    Evaluates multimodal hate speech detection on ~10K memes.
    Primary metric: AUROC. Secondary: F1, accuracy.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/hateful_memes",
        split: str = "val",
        tokenizer_name: str = "roberta-base",
        max_text_length: int = 77,
        batch_size: int = 16,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer_name = tokenizer_name
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.primary_metric = "auroc"

    def _build_dataloader(self) -> DataLoader:
        """Build evaluation dataloader."""
        transforms = build_image_transforms({"image_size": 224}, split="val")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        dataset = HatefulMemesDataset(
            data_dir=self.data_dir,
            split=self.split,
            image_transform=transforms,
            tokenizer=tokenizer,
            max_text_length=self.max_text_length,
        )

        return build_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @torch.no_grad()
    def run(self, model: torch.nn.Module) -> dict[str, float]:
        """Run the Hateful Memes benchmark.

        Args:
            model: Trained multimodal model.

        Returns:
            Dict of benchmark metrics (auroc, f1, accuracy, precision, recall).
        """
        device = get_device()
        model = model.to(device)
        model.eval()

        dataloader = self._build_dataloader()
        all_logits = []
        all_labels = []

        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch)
            all_logits.append(outputs["logits"].cpu())
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels) if all_labels else None

        if labels is None:
            logger.warning("No labels in test split — cannot compute metrics")
            return {}

        metrics = compute_classification_metrics(logits, labels)
        logger.info(
            f"Hateful Memes [{self.split}]: "
            f"AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}"
        )
        return metrics
