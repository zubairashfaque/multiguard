"""Model evaluator for running benchmarks and computing metrics."""

from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.training.metrics import compute_classification_metrics
from src.utils.device import get_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluate a trained multimodal model on test data.

    Runs inference, computes metrics, and generates reports.
    """

    def __init__(self, model: torch.nn.Module, config: DictConfig | None = None) -> None:
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run evaluation on a dataloader.

        Args:
            dataloader: Test or validation DataLoader.

        Returns:
            Dict of evaluation metrics.
        """
        all_preds = []
        all_labels = []

        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = self.model(batch)
            logits = outputs["logits"]
            all_preds.append(logits.cpu())
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu())

        predictions = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0) if all_labels else None

        if labels is not None:
            metrics = compute_classification_metrics(predictions, labels)
            logger.info(f"Evaluation results: {metrics}")
            return metrics

        logger.warning("No labels found in data, returning empty metrics")
        return {}
