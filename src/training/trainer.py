"""Base training loop for multimodal models."""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.utils.device import get_device, log_vram_usage
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MultimodalTrainer:
    """Base trainer for multimodal classification and retrieval.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        config: DictConfig | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        self.global_step = 0
        self.best_metric = 0.0

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict of training metrics for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = outputs.get("loss", torch.tensor(0.0))
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f}")
        log_vram_usage(f"epoch_{epoch}")
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation.

        Returns:
            Dict of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = self.model(batch)
            loss = outputs.get("loss", torch.tensor(0.0))
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Validation: val_loss={avg_loss:.4f}")
        return {"val_loss": avg_loss}

    def save_checkpoint(self, path: str | Path, metrics: dict[str, float]) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "metrics": metrics,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")

    def train(self, num_epochs: int = 10) -> None:
        """Full training loop."""
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch}/{num_epochs}: {train_metrics} | {val_metrics}")
