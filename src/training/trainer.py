"""Base training loop for multimodal models."""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping, WandbCallback
from src.training.metrics import compute_classification_metrics
from src.utils.device import get_device, log_vram_usage
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MultimodalTrainer:
    """Base trainer for multimodal classification and retrieval.

    Handles training loop, validation, checkpointing, mixed precision,
    gradient accumulation, early stopping, and W&B logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        config: DictConfig | None = None,
        loss_fn: torch.nn.Module | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.device = get_device()
        self.model.to(self.device)
        self.global_step = 0
        self.best_metric = float("inf")

        # Loss function
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()

        # Training config
        training_cfg = config.get("training", {}) if config else {}
        self.grad_accum_steps = training_cfg.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
        self.fp16 = training_cfg.get("fp16", False)
        self.logging_steps = training_cfg.get("logging_steps", 50)
        self.eval_steps = training_cfg.get("eval_steps", 0)
        self.save_steps = training_cfg.get("save_steps", 0)
        self.output_dir = Path(training_cfg.get("output_dir", "models/checkpoints"))

        # Mixed precision
        self.scaler = (
            torch.amp.GradScaler("cuda") if self.fp16 and self.device.type == "cuda" else None
        )

        # Callbacks
        self.early_stopping = EarlyStopping(patience=training_cfg.get("patience", 5), mode="min")
        self.wandb_cb = WandbCallback()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch with gradient accumulation and mixed precision.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict of training metrics for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass with optional mixed precision
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs["logits"], batch["labels"])
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                    self.global_step += 1
            else:
                outputs = self.model(batch)
                loss = self.loss_fn(outputs["logits"], batch["labels"])
                loss = loss / self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                    self.global_step += 1

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            # Step-level logging
            if self.global_step > 0 and self.global_step % self.logging_steps == 0:
                step_loss = total_loss / num_batches
                self.wandb_cb.log(
                    {"train/loss": step_loss, "train/lr": self.optimizer.param_groups[0]["lr"]},
                    self.global_step,
                )

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f}")
        log_vram_usage(f"epoch_{epoch}")
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation with metric computation.

        Returns:
            Dict of validation metrics (loss, accuracy, f1, auroc).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_logits = []
        all_labels = []

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs["logits"], batch["labels"])
            else:
                outputs = self.model(batch)
                loss = self.loss_fn(outputs["logits"], batch["labels"])

            total_loss += loss.item()
            num_batches += 1
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(batch["labels"].cpu())

        avg_loss = total_loss / max(num_batches, 1)

        # Compute classification metrics
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        metrics = compute_classification_metrics(logits, labels)
        metrics["val_loss"] = avg_loss

        logger.info(
            f"Validation: loss={avg_loss:.4f}, acc={metrics.get('accuracy', 0):.4f}, "
            f"f1={metrics.get('f1', 0):.4f}, auroc={metrics.get('auroc', 0):.4f}"
        )
        return metrics

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

    def train(self, num_epochs: int = 10) -> dict[str, float]:
        """Full training loop with validation, early stopping, and checkpointing.

        Args:
            num_epochs: Number of epochs to train.

        Returns:
            Best validation metrics.
        """
        best_metrics = {}

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            # Log to W&B
            all_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            all_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            all_metrics["epoch"] = epoch
            self.wandb_cb.log(all_metrics, self.global_step)

            logger.info(f"Epoch {epoch}/{num_epochs}: {train_metrics} | {val_metrics}")

            # Save best model
            val_loss = val_metrics["val_loss"]
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                best_metrics = val_metrics
                self.save_checkpoint(self.output_dir / "best_model.pt", val_metrics)
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save final checkpoint
        self.save_checkpoint(self.output_dir / "last_model.pt", val_metrics)
        return best_metrics
