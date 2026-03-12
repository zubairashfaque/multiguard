"""Training callbacks for logging, checkpointing, and early stopping."""

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return True

        return False


class WandbCallback:
    """Log metrics to Weights & Biases."""

    def __init__(self) -> None:
        self._wandb: Any = None

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to W&B."""
        if self._wandb is None:
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                return
        self._wandb.log(metrics, step=step)
