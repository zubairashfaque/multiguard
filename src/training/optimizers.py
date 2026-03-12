"""Optimizer and scheduler factories."""

from typing import Any

import torch.nn as nn
import torch.optim as optim

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> optim.Optimizer:
    """Build optimizer from config.

    Args:
        model: Model whose parameters to optimize.
        name: Optimizer name (adam, adamw, sgd).
        lr: Learning rate.
        weight_decay: Weight decay coefficient.

    Returns:
        Configured optimizer.
    """
    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")

    optimizer = optimizers[name](params, lr=lr, weight_decay=weight_decay, **kwargs)
    logger.info(f"Optimizer: {name}, lr={lr}, weight_decay={weight_decay}")
    return optimizer


def build_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    num_training_steps: int = 1000,
    warmup_ratio: float = 0.1,
) -> optim.lr_scheduler.LRScheduler:
    """Build learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        name: Scheduler name (cosine, linear, step).
        num_training_steps: Total number of training steps.
        warmup_ratio: Fraction of steps for warmup.

    Returns:
        Configured scheduler.
    """
    warmup_steps = int(num_training_steps * warmup_ratio)

    if name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps - warmup_steps
        )
    elif name == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
    elif name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_training_steps // 3)
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    logger.info(f"Scheduler: {name}, warmup_steps={warmup_steps}")
    return scheduler
