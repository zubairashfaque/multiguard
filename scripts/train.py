#!/usr/bin/env python3
"""Training entry point for MultiGuard.

Usage:
    python scripts/train.py --config configs/train/baseline.yaml
    python scripts/train.py --config configs/train/fusion_ablation.yaml
    python scripts/train.py --config configs/train/distillation.yaml
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def train_baseline(config) -> None:
    """Run baseline multimodal training (late fusion)."""
    logger.info("Starting baseline training...")
    raise NotImplementedError("Implement after data pipeline and model factory are ready")


def train_fusion_ablation(config) -> None:
    """Run fusion ablation study (cross-attention, tensor fusion)."""
    logger.info("Starting fusion ablation training...")
    raise NotImplementedError("Implement after fusion modules are tested")


def train_distillation(config) -> None:
    """Run knowledge distillation training."""
    logger.info("Starting distillation training...")
    raise NotImplementedError("Implement after teacher model is trained")


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard training")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        wandb_enabled=config.get("logging", {}).get("wandb", {}).get("enabled", False),
        wandb_project=config.get("logging", {}).get("wandb", {}).get("project"),
        wandb_entity=config.get("logging", {}).get("wandb", {}).get("entity"),
    )
    set_seed(config.get("project", {}).get("seed", 42))

    training_type = config.training.get("type", "baseline")
    logger.info(f"Starting {training_type} training...")

    if training_type == "baseline":
        train_baseline(config)
    elif training_type == "fusion_ablation":
        train_fusion_ablation(config)
    elif training_type == "distillation":
        train_distillation(config)
    else:
        raise ValueError(f"Unknown training type: {training_type}")


if __name__ == "__main__":
    main()
