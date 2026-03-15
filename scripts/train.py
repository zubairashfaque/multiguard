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

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.data.augmentation import build_image_transforms  # noqa: E402
from src.data.loaders import HatefulMemesDataset, build_dataloader  # noqa: E402
from src.models.model_factory import build_model  # noqa: E402
from src.training.optimizers import build_optimizer, build_scheduler  # noqa: E402
from src.training.trainer import MultimodalTrainer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _build_datasets(config):
    """Build train and validation datasets from config."""
    data_cfg = config.data
    model_cfg = config.model
    data_dir = Path(config.paths.data_dir) / "raw" / "hateful_memes"

    # Image transforms
    aug_cfg = {"image_size": data_cfg.get("image_size", 224)}
    train_transforms = build_image_transforms(aug_cfg, split="train")
    val_transforms = build_image_transforms(aug_cfg, split="val")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.text_backbone)
    max_text_length = data_cfg.get("max_text_length", 77)

    train_dataset = HatefulMemesDataset(
        data_dir=data_dir,
        split=data_cfg.get("train_split", "train"),
        image_transform=train_transforms,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )

    val_dataset = HatefulMemesDataset(
        data_dir=data_dir,
        split=data_cfg.get("val_split", "dev"),
        image_transform=val_transforms,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )

    return train_dataset, val_dataset


def _build_loaders(config, train_dataset, val_dataset):
    """Build DataLoaders from datasets."""
    training_cfg = config.training
    device_cfg = config.get("device", {})
    batch_size = training_cfg.get("batch_size", 32)
    num_workers = device_cfg.get("num_workers", 4)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def train_baseline(config) -> None:
    """Run baseline multimodal training (late fusion)."""
    logger.info("Starting baseline training...")

    # Data
    train_dataset, val_dataset = _build_datasets(config)
    train_loader, val_loader = _build_loaders(config, train_dataset, val_dataset)
    logger.info(f"Data: {len(train_dataset)} train, {len(val_dataset)} val")

    # Model
    model = build_model(config)

    # Optimizer & scheduler
    training_cfg = config.training
    optimizer = build_optimizer(
        model,
        name="adamw",
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )
    num_training_steps = len(train_loader) * training_cfg.get("epochs", 15)
    scheduler = build_scheduler(
        optimizer,
        name=training_cfg.get("lr_scheduler", "cosine"),
        num_training_steps=num_training_steps,
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
    )

    # Train
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    best_metrics = trainer.train(num_epochs=training_cfg.get("epochs", 15))
    logger.info(f"Training complete. Best metrics: {best_metrics}")


def train_fusion_ablation(config) -> None:
    """Run fusion ablation study (cross-attention, tensor fusion)."""
    logger.info("Starting fusion ablation training...")

    # Same pipeline as baseline — config determines the fusion strategy
    train_dataset, val_dataset = _build_datasets(config)
    train_loader, val_loader = _build_loaders(config, train_dataset, val_dataset)
    logger.info(f"Data: {len(train_dataset)} train, {len(val_dataset)} val")

    model = build_model(config)

    training_cfg = config.training
    optimizer = build_optimizer(
        model,
        name="adamw",
        lr=training_cfg.get("learning_rate", 5e-4),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )
    num_training_steps = len(train_loader) * training_cfg.get("epochs", 20)
    scheduler = build_scheduler(
        optimizer,
        name=training_cfg.get("lr_scheduler", "cosine"),
        num_training_steps=num_training_steps,
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
    )

    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    best_metrics = trainer.train(num_epochs=training_cfg.get("epochs", 20))
    logger.info(f"Fusion ablation complete. Best metrics: {best_metrics}")


def train_distillation(config) -> None:
    """Run knowledge distillation training."""
    from src.training.distillation_trainer import DistillationTrainer

    logger.info("Starting distillation training...")

    train_dataset, val_dataset = _build_datasets(config)
    train_loader, val_loader = _build_loaders(config, train_dataset, val_dataset)

    # Load teacher model from checkpoint
    teacher_cfg = config.teacher
    teacher_ckpt = Path(teacher_cfg.checkpoint)
    if not teacher_ckpt.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_ckpt}. "
            "Train a baseline or fusion model first."
        )

    # Build teacher from the baseline config (matching checkpoint architecture)
    teacher_config_path = teacher_cfg.get("config", "configs/train/baseline.yaml")
    from src.utils.config import load_config as _load_config

    teacher_full_config = _load_config(teacher_config_path)
    teacher_model = build_model(teacher_full_config)

    # Trigger lazy backbone loading with a dummy forward pass
    dummy_batch = {
        "pixel_values": torch.randn(1, 3, 224, 224),
        "input_ids": torch.randint(0, 30000, (1, 77)),
        "attention_mask": torch.ones(1, 77, dtype=torch.long),
    }
    with torch.no_grad():
        teacher_model(dummy_batch)

    checkpoint = torch.load(teacher_ckpt, map_location="cpu", weights_only=True)
    teacher_model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded teacher from {teacher_ckpt}")

    # Build student model (uses 'student' config section)
    student_model = build_model(config)

    training_cfg = config.training
    optimizer = build_optimizer(
        student_model,
        name="adamw",
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )
    num_training_steps = len(train_loader) * training_cfg.get("epochs", 20)
    scheduler = build_scheduler(
        optimizer,
        name=training_cfg.get("lr_scheduler", "cosine"),
        num_training_steps=num_training_steps,
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
    )

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=training_cfg.get("temperature", 4.0),
        alpha=training_cfg.get("alpha", 0.5),
        model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    best_metrics = trainer.train(num_epochs=training_cfg.get("epochs", 20))
    logger.info(f"Distillation complete. Best metrics: {best_metrics}")


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
