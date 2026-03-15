"""Integration tests for the training loop."""

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_factory import build_model
from src.training.optimizers import build_optimizer
from src.training.trainer import MultimodalTrainer


def _make_dummy_loader(batch_size=4, n_samples=16):
    """Create a DataLoader with dummy multimodal data."""
    dataset = TensorDataset(
        torch.randn(n_samples, 3, 224, 224),  # pixel_values
        torch.randint(0, 30000, (n_samples, 77)),  # input_ids
        torch.ones(n_samples, 77, dtype=torch.long),  # attention_mask
        torch.randint(0, 2, (n_samples,)),  # labels
    )

    def collate_fn(batch):
        pixel_values, input_ids, attention_mask, labels = zip(*batch)
        return {
            "pixel_values": torch.stack(pixel_values),
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


@pytest.fixture
def dummy_config():
    """Config for integration testing (CPU, no wandb)."""
    return OmegaConf.create(
        {
            "model": {
                "vision_backbone": "openai/clip-vit-base-patch16",
                "vision_dim": 512,
                "text_backbone": "roberta-base",
                "text_dim": 768,
                "fusion": "late_fusion",
                "num_labels": 2,
                "dropout": 0.1,
                "freeze_backbones": True,
            },
            "training": {
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "fp16": False,
                "logging_steps": 10,
                "output_dir": "models/checkpoints/test",
                "patience": 3,
            },
            "logging": {"level": "WARNING", "wandb": {"enabled": False}},
        }
    )


@pytest.mark.integration
class TestTrainingLoop:
    """Test the full training loop with tiny data."""

    def test_single_epoch_baseline(self, dummy_config):
        """Test one training epoch completes without errors."""
        model = build_model(dummy_config)
        train_loader = _make_dummy_loader(batch_size=4, n_samples=8)
        val_loader = _make_dummy_loader(batch_size=4, n_samples=4)

        optimizer = build_optimizer(model, name="adamw", lr=1e-3)

        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=dummy_config,
        )

        train_metrics = trainer.train_epoch(epoch=1)
        assert "train_loss" in train_metrics
        assert train_metrics["train_loss"] > 0

    def test_validation(self, dummy_config):
        """Test validation returns proper metrics."""
        model = build_model(dummy_config)
        train_loader = _make_dummy_loader(batch_size=4, n_samples=8)
        val_loader = _make_dummy_loader(batch_size=4, n_samples=8)

        optimizer = build_optimizer(model, name="adamw", lr=1e-3)

        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=dummy_config,
        )

        val_metrics = trainer.validate()
        assert "val_loss" in val_metrics
        assert "accuracy" in val_metrics
        assert "f1" in val_metrics
        assert "auroc" in val_metrics

    def test_checkpoint_save_load(self, dummy_config, tmp_path):
        """Test checkpoint saving and loading."""
        model = build_model(dummy_config)
        train_loader = _make_dummy_loader(batch_size=4, n_samples=8)
        val_loader = _make_dummy_loader(batch_size=4, n_samples=4)

        optimizer = build_optimizer(model, name="adamw", lr=1e-3)

        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=dummy_config,
        )

        ckpt_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(ckpt_path, {"test_metric": 0.95})

        assert ckpt_path.exists()
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["metrics"]["test_metric"] == 0.95
