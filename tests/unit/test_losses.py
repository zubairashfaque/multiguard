"""Tests for loss functions."""

import pytest
import torch


class TestMultiTaskLoss:
    """Test suite for MultiTaskLoss."""

    def test_fixed_weights(self):
        """Test multi-task loss with fixed weights."""
        from src.models.losses import MultiTaskLoss

        loss_fn = MultiTaskLoss(
            task_names=["cls", "retrieval"],
            task_weights={"cls": 1.0, "retrieval": 0.5},
        )
        losses = {
            "cls": torch.tensor(0.5),
            "retrieval": torch.tensor(0.3),
        }
        total = loss_fn(losses)
        expected = 1.0 * 0.5 + 0.5 * 0.3
        assert torch.isclose(total, torch.tensor(expected))

    def test_learnable_weights(self):
        """Test multi-task loss with learnable weights."""
        from src.models.losses import MultiTaskLoss

        loss_fn = MultiTaskLoss(
            task_names=["cls", "retrieval"],
            learnable_weights=True,
        )
        losses = {
            "cls": torch.tensor(0.5),
            "retrieval": torch.tensor(0.3),
        }
        total = loss_fn(losses)
        assert total.requires_grad


class TestInfoNCELoss:
    """Test suite for InfoNCE contrastive loss."""

    def test_perfect_alignment(self):
        """Test loss is low when embeddings are perfectly aligned."""
        from src.models.losses import InfoNCELoss

        loss_fn = InfoNCELoss(temperature=0.07)
        embeddings = torch.nn.functional.normalize(torch.randn(8, 128), dim=-1)
        loss = loss_fn(embeddings, embeddings)
        assert loss.item() < 1.0  # Should be very low for identical pairs

    def test_output_is_scalar(self):
        """Test loss returns a scalar."""
        from src.models.losses import InfoNCELoss

        loss_fn = InfoNCELoss()
        a = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
        b = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
        loss = loss_fn(a, b)
        assert loss.dim() == 0


class TestFocalLoss:
    """Test suite for FocalLoss."""

    def test_output_is_scalar(self):
        """Test focal loss returns a scalar."""
        from src.models.losses import FocalLoss

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0
