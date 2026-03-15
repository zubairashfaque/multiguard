"""Smoke tests for the full pipeline end-to-end."""

import pytest
import torch

from src.models.model_factory import build_model
from src.utils.config import load_config


@pytest.mark.smoke
class TestFullPipeline:
    """End-to-end smoke tests."""

    def test_model_forward_pass(self):
        """Test that a model can be built and run a forward pass."""
        config = load_config("configs/train/baseline.yaml")
        model = build_model(config.model)
        model.eval()

        dummy_batch = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 30000, (1, 77)),
            "attention_mask": torch.ones(1, 77, dtype=torch.long),
        }

        with torch.no_grad():
            outputs = model(dummy_batch)

        assert "logits" in outputs
        assert outputs["logits"].shape == (1, 2)
        assert "fused_features" in outputs

    def test_predict_with_explanation(self):
        """Test prediction produces valid logits and features for explainability."""
        config = load_config("configs/train/baseline.yaml")
        model = build_model(config.model)
        model.eval()

        dummy_batch = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 30000, (1, 77)),
            "attention_mask": torch.ones(1, 77, dtype=torch.long),
        }

        with torch.no_grad():
            outputs = model(dummy_batch)

        probs = torch.softmax(outputs["logits"], dim=-1)
        assert probs.shape == (1, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
