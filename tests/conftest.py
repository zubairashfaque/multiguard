"""Shared test fixtures for MultiGuard test suite."""

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def tiny_config():
    """Minimal config for testing."""
    return OmegaConf.create(
        {
            "project": {"name": "multiguard-test", "version": "0.1.0", "seed": 42},
            "device": {"accelerator": "cpu", "precision": "32", "num_workers": 0},
            "logging": {
                "level": "DEBUG",
                "wandb": {"enabled": False, "project": "test", "entity": "test"},
            },
            "paths": {
                "data_dir": "data/",
                "output_dir": "models/",
                "reports_dir": "reports/",
            },
            "model": {
                "vision_backbone": "openai/clip-vit-large-patch14",
                "text_backbone": "roberta-base",
                "fusion": "late_fusion",
                "num_labels": 2,
                "dropout": 0.1,
            },
        }
    )


@pytest.fixture
def dummy_image_tensor():
    """Create a dummy image tensor [B, C, H, W]."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def dummy_text_tokens():
    """Create dummy tokenized text (input_ids + attention_mask)."""
    return {
        "input_ids": torch.randint(0, 30000, (2, 128)),
        "attention_mask": torch.ones(2, 128, dtype=torch.long),
    }


@pytest.fixture
def dummy_multimodal_batch(dummy_image_tensor, dummy_text_tokens):
    """Create a complete multimodal batch."""
    return {
        "pixel_values": dummy_image_tensor,
        "input_ids": dummy_text_tokens["input_ids"],
        "attention_mask": dummy_text_tokens["attention_mask"],
        "labels": torch.randint(0, 2, (2,)),
    }


@pytest.fixture
def dummy_vision_features():
    """Dummy vision features [B, D]."""
    return torch.randn(4, 768)


@pytest.fixture
def dummy_text_features():
    """Dummy text features [B, D]."""
    return torch.randn(4, 768)
