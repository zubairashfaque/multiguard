"""Tests for data loading and dataset classes."""

import pytest


class TestMultimodalDataset:
    """Test suite for MultimodalDataset."""

    def test_dataset_initialization(self):
        """Test dataset can be initialized with valid params."""
        from src.data.loaders import MultimodalDataset

        dataset = MultimodalDataset(data_dir="data/raw", split="train")
        assert dataset.split == "train"
        assert len(dataset) == 0  # No data loaded yet

    def test_build_dataloader(self):
        """Test DataLoader creation with standard settings."""
        from torch.utils.data import TensorDataset

        import torch
        from src.data.loaders import build_dataloader

        dummy = TensorDataset(torch.randn(10, 3), torch.randint(0, 2, (10,)))
        loader = build_dataloader(dummy, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        assert batch[0].shape[0] <= 4


class TestValidators:
    """Test suite for data validators."""

    def test_validate_valid_sample(self):
        """Test validation passes for valid sample."""
        from src.data.validators import validate_sample

        sample = {"text": "Hello world", "img": "test.jpg", "label": 0}
        assert validate_sample(sample) is True

    def test_validate_missing_text(self):
        """Test validation fails for missing text."""
        from src.data.validators import validate_sample

        sample = {"img": "test.jpg", "label": 0}
        assert validate_sample(sample) is False

    def test_validate_empty_text(self):
        """Test validation fails for empty text."""
        from src.data.validators import validate_sample

        sample = {"text": "  ", "img": "test.jpg", "label": 0}
        assert validate_sample(sample) is False

    def test_validate_invalid_image_extension(self):
        """Test validation fails for invalid image extension."""
        from src.data.validators import validate_sample

        sample = {"text": "Hello", "img": "test.txt", "label": 0}
        assert validate_sample(sample) is False
