"""Tests for evaluation metrics."""

import pytest
import torch


class TestClassificationMetrics:
    """Test suite for classification metrics."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        from src.training.metrics import compute_classification_metrics

        predictions = torch.tensor([[0.0, 10.0], [10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
        labels = torch.tensor([1, 0, 1, 0])
        metrics = compute_classification_metrics(predictions, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_random_predictions(self):
        """Test metrics return valid values for random predictions."""
        from src.training.metrics import compute_classification_metrics

        predictions = torch.randn(100, 2)
        labels = torch.randint(0, 2, (100,))
        metrics = compute_classification_metrics(predictions, labels)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert "auroc" in metrics

    def test_all_metrics_present(self):
        """Test all expected metrics are computed."""
        from src.training.metrics import compute_classification_metrics

        predictions = torch.randn(20, 2)
        labels = torch.randint(0, 2, (20,))
        metrics = compute_classification_metrics(predictions, labels)
        expected_keys = {"accuracy", "f1", "precision", "recall", "auroc"}
        assert expected_keys == set(metrics.keys())
