"""Tests for model architectures (fusion, heads)."""

import pytest
import torch


class TestLateFusion:
    """Test suite for LateFusion module."""

    def test_output_shape(self, dummy_vision_features, dummy_text_features):
        """Test late fusion produces correct output shape."""
        from src.models.fusion.late_fusion import LateFusion

        fusion = LateFusion(vision_dim=768, text_dim=768, hidden_dims=[512, 256])
        output = fusion(dummy_vision_features, dummy_text_features)
        assert output.shape == (4, 256)

    def test_output_dim_attribute(self):
        """Test output_dim is set correctly."""
        from src.models.fusion.late_fusion import LateFusion

        fusion = LateFusion(hidden_dims=[512, 128])
        assert fusion.output_dim == 128


class TestCrossAttentionFusion:
    """Test suite for CrossAttentionFusion module."""

    def test_output_shape(self, dummy_vision_features, dummy_text_features):
        """Test cross-attention fusion produces correct output shape."""
        from src.models.fusion.cross_attention import CrossAttentionFusion

        fusion = CrossAttentionFusion(hidden_dim=768, num_heads=8, num_layers=2)
        output = fusion(dummy_vision_features, dummy_text_features)
        assert output.shape == (4, 768)


class TestTensorFusion:
    """Test suite for TensorFusion module."""

    def test_output_shape(self, dummy_vision_features, dummy_text_features):
        """Test tensor fusion produces correct output shape."""
        from src.models.fusion.tensor_fusion import TensorFusion

        fusion = TensorFusion(vision_dim=768, text_dim=768, hidden_dim=512, rank=32)
        output = fusion(dummy_vision_features, dummy_text_features)
        assert output.shape == (4, 512)


class TestClassificationHead:
    """Test suite for ClassificationHead."""

    def test_binary_classification(self):
        """Test binary classification output shape."""
        from src.models.heads.classifier import ClassificationHead

        head = ClassificationHead(input_dim=256, num_labels=2)
        features = torch.randn(4, 256)
        logits = head(features)
        assert logits.shape == (4, 2)

    def test_multiclass_classification(self):
        """Test multi-class classification output shape."""
        from src.models.heads.classifier import ClassificationHead

        head = ClassificationHead(input_dim=256, num_labels=10)
        features = torch.randn(4, 256)
        logits = head(features)
        assert logits.shape == (4, 10)


class TestRetrievalHead:
    """Test suite for RetrievalHead."""

    def test_normalized_output(self):
        """Test retrieval head produces normalized embeddings."""
        from src.models.heads.retrieval import RetrievalHead

        head = RetrievalHead(input_dim=256, embedding_dim=128)
        features = torch.randn(4, 256)
        embeddings = head(features)
        norms = torch.norm(embeddings, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestVQAHead:
    """Test suite for VQAHead."""

    def test_output_shape(self):
        """Test VQA head output matches answer vocab size."""
        from src.models.heads.vqa import VQAHead

        head = VQAHead(input_dim=256, answer_vocab_size=3129)
        features = torch.randn(4, 256)
        logits = head(features)
        assert logits.shape == (4, 3129)
