"""Smoke tests for the full pipeline end-to-end."""

import pytest


@pytest.mark.smoke
class TestFullPipeline:
    """End-to-end smoke tests."""

    @pytest.mark.skip(reason="Implement after all components are ready")
    def test_ingest_train_evaluate(self):
        """Test full pipeline: ingest -> train -> evaluate."""
        pass

    @pytest.mark.skip(reason="Implement after all components are ready")
    def test_predict_with_explanation(self):
        """Test prediction with explainability output."""
        pass
