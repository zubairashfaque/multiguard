"""Integration tests for the FastAPI serving endpoints."""

import pytest


@pytest.mark.integration
class TestServingAPI:
    """Test API endpoints."""

    @pytest.mark.skip(reason="Requires model loading — implement after model is trained")
    def test_health_endpoint(self):
        """Test /health returns 200."""
        pass

    @pytest.mark.skip(reason="Requires model loading — implement after model is trained")
    def test_predict_endpoint(self):
        """Test /api/v1/predict returns valid response."""
        pass
