"""Integration tests for the FastAPI serving endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.mark.integration
class TestServingAPI:
    """Test API endpoints."""

    @pytest.fixture(autouse=True)
    def client(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test /health returns 200."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "multiguard"

    def test_ready_endpoint_no_model(self):
        """Test /ready returns not_ready when no model is loaded."""
        response = self.client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "multiguard"
        assert "model_loaded" in data
