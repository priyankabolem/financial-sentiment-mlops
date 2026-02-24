"""Integration tests for API."""

import pytest
from fastapi.testclient import TestClient
from src.deployment.api import app


client = TestClient(app)


class TestAPI:
    """Test API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_endpoint(self):
        """Test prediction endpoint."""
        payload = {"text": "Stock market is going up with strong gains"}
        response = client.post("/predict", json=payload)

        assert response.status_code in [200, 503]  # 503 if model not loaded
        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert "confidence" in data
            assert "probabilities" in data

    def test_predict_batch_endpoint(self):
        """Test batch prediction endpoint."""
        payload = {
            "texts": [
                "Stock is rising",
                "Market crash expected",
                "Neutral outlook for economy",
            ]
        }
        response = client.post("/predict/batch", json=payload)

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_count" in data
            assert data["total_count"] == 3

    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code in [200, 503]

    def test_invalid_input(self):
        """Test invalid input handling."""
        payload = {"text": ""}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
