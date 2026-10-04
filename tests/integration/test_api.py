"""
tests/integration/test_api.py
------------------------------
Integration tests for FastAPI inference & security API endpoints:
  - GET  /health
  - GET  /metrics
  - GET  /security/audit
  - POST /predict
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"mlsecops_" in response.content

    def test_security_audit_endpoint(self):
        response = client.get("/security/audit")
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data

    def test_predict_endpoint(self):
        payload = {
            "features": [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 0.0,
                          0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 0.0]]
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "latency_ms" in data
