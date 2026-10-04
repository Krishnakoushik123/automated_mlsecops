"""
tests/unit/test_monitoring.py
------------------------------
Unit tests for runtime monitoring: drift detection, anomaly detection, auto-response.
"""

import numpy as np
import pytest

from src.monitoring.drift_detector import DriftDetector
from src.monitoring.anomaly_detector import AnomalyDetector
from src.monitoring.auto_responder import AutoResponder


class TestDriftDetector:
    def test_no_drift_on_same_distribution(self):
        np.random.seed(42)
        ref = np.random.randn(200, 5)
        curr = np.random.randn(200, 5)

        detector = DriftDetector(ref)
        report = detector.detect_drift(curr)
        assert report.is_drifted is False
        assert report.drift_score < 0.3

    def test_detects_drift_on_shifted_distribution(self):
        np.random.seed(42)
        ref = np.random.randn(200, 5)
        curr = np.random.randn(200, 5) + 5.0  # Massive shift

        detector = DriftDetector(ref)
        report = detector.detect_drift(curr)
        assert report.is_drifted is True
        assert len(report.drifted_features) > 0


class TestAnomalyDetector:
    def test_normal_input(self):
        ref = np.random.randn(200, 5)
        detector = AnomalyDetector(ref)

        normal_sample = np.random.randn(1, 5)
        report = detector.evaluate_sample(normal_sample)
        assert report.is_anomaly is False

    def test_out_of_bounds_anomaly(self):
        ref = np.random.randn(200, 5)
        detector = AnomalyDetector(ref)

        extreme_sample = np.array([[50.0, -100.0, 30.0, 40.0, -80.0]])
        report = detector.evaluate_sample(extreme_sample)
        assert report.is_anomaly is True
        assert len(report.out_of_bounds_features) > 0


class TestAutoResponder:
    def test_handles_input_anomaly(self, tmp_path):
        responder = AutoResponder(incident_dir=str(tmp_path))
        X = np.random.randn(5, 5)
        
        res = responder.handle_input_anomaly(X, anomaly_score=0.90, threshold=0.75)
        assert res.triggered is True
        assert res.action_taken == "fallback_conservative_prediction"
        assert res.fallback_prediction is not None
        assert len(list(tmp_path.glob("*.json"))) == 1
