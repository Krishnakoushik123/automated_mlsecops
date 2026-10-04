"""
tests/unit/test_security.py
----------------------------
Unit tests for security gates (dependency audit, secrets scan, data integrity, FGSM robustness, model integrity).
"""

import numpy as np
import pytest
from pathlib import Path

from src.security.gates import (
    gate_data_integrity,
    gate_adversarial_robustness,
    gate_model_integrity,
    run_all_gates,
)
from src.training.trainer import build_model


class TestDataIntegrityGate:
    def test_clean_data_passes(self, base_config):
        X = np.random.randn(100, 6)
        y = np.random.choice([0, 1], size=100)
        res = gate_data_integrity(X, y, X, y, base_config)
        assert res.gate == "data_integrity"
        assert res.passed is True
        assert res.score > 0.8

    def test_poisoned_data_fails(self, base_config):
        X_ref = np.random.randn(100, 6)
        y_ref = np.array([0] * 50 + [1] * 50)

        # Extreme shift / flipped labels
        X_poisoned = np.random.randn(100, 6) + 15.0
        y_poisoned = np.array([1] * 99 + [0] * 1)

        res = gate_data_integrity(X_poisoned, y_poisoned, X_ref, y_ref, base_config)
        assert res.passed is False


class TestAdversarialRobustnessGate:
    def test_robustness_on_trained_model(self, base_config):
        X = np.random.randn(100, 6)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = build_model("logistic_regression", {"C": 1.0}, seed=42)
        model.fit(X, y)

        res = gate_adversarial_robustness(model, X, y, base_config, epsilon=0.05)
        assert res.gate == "adversarial_robustness"
        assert res.score is not None
        assert 0.0 <= res.score <= 1.0


class TestModelIntegrityGate:
    def test_checksum_creation_and_verification(self, tmp_path):
        dummy_model = tmp_path / "model.joblib"
        dummy_model.write_bytes(b"fake_model_binary_bytes_12345")

        res1 = gate_model_integrity(dummy_model)
        assert res1.passed is True

        res2 = gate_model_integrity(dummy_model)
        assert res2.passed is True

    def test_corrupted_model_fails(self, tmp_path):
        dummy_model = tmp_path / "model.joblib"
        dummy_model.write_bytes(b"original_content")

        gate_model_integrity(dummy_model)

        # Corrupt file
        dummy_model.write_bytes(b"tampered_content")

        res = gate_model_integrity(dummy_model)
        assert res.passed is False
