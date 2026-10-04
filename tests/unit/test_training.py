"""
tests/unit/test_training.py
---------------------------
Unit tests for the training module.
"""

import numpy as np
import pytest

from src.training.trainer import build_model, compute_metrics, train_model
from src.data.loader import load_dataset
from src.features.preprocessing import split_dataset, fit_transform_splits


@pytest.fixture
def mini_config():
    return {
        "project": {"random_seed": 42, "log_level": "WARNING"},
        "dataset": {
            "name": "credit_fraud",
            "source": "sklearn",
            "n_samples": 300,
            "n_features": 6,
            "n_informative": 3,
            "class_weights": [0.7, 0.3],
            "test_size": 0.2,
            "validation_size": 0.1,
        },
        "features": {"numerical_strategy": "median", "scale": "standard", "pca_components": None},
        "training": {
            "algorithms": ["logistic_regression"],
            "hyperparameters": {
                "logistic_regression": {"C": 1.0, "max_iter": 200},
                "random_forest": {"n_estimators": 10, "max_depth": 3},
            },
            "cv_folds": 2,
        },
        "mlflow": {"experiment_prefix": "unit_test", "model_name": "unit_test_model", "auto_log": False},
        "paths": {
            "data_raw": "data/raw",
            "data_processed": "data/processed",
            "models": "models",
            "results": "results",
            "mlflow_uri": "mlruns",
        },
    }


@pytest.fixture
def transformed_splits(mini_config):
    X, y = load_dataset(mini_config)
    splits = split_dataset(X, y, mini_config)
    t_splits, _ = fit_transform_splits(splits, config=mini_config)
    return t_splits


class TestBuildModel:
    def test_logistic_regression(self, mini_config):
        model = build_model("logistic_regression", {"C": 0.5, "max_iter": 100}, seed=42)
        assert hasattr(model, "fit")

    def test_random_forest(self, mini_config):
        model = build_model("random_forest", {"n_estimators": 5}, seed=42)
        assert hasattr(model, "fit")

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            build_model("super_net", {}, seed=42)


class TestComputeMetrics:
    def test_returns_expected_keys(self, transformed_splits, mini_config):
        X_tr, y_tr = transformed_splits["train"]
        model = build_model("logistic_regression", {"C": 1.0, "max_iter": 200}, seed=42)
        model.fit(X_tr, y_tr)
        X_test, y_test = transformed_splits["test"]
        metrics = compute_metrics(model, X_test, y_test, n_latency_samples=10)
        for key in ("accuracy", "precision", "recall", "f1", "latency_p50_ms", "latency_p99_ms"):
            assert key in metrics, f"Missing metric: {key}"

    def test_metric_ranges(self, transformed_splits, mini_config):
        X_tr, y_tr = transformed_splits["train"]
        model = build_model("logistic_regression", {"C": 1.0, "max_iter": 200}, seed=42)
        model.fit(X_tr, y_tr)
        X_test, y_test = transformed_splits["test"]
        metrics = compute_metrics(model, X_test, y_test, n_latency_samples=10)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert metrics["latency_p50_ms"] >= 0


class TestTrainModel:
    def test_train_model_returns_dict(self, mini_config, transformed_splits):
        result = train_model("logistic_regression", transformed_splits, config=mini_config)
        assert "algorithm" in result
        assert "metrics" in result
        assert "run_id" in result

    def test_test_f1_in_metrics(self, mini_config, transformed_splits):
        result = train_model("logistic_regression", transformed_splits, config=mini_config)
        assert "test_f1" in result["metrics"]

    def test_training_time_positive(self, mini_config, transformed_splits):
        result = train_model("logistic_regression", transformed_splits, config=mini_config)
        assert result["metrics"]["training_time_seconds"] > 0
