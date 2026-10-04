"""
tests/conftest.py
-----------------
Shared pytest fixtures used across unit and integration tests.
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def base_config():
    """Minimal config fixture usable by all test modules."""
    return {
        "project": {"random_seed": 42, "log_level": "WARNING"},
        "dataset": {
            "name": "credit_fraud",
            "source": "sklearn",
            "n_samples": 300,
            "n_features": 8,
            "n_informative": 4,
            "class_weights": [0.8, 0.2],
            "test_size": 0.20,
            "validation_size": 0.10,
        },
        "features": {
            "numerical_strategy": "median",
            "scale": "standard",
            "pca_components": None,
        },
        "training": {
            "algorithms": ["logistic_regression"],
            "hyperparameters": {
                "logistic_regression": {"C": 1.0, "max_iter": 200},
                "random_forest": {"n_estimators": 10, "max_depth": 3},
                "gradient_boosting": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
            },
            "cv_folds": 2,
        },
        "mlflow": {
            "experiment_prefix": "test",
            "model_name": "test_credit_fraud",
            "auto_log": False,
        },
        "paths": {
            "data_raw":       "data/raw",
            "data_processed": "data/processed",
            "models":         "models",
            "results":        "results",
            "mlflow_uri":     "mlruns",
        },
        "security": {
            "dependency_audit":      False,
            "secrets_scan":          False,
            "data_integrity":        True,
            "adversarial_robustness": False,
            "model_integrity":       False,
            "max_critical_vulns":    0,
            "max_high_vulns":        2,
            "min_robustness_score":  0.70,
            "poisoning_threshold":   0.05,
        },
    }
