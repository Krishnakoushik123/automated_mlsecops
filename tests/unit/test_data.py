"""
tests/unit/test_data.py
-----------------------
Unit tests for data loading and validation.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_dataset, _load_sklearn
from src.data.validator import validate_dataset, _check_duplicates, _check_missing, _check_class_balance


@pytest.fixture
def sample_config():
    return {
        "project": {"random_seed": 42, "log_level": "WARNING"},
        "dataset": {
            "name": "credit_fraud",
            "source": "sklearn",
            "n_samples": 500,
            "n_features": 10,
            "n_informative": 5,
            "class_weights": [0.9, 0.1],
            "test_size": 0.2,
            "validation_size": 0.1,
        },
        "paths": {"data_raw": "data/raw", "data_processed": "data/processed", "models": "models",
                  "results": "results", "mlflow_uri": "mlruns"},
        "features": {"numerical_strategy": "median", "scale": "standard", "pca_components": None},
        "training": {
            "algorithms": ["logistic_regression"],
            "hyperparameters": {"logistic_regression": {"C": 1.0, "max_iter": 200}},
            "cv_folds": 3,
        },
        "mlflow": {"experiment_prefix": "test", "model_name": "test_model", "auto_log": False},
    }


@pytest.fixture
def sample_data(sample_config):
    X, y = load_dataset(sample_config)
    return X, y


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestLoader:
    def test_returns_dataframe_and_series(self, sample_data):
        X, y = sample_data
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_shape_matches_config(self, sample_config, sample_data):
        X, y = sample_data
        n = sample_config["dataset"]["n_samples"]
        k = sample_config["dataset"]["n_features"]
        assert len(X) == n
        assert X.shape[1] == k
        assert len(y) == n

    def test_binary_target(self, sample_data):
        _, y = sample_data
        assert set(y.unique()).issubset({0, 1})

    def test_deterministic_with_seed(self, sample_config):
        X1, y1 = load_dataset(sample_config)
        X2, y2 = load_dataset(sample_config)
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_no_nulls(self, sample_data):
        X, _ = sample_data
        assert X.isnull().sum().sum() == 0

    def test_feature_names(self, sample_data):
        X, _ = sample_data
        assert all(col.startswith("feature_") for col in X.columns)


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    def test_clean_data_passes(self, sample_config, sample_data):
        X, y = sample_data
        passed, report = validate_dataset(X, y, sample_config)
        assert passed, f"Expected validation to pass. Report: {report}"

    def test_missing_values_detected(self, sample_config, sample_data):
        X, y = sample_data
        X_dirty = X.copy()
        X_dirty.iloc[0, 0] = np.nan
        passed, report = validate_dataset(X_dirty, y, sample_config)
        assert not report["schema"]["passed"] or not report["missing"]["passed"]

    def test_duplicate_detection(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        result = _check_duplicates(df)
        assert result["n_duplicates"] == 1
        assert not result["passed"]

    def test_missing_check_clean(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = _check_missing(df)
        assert result["total_missing"] == 0
        assert result["passed"]

    def test_missing_check_dirty(self):
        df = pd.DataFrame({"a": [1.0, None], "b": [3.0, 4.0]})
        result = _check_missing(df)
        assert result["total_missing"] == 1
        assert not result["passed"]

    def test_class_balance_severe_imbalance(self):
        y = pd.Series([0] * 99 + [1] * 1)
        result = _check_class_balance(y, warn_threshold=0.10)
        assert not result["passed"]

    def test_class_balance_acceptable(self):
        y = pd.Series([0] * 60 + [1] * 40)
        result = _check_class_balance(y, warn_threshold=0.10)
        assert result["passed"]
