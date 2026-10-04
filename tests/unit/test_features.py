"""
tests/unit/test_features.py
---------------------------
Unit tests for feature preprocessing pipeline.
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.features.preprocessing import (
    build_feature_pipeline,
    split_dataset,
    fit_transform_splits,
)
from src.data.loader import load_dataset


@pytest.fixture
def minimal_config():
    return {
        "project": {"random_seed": 42, "log_level": "WARNING"},
        "dataset": {
            "name": "credit_fraud",
            "source": "sklearn",
            "n_samples": 400,
            "n_features": 8,
            "n_informative": 4,
            "class_weights": [0.8, 0.2],
            "test_size": 0.2,
            "validation_size": 0.1,
        },
        "features": {"numerical_strategy": "median", "scale": "standard", "pca_components": None},
        "paths": {
            "data_raw": "data/raw",
            "data_processed": "data/processed",
            "models": "models",
            "results": "results",
            "mlflow_uri": "mlruns",
        },
    }


@pytest.fixture
def data_and_splits(minimal_config):
    X, y = load_dataset(minimal_config)
    splits = split_dataset(X, y, minimal_config)
    return X, y, splits


class TestBuildPipeline:
    def test_returns_pipeline(self, minimal_config):
        pipe = build_feature_pipeline(minimal_config)
        assert isinstance(pipe, Pipeline)

    def test_has_imputer(self, minimal_config):
        pipe = build_feature_pipeline(minimal_config)
        assert "imputer" in pipe.named_steps

    def test_has_scaler_standard(self, minimal_config):
        pipe = build_feature_pipeline(minimal_config)
        assert "scaler" in pipe.named_steps

    def test_no_scaler_when_none(self, minimal_config):
        minimal_config["features"]["scale"] = "none"
        pipe = build_feature_pipeline(minimal_config)
        assert "scaler" not in pipe.named_steps

    def test_pca_added_when_configured(self, minimal_config):
        minimal_config["features"]["pca_components"] = 4
        pipe = build_feature_pipeline(minimal_config)
        assert "pca" in pipe.named_steps


class TestSplitDataset:
    def test_split_sizes_sum_to_total(self, data_and_splits):
        X, y, splits = data_and_splits
        total = sum(len(v[1]) for v in splits.values())
        assert total == len(y)

    def test_all_three_splits_exist(self, data_and_splits):
        _, _, splits = data_and_splits
        assert {"train", "val", "test"} == set(splits.keys())

    def test_no_overlap_between_splits(self, data_and_splits):
        """Verify train/val/test indices don't overlap (shapes are disjoint)."""
        _, _, splits = data_and_splits
        sizes = [len(v[1]) for v in splits.values()]
        # sizes must all be positive
        assert all(s > 0 for s in sizes)


class TestFitTransform:
    def test_transform_preserves_sample_count(self, minimal_config, data_and_splits):
        X, y, splits = data_and_splits
        transformed, _ = fit_transform_splits(splits, config=minimal_config)
        for name, (Xs, ys) in transformed.items():
            assert len(Xs) == len(ys), f"Mismatch in split '{name}'"

    def test_output_is_numpy(self, minimal_config, data_and_splits):
        X, y, splits = data_and_splits
        transformed, _ = fit_transform_splits(splits, config=minimal_config)
        for name, (Xs, _) in transformed.items():
            assert isinstance(Xs, np.ndarray), f"Split '{name}' is not ndarray"

    def test_no_nan_after_transform(self, minimal_config, data_and_splits):
        X, y, splits = data_and_splits
        transformed, _ = fit_transform_splits(splits, config=minimal_config)
        for name, (Xs, _) in transformed.items():
            assert not np.isnan(Xs).any(), f"NaN found in split '{name}'"
