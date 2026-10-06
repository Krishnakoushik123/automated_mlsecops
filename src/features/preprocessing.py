"""
src/features/preprocessing.py
------------------------------
Reproducible feature preprocessing pipeline built on scikit-learn's Pipeline.

Outputs
-------
* Fitted sklearn Pipeline (pickled by training stage)
* Processed train/val/test splits saved to data/processed/

Run standalone:
    python -m src.features.preprocessing
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_feature_pipeline(config: dict | None = None) -> Pipeline:
    """Construct a (Imputer â†’ Scaler [â†’ PCA]) sklearn Pipeline."""
    if config is None:
        config = load_config()

    feat_cfg = config["features"]
    seed = config["project"]["random_seed"]
    steps = []

    # Step 1 â€“ imputation
    strategy = feat_cfg.get("numerical_strategy", "median")
    steps.append(("imputer", SimpleImputer(strategy=strategy)))

    # Step 2 â€“ scaling
    scaler_name = feat_cfg.get("scale", "standard")
    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "none": None,
    }
    scaler = scaler_map.get(scaler_name)
    if scaler is not None:
        steps.append(("scaler", scaler))

    # Step 3 â€“ optional PCA
    n_components = feat_cfg.get("pca_components")
    if n_components:
        steps.append(("pca", PCA(n_components=int(n_components), random_state=seed)))

    pipeline = Pipeline(steps)
    logger.info("Feature pipeline built: %s", [s[0] for s in steps])
    return pipeline


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return dict with keys 'train', 'val', 'test'."""
    if config is None:
        config = load_config()

    ds_cfg = config["dataset"]
    seed = config["project"]["random_seed"]
    test_size: float = ds_cfg.get("test_size", 0.20)
    val_size: float = ds_cfg.get("validation_size", 0.10)

    X_arr = X.values
    y_arr = y.values

    # Determine whether to stratify (only for classification)
    task_type = config.get("dataset", {}).get("task_type", "classification")
    stratify_arr = y_arr if task_type == "classification" else None

    # First split off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=seed, stratify=stratify_arr
    )

    # Then split validation from remaining
    val_fraction = val_size / (1.0 - test_size)
    stratify_trainval = y_trainval if task_type == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify_trainval,
    )

    logger.info(
        "Split sizes â€“ train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }


# ---------------------------------------------------------------------------
# Fit-transform + persist
# ---------------------------------------------------------------------------

def fit_transform_splits(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pipeline: Pipeline | None = None,
    config: dict | None = None,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Pipeline]:
    """Fit pipeline on train split, transform all splits. Persist results."""
    if config is None:
        config = load_config()
    if pipeline is None:
        pipeline = build_feature_pipeline(config)

    X_train, y_train = splits["train"]

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    logger.info("Feature pipeline fitted in %.3fs", fit_time)

    transformed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split_name, (X_split, y_split) in splits.items():
        transformed[split_name] = (pipeline.transform(X_split), y_split)

    # Persist processed arrays
    proc_dir = Path(config["paths"]["data_processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    for split_name, (X_s, y_s) in transformed.items():
        np.save(proc_dir / f"X_{split_name}.npy", X_s)
        np.save(proc_dir / f"y_{split_name}.npy", y_s)
    logger.info("Processed splits saved â†’ %s", proc_dir)

    # Persist pipeline
    models_dir = Path(config["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = models_dir / "feature_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info("Feature pipeline saved â†’ %s", pipeline_path)

    return transformed, pipeline


def load_processed_splits(
    config: dict | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load processed numpy arrays from disk."""
    if config is None:
        config = load_config()
    proc_dir = Path(config["paths"]["data_processed"])
    splits = {}
    for split_name in ("train", "val", "test"):
        X = np.load(proc_dir / f"X_{split_name}.npy")
        y = np.load(proc_dir / f"y_{split_name}.npy")
        splits[split_name] = (X, y)
    return splits


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    cfg = load_config()
    from src.data.loader import load_dataset
    from src.data.validator import validate_dataset

    X, y = load_dataset(cfg)
    passed, _ = validate_dataset(X, y, cfg)
    if not passed:
        raise SystemExit("Data validation failed â€“ aborting feature pipeline")

    splits = split_dataset(X, y, cfg)
    transformed_splits, pipe = fit_transform_splits(splits, config=cfg)
    print("âœ“  Feature pipeline complete")
    for name, (Xs, ys) in transformed_splits.items():
        print(f"   {name}: X={Xs.shape}  y={ys.shape}")

