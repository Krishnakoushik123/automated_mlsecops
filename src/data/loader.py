"""
src/data/loader.py
------------------
Loads the raw dataset from the configured source (sklearn built-ins or CSV)
and persists it as Parquet in data/processed/.

Run standalone:
    python -m src.data.loader
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification,
    load_breast_cancer,
    load_wine,
)

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(config: dict | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) as a DataFrame and Series.

    The dataset source is determined by ``config.dataset.source``.
    """
    if config is None:
        config = load_config()

    ds_cfg = config["dataset"]
    source: str = ds_cfg.get("source", "sklearn")
    seed: int = config["project"]["random_seed"]

    logger.info("Loading dataset â€“ source=%s", source)
    t0 = time.perf_counter()

    if source == "sklearn":
        X, y = _load_sklearn(ds_cfg, seed)
    elif source == "file":
        X, y = _load_file(ds_cfg)
    else:
        raise ValueError(f"Unknown dataset source: {source!r}")

    elapsed = time.perf_counter() - t0
    logger.info(
        "Dataset loaded: %d samples, %d features in %.3fs",
        len(X), X.shape[1], elapsed,
    )
    return X, y


import hashlib
import re

def generate_dataset_id(file_name: str, file_content: bytes | None = None) -> str:
    """Generate a stable, deterministic dataset ID based on filename and optional content hash."""
    clean_stem = re.sub(r'[^a-zA-Z0-9_]', '_', Path(file_name).stem.lower()).strip('_')
    if not clean_stem:
        clean_stem = "dataset"
    if file_content is not None and len(file_content) > 0:
        h = hashlib.sha256(file_content).hexdigest()[:8]
    else:
        h = hashlib.md5(file_name.encode("utf-8")).hexdigest()[:8]
    return f"ds_{clean_stem}_{h}"


def save_raw(X: pd.DataFrame, y: pd.Series, config: dict | None = None) -> Path:
    """Persist the raw dataset to data/raw/ as Parquet and return the path."""
    if config is None:
        config = load_config()

    raw_dir = Path(config["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = X.copy()
    df["target"] = y.values
    out_path = raw_dir / "dataset.parquet"
    df.to_parquet(out_path, index=False)
    
    # Also preserve the original raw snapshot
    raw_snapshot = raw_dir / "raw_dataset.parquet"
    df.to_parquet(raw_snapshot, index=False)
    logger.info("Raw dataset saved -> %s and %s", out_path, raw_snapshot)
    return out_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_sklearn(ds_cfg: dict, seed: int) -> Tuple[pd.DataFrame, pd.Series]:
    name = ds_cfg.get("name", "credit_fraud")

    if name == "breast_cancer":
        bunch = load_breast_cancer(as_frame=True)
        return bunch.data, bunch.target.rename("target")

    if name == "wine":
        bunch = load_wine(as_frame=True)
        return bunch.data, bunch.target.rename("target")

    # Default: synthetic binary classification (credit-fraud-like)
    n_samples = ds_cfg.get("n_samples", 5000)
    n_features = ds_cfg.get("n_features", 20)
    n_informative = ds_cfg.get("n_informative", 10)
    weights = ds_cfg.get("class_weights", [0.95, 0.05])

    n_redundant = ds_cfg.get("n_redundant", max(0, min(4, n_features - n_informative - 1)))

    X_arr, y_arr = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=2,
        weights=weights,
        flip_y=0.01,
        random_state=seed,
    )
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="target")
    return X, y


def _load_file(ds_cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
    file_path = Path(ds_cfg["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(f"Uploaded dataset file not found at: {file_path}")
    target_col = ds_cfg.get("target_column", "target")
    task_type = ds_cfg.get("task_type", "classification")

    ext = file_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats are: CSV, Excel (.xlsx, .xls), JSON, Parquet.")

    if target_col not in df.columns:
        cols_str = ", ".join([f"'{c}'" for c in df.columns[:10]])
        if len(df.columns) > 10:
            cols_str += f", ... ({len(df.columns)} total)"
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: [{cols_str}]")

    # Extract target column
    y_raw = df.pop(target_col)

    # Drop rows where target is NaN/null
    valid_mask = y_raw.notnull()
    if not valid_mask.any():
        raise ValueError(f"Target column '{target_col}' contains only missing/null values.")

    df = df[valid_mask].copy()
    y_raw = y_raw[valid_mask].copy()

    # Process target column according to task_type
    if task_type == "classification":
        # Convert string / boolean / object / float labels to integer class indices
        if not np.issubdtype(y_raw.dtype, np.integer):
            categories, codes = np.unique(y_raw.astype(str).values, return_inverse=True)
            y = pd.Series(codes, index=y_raw.index, name="target", dtype=int)
            logger.info("Encoded classification target '%s' into %d integer classes: %s", target_col, len(categories), categories.tolist())
        else:
            y = pd.Series(y_raw.values, index=y_raw.index, name="target", dtype=int)

        n_classes = len(np.unique(y))
        if n_classes < 2:
            raise ValueError(f"Classification target column '{target_col}' must contain at least 2 distinct classes, found {n_classes}.")
    else:
        # Task type: regression
        y_numeric = pd.to_numeric(y_raw, errors="coerce")
        num_mask = y_numeric.notnull()
        if not num_mask.any():
            raise ValueError(f"Regression target column '{target_col}' contains no valid numeric values.")

        df = df[num_mask].copy()
        y = pd.Series(y_numeric[num_mask].values, index=df.index, name="target", dtype=float)

    # Auto-convert categorical/string feature columns to one-hot numeric dummies
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    # Ensure all remaining feature columns are numeric without overriding NaNs with 0
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.shape[1] == 0:
        raise ValueError("No feature columns remaining after dataset preprocessing.")

    return df, y


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    X, y = load_dataset(cfg)
    path = save_raw(X, y, cfg)
    print(f"âœ“  Dataset saved to {path}")
    print(f"   Shape  : {X.shape}")
    print(f"   Classes: {y.value_counts().to_dict()}")

