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
    elif source == "csv":
        X, y = _load_csv(ds_cfg)
    else:
        raise ValueError(f"Unknown dataset source: {source!r}")

    elapsed = time.perf_counter() - t0
    logger.info(
        "Dataset loaded: %d samples, %d features in %.3fs",
        len(X), X.shape[1], elapsed,
    )
    return X, y


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
    logger.info("Raw dataset saved â†’ %s", out_path)
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


def _load_csv(ds_cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = Path(ds_cfg["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    target_col = ds_cfg.get("target_column", "target")
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not in CSV")
    y = df.pop(target_col).rename("target")
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

