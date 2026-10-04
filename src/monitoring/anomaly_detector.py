"""
src/monitoring/anomaly_detector.py
----------------------------------
Detects anomalous runtime inference inputs, out-of-bounds features,
and potential adversarial queries using statistical bounds and Isolation Forest.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class AnomalyReport:
    is_anomaly: bool
    anomaly_score: float             # Normalized anomaly score [0, 1]
    out_of_bounds_features: List[str]
    isolation_forest_flag: bool
    details: Dict[str, Any]


class AnomalyDetector:
    """Runtime input anomaly and out-of-distribution detector."""

    def __init__(self, reference_data: np.ndarray | pd.DataFrame, contamination: float = 0.05, random_seed: int = 42):
        if isinstance(reference_data, pd.DataFrame):
            self.feature_names = list(reference_data.columns)
            ref_arr = reference_data.values
        else:
            ref_arr = reference_data
            self.feature_names = [f"feature_{i:02d}" for i in range(ref_arr.shape[1])]

        self.means = np.mean(ref_arr, axis=0)
        self.stds = np.std(ref_arr, axis=0) + 1e-8
        self.mins = np.min(ref_arr, axis=0) - 3 * self.stds
        self.maxs = np.max(ref_arr, axis=0) + 3 * self.stds

        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_seed,
            n_estimators=50,
        )
        self.iso_forest.fit(ref_arr)

    def evaluate_sample(self, x: np.ndarray | List[float] | pd.DataFrame) -> AnomalyReport:
        """Evaluate a single input vector or 2D batch for anomalies."""
        if isinstance(x, list):
            arr = np.array(x, dtype=float).reshape(1, -1)
        elif isinstance(x, pd.DataFrame):
            arr = x.values
        elif x.ndim == 1:
            arr = x.reshape(1, -1)
        else:
            arr = x

        # 1. Check feature bounds (z-score > 4 or outside min-max limits)
        z_scores = np.abs((arr - self.means) / self.stds)
        oob_mask = (z_scores > 4.5) | (arr < self.mins) | (arr > self.maxs)
        
        out_of_bounds_features = []
        for col_idx in range(arr.shape[1]):
            if np.any(oob_mask[:, col_idx]):
                feature_name = self.feature_names[col_idx] if col_idx < len(self.feature_names) else f"feature_{col_idx:02d}"
                out_of_bounds_features.append(feature_name)

        # 2. Isolation Forest prediction (-1 for outlier, 1 for inlier)
        preds = self.iso_forest.predict(arr)
        raw_scores = self.iso_forest.decision_function(arr)
        
        iso_flag = bool(np.any(preds == -1))
        # Convert decision function score (higher = normal) to anomaly score [0, 1]
        mean_raw_score = float(np.mean(raw_scores))
        anomaly_score = float(np.clip(0.5 - mean_raw_score, 0.0, 1.0))

        is_anomaly = iso_flag or len(out_of_bounds_features) > (arr.shape[1] // 3)

        return AnomalyReport(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            out_of_bounds_features=out_of_bounds_features,
            isolation_forest_flag=iso_flag,
            details={
                "max_z_score": float(np.max(z_scores)),
                "oob_count": len(out_of_bounds_features),
                "decision_score": mean_raw_score,
            },
        )
