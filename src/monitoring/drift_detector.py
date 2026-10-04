"""
src/monitoring/drift_detector.py
--------------------------------
Data drift detection engine comparing production inference features
against reference training distributions using statistical tests:
  - Kolmogorov-Smirnov (KS) test p-values
  - Population Stability Index (PSI)
  - Wasserstein Distance
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class DriftReport:
    is_drifted: bool
    drift_score: float                # 0.0 to 1.0 overall drift score
    drifted_features: List[str]
    feature_reports: Dict[str, Dict[str, float]]
    total_features: int


class DriftDetector:
    """Statistical data drift detector."""

    def __init__(self, reference_data: pd.DataFrame | np.ndarray, feature_names: Optional[List[str]] = None):
        if isinstance(reference_data, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i:02d}" for i in range(reference_data.shape[1])]
            self.ref_df = pd.DataFrame(reference_data, columns=feature_names)
        else:
            self.ref_df = reference_data.copy()
            feature_names = list(self.ref_df.columns)

        self.feature_names = feature_names

    def detect_drift(
        self,
        current_data: pd.DataFrame | np.ndarray,
        ks_alpha: float = 0.05,
        psi_threshold: float = 0.2,
    ) -> DriftReport:
        """Detect drift between reference and current data batch."""
        if isinstance(current_data, np.ndarray):
            curr_df = pd.DataFrame(current_data, columns=self.feature_names)
        else:
            curr_df = current_data.copy()

        drifted_features: List[str] = []
        feature_reports: Dict[str, Dict[str, float]] = {}

        for col in self.feature_names:
            if col not in curr_df.columns:
                continue

            ref_vals = self.ref_df[col].dropna().values
            curr_vals = curr_df[col].dropna().values

            if len(ref_vals) == 0 or len(curr_vals) == 0:
                continue

            # 1. KS Test
            ks_stat, p_val = stats.ks_2samp(ref_vals, curr_vals)

            # 2. Wasserstein Distance
            w_dist = float(stats.wasserstein_distance(ref_vals, curr_vals))

            # 3. PSI
            psi_val = self._compute_psi(ref_vals, curr_vals)

            feature_is_drifted = bool(p_val < ks_alpha or psi_val > psi_threshold)
            if feature_is_drifted:
                drifted_features.append(col)

            feature_reports[col] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "wasserstein_distance": w_dist,
                "psi": psi_val,
                "is_drifted": feature_is_drifted,
            }

        drift_ratio = len(drifted_features) / max(1, len(self.feature_names))
        is_overall_drifted = drift_ratio >= 0.3  # > 30% features drifted

        logger.info(
            "Drift check complete: %d/%d features drifted (overall_drifted=%s)",
            len(drifted_features), len(self.feature_names), is_overall_drifted
        )

        return DriftReport(
            is_drifted=is_overall_drifted,
            drift_score=float(drift_ratio),
            drifted_features=drifted_features,
            feature_reports=feature_reports,
            total_features=len(self.feature_names),
        )

    def _compute_psi(self, ref: np.ndarray, curr: np.ndarray, num_bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        eps = 1e-4
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref, quantiles)
        bins[0] -= eps
        bins[-1] += eps

        ref_counts, _ = np.histogram(ref, bins=bins)
        curr_counts, _ = np.histogram(curr, bins=bins)

        ref_pct = ref_counts / max(1, len(ref)) + eps
        curr_pct = curr_counts / max(1, len(curr)) + eps

        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(np.clip(psi, 0.0, 10.0))
