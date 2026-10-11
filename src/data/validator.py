"""
src/data/validator.py
---------------------
Pandera-based schema + quality validation for the raw dataset.

Checks performed
----------------
1. Column presence and dtype coercion
2. No NaN in critical columns (configurable)
3. Target distribution (class imbalance warning)
4. Duplicate row detection
5. Out-of-range value detection via Pandera checks

Run standalone:
    python -m src.data.validator
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema factory  (generated from data characteristics)
# ---------------------------------------------------------------------------

def build_schema(df: pd.DataFrame, target_col: str = "target") -> DataFrameSchema:
    """Dynamically build a Pandera schema from the DataFrame columns."""
    columns: Dict[str, Column] = {}
    for col in df.columns:
        if col == target_col:
            continue
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.floating):
            columns[col] = Column(
                float,
                checks=[
                    Check(lambda s: np.isfinite(s.dropna()).all(), error=f"{col}: contains inf"),
                ],
                nullable=True,
                coerce=True,
            )
        elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
            columns[col] = Column(int, nullable=True, coerce=True)
        else:
            columns[col] = Column(object, nullable=True, coerce=False)

    # Add target column (float if regression/continuous, int if classification/discrete)
    target_dtype = df[target_col].dtype
    if np.issubdtype(target_dtype, np.floating):
        columns[target_col] = Column(float, nullable=False, coerce=True)
    else:
        columns[target_col] = Column(int, nullable=False, coerce=True)

    return DataFrameSchema(columns, strict=False)


# ---------------------------------------------------------------------------
# Quality checks  (beyond Pandera schema)
# ---------------------------------------------------------------------------

def _check_duplicates(df: pd.DataFrame) -> Dict:
    n_dup = df.duplicated().sum()
    result = {"n_duplicates": int(n_dup), "passed": n_dup == 0}
    if n_dup > 0:
        logger.warning("Found %d duplicate rows", n_dup)
    return result


def _check_class_balance(y: pd.Series, warn_threshold: float = 0.10) -> Dict:
    # Check if target is continuous
    if len(np.unique(y)) > 20 or not np.issubdtype(y.dtype, np.integer):
        return {"passed": True, "task": "regression", "note": "Continuous target"}

    counts = y.value_counts(normalize=True)
    min_ratio = float(counts.min())
    passed = min_ratio >= warn_threshold
    result = {
        "class_distribution": counts.to_dict(),
        "min_class_ratio": min_ratio,
        "passed": passed,
    }
    if not passed:
        logger.warning(
            "Severe class imbalance detected: min class ratio=%.3f (threshold=%.2f)",
            min_ratio,
            warn_threshold,
        )
    return result


def _check_missing(df: pd.DataFrame) -> Dict:
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    missing_ratio = total_missing / (len(df) * max(1, len(df.columns)))
    passed = (total_missing == 0)
    result = {
        "total_missing": total_missing,
        "missing_ratio": missing_ratio,
        "by_column": missing[missing > 0].to_dict(),
        "passed": passed,
    }
    if total_missing > 0:
        logger.warning("Missing values detected: %s (handled by SimpleImputer)", result["by_column"])
    return result


def get_quality_thresholds(config: dict | None = None) -> Dict[str, Any]:
    """Retrieve configurable data quality thresholds with robust defaults."""
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}

    dq_cfg = (config or {}).get("data_quality", {})
    return {
        "min_rows": dq_cfg.get("min_rows", 20),
        "min_features": dq_cfg.get("min_features", 2),
        "max_missing_ratio": dq_cfg.get("max_missing_ratio", 0.50),
        "max_column_missing_ratio": dq_cfg.get("max_column_missing_ratio", 0.80),
        "imbalance_threshold": dq_cfg.get("imbalance_threshold", 0.20),
        "outlier_std_dev": dq_cfg.get("outlier_std_dev", 4.0),
        "deduplicate": dq_cfg.get("deduplicate", True),
        "handle_outliers": dq_cfg.get("handle_outliers", True),
    }


def check_rejection_criteria(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict | None = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Verify if dataset meets minimum viability thresholds or must be rejected.
    
    Returns:
        (passed, rejection_reasons, details_dict)
    """
    thresholds = get_quality_thresholds(config)
    task_type = (config or {}).get("dataset", {}).get("task_type", "classification")
    reasons: List[str] = []

    # 1. Row count check
    n_rows = len(X)
    if n_rows < thresholds["min_rows"]:
        reasons.append(
            f"Insufficient samples: dataset has {n_rows} rows; minimum required is {thresholds['min_rows']}."
        )

    # 2. Feature count check
    n_features = X.shape[1]
    if n_features < thresholds["min_features"]:
        reasons.append(
            f"Insufficient features: dataset has {n_features} features; minimum required is {thresholds['min_features']}."
        )

    # 3. Target validity
    if y is None or len(y) == 0:
        reasons.append("Target column is empty.")
    else:
        valid_y = y.dropna()
        if len(valid_y) == 0:
            reasons.append("Target column contains only null or missing values.")
        elif (y.isnull().sum() / len(y)) > thresholds["max_missing_ratio"]:
            reasons.append(
                f"Target column missingness ({y.isnull().mean():.1%}) exceeds threshold ({thresholds['max_missing_ratio']:.1%})."
            )
        elif task_type == "classification":
            unique_classes = np.unique(valid_y)
            if len(unique_classes) < 2:
                reasons.append(
                    f"Classification target must have at least 2 distinct classes; found {len(unique_classes)}."
                )
            else:
                counts = valid_y.value_counts()
                if (counts < 2).any():
                    reasons.append(
                        "Each classification class must have at least 2 samples for validation splitting."
                    )
        else:
            # Regression target
            y_num = pd.to_numeric(valid_y, errors="coerce").dropna()
            if len(y_num) == 0:
                reasons.append("Regression target contains no valid numeric values.")
            elif float(np.var(y_num)) == 0.0:
                reasons.append("Regression target has zero variance (constant value).")

    # 4. Overall missing ratio check
    total_cells = max(1, len(X) * max(1, n_features))
    missing_ratio = float(X.isnull().sum().sum() / total_cells)
    if missing_ratio > thresholds["max_missing_ratio"]:
        reasons.append(
            f"Excessive missing values: overall missing ratio is {missing_ratio:.1%}, exceeding threshold {thresholds['max_missing_ratio']:.1%}."
        )

    # 5. Unrecoverable feature columns (all columns missing > max_column_missing_ratio)
    col_missing_ratios = X.isnull().mean()
    high_missing_cols = [c for c, r in col_missing_ratios.items() if r > thresholds["max_column_missing_ratio"]]
    if len(high_missing_cols) == n_features and n_features > 0:
        reasons.append(
            f"All {n_features} feature columns exceed maximum allowable column missingness of {thresholds['max_column_missing_ratio']:.1%}."
        )

    passed = len(reasons) == 0
    details = {
        "passed": passed,
        "n_rows": n_rows,
        "n_features": n_features,
        "missing_ratio": missing_ratio,
        "rejection_reasons": reasons,
        "thresholds": thresholds,
    }
    return passed, reasons, details


def clean_and_repair_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict | None = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Clean repairable dataset safely, preserving data integrity and reporting all actions.
    
    Handles:
      - Rows with missing target values
      - Unrepairable columns with excessive missingness (> max_column_missing_ratio)
      - Duplicate rows
      - Categorical and string features (encoding)
      - Infinite values and extreme noisy outliers (winsorizing/capping)
    
    Returns:
        (X_cleaned, y_cleaned, quality_report)
    """
    thresholds = get_quality_thresholds(config)
    task_type = (config or {}).get("dataset", {}).get("task_type", "classification")
    actions: List[str] = []

    # Record BEFORE quality
    total_cells_before = max(1, len(X) * max(1, X.shape[1]))
    before_stats = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "duplicate_rows": int(X.duplicated().sum()),
        "total_missing": int(X.isnull().sum().sum()),
        "missing_ratio": round(float(X.isnull().sum().sum() / total_cells_before), 4),
        "class_distribution": y.value_counts().to_dict() if task_type == "classification" and y.notnull().any() else None,
    }

    X_clean = X.copy()
    y_clean = y.copy()

    # 1. Missing target handling
    valid_y_mask = y_clean.notnull()
    if not valid_y_mask.all():
        n_dropped = int((~valid_y_mask).sum())
        X_clean = X_clean[valid_y_mask].copy()
        y_clean = y_clean[valid_y_mask].copy()
        actions.append(f"Dropped {n_dropped} row(s) with missing target values")

    # 2. Drop unrepairable columns with excessive missingness
    col_missing = X_clean.isnull().mean()
    drop_cols = [c for c, r in col_missing.items() if r > thresholds["max_column_missing_ratio"]]
    if drop_cols:
        X_clean = X_clean.drop(columns=drop_cols)
        actions.append(
            f"Dropped {len(drop_cols)} unrepairable column(s) exceeding {thresholds['max_column_missing_ratio']:.0%} missing values: {drop_cols}"
        )

    # 3. Deduplication
    if thresholds["deduplicate"]:
        n_dups = int(X_clean.duplicated().sum())
        if n_dups > 0:
            dup_mask = ~X_clean.duplicated()
            X_clean = X_clean[dup_mask].copy()
            y_clean = y_clean[dup_mask].copy()
            actions.append(f"Removed {n_dups} duplicate row(s)")

    # 4. Categorical / string feature encoding
    cat_cols = X_clean.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if len(cat_cols) > 0:
        X_clean = pd.get_dummies(X_clean, columns=cat_cols, drop_first=True, dtype=float)
        actions.append(f"One-hot encoded {len(cat_cols)} categorical feature(s): {cat_cols}")

    # 5. Outlier & Infinity handling for numeric features
    num_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts = 0
    inf_cols = []
    
    for col in num_cols:
        # Replace inf/-inf with NaN
        inf_mask = np.isinf(X_clean[col])
        if inf_mask.any():
            X_clean.loc[inf_mask, col] = np.nan
            inf_cols.append(col)

        # Capping extreme outliers (robust IQR + std dev winsorization)
        if thresholds["handle_outliers"]:
            s = X_clean[col].dropna()
            if len(s) >= 4:
                q25, q75 = s.quantile(0.25), s.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - 3.0 * iqr
                    upper_bound = q75 + 3.0 * iqr
                else:
                    std_val = s.std()
                    mean_val = s.mean()
                    k = thresholds["outlier_std_dev"]
                    lower_bound = mean_val - k * max(std_val, 1e-6)
                    upper_bound = mean_val + k * max(std_val, 1e-6)

                outlier_mask = (X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)
                n_col_outliers = int(outlier_mask.sum())
                if n_col_outliers > 0:
                    X_clean[col] = X_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    outlier_counts += n_col_outliers

    if inf_cols:
        actions.append(f"Sanitized infinite values to NaN across {len(inf_cols)} column(s)")
    if outlier_counts > 0:
        actions.append(f"Safely capped {outlier_counts} extreme noisy outlier values using std dev threshold {thresholds['outlier_std_dev']}")

    # Record AFTER quality
    total_cells_after = max(1, len(X_clean) * max(1, X_clean.shape[1]))
    after_stats = {
        "rows": int(len(X_clean)),
        "features": int(X_clean.shape[1]),
        "duplicate_rows": int(X_clean.duplicated().sum()),
        "total_missing": int(X_clean.isnull().sum().sum()),
        "missing_ratio": round(float(X_clean.isnull().sum().sum() / total_cells_after), 4),
        "actions_applied": actions,
    }

    quality_report = {
        "before": before_stats,
        "after": after_stats,
        "actions": actions,
        "passed": True,
    }
    logger.info("Dataset cleaning complete. %d actions performed: %s", len(actions), actions)
    return X_clean, y_clean, quality_report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict | None = None,
) -> Tuple[bool, Dict]:
    """Run all validation checks.  Returns (overall_pass, report_dict)."""
    if config is None:
        config = load_config()

    thresholds = get_quality_thresholds(config)
    logger.info("Starting dataset validation (%d rows, %d features)", len(X), X.shape[1])

    # Rejection criteria check
    rejection_passed, reasons, rejection_details = check_rejection_criteria(X, y, config)

    df = X.copy()
    df["target"] = y.values

    report: Dict = {
        "rejection_check": rejection_details,
        "rejection_reasons": reasons,
    }

    # 1. Schema validation
    try:
        schema = build_schema(df)
        schema.validate(df, lazy=True)
        report["schema"] = {"passed": True, "errors": []}
        logger.info("Schema validation: PASSED")
    except pa.errors.SchemaErrors as exc:
        errors = exc.failure_cases.to_dict(orient="records")
        report["schema"] = {"passed": False, "errors": errors}
        logger.error("Schema validation: FAILED – %d error(s)", len(errors))

    # 2. Missing values
    report["missing"] = _check_missing(df)

    # 3. Duplicates
    report["duplicates"] = _check_duplicates(df)

    # 4. Class balance
    report["class_balance"] = _check_class_balance(y, warn_threshold=thresholds["imbalance_threshold"])

    # Overall result — schema failure, rejection criteria, or excessive missing ratio blocks execution
    schema_passed = report["schema"].get("passed", False)
    missing_ok = report["missing"].get("missing_ratio", 0.0) <= thresholds["max_missing_ratio"]
    overall = rejection_passed and schema_passed and missing_ok
    report["overall_passed"] = overall

    if overall:
        logger.info("All data validation checks PASSED")
    else:
        logger.error("Data validation FAILED (schema_passed=%s, missing_ok=%s, rejection_passed=%s, reasons=%s)",
                     schema_passed, missing_ok, rejection_passed, reasons)

    return overall, report


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.loader import load_dataset, save_raw

    cfg = load_config()
    X, y = load_dataset(cfg)
    save_raw(X, y, cfg)
    passed, report = validate_dataset(X, y, cfg)
    import json, sys
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if passed else 1)
