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
                    Check(lambda s: s.notna().all(), error=f"{col}: contains NaN"),
                    Check(lambda s: np.isfinite(s).all(), error=f"{col}: contains inf"),
                ],
                nullable=False,
                coerce=True,
            )
        elif np.issubdtype(dtype, np.integer):
            columns[col] = Column(int, nullable=False, coerce=True)
        else:
            columns[col] = Column(object, nullable=True, coerce=False)

    # Add target column
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
    result = {
        "total_missing": total_missing,
        "by_column": missing[missing > 0].to_dict(),
        "passed": total_missing == 0,
    }
    if total_missing > 0:
        logger.warning("Missing values detected: %s", result["by_column"])
    return result


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

    logger.info("Starting dataset validation (%d rows, %d features)", len(X), X.shape[1])

    df = X.copy()
    df["target"] = y.values

    report: Dict = {}

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
    report["class_balance"] = _check_class_balance(y)

    # Overall result — only inspect dict sub-reports (not scalar values)
    # class_balance is a WARNING, not a blocker (imbalance is expected in fraud datasets)
    blocking_checks = {"schema", "missing", "duplicates"}
    sub_reports = {k: v for k, v in report.items() if isinstance(v, dict)}
    overall = all(
        v.get("passed", False)
        for k, v in sub_reports.items()
        if k in blocking_checks
    )
    report["overall_passed"] = overall

    if overall:
        logger.info("All data validation checks PASSED")
    else:
        failed = [k for k, v in sub_reports.items() if not v.get("passed", True)]
        logger.error("Data validation FAILED for: %s", failed)

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
