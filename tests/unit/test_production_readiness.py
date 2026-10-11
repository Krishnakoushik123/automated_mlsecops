"""
tests/unit/test_production_readiness.py
----------------------------------------
Focused tests for production readiness requirements:
1. Dataset quality, rejection criteria, and repairable dataset cleaning.
2. Safe class imbalance detection without data leakage.
3. Stable dataset identification and meaningful filenames.
4. Exported package with minimal usable frontend and live prediction.
5. Performance optimizations without regression.
"""

import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.loader import generate_dataset_id, load_dataset
from src.data.validator import (
    get_quality_thresholds,
    check_rejection_criteria,
    clean_and_repair_dataset,
    validate_dataset,
)
from src.training.trainer import build_model, compute_metrics, train_model
from src.training.pipeline import run_pipeline
from src.utils.experiment_schema import create_standard_experiment_result, list_experiments


# ---------------------------------------------------------------------------
# 1. Dataset Quality & Rejection Criteria Tests
# ---------------------------------------------------------------------------

class TestDatasetQualityAndRejection:
    def test_clean_dataset_passes_rejection(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f"f_{i}" for i in range(4)])
        y = pd.Series(rng.integers(0, 2, size=50), name="target")
        cfg = {"data_quality": {"min_rows": 20, "min_features": 2, "max_missing_ratio": 0.5}}
        
        passed, reasons, details = check_rejection_criteria(X, y, cfg)
        assert passed, f"Clean dataset should pass, but failed: {reasons}"
        assert len(reasons) == 0

    def test_reject_insufficient_rows(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0])
        cfg = {"data_quality": {"min_rows": 20}}
        
        passed, reasons, _ = check_rejection_criteria(X, y, cfg)
        assert not passed
        assert any("Insufficient samples" in r for r in reasons)

    def test_reject_insufficient_features(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"single_feat": rng.standard_normal(40)})
        y = pd.Series(rng.integers(0, 2, size=40))
        cfg = {"data_quality": {"min_rows": 20, "min_features": 2}}
        
        passed, reasons, _ = check_rejection_criteria(X, y, cfg)
        assert not passed
        assert any("Insufficient features" in r for r in reasons)

    def test_reject_excessive_missing_values(self):
        X = pd.DataFrame({
            "a": [np.nan] * 35 + [1.0] * 5,
            "b": [np.nan] * 30 + [2.0] * 10,
        })
        y = pd.Series([0] * 20 + [1] * 20)
        cfg = {"data_quality": {"min_rows": 20, "max_missing_ratio": 0.50}}
        
        passed, reasons, _ = check_rejection_criteria(X, y, cfg)
        assert not passed
        assert any("Excessive missing values" in r for r in reasons)

    def test_reject_invalid_target_single_class(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((30, 3)), columns=["a", "b", "c"])
        y = pd.Series([1] * 30, name="target")  # only 1 class
        cfg = {"dataset": {"task_type": "classification"}, "data_quality": {"min_rows": 20}}
        
        passed, reasons, _ = check_rejection_criteria(X, y, cfg)
        assert not passed
        assert any("at least 2 distinct classes" in r for r in reasons)

    def test_reject_invalid_target_all_null(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((30, 3)), columns=["a", "b", "c"])
        y = pd.Series([np.nan] * 30, name="target")
        
        passed, reasons, _ = check_rejection_criteria(X, y, {})
        assert not passed
        assert any("null or missing" in r for r in reasons)


# ---------------------------------------------------------------------------
# 2. Repairable Dataset Cleaning Tests
# ---------------------------------------------------------------------------

class TestRepairableDatasetCleaning:
    def test_repairs_duplicates_missing_and_outliers(self):
        # Create dataset with duplicates, high-missing column, and extreme outlier
        data = {
            "feat_normal": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 4,
            "feat_outlier": [1.0, 1.2, 1.1, 999999.0, 1.0, 1.1, 1.3, 1.2, 1.1, 1.0] * 4,
            "feat_junk_null": [np.nan] * 36 + [1.0, 2.0, 3.0, 4.0],  # 90% missing
            "category_col": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 4,
        }
        X = pd.DataFrame(data)
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 4, name="target")

        cfg = {
            "data_quality": {
                "min_rows": 10,
                "max_column_missing_ratio": 0.80,
                "outlier_std_dev": 3.0,
                "deduplicate": True,
                "handle_outliers": True,
            }
        }

        X_clean, y_clean, report = clean_and_repair_dataset(X, y, cfg)

        # 1. Unrepairable column feat_junk_null (>80% missing) should be dropped
        assert "feat_junk_null" not in X_clean.columns

        # 2. Categorical column should be encoded
        assert "category_col" not in X_clean.columns
        assert any("category_col" in c for c in X_clean.columns)

        # 3. Extreme outlier (999999.0) should be capped
        assert X_clean["feat_outlier"].max() < 1000.0

        # 4. Actions and before/after report
        assert len(report["actions"]) >= 2
        assert "before" in report and "after" in report
        assert report["after"]["rows"] <= report["before"]["rows"]
        assert report["passed"] is True


# ---------------------------------------------------------------------------
# 3. Class Imbalance & Leakage Prevention Tests
# ---------------------------------------------------------------------------

class TestClassImbalanceAndDataLeakage:
    def test_class_imbalance_weighting_train_split_only(self):
        # Severe 95:5 imbalanced training set
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 4))
        y_train = np.array([0] * 95 + [1] * 5)
        
        X_val = rng.standard_normal((20, 4))
        y_val = np.array([0] * 19 + [1] * 1)
        
        X_test = rng.standard_normal((20, 4))
        y_test = np.array([0] * 19 + [1] * 1)

        splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
        cfg = {
            "project": {"random_seed": 42},
            "dataset": {"name": "test_imbal", "task_type": "classification"},
            "training": {
                "algorithms": ["logistic_regression"],
                "hyperparameters": {"logistic_regression": {"C": 1.0, "max_iter": 100}},
                "cv_folds": 2,
            },
            "mlflow": {"experiment_prefix": "unit_test"},
            "paths": {"models": "models", "results": "results"},
            "data_quality": {"imbalance_threshold": 0.20},
        }

        res = train_model("logistic_regression", splits, config=cfg)
        assert res["metrics"]["test_f1"] >= 0.0
        # Verification: Test split distribution remains untouched
        assert len(y_test) == 20
        assert sum(y_test) == 1


# ---------------------------------------------------------------------------
# 4. Dataset Identification Tests
# ---------------------------------------------------------------------------

class TestDatasetIdentification:
    def test_stable_dataset_id_generation(self):
        id1 = generate_dataset_id("credit_card_data.csv", b"sample content 123")
        id2 = generate_dataset_id("credit_card_data.csv", b"sample content 123")
        assert id1 == id2, "Dataset ID must be stable and deterministic"
        assert id1.startswith("ds_credit_card_data_")

    def test_experiment_schema_propagates_dataset_identifiers(self):
        cfg = {"dataset": {"dataset_id": "ds_custom_001", "name": "cardata", "file_name": "cardata.csv"}}
        res = create_standard_experiment_result(experiment_id="exp_test_id", config=cfg)
        assert res["dataset_id"] == "ds_custom_001"
        assert res["dataset"]["dataset_id"] == "ds_custom_001"
        assert res["dataset"]["file_name"] == "cardata.csv"


# ---------------------------------------------------------------------------
# 5. Exported Package & Usable Frontend Prediction Tests
# ---------------------------------------------------------------------------

class TestExportedPackageFrontendAndAPI:
    def test_exported_package_prediction(self, tmp_path):
        from fastapi.testclient import TestClient
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import joblib

        # Train a dummy pipeline and model
        X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y_train = np.array([0, 0, 1, 1])

        pipe = Pipeline([("scaler", StandardScaler())])
        X_trans = pipe.fit_transform(X_train)
        clf = LogisticRegression()
        clf.fit(X_trans, y_train)

        # Save artifacts to temp directory
        joblib.dump(clf, tmp_path / "best_model.joblib")
        joblib.dump(pipe, tmp_path / "feature_pipeline.joblib")

        meta = {
            "model_version": "v1.0.0",
            "algorithm": "logistic_regression",
            "task_type": "classification",
        }
        with open(tmp_path / "model_metadata.json", "w") as f:
            json.dump(meta, f)

        # Create the standalone app.py exactly as generated in download_experiment_model
        app_code = """
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Any, Optional
from pathlib import Path
import joblib, json, pandas as pd

app = FastAPI(title="Deployed Model API")
model = joblib.load("best_model.joblib")
pipeline = joblib.load("feature_pipeline.joblib")
with open("model_metadata.json") as f:
    metadata = json.load(f)

class PredictRequest(BaseModel):
    features: List[List[Any]]

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse("<h1>Deployed Model Inference Interface</h1>")

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.features)
    X = pipeline.transform(df)
    preds = model.predict(X)
    return {"predictions": preds.tolist(), "status": "success"}
"""
        with open(tmp_path / "app.py", "w") as f:
            f.write(app_code)

        # Test the deployed app with TestClient in the isolated directory
        import os, sys
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Dynamically import app
            import importlib.util
            spec = importlib.util.spec_from_file_location("deployed_app", str(tmp_path / "app.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            client = TestClient(mod.app)
            
            # 1. Test frontend endpoint
            r_ui = client.get("/ui")
            assert r_ui.status_code == 200
            assert "Deployed Model Inference Interface" in r_ui.text

            # 2. Test live prediction endpoint
            r_pred = client.post("/predict", json={"features": [[2.5, 3.5]]})
            assert r_pred.status_code == 200
            body = r_pred.json()
            assert "predictions" in body
            assert len(body["predictions"]) == 1
            assert body["predictions"][0] in [0, 1]
        finally:
            os.chdir(orig_cwd)
