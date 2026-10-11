"""
src/training/pipeline.py
------------------------
End-to-end Phase 1 pipeline orchestrator:

    DATA LOAD -> VALIDATE -> FEATURE PIPELINE -> TRAIN -> REGISTER

Run:
    python -m src.training.pipeline
    python -m src.training.pipeline --register --stage Staging
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np

from src.data.loader import load_dataset, save_raw
from src.data.validator import validate_dataset
from src.features.preprocessing import fit_transform_splits, split_dataset
from src.training.trainer import train_all, build_model
from src.training.registry import register_model, transition_model_stage
from src.security.gates import run_all_gates
from src.utils.config import load_config, get_mlflow_uri
from src.utils.experiment_schema import create_standard_experiment_result, save_experiment_result
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(
    register: bool = False,
    stage: str = "Staging",
    experiment_id: str | None = None,
    experiment_name: str = "pipeline_experiment",
    config: dict | None = None,
    stage_callback: Any = None,
) -> dict:
    """Execute the full MLSecOps pipeline with genuine stage reporting and return standardized result dict."""
    if config is None:
        config = load_config()

    def notify_stage(stg_name: str, details: str = ""):
        if stage_callback is not None:
            try:
                stage_callback(stg_name, details)
            except Exception as cb_err:
                logger.warning("Stage callback error: %s", cb_err)

    mlflow.set_tracking_uri(get_mlflow_uri(config))
    t_pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 1 – Data Ingestion / Upload
    # ------------------------------------------------------------------
    notify_stage("Upload", "Ingesting and snapshotting raw dataset")
    logger.info("=" * 60)
    logger.info("STAGE 1 / DATA INGESTION (UPLOAD)")
    logger.info("=" * 60)
    X, y = load_dataset(config)
    save_raw(X, y, config)

    # ------------------------------------------------------------------
    # Stage 2 – Data Validation / Quality Check
    # ------------------------------------------------------------------
    notify_stage("Quality Check", "Verifying schema, null bounds, and rejection criteria")
    logger.info("=" * 60)
    logger.info("STAGE 2 / DATA VALIDATION (QUALITY CHECK)")
    logger.info("=" * 60)
    passed, val_report = validate_dataset(X, y, config)
    if not passed:
        reasons = val_report.get("rejection_reasons", ["Dataset rejected: failed quality criteria"])
        err_msg = "Data quality check failed:\n - " + "\n - ".join(reasons)
        logger.error(err_msg)
        raise ValueError(err_msg)

    # ------------------------------------------------------------------
    # Stage 3 – Data Cleaning & Repair
    # ------------------------------------------------------------------
    notify_stage("Cleaning", "Safely repairing duplicates, missing values, outliers, and encoding categories")
    logger.info("=" * 60)
    logger.info("STAGE 3 / DATA CLEANING")
    logger.info("=" * 60)
    from src.data.validator import clean_and_repair_dataset
    X_clean, y_clean, cleaning_report = clean_and_repair_dataset(X, y, config)
    val_report["cleaning"] = cleaning_report

    # ------------------------------------------------------------------
    # Feature Pipeline & Splitting
    # ------------------------------------------------------------------
    splits = split_dataset(X_clean, y_clean, config)
    transformed_splits, feature_pipe = fit_transform_splits(splits, config=config)

    # ------------------------------------------------------------------
    # Stage 4 – Model Training
    # ------------------------------------------------------------------
    notify_stage("Training", "Training configured ML models with cross-validation and class weighting")
    logger.info("=" * 60)
    logger.info("STAGE 4 / MODEL TRAINING")
    logger.info("=" * 60)
    results = train_all(transformed_splits, config)

    # ------------------------------------------------------------------
    # Stage 5 – Evaluation
    # ------------------------------------------------------------------
    notify_stage("Evaluation", "Evaluating held-out validation & test sets and optimizing decision thresholds")
    logger.info("=" * 60)
    logger.info("STAGE 5 / MODEL EVALUATION")
    logger.info("=" * 60)
    task_type = config.get("dataset", {}).get("task_type", "classification")
    best = None
    if results:
        if task_type == "classification":
            opt_metric = f"test_{config['training'].get('optimization_metric', 'f1')}"
            best = max(results, key=lambda r: r["metrics"].get(opt_metric, 0))
        else:
            opt_metric = f"test_{config['training'].get('optimization_metric', 'rmse')}"
            if opt_metric in ["test_rmse", "test_mae"]:
                best = min(results, key=lambda r: r["metrics"].get(opt_metric, float('inf')))
            else:
                best = max(results, key=lambda r: r["metrics"].get(opt_metric, -float('inf')))

    # ------------------------------------------------------------------
    # Stage 6 – Security Gates Verification
    # ------------------------------------------------------------------
    notify_stage("Security", "Running automated security gates: dependencies, secrets, poisoning, robustness")
    logger.info("=" * 60)
    logger.info("STAGE 6 / SECURITY GATES")
    logger.info("=" * 60)
    sec_results = []
    if results and best is not None:
        best_algo = best["algorithm"]
        best_model_path = Path(best["model_path"])
        # Performance optimization: load already-trained best model instead of re-fitting
        best_model_obj = joblib.load(best_model_path)

        sec_results = run_all_gates(
            config=config,
            model=best_model_obj,
            model_path=best_model_path,
            X_train=transformed_splits["train"][0],
            y_train=transformed_splits["train"][1],
            X_test=transformed_splits["test"][0],
            y_test=transformed_splits["test"][1],
        )

        # Save model artifacts to the experiment results directory
        if experiment_id:
            exp_dir = Path(config["paths"]["results"]) / experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model_obj, exp_dir / "best_model.joblib")
            
            # Save feature pipeline
            feature_pipe_path = Path(config["paths"]["models"]) / "feature_pipeline.joblib"
            if feature_pipe_path.exists():
                import shutil
                shutil.copy2(feature_pipe_path, exp_dir / "feature_pipeline.joblib")
            
            # Also overwrite the global 'latest' model for default API startup
            joblib.dump(best_model_obj, Path(config["paths"]["models"]) / "best_model.joblib")

    # ------------------------------------------------------------------
    # Stage 7 – Packaging & Model Registry
    # ------------------------------------------------------------------
    notify_stage("Packaging", "Persisting model registry, metadata, and deployable artifacts")
    registry_info = None
    if register and results and best is not None:
        logger.info("=" * 60)
        logger.info("STAGE 7 / MODEL PACKAGING & REGISTRY")
        logger.info("=" * 60)
        logger.info(
            "Best model: %s  (%s=%.4f)",
            best["algorithm"],
            opt_metric,
            best["metrics"].get(opt_metric, 0),
        )
        try:
            version = register_model(
                run_id=best["run_id"],
                artifact_path=best["algorithm"],
                config=config,
            )
            transition_model_stage(
                model_name=config["mlflow"]["model_name"],
                version=version,
                stage=stage,
                config=config,
            )
            registry_info = {
                "model_name": config["mlflow"]["model_name"],
                "version":    version,
                "stage":      stage,
                "algorithm":  best["algorithm"],
                "run_id":     best["run_id"],
            }
        except Exception as exc:
            logger.error("Registry step failed: %s", exc)

    pipeline_time = time.perf_counter() - t_pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.2fs", pipeline_time)
    logger.info("=" * 60)

    # Construct and persist standardized experiment result payload
    task_type = config.get("dataset", {}).get("task_type", "classification")
    file_path_str = config["dataset"].get("file_path", "")
    file_name_str = config["dataset"].get("file_name") or (Path(file_path_str).name if file_path_str else "dataset.csv")
    from src.data.loader import generate_dataset_id
    dataset_id_str = config["dataset"].get("dataset_id") or generate_dataset_id(file_name_str)

    dataset_meta = {
        "name": Path(file_name_str).stem if file_name_str else config["dataset"].get("name", "dataset"),
        "file_name": file_name_str,
        "dataset_id": dataset_id_str,
        "source": config["dataset"].get("source", "sklearn"),
        "file_path": file_path_str,
        "target_column": config["dataset"].get("target_column", "target"),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "task_type": task_type,
    }
    if task_type == "classification":
        dataset_meta["n_classes"] = len(np.unique(y))
        dataset_meta["class_distribution"] = {str(k): int(v) for k, v in y.value_counts().items()}

    std_result = create_standard_experiment_result(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        config=config,
        dataset_meta=dataset_meta,
        val_report=val_report,
        training_results=results,
        security_results=sec_results,
        registry_info=registry_info,
        pipeline_time_sec=pipeline_time,
        status="COMPLETED",
    )

    save_experiment_result(std_result, results_dir=config["paths"]["results"])

    # Also output summary dict for backward compatibility
    summary = {
        "pipeline_time_seconds": pipeline_time,
        "validation_passed":     passed,
        "training_results":      results,
        "registry":              registry_info,
        "standard_result":       std_result,
    }

    results_dir = Path(config["paths"]["results"]) / "json"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "pipeline_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Pipeline summary -> %s", summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run Phase 1 MLSecOps pipeline")
    parser.add_argument(
        "--register", action="store_true",
        help="Register best model in MLflow registry after training"
    )
    parser.add_argument(
        "--stage", default="Staging",
        choices=["Staging", "Production"],
        help="Registry stage for the registered model (default: Staging)"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to alternate config.yaml"
    )
    args = parser.parse_args()

    from src.utils.config import load_config as _load
    cfg = _load(args.config)

    summary = run_pipeline(
        register=args.register,
        stage=args.stage,
        config=cfg,
    )

    # Pretty-print results table
    from tabulate import tabulate
    rows = []
    for r in summary["training_results"]:
        m = r["metrics"]
        rows.append([
            r["algorithm"],
            f"{m.get('test_f1', 0):.4f}",
            f"{m.get('test_accuracy', 0):.4f}",
            f"{m.get('test_roc_auc', 0):.4f}",
            f"{m.get('training_time_seconds', 0):.2f}s",
            f"{m.get('test_latency_p99_ms', 0):.2f}ms",
        ])
    print("\n" + tabulate(rows, headers=["Algorithm", "F1", "Accuracy", "ROC-AUC", "Train Time", "P99 Latency"]))
    if summary["registry"]:
        print(f"\n✓  Registered: {summary['registry']}")
