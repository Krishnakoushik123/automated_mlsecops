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
) -> dict:
    """Execute the full MLSecOps pipeline and return standardized result dict."""
    if config is None:
        config = load_config()

    mlflow.set_tracking_uri(get_mlflow_uri(config))
    t_pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 1 – Data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1 / DATA INGESTION")
    logger.info("=" * 60)
    X, y = load_dataset(config)
    save_raw(X, y, config)

    # ------------------------------------------------------------------
    # Stage 2 – Validation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2 / DATA VALIDATION")
    logger.info("=" * 60)
    passed, val_report = validate_dataset(X, y, config)
    if not passed:
        raise RuntimeError("Data validation failed – pipeline aborted.\n" + json.dumps(val_report, default=str))

    # ------------------------------------------------------------------
    # Stage 3 – Feature pipeline
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3 / FEATURE PIPELINE")
    logger.info("=" * 60)
    splits = split_dataset(X, y, config)
    transformed_splits, feature_pipe = fit_transform_splits(splits, config=config)

    # ------------------------------------------------------------------
    # Stage 4 – Training
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 4 / MODEL TRAINING")
    logger.info("=" * 60)
    results = train_all(transformed_splits, config)

    # ------------------------------------------------------------------
    # Stage 5 – Security Gates Verification
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 5 / SECURITY GATES")
    logger.info("=" * 60)
    sec_results = []
    if results:
        task_type = config.get("dataset", {}).get("task_type", "classification")
        if task_type == "classification":
            opt_metric = f"test_{config['training'].get('optimization_metric', 'f1')}"
            best = max(results, key=lambda r: r["metrics"].get(opt_metric, 0))
        else:
            opt_metric = f"test_{config['training'].get('optimization_metric', 'rmse')}"
            if opt_metric in ["test_rmse", "test_mae"]:
                best = min(results, key=lambda r: r["metrics"].get(opt_metric, float('inf')))
            else:
                best = max(results, key=lambda r: r["metrics"].get(opt_metric, -float('inf')))
                
        best_algo = best["algorithm"]
        best_model_path = Path(best["model_path"])
        best_model_obj = build_model(
            best_algo,
            config["training"]["hyperparameters"].get(best_algo, {}),
            seed=config["project"]["random_seed"],
        )
        best_model_obj.fit(transformed_splits["train"][0], transformed_splits["train"][1])

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
            
            # Also overwrite the global 'latest' model for the default API startup
            joblib.dump(best_model_obj, Path(config["paths"]["models"]) / "best_model.joblib")

    # ------------------------------------------------------------------
    # Stage 6 – (Optional) Model Registry
    # ------------------------------------------------------------------
    registry_info = None
    if register and results:
        logger.info("=" * 60)
        logger.info("STAGE 6 / MODEL REGISTRY")
        logger.info("=" * 60)
        if task_type == "classification":
            opt_metric = f"test_{config['training'].get('optimization_metric', 'f1')}"
            best = max(results, key=lambda r: r["metrics"].get(opt_metric, 0))
        else:
            opt_metric = f"test_{config['training'].get('optimization_metric', 'rmse')}"
            if opt_metric in ["test_rmse", "test_mae"]:
                best = min(results, key=lambda r: r["metrics"].get(opt_metric, float('inf')))
            else:
                best = max(results, key=lambda r: r["metrics"].get(opt_metric, -float('inf')))
                
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
    dataset_meta = {
        "name": config["dataset"].get("name", "credit_fraud"),
        "source": config["dataset"].get("source", "sklearn"),
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
