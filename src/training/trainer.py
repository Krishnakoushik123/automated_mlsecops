"""
src/training/trainer.py
-----------------------
Trains configured algorithms, evaluates on val+test splits,
and logs everything to MLflow.

Metrics collected (per model)
------------------------------
accuracy, precision, recall, f1, roc_auc,
training_time_seconds, inference_latency_ms (p50/p95/p99)

Run standalone:
    python -m src.training.trainer
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import load_config, get_mlflow_uri
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

_ALGORITHM_MAP: Dict[str, Any] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_model(algorithm: str, hyperparams: dict, seed: int, class_weight: str | None = "balanced") -> Any:
    """Instantiate a sklearn estimator from config with optional class weighting."""
    cls = _ALGORITHM_MAP.get(algorithm)
    if cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    hp = dict(hyperparams)
    
    # Inject random_state and class_weight where supported
    try:
        import inspect
        params = inspect.signature(cls.__init__).parameters
        if "random_state" in params:
            hp.setdefault("random_state", seed)
        if "class_weight" in params and class_weight is not None:
            hp.setdefault("class_weight", class_weight)
    except Exception:
        pass
    return cls(**hp)


# ---------------------------------------------------------------------------
# Threshold Optimization & Metrics
# ---------------------------------------------------------------------------

def find_optimal_threshold(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Find threshold in [0.05, 0.95] that maximizes F1 score on validation set for binary tasks."""
    unique_classes = np.unique(y_val)
    if len(unique_classes) != 2 or not hasattr(model, "predict_proba"):
        return 0.5

    try:
        from sklearn.metrics import precision_recall_curve
        probs = model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
        if len(thresholds) == 0:
            return 0.5
        f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        best_idx = int(np.argmax(f1s))
        best_thresh = float(thresholds[best_idx])
        return float(np.clip(best_thresh, 0.05, 0.95))
    except Exception:
        return 0.5


def compute_metrics(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    n_latency_samples: int = 200,
) -> Dict[str, Any]:
    """Compute classification metrics (binary & multiclass) + inference latency."""
    unique_classes = np.unique(y)
    is_binary = len(unique_classes) <= 2
    
    y_prob = None
    if is_binary and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)

    avg_strategy = "binary" if is_binary else "weighted"

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "accuracy":  float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0, average=avg_strategy)),
        "recall":    float(recall_score(y, y_pred, zero_division=0, average=avg_strategy)),
        "f1":        float(f1_score(y, y_pred, zero_division=0, average=avg_strategy)),
        "optimal_threshold": float(threshold),
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        try:
            if is_binary:
                metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y, y_prob, multi_class="ovr", average="weighted"))
        except ValueError:
            metrics["roc_auc"] = 0.5
    else:
        metrics["roc_auc"] = 0.5

    # Inference latency: run n_latency_samples single-sample predictions
    latencies: List[float] = []
    for _ in range(n_latency_samples):
        idx = np.random.randint(0, len(X))
        sample = X[idx : idx + 1]
        t0 = time.perf_counter()
        model.predict(sample)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies_arr = np.array(latencies)
    metrics["latency_p50_ms"] = float(np.percentile(latencies_arr, 50))
    metrics["latency_p95_ms"] = float(np.percentile(latencies_arr, 95))
    metrics["latency_p99_ms"] = float(np.percentile(latencies_arr, 99))

    return metrics


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
) -> Dict[str, float]:
    """Return mean CV metrics."""
    is_binary = len(np.unique(y)) <= 2
    avg_strat = "binary" if is_binary else "weighted"

    from sklearn.metrics import make_scorer
    scoring = {
        "accuracy":  "accuracy",
        "precision": make_scorer(precision_score, zero_division=0, average=avg_strat),
        "recall":    make_scorer(recall_score, zero_division=0, average=avg_strat),
        "f1":        make_scorer(f1_score, zero_division=0, average=avg_strat),
        "roc_auc":   "roc_auc" if is_binary else "roc_auc_ovr",
    }
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        f"cv_{k}": float(np.mean(v))
        for k, v in results.items()
        if k.startswith("test_")
        for k in [k[5:]]  # strip "test_" prefix
    }


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_model(
    algorithm: str,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    config: dict | None = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train one algorithm, log to MLflow, return result dict."""
    if config is None:
        config = load_config()

    tr_cfg = config["training"]
    ml_cfg = config["mlflow"]
    seed = config["project"]["random_seed"]
    np.random.seed(seed)

    hyperparams = tr_cfg["hyperparameters"].get(algorithm, {})
    model = build_model(algorithm, hyperparams, seed)

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    mlflow.set_tracking_uri(get_mlflow_uri(config))
    experiment_name = f"{ml_cfg['experiment_prefix']}-{config['dataset']['name']}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name or algorithm) as run:
        # Log hyperparams
        mlflow.log_params({f"hp_{k}": v for k, v in hyperparams.items()})
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("random_seed", seed)

        # --- Training ---
        logger.info("Training %s …", algorithm)
        t_start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - t_start
        logger.info("Training %s complete in %.3fs", algorithm, training_time)

        # --- Cross-validation (on train set) ---
        cv_metrics = cross_validate_model(model, X_train, y_train, tr_cfg["cv_folds"], seed)
        logger.info("CV results: %s", cv_metrics)

        # --- Find optimal threshold on validation set ---
        opt_thresh = find_optimal_threshold(model, X_val, y_val)
        logger.info("Optimal threshold for %s: %.3f", algorithm, opt_thresh)

        # --- Held-out eval with optimal & baseline threshold ---
        val_metrics  = {f"val_{k}":  v for k, v in compute_metrics(model, X_val,  y_val, threshold=opt_thresh).items() if k != "confusion_matrix"}
        test_eval    = compute_metrics(model, X_test, y_test, threshold=opt_thresh)
        base_eval    = compute_metrics(model, X_test, y_test, threshold=0.5)

        test_metrics = {f"test_{k}": v for k, v in test_eval.items() if k != "confusion_matrix"}
        test_metrics["confusion_matrix"] = test_eval.get("confusion_matrix")
        test_metrics["baseline_test_f1"] = base_eval["f1"]
        test_metrics["baseline_test_precision"] = base_eval["precision"]
        test_metrics["baseline_test_recall"] = base_eval["recall"]
        test_metrics["optimal_threshold"] = opt_thresh

        all_metrics = {
            "training_time_seconds": training_time,
            **{f"cv_{k}": v for k, v in cv_metrics.items()},
            **val_metrics,
            **{k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
        }
        # Log numeric metrics to MLflow
        mlflow.log_metrics({k: v for k, v in all_metrics.items() if isinstance(v, (int, float))})

        # --- Persist model artifact ---
        models_dir = Path(config["paths"]["models"])
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{algorithm}.joblib"
        joblib.dump(model, model_path)
        try:
            mlflow.sklearn.log_model(model, artifact_path=algorithm, skops_trusted_types=["sklearn.tree._tree.Tree"])
        except Exception:
            try:
                mlflow.sklearn.log_model(model, artifact_path=algorithm)
            except Exception as exc:
                logger.warning("MLflow sklearn log_model warning: %s", exc)

        run_id = run.info.run_id
        logger.info("MLflow run_id=%s  experiment=%s", run_id, experiment_name)

    result = {
        "algorithm":    algorithm,
        "run_id":       run_id,
        "model_path":   str(model_path),
        "metrics":      all_metrics,
        "hyperparams":  hyperparams,
    }
    return result


def train_all(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    config: dict | None = None,
) -> List[Dict[str, Any]]:
    """Train all configured algorithms and return list of result dicts."""
    if config is None:
        config = load_config()

    algorithms = config["training"]["algorithms"]
    results = []
    for algo in algorithms:
        try:
            result = train_model(algo, splits, config)
            results.append(result)
        except Exception as exc:
            logger.error("Failed to train %s: %s", algo, exc, exc_info=True)

    # Save summary JSON
    results_dir = Path(config["paths"]["results"]) / "json"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "training_results.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Training summary saved -> %s", summary_path)
    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    from src.data.loader import load_dataset
    from src.data.validator import validate_dataset
    from src.features.preprocessing import split_dataset, fit_transform_splits

    cfg = load_config()

    # Configure MLflow tracking URI
    mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])

    X, y = load_dataset(cfg)
    passed, _ = validate_dataset(X, y, cfg)
    if not passed:
        raise SystemExit("Data validation failed")

    splits = split_dataset(X, y, cfg)
    transformed_splits, _ = fit_transform_splits(splits, config=cfg)
    results = train_all(transformed_splits, cfg)

    print("\n✓  Training complete")
    from tabulate import tabulate
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append([
            r["algorithm"],
            f"{m.get('test_f1', 0):.4f}",
            f"{m.get('test_accuracy', 0):.4f}",
            f"{m.get('test_roc_auc', 0):.4f}",
            f"{m.get('training_time_seconds', 0):.2f}s",
            f"{m.get('test_latency_p99_ms', 0):.2f}ms",
        ])
    print(tabulate(rows, headers=["Algorithm", "F1", "Accuracy", "ROC-AUC", "Train Time", "Latency P99"]))
