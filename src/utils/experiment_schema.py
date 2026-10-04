"""
src/utils/experiment_schema.py
------------------------------
Standardized experiment result model and persistence layer.

Ensures every pipeline / experiment produces a consistent JSON payload containing:
  - experiment_id, name, created_at, status
  - dataset details & data quality validation report
  - feature engineering pipeline summary
  - models comparison & hyperparameter tuning
  - best model selection & threshold optimization
  - security gate audit results (5 security gates)
  - resource utilization (CPU, memory, training time, latency)
  - runtime monitoring baseline
  - deployment & registry status
  - research comparison metrics
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_standard_experiment_result(
    experiment_id: Optional[str] = None,
    experiment_name: str = "default_experiment",
    config: Optional[dict] = None,
    dataset_meta: Optional[dict] = None,
    val_report: Optional[dict] = None,
    training_results: Optional[List[dict]] = None,
    security_results: Optional[List[dict]] = None,
    registry_info: Optional[dict] = None,
    pipeline_time_sec: float = 0.0,
    status: str = "COMPLETED",
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a standardized experiment result object."""
    if config is None:
        config = load_config()

    exp_id = experiment_id or f"exp_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Process models
    models_summary = {}
    best_model_info = None
    best_f1 = -1.0

    if training_results:
        for item in training_results:
            algo = item.get("algorithm", "unknown")
            metrics = item.get("metrics", {})
            models_summary[algo] = {
                "algorithm": algo,
                "run_id": item.get("run_id"),
                "model_path": item.get("model_path"),
                "hyperparams": item.get("hyperparams", {}),
                "metrics": {
                    "accuracy": metrics.get("test_accuracy", 0.0),
                    "precision": metrics.get("test_precision", 0.0),
                    "recall": metrics.get("test_recall", 0.0),
                    "f1": metrics.get("test_f1", 0.0),
                    "roc_auc": metrics.get("test_roc_auc", 0.5),
                    "latency_p50_ms": metrics.get("test_latency_p50_ms", 0.0),
                    "latency_p95_ms": metrics.get("test_latency_p95_ms", 0.0),
                    "latency_p99_ms": metrics.get("test_latency_p99_ms", 0.0),
                    "training_time_sec": metrics.get("training_time_seconds", 0.0),
                    "optimal_threshold": metrics.get("optimal_threshold", 0.5),
                    "baseline_f1": metrics.get("baseline_test_f1", 0.0),
                    "baseline_precision": metrics.get("baseline_test_precision", 0.0),
                    "baseline_recall": metrics.get("baseline_test_recall", 0.0),
                    "confusion_matrix": metrics.get("confusion_matrix"),
                },
            }
            if metrics.get("test_f1", 0.0) > best_f1:
                best_f1 = metrics.get("test_f1", 0.0)
                best_model_info = {
                    "algorithm": algo,
                    "run_id": item.get("run_id"),
                    "model_path": item.get("model_path"),
                    "optimal_threshold": metrics.get("optimal_threshold", 0.5),
                    "metrics": models_summary[algo]["metrics"],
                }

    # Security summary
    sec_gates_formatted = []
    sec_score_sum = 0.0
    sec_passed_count = 0
    if security_results:
        for r in security_results:
            is_passed = getattr(r, "passed", False) if hasattr(r, "passed") else r.get("passed", False)
            score_val = getattr(r, "score", 0.0) if hasattr(r, "score") else r.get("score", 0.0)
            gate_name = getattr(r, "gate", "unknown") if hasattr(r, "gate") else r.get("gate", "unknown")
            dur = getattr(r, "duration_seconds", 0.0) if hasattr(r, "duration_seconds") else r.get("duration_seconds", 0.0)
            details_dict = getattr(r, "details", {}) if hasattr(r, "details") else r.get("details", {})

            if is_passed:
                sec_passed_count += 1
            sec_score_sum += (score_val or 0.0)

            sec_gates_formatted.append({
                "gate": gate_name,
                "passed": is_passed,
                "score": score_val,
                "duration_seconds": dur,
                "details": details_dict,
            })

    total_gates = len(sec_gates_formatted) if sec_gates_formatted else 5
    overall_sec_passed = (sec_passed_count == total_gates) if sec_gates_formatted else False
    overall_sec_score = float(sec_score_sum / total_gates) if sec_gates_formatted else 0.0

    # Resource metrics
    vm = psutil.virtual_memory()
    resources = {
        "pipeline_execution_time_sec": pipeline_time_sec,
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "ram_used_gb": round(vm.used / (1024 ** 3), 2),
        "ram_usage_percent": vm.percent,
    }

    result = {
        "experiment_id": exp_id,
        "experiment_name": experiment_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "error_message": error_message,
        "config": config,
        "dataset": dataset_meta or {
            "name": config.get("dataset", {}).get("name", "credit_fraud"),
            "source": config.get("dataset", {}).get("source", "sklearn"),
        },
        "data_quality": val_report or {"passed": True},
        "models": models_summary,
        "best_model": best_model_info,
        "security": {
            "overall_passed": overall_sec_passed,
            "overall_score": overall_sec_score,
            "gate_results": sec_gates_formatted,
        },
        "resources": resources,
        "monitoring": {
            "drift_threshold": config.get("monitoring", {}).get("drift_threshold", 0.10),
            "anomaly_detector_active": True,
        },
        "deployment": {
            "stage": registry_info.get("stage", "Staging") if registry_info else "Staging",
            "model_version": registry_info.get("version", "v1.0.0") if registry_info else "v1.0.0",
            "registered": bool(registry_info),
            "api_endpoint": "http://localhost:8000/predict",
        },
        "research_metrics": {
            "f1_score": best_model_info["metrics"]["f1"] if best_model_info else 0.0,
            "precision": best_model_info["metrics"]["precision"] if best_model_info else 0.0,
            "recall": best_model_info["metrics"]["recall"] if best_model_info else 0.0,
            "accuracy": best_model_info["metrics"]["accuracy"] if best_model_info else 0.0,
            "roc_auc": best_model_info["metrics"]["roc_auc"] if best_model_info else 0.5,
            "security_compliance": overall_sec_score,
            "latency_p50_ms": best_model_info["metrics"]["latency_p50_ms"] if best_model_info else 0.0,
        },
    }

    return result


def save_experiment_result(result: Dict[str, Any], results_dir: str = "results") -> Path:
    """Save experiment result JSON to results/experiments/<exp_id>.json and latest_experiment.json."""
    base_dir = Path(results_dir)
    exp_dir = base_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_id = result["experiment_id"]
    file_path = exp_dir / f"{exp_id}.json"

    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    # Update latest_experiment.json
    latest_path = base_dir / "latest_experiment.json"
    with open(latest_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    logger.info("Saved standardized experiment result -> %s", file_path)
    return file_path


def load_experiment_result(experiment_id: str, results_dir: str = "results") -> Optional[Dict[str, Any]]:
    """Load an experiment result by ID or 'latest'."""
    base_dir = Path(results_dir)
    if experiment_id == "latest":
        latest_path = base_dir / "latest_experiment.json"
        if latest_path.exists():
            with open(latest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    file_path = base_dir / "experiments" / f"{experiment_id}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def list_experiments(results_dir: str = "results") -> List[Dict[str, Any]]:
    """List all saved experiment summaries."""
    exp_dir = Path(results_dir) / "experiments"
    if not exp_dir.exists():
        return []

    summaries = []
    for p in sorted(exp_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                summaries.append({
                    "experiment_id": data.get("experiment_id"),
                    "experiment_name": data.get("experiment_name"),
                    "created_at": data.get("created_at"),
                    "status": data.get("status"),
                    "best_model": data.get("best_model", {}).get("algorithm") if data.get("best_model") else None,
                    "f1_score": data.get("research_metrics", {}).get("f1_score", 0.0),
                    "security_score": data.get("security", {}).get("overall_score", 0.0),
                })
        except Exception:
            pass
    return summaries
