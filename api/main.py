"""
api/main.py
-----------
FastAPI Production Inference Service with integrated MLSecOps Security & Monitoring:
  - Input payload schema validation & sanitization
  - Feature preprocessing pipeline & ML model prediction
  - Real-time input anomaly & adversarial input scanning
  - Automated threat mitigation & fallback response
  - Prometheus monitoring endpoint (/metrics)
  - Security audit report endpoint (/security/audit)
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import uuid
from fastapi import FastAPI, HTTPException, Request, Response, status, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score

from src.monitoring.anomaly_detector import AnomalyDetector
from src.monitoring.auto_responder import AutoResponder
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.prometheus_exporter import (
    get_metrics_payload,
    record_anomaly,
    record_inference,
    update_security_gate_status,
)
from src.security.gates import gate_dependency_audit, gate_secrets_scan
from src.training.pipeline import run_pipeline
from src.utils.config import load_config
from src.utils.experiment_schema import list_experiments, load_experiment_result
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="automated-mlsecops Inference & Security API",
    version="1.0.0",
    description="Production ML inference service with runtime security-by-design & anomaly auto-response",
)

# Global model state
_STATE: Dict[str, Any] = {
    "config": None,
    "model": None,
    "feature_pipeline": None,
    "anomaly_detector": None,
    "drift_detector": None,
    "auto_responder": None,
    "model_version": "v1.0.0",
    "prediction_logs": [],
    "ground_truth_logs": [],
}

# In-memory store for running experiments
_RUNNING_EXPERIMENTS: Dict[str, Any] = {}


# --- Request & Response Models ---

class PredictRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description="2D array of numerical feature vectors, e.g. [[0.5, -1.2, 0.8, ...]]",
        example=[[0.12, -0.45, 1.23, 0.05, -0.88, 0.34, 0.91, -0.12, 0.45, 0.67, -0.23, 0.11, -0.78, 0.89, -0.34, 0.56, -0.12, 0.78, -0.90, 0.12]],
    )


class PredictResponse(BaseModel):
    status: str
    predictions: List[Any]
    probabilities: Optional[List[float]] = None
    anomaly_detected: bool
    action_taken: str
    model_version: str
    latency_ms: float


class GroundTruthRequest(BaseModel):
    prediction_indices: List[int]
    ground_truth: List[Any]


# --- Startup Event ---

@app.on_event("startup")
def load_artifacts():
    """Load configuration, model pipeline, and initialize runtime security detectors."""
    cfg = load_config()
    _STATE["config"] = cfg

    models_dir = Path(cfg["paths"]["models"])
    data_dir = Path(cfg["paths"]["data_processed"])

    # Load feature pipeline
    pipe_path = models_dir / "feature_pipeline.joblib"
    if pipe_path.exists():
        _STATE["feature_pipeline"] = joblib.load(pipe_path)
        logger.info("Loaded feature pipeline from %s", pipe_path)

    # Load latest trained model
    model_path = models_dir / "best_model.joblib"
    if not model_path.exists():
        saved_models = list(models_dir.glob("*.joblib"))
        if saved_models:
            model_path = saved_models[0]

    if model_path.exists():
        _STATE["model"] = joblib.load(model_path)
        logger.info("Loaded inference model from %s", model_path)

    # Initialize AnomalyDetector and DriftDetector using reference training data
    train_path = data_dir / "train.parquet"
    X_train_npy = data_dir / "X_train.npy"
    
    X_train_arr = None
    if train_path.exists():
        df_train = pd.read_parquet(train_path)
        X_train_arr = df_train.drop(columns=["target"], errors="ignore").values
    elif X_train_npy.exists():
        X_train_arr = np.load(X_train_npy)

    if X_train_arr is not None and len(X_train_arr) > 0:
        _STATE["anomaly_detector"] = AnomalyDetector(X_train_arr)
        _STATE["drift_detector"] = DriftDetector(X_train_arr)
        logger.info("Initialized AnomalyDetector & DriftDetector with baseline shape: %s", X_train_arr.shape)

    _STATE["auto_responder"] = AutoResponder()
    logger.info("API Startup Complete.")


# --- Endpoints ---

@app.get("/", tags=["General"])
def root():
    return {
        "service": "automated-mlsecops Inference & Security API",
        "status": "running",
        "model_version": _STATE["model_version"],
        "docs": "/docs",
    }


@app.get("/health", tags=["General"])
def health():
    model_loaded = _STATE["model"] is not None
    pipe_loaded = _STATE["feature_pipeline"] is not None
    return {
        "status": "healthy" if (model_loaded or pipe_loaded) else "degraded",
        "model_loaded": model_loaded,
        "feature_pipeline_loaded": pipe_loaded,
        "anomaly_detector_active": _STATE["anomaly_detector"] is not None,
        "drift_detector_active": _STATE["drift_detector"] is not None,
        "model_version": _STATE["model_version"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(payload: PredictRequest):
    t0 = time.perf_counter()
    model = _STATE["model"]
    pipeline = _STATE["feature_pipeline"]
    anomaly_detector = _STATE["anomaly_detector"]
    auto_responder = _STATE["auto_responder"]

    if not payload.features or len(payload.features) == 0:
        raise HTTPException(status_code=400, detail="Empty feature payload")

    X_raw = np.array(payload.features, dtype=float)

    # 1. Feature Preprocessing
    try:
        if pipeline is not None:
            X_trans = pipeline.transform(X_raw)
        else:
            X_trans = X_raw
    except Exception as exc:
        logger.error("Feature transform error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Feature preprocessing error: {exc}")

    # 2. Anomaly & Out-of-Bounds Detection
    is_anomaly = False
    action_taken = "inference_normal"
    if anomaly_detector is not None:
        try:
            anom_report = anomaly_detector.evaluate_sample(X_trans)
        except Exception as anom_exc:
            # Anomaly detector baseline may have different feature dimensions than
            # the currently loaded model (e.g., after loading a new experiment).
            # Degrade gracefully: skip detection rather than returning 500.
            logger.warning(
                "Anomaly detector skipped (dimension mismatch or error): %s. "
                "Reload the server or retrain to re-initialize the detector.",
                anom_exc,
            )
            anom_report = None
        if anom_report is not None and anom_report.is_anomaly:
            is_anomaly = True
            record_anomaly("ood_input")
            
            resp = auto_responder.handle_input_anomaly(X_trans, anom_report.anomaly_score)
            if resp.triggered and resp.fallback_prediction is not None:
                latency = (time.perf_counter() - t0) * 1000
                record_inference("anomaly_fallback", latency / 1000.0, prediction=1, model_version=_STATE["model_version"])
                
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "latency_ms": latency,
                    "status": "anomaly_mitigated",
                    "predictions": resp.fallback_prediction.tolist(),
                    "features": payload.features,
                    "anomaly_detected": True,
                    "model_version": _STATE["model_version"],
                }
                _STATE["prediction_logs"].append(log_entry)
                
                return PredictResponse(
                    status="anomaly_mitigated",
                    predictions=resp.fallback_prediction.tolist(),
                    probabilities=[1.0] * len(X_trans),
                    anomaly_detected=True,
                    action_taken=resp.action_taken,
                    model_version=_STATE["model_version"],
                    latency_ms=latency,
                )

    # 3. Model Prediction
    if model is None:
        preds = np.zeros(len(X_trans), dtype=int)
        probs = [0.5] * len(X_trans)
        action_taken = "heuristic_fallback"
    else:
        preds = model.predict(X_trans)
        if hasattr(model, "predict_proba"):
            try:
                probs_arr = model.predict_proba(X_trans)
                if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
                    probs = probs_arr[:, 1].tolist()
                else:
                    probs = probs_arr.tolist()
            except Exception:
                probs = None
        else:
            probs = None

    latency = (time.perf_counter() - t0) * 1000
    pred_val = int(preds[0]) if np.issubdtype(preds.dtype, np.integer) else float(preds[0])
    record_inference("success", latency / 1000.0, prediction=pred_val, model_version=_STATE["model_version"])

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latency_ms": latency,
        "status": "success",
        "predictions": preds.tolist(),
        "features": payload.features,
        "anomaly_detected": is_anomaly,
        "model_version": _STATE["model_version"],
    }
    _STATE["prediction_logs"].append(log_entry)

    return PredictResponse(
        status="success",
        predictions=preds.tolist(),
        probabilities=probs,
        anomaly_detected=is_anomaly,
        action_taken=action_taken,
        model_version=_STATE["model_version"],
        latency_ms=latency,
    )


@app.post("/monitoring/ground_truth", tags=["Monitoring"])
def submit_ground_truth(payload: GroundTruthRequest):
    """Submit ground truth labels for monitoring predictions to measure actual performance drift."""
    for idx, label in zip(payload.prediction_indices, payload.ground_truth):
        if 0 <= idx < len(_STATE["prediction_logs"]):
            _STATE["ground_truth_logs"].append({
                "log_index": idx,
                "prediction": _STATE["prediction_logs"][idx]["predictions"],
                "y_true": label,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
    return {"status": "success", "labels_received": len(payload.ground_truth), "total_labeled": len(_STATE["ground_truth_logs"])}


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    content, content_type = get_metrics_payload()
    return Response(content=content, media_type=content_type)


@app.get("/monitoring/stats", tags=["Monitoring"])
def monitoring_stats():
    """JSON formatted genuine monitoring stats for UI dashboard."""
    import psutil
    vm = psutil.virtual_memory()

    logs = _STATE["prediction_logs"]
    total_requests = len(logs)

    if total_requests == 0:
        avg_latency = None
        error_count = 0
        error_rate_pct = 0.0
        latest_activity = None
    else:
        avg_latency = float(np.mean([l["latency_ms"] for l in logs]))
        error_count = sum(1 for l in logs if l["status"] not in ["success", "anomaly_mitigated"])
        error_rate_pct = float((error_count / total_requests) * 100)
        latest_activity = logs[-1]["timestamp"]

    # System metrics
    system_metrics = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round(vm.used / (1024 ** 3), 2),
        "ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "ram_usage_percent": vm.percent,
    }

    # Data Drift Analysis
    drift_detector = _STATE["drift_detector"]
    drift_info = {
        "status": "active" if drift_detector is not None else "no_baseline",
        "has_baseline": drift_detector is not None,
        "sample_count": total_requests,
        "is_drifted": False,
        "drift_score": None,
        "drifted_features": [],
        "feature_reports": {},
    }

    if drift_detector is not None and total_requests >= 3:
        all_features = []
        for l in logs:
            all_features.extend(l["features"])
        try:
            arr_curr = np.array(all_features, dtype=float)
            report = drift_detector.detect_drift(arr_curr)
            drift_info.update({
                "status": "evaluated",
                "is_drifted": report.is_drifted,
                "drift_score": round(report.drift_score, 4),
                "drifted_features": report.drifted_features,
                "feature_reports": report.feature_reports,
            })
        except Exception as exc:
            logger.warning("Drift detection error: %s", exc)
            drift_info["status"] = "error"
    elif total_requests < 3:
        drift_info["status"] = "insufficient_samples"

    # Performance Drift Analysis (only when ground truth is submitted)
    gt_logs = _STATE["ground_truth_logs"]
    if len(gt_logs) > 0:
        y_preds = [g["prediction"][0] for g in gt_logs if len(g["prediction"]) > 0]
        y_trues = [g["y_true"] for g in gt_logs]
        if len(y_preds) > 0 and len(y_preds) == len(y_trues):
            try:
                acc = float(accuracy_score(y_trues, y_preds))
                f1_val = float(f1_score(y_trues, y_preds, zero_division=0, average="weighted"))
                perf_drift = {
                    "status": "evaluated",
                    "sample_count": len(gt_logs),
                    "current_accuracy": acc,
                    "current_f1": f1_val,
                }
            except Exception:
                perf_drift = {"status": "unavailable_error"}
        else:
            perf_drift = {"status": "unavailable_mismatch"}
    else:
        perf_drift = {
            "status": "unavailable_no_ground_truth",
            "message": "Ground-truth labels have not been provided for prediction inputs yet."
        }

    return {
        "health": {
            "status": "healthy" if (_STATE["model"] is not None or _STATE["feature_pipeline"] is not None) else "degraded",
            "model_loaded": _STATE["model"] is not None,
            "pipeline_loaded": _STATE["feature_pipeline"] is not None,
            "active_model_version": _STATE["model_version"],
        },
        "traffic": {
            "total_requests": total_requests,
            "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
            "error_count": error_count,
            "error_rate_pct": round(error_rate_pct, 2),
            "latest_activity": latest_activity,
            "model_version": _STATE["model_version"],
        },
        "system": system_metrics,
        "drift": drift_info,
        "performance_drift": perf_drift,
        "request_history": logs[-20:],  # Recent 20 logs for timeline graph
    }


@app.get("/security/audit", tags=["Security"])
def security_audit():
    """Run real-time security audit scan of workspace."""
    cfg = _STATE["config"] or load_config()
    dep_audit = gate_dependency_audit(cfg)
    secrets_scan = gate_secrets_scan(cfg)

    update_security_gate_status("dependency_audit", dep_audit.passed)
    update_security_gate_status("secrets_scan", secrets_scan.passed)

    return {
        "overall_status": "PASS" if (dep_audit.passed and secrets_scan.passed) else "FAIL",
        "dependency_audit": {
            "passed": dep_audit.passed,
            "score": dep_audit.score,
            "details": dep_audit.details,
        },
        "secrets_scan": {
            "passed": secrets_scan.passed,
            "score": secrets_scan.score,
            "details": secrets_scan.details,
        },
    }

# --- Experiment & Pipeline Endpoints ---

def _background_run_pipeline(experiment_id: str, config: dict):
    """Background task to run the pipeline with genuine stage reporting."""
    def _stage_cb(stage_name: str, details: str = ""):
        if experiment_id in _RUNNING_EXPERIMENTS:
            _RUNNING_EXPERIMENTS[experiment_id]["stage"] = stage_name
            _RUNNING_EXPERIMENTS[experiment_id]["stage_details"] = details

    try:
        _RUNNING_EXPERIMENTS[experiment_id]["status"] = "RUNNING"
        _RUNNING_EXPERIMENTS[experiment_id]["stage"] = "Upload"
        _RUNNING_EXPERIMENTS[experiment_id]["stage_details"] = "Starting pipeline..."
        run_pipeline(register=True, stage="Staging", experiment_id=experiment_id, config=config, stage_callback=_stage_cb)
        _RUNNING_EXPERIMENTS[experiment_id]["status"] = "COMPLETED"
        _RUNNING_EXPERIMENTS[experiment_id]["stage"] = "Packaging"
        _RUNNING_EXPERIMENTS[experiment_id]["stage_details"] = "Pipeline execution complete"
    except Exception as exc:
        logger.error("Pipeline failed for %s: %s", experiment_id, exc, exc_info=True)
        _RUNNING_EXPERIMENTS[experiment_id]["status"] = "FAILED"
        _RUNNING_EXPERIMENTS[experiment_id]["error"] = str(exc)
        
        # Save a failed experiment result
        from src.utils.experiment_schema import create_standard_experiment_result, save_experiment_result
        failed_res = create_standard_experiment_result(
            experiment_id=experiment_id,
            status="FAILED",
            error_message=str(exc),
            config=config,
        )
        save_experiment_result(failed_res, config["paths"]["results"])


class RunPipelineRequest(BaseModel):
    target_column: Optional[str] = "target"
    opt_metric: Optional[str] = "f1"
    models_to_train: Optional[List[str]] = ["logistic_regression", "random_forest", "gradient_boosting"]
    task_type: Optional[str] = "classification"

@app.post("/experiments/run", tags=["Experiments"])
def run_experiment(background_tasks: BackgroundTasks, req: RunPipelineRequest = None):
    """Trigger a new MLSecOps pipeline run."""
    # Prevent duplicate runs if a pipeline is actively running
    for exp_k, exp_v in _RUNNING_EXPERIMENTS.items():
        if exp_v.get("status") in ["RUNNING", "QUEUED"]:
            if time.time() - exp_v.get("start_time", 0) < 600:
                logger.info("Pipeline already running: %s (stage=%s)", exp_k, exp_v.get("stage", "Upload"))
                return {
                    "experiment_id": exp_k,
                    "status": exp_v["status"],
                    "stage": exp_v.get("stage", "Upload"),
                    "stage_details": exp_v.get("stage_details", ""),
                    "message": "Pipeline already running",
                }

    cfg = load_config()
    if req:
        # Override config based on request
        cfg["dataset"]["target_column"] = req.target_column
        cfg["dataset"]["task_type"] = req.task_type
        cfg["training"]["optimization_metric"] = req.opt_metric
        cfg["training"]["algorithms"] = req.models_to_train
        
    experiment_id = f"exp_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    _RUNNING_EXPERIMENTS[experiment_id] = {
        "status": "QUEUED",
        "stage": "Upload",
        "stage_details": "Queued for processing",
        "start_time": time.time(),
        "error": None
    }
    
    background_tasks.add_task(_background_run_pipeline, experiment_id, cfg)
    
    return {"experiment_id": experiment_id, "status": "QUEUED", "stage": "Upload"}


@app.get("/experiments", tags=["Experiments"])
def get_experiments():
    """List all completed experiments."""
    cfg = load_config()
    exps = list_experiments(results_dir=cfg["paths"]["results"])
    return {"experiments": exps}


@app.get("/experiments/{experiment_id}", tags=["Experiments"])
def get_experiment(experiment_id: str):
    """Get the status and results of a specific experiment."""
    cfg = load_config()
    
    # Check if it's currently running in memory
    if experiment_id in _RUNNING_EXPERIMENTS:
        status_info = _RUNNING_EXPERIMENTS[experiment_id]
        if status_info["status"] in ["QUEUED", "RUNNING"]:
            return {
                "experiment_id": experiment_id,
                "status": status_info["status"],
                "stage": status_info.get("stage", "Upload"),
                "stage_details": status_info.get("stage_details", ""),
            }
            
    # Try to load from disk
    result = load_experiment_result(experiment_id, results_dir=cfg["paths"]["results"])
    if result:
        return result
        
    if experiment_id in _RUNNING_EXPERIMENTS and status_info["status"] == "FAILED":
        return {
            "experiment_id": experiment_id,
            "status": "FAILED",
            "stage": status_info.get("stage", "Unknown"),
            "error": status_info["error"],
        }
        
    raise HTTPException(status_code=404, detail="Experiment not found")


@app.get("/experiments/{experiment_id}/model/download", tags=["Experiments"])
def download_experiment_model(experiment_id: str):
    """Download the model artifacts (zip) for a specific experiment."""
    cfg = load_config()
    exp_dir = Path(cfg["paths"]["results"]) / experiment_id
    
    if not exp_dir.exists():
        # Fallback to checking results/experiments/<exp_id>.json
        json_path = Path(cfg["paths"]["results"]) / "experiments" / f"{experiment_id}.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Experiment artifacts not found")

    model_path = exp_dir / "best_model.joblib"
    if not model_path.exists():
        # Fallback to global best model
        model_path = Path(cfg["paths"]["models"]) / "best_model.joblib"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found for this experiment")
        
    pipe_path = exp_dir / "feature_pipeline.joblib"
    if not pipe_path.exists():
        pipe_path = Path(cfg["paths"]["models"]) / "feature_pipeline.joblib"

    # Try loading experiment result payload
    exp_data = load_experiment_result(experiment_id, results_dir=cfg["paths"]["results"]) or {}
    bm = exp_data.get("best_model", {})
    sec = exp_data.get("security", {})
    models_data = exp_data.get("models", {})
    ds = exp_data.get("dataset", {})
    model_ver = exp_data.get("model_version") or exp_data.get("deployment", {}).get("model_version", "v1")

    # Generate metadata files
    model_metadata = {
        "experiment_id": experiment_id,
        "model_version": model_ver,
        "algorithm": bm.get("algorithm", "unknown"),
        "task_type": ds.get("task_type", "classification"),
        "created_at": exp_data.get("created_at"),
        "dataset": ds,
        "hyperparams": bm.get("hyperparams", {}),
        "optimal_threshold": bm.get("optimal_threshold", 0.5),
        "metrics": bm.get("metrics", {}),
    }

    eval_results = {
        "best_model": bm,
        "all_models_trained": models_data,
        "research_metrics": exp_data.get("research_metrics", {}),
    }

    sec_results = {
        "overall_passed": sec.get("overall_passed", False),
        "overall_score": sec.get("overall_score", 0.0),
        "gate_results": sec.get("gate_results", []),
    }

    # Generate minimal usable HTML frontend for the model
    sample_feature_count = max(1, ds.get("n_features", 4))
    sample_vector = [0.5] * sample_feature_count

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Inference Interface - {model_ver}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #0F172A;
            color: #F8FAFC;
            margin: 0;
            padding: 30px 20px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 720px;
            width: 100%;
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 28px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        }}
        h1 {{ margin-top: 0; font-size: 24px; color: #F8FAFC; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(99, 102, 241, 0.2);
            color: #818CF8;
            border: 1px solid rgba(99, 102, 241, 0.4);
            margin-right: 8px;
        }}
        .meta-row {{ margin-bottom: 20px; }}
        label {{ font-size: 14px; font-weight: 600; color: #94A3B8; display: block; margin-bottom: 8px; }}
        textarea {{
            width: 100%;
            height: 90px;
            background: #0B0F19;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #F8FAFC;
            padding: 12px;
            font-family: monospace;
            font-size: 13px;
            box-sizing: border-box;
            resize: vertical;
        }}
        textarea:focus {{ outline: none; border-color: #6366F1; }}
        .btn-group {{ display: flex; gap: 10px; margin-top: 14px; }}
        button {{
            padding: 10px 18px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            transition: 0.15s;
        }}
        .btn-primary {{
            background: #6366F1;
            color: white;
            flex: 1;
        }}
        .btn-primary:hover {{ background: #4F46E5; }}
        .btn-secondary {{
            background: #334155;
            color: #E2E8F0;
        }}
        .btn-secondary:hover {{ background: #475569; }}
        .result-card {{
            margin-top: 24px;
            background: #0B0F19;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            display: none;
        }}
        .result-title {{ font-size: 13px; color: #94A3B8; text-transform: uppercase; font-weight: 600; }}
        .result-value {{ font-size: 26px; font-weight: 700; color: #10B981; margin: 8px 0; }}
        .latency {{ font-size: 12px; color: #94A3B8; }}
        .error {{ color: #EF4444; margin-top: 10px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Deployed Model Inference</h1>
        <div class="meta-row">
            <span class="badge">Model: {model_ver}</span>
            <span class="badge">Algorithm: {bm.get('algorithm', 'unknown')}</span>
            <span class="badge">Task: {ds.get('task_type', 'classification')}</span>
        </div>
        <form id="pred-form" onsubmit="event.preventDefault(); makePrediction();">
            <label for="features-input">Input Features (comma-separated or JSON array):</label>
            <textarea id="features-input" placeholder="e.g. 0.5, -1.2, 0.8, 1.4"></textarea>
            <div class="btn-group">
                <button type="button" class="btn-secondary" onclick="loadSample()">Load Sample</button>
                <button type="submit" id="predict-btn" class="btn-primary">Generate Prediction</button>
            </div>
        </form>
        <div id="error-box" class="error"></div>
        <div id="result-box" class="result-card">
            <div class="result-title">Prediction Result</div>
            <div id="pred-val" class="result-value">-</div>
            <div id="prob-val" style="color: #E2E8F0; font-size: 14px; margin-bottom: 6px;"></div>
            <div id="latency-val" class="latency"></div>
        </div>
    </div>
    <script>
        function loadSample() {{
            const sample = {json.dumps(sample_vector)};
            document.getElementById('features-input').value = sample.join(', ');
        }}
        async function makePrediction() {{
            const errBox = document.getElementById('error-box');
            const resBox = document.getElementById('result-box');
            errBox.innerText = '';
            const raw = document.getElementById('features-input').value.trim();
            if (!raw) {{ errBox.innerText = 'Please enter feature values'; return; }}
            let features;
            try {{
                if (raw.startsWith('[') && raw.endsWith(']')) {{
                    features = JSON.parse(raw);
                    if (!Array.isArray(features[0])) features = [features];
                }} else {{
                    features = [raw.split(',').map(s => Number(s.trim()))];
                }}
            }} catch(e) {{
                errBox.innerText = 'Invalid input format. Use comma-separated numbers or JSON array.';
                return;
            }}
            const btn = document.getElementById('predict-btn');
            btn.disabled = true;
            btn.innerText = 'Predicting...';
            const t0 = performance.now();
            try {{
                const res = await fetch('/predict', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ features: features }})
                }});
                const data = await res.json();
                const latency = Math.round(performance.now() - t0);
                if (!res.ok) throw new Error(data.detail || 'Prediction failed');
                document.getElementById('pred-val').innerText = 'Predicted: ' + JSON.stringify(data.predictions);
                if (data.probabilities) {{
                    document.getElementById('prob-val').innerText = 'Probabilities: ' + JSON.stringify(data.probabilities);
                }} else {{
                    document.getElementById('prob-val').innerText = '';
                }}
                document.getElementById('latency-val').innerText = 'Roundtrip: ' + latency + ' ms';
                resBox.style.display = 'block';
            }} catch(err) {{
                errBox.innerText = 'Error: ' + err.message;
            }} finally {{
                btn.disabled = false;
                btn.innerText = 'Generate Prediction';
            }}
        }}
    </script>
</body>
</html>"""

    # Generate the deployable FastAPI prediction server code
    fastapi_code = f"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Any, Optional
from pathlib import Path
import joblib
import json
import pandas as pd

app = FastAPI(
    title="Deployed Secure ML Model API",
    version="{model_ver}",
    description="Executable prediction endpoint with interactive frontend generated by Secure MLOps Pipeline"
)

# Load artifacts
model = joblib.load("best_model.joblib")
try:
    pipeline = joblib.load("feature_pipeline.joblib")
except Exception:
    pipeline = None

try:
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
except Exception:
    metadata = {{}}

class PredictRequest(BaseModel):
    features: List[List[Any]]

class PredictResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[Any]] = None
    model_version: str

@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    ui_path = Path("index.html")
    if ui_path.exists():
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Deployed Secure ML Model API</h2><p>Visit <a href='/docs'>/docs</a></p>")

@app.get("/api/info")
def root():
    return {{
        "service": "Deployed Secure ML API",
        "status": "online",
        "model_version": metadata.get("model_version", "{model_ver}"),
        "algorithm": metadata.get("algorithm", "unknown"),
        "task_type": metadata.get("task_type", "classification")
    }}

@app.get("/health")
def health():
    return {{"status": "healthy", "model_loaded": model is not None, "pipeline_loaded": pipeline is not None}}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.features or len(req.features) == 0:
        raise HTTPException(status_code=400, detail="Empty features payload")
    
    df = pd.DataFrame(req.features)
    if pipeline is not None:
        X = pipeline.transform(df)
    else:
        X = df.values

    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X).tolist()
        except Exception:
            probs = None

    return PredictResponse(
        predictions=preds.tolist(),
        probabilities=probs,
        model_version=metadata.get("model_version", "{model_ver}")
    )
"""

    requirements_txt = """fastapi>=0.103.2
uvicorn>=0.23.2
scikit-learn>=1.3.1
pandas>=2.1.1
numpy>=1.26.0
joblib>=1.3.2
pydantic>=2.4.2
"""

    config_json = {
        "model_version": model_ver,
        "experiment_id": experiment_id,
        "algorithm": bm.get("algorithm", "unknown"),
        "task_type": ds.get("task_type", "classification"),
        "optimal_threshold": bm.get("optimal_threshold", 0.5),
        "dataset_name": ds.get("file_name") or ds.get("name", "dataset.csv"),
        "n_features": sample_feature_count,
    }

    readme_md = f"""# Deployed ML Model Package ({model_ver})

**Experiment ID:** `{experiment_id}`
**Algorithm:** `{bm.get('algorithm', 'N/A')}`
**Task Type:** `{ds.get('task_type', 'classification')}`
**Created At:** `{exp_data.get('created_at', 'N/A')}`

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI Inference Server & Web Frontend
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Open Interactive Web Frontend
Open your web browser and navigate to:
**http://localhost:8000/ui** (or open `index.html` directly)

You can click **"Load Sample"** to load example features and **"Generate Prediction"** to test live inferences!

### 4. Send API Prediction Request via cURL
```bash
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"features": [{sample_vector}]}}'
```

---

## 📦 Package Contents
- `index.html`: Interactive web frontend interface
- `app.py`: Executable FastAPI inference service serving API + UI
- `best_model.joblib`: Trained machine learning model
- `feature_pipeline.joblib`: Preprocessing transformer pipeline
- `config.json`: Deployment configuration
- `model_metadata.json`: Full model version and training metadata
- `evaluation_results.json`: Model performance metrics breakdown
- `security_results.json`: Security gates compliance report
- `requirements.txt`: Python package requirements
- `README.md`: Run and prediction instructions
"""

    import io
    import zipfile
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(model_path, "best_model.joblib")
        if pipe_path.exists():
            zip_file.write(pipe_path, "feature_pipeline.joblib")
            
        zip_file.writestr("index.html", index_html.strip())
        zip_file.writestr("app.py", fastapi_code.strip())
        zip_file.writestr("config.json", json.dumps(config_json, indent=2))
        zip_file.writestr("requirements.txt", requirements_txt.strip())
        zip_file.writestr("README.md", readme_md.strip())
        zip_file.writestr("model_metadata.json", json.dumps(model_metadata, indent=2, default=str))
        zip_file.writestr("evaluation_results.json", json.dumps(eval_results, indent=2, default=str))
        zip_file.writestr("security_results.json", json.dumps(sec_results, indent=2, default=str))
        zip_file.writestr("experiment_result.json", json.dumps(exp_data, indent=2, default=str))
            
    zip_buffer.seek(0)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=ml_project_{experiment_id}.zip"}
    )

@app.post("/experiments/{experiment_id}/model/load", tags=["Experiments"])
def load_experiment_model(experiment_id: str):
    """Load the specific experiment's model into memory for predictions."""
    cfg = load_config()
    exp_dir = Path(cfg["paths"]["results"]) / experiment_id
    
    model_path = exp_dir / "best_model.joblib"
    pipe_path = exp_dir / "feature_pipeline.joblib"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found")
        
    _STATE["model"] = joblib.load(model_path)
    if pipe_path.exists():
        _STATE["feature_pipeline"] = joblib.load(pipe_path)

    # Reset anomaly/drift detectors — their baselines were trained on the previous
    # model's feature space and are invalid after loading a new experiment model.
    # They will be re-initialized on the next full server startup or pipeline run.
    _STATE["anomaly_detector"] = None
    _STATE["drift_detector"] = None
    logger.info(
        "Loaded experiment %s model. Anomaly/drift detectors reset "
        "(will reinitialize on next server start or new training run).",
        experiment_id,
    )
        
    _STATE["model_version"] = f"exp_{experiment_id}"
    return {"status": "success", "message": f"Loaded model from experiment {experiment_id}"}


@app.post("/data/upload", tags=["Data"])
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = "target",
    task_type: str = "classification",
):
    """Upload a new dataset file and update config so the next pipeline run uses it."""
    # Validate file extension
    allowed_exts = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: CSV, Excel (.xlsx/.xls), JSON, Parquet."
        )

    cfg = load_config()
    raw_dir = Path(cfg["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_path = raw_dir / file.filename
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with open(file_path, "wb") as f:
        f.write(content)

    import yaml
    from src.data.loader import generate_dataset_id
    dataset_id = generate_dataset_id(file.filename, content)

    # Update config.yaml with new file path, target_column, task_type and dataset identifiers
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)

        yaml_cfg["dataset"]["source"] = "file"
        yaml_cfg["dataset"]["file_path"] = str(file_path)
        yaml_cfg["dataset"]["file_name"] = file.filename
        yaml_cfg["dataset"]["dataset_id"] = dataset_id
        yaml_cfg["dataset"]["name"] = Path(file.filename).stem
        yaml_cfg["dataset"]["target_column"] = target_column
        yaml_cfg["dataset"]["task_type"] = task_type

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)

    # Clear LRU cache so the next load_config() reads updated YAML
    load_config.cache_clear()

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "file_path": str(file_path),
        "file_name": file.filename,
        "file_size_bytes": len(content),
        "target_column": target_column,
        "task_type": task_type,
    }

