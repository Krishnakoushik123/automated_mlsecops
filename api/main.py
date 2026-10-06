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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import uuid
from fastapi import FastAPI, HTTPException, Request, Response, status, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field

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
    "auto_responder": None,
    "model_version": "v1.0.0",
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
    predictions: List[int]
    probabilities: Optional[List[float]] = None
    anomaly_detected: bool
    action_taken: str
    model_version: str
    latency_ms: float


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
        # Fallback to any saved model in directory
        saved_models = list(models_dir.glob("*.joblib"))
        if saved_models:
            model_path = saved_models[0]

    if model_path.exists():
        _STATE["model"] = joblib.load(model_path)
        logger.info("Loaded inference model from %s", model_path)

    # Initialize AnomalyDetector using processed training dataset as baseline
    train_path = data_dir / "train.parquet"
    if train_path.exists():
        df_train = pd.read_parquet(train_path)
        y_col = "target"
        X_train_arr = df_train.drop(columns=[y_col], errors="ignore").values
        _STATE["anomaly_detector"] = AnomalyDetector(X_train_arr)
        logger.info("Initialized AnomalyDetector with reference data shape: %s", X_train_arr.shape)

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
        anom_report = anomaly_detector.evaluate_sample(X_trans)
        if anom_report.is_anomaly:
            is_anomaly = True
            record_anomaly("ood_input")
            
            # Trigger AutoResponder mitigation
            resp = auto_responder.handle_input_anomaly(X_trans, anom_report.anomaly_score)
            if resp.triggered and resp.fallback_prediction is not None:
                latency = (time.perf_counter() - t0) * 1000
                record_inference("anomaly_fallback", latency / 1000.0, prediction=1)
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
        # Heuristic fallback if model not loaded
        preds = np.zeros(len(X_trans), dtype=int)
        probs = [0.5] * len(X_trans)
        action_taken = "heuristic_fallback"
    else:
        preds = model.predict(X_trans)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_trans)[:, 1].tolist()
        else:
            probs = None

    latency = (time.perf_counter() - t0) * 1000
    record_inference("success", latency / 1000.0, prediction=int(preds[0]))

    return PredictResponse(
        status="success",
        predictions=preds.tolist(),
        probabilities=probs,
        anomaly_detected=is_anomaly,
        action_taken=action_taken,
        model_version=_STATE["model_version"],
        latency_ms=latency,
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    content, content_type = get_metrics_payload()
    return Response(content=content, media_type=content_type)

@app.get("/monitoring/stats", tags=["Monitoring"])
def monitoring_stats():
    """JSON formatted monitoring stats for UI dashboard."""
    # We parse the basic metrics from prometheus output for simplicity
    content, _ = get_metrics_payload()
    lines = content.decode("utf-8").split("\n")
    
    stats = {
        "inference_requests": 0,
        "anomalies_detected": 0,
        "avg_latency_ms": 0.0,
        "security_gates_passed": 0
    }
    
    latency_sum = 0
    latency_count = 0
    
    for line in lines:
        if line.startswith("mlsecops_inference_requests_total"):
            try:
                val = float(line.split(" ")[1])
                stats["inference_requests"] += int(val)
            except: pass
        elif line.startswith("mlsecops_input_anomalies_total"):
            try:
                val = float(line.split(" ")[1])
                stats["anomalies_detected"] += int(val)
            except: pass
        elif line.startswith("mlsecops_inference_latency_seconds_sum"):
            try:
                latency_sum = float(line.split(" ")[1])
            except: pass
        elif line.startswith("mlsecops_inference_latency_seconds_count"):
            try:
                latency_count = float(line.split(" ")[1])
            except: pass
            
    if latency_count > 0:
        stats["avg_latency_ms"] = (latency_sum / latency_count) * 1000
        
    return stats


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
    """Background task to run the pipeline."""
    try:
        _RUNNING_EXPERIMENTS[experiment_id]["status"] = "RUNNING"
        run_pipeline(register=True, stage="Staging", experiment_id=experiment_id, config=config)
        _RUNNING_EXPERIMENTS[experiment_id]["status"] = "COMPLETED"
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
        "start_time": time.time(),
        "error": None
    }
    
    background_tasks.add_task(_background_run_pipeline, experiment_id, cfg)
    
    return {"experiment_id": experiment_id, "status": "QUEUED"}


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
            return {"experiment_id": experiment_id, "status": status_info["status"]}
            
    # Try to load from disk
    result = load_experiment_result(experiment_id, results_dir=cfg["paths"]["results"])
    if result:
        return result
        
    if experiment_id in _RUNNING_EXPERIMENTS and status_info["status"] == "FAILED":
        return {"experiment_id": experiment_id, "status": "FAILED", "error": status_info["error"]}
        
    raise HTTPException(status_code=404, detail="Experiment not found")


@app.get("/experiments/{experiment_id}/model/download", tags=["Experiments"])
def download_experiment_model(experiment_id: str):
    """Download the model artifacts (zip) for a specific experiment."""
    cfg = load_config()
    exp_dir = Path(cfg["paths"]["results"]) / experiment_id
    
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment artifacts not found")
        
    model_path = exp_dir / "best_model.joblib"
    pipe_path = exp_dir / "feature_pipeline.joblib"
    result_path = exp_dir / "experiment_result.json"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found for this experiment")
        
    # Generate the deployable project files
    fastapi_code = f"""
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List, Any

app = FastAPI(title="Deployed ML Model")
model = joblib.load("best_model.joblib")
try:
    pipeline = joblib.load("feature_pipeline.joblib")
except:
    pipeline = None

class PredictRequest(BaseModel):
    features: List[List[Any]]

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.features)
    if pipeline:
        X = pipeline.transform(df)
    else:
        X = df.values
    preds = model.predict(X)
    return {{"predictions": preds.tolist()}}
"""
    
    requirements_txt = """fastapi==0.103.2
uvicorn==0.23.2
scikit-learn==1.3.1
pandas==2.1.1
numpy==1.26.0
joblib==1.3.2
pydantic==2.4.2
xgboost==2.0.0
"""
    
    readme_md = f"""# Deployed ML Model: {experiment_id}
    
## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Run server: `uvicorn app:app --host 0.0.0.0 --port 8000`

## API Usage
POST to `/predict` with payload:
```json
{{
    "features": [
        [feature1, feature2, ...]
    ]
}}
```
"""

    import io
    import zipfile
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(model_path, "best_model.joblib")
        if pipe_path.exists():
            zip_file.write(pipe_path, "feature_pipeline.joblib")
        if result_path.exists():
            zip_file.write(result_path, "experiment_result.json")
            
        zip_file.writestr("app.py", fastapi_code.strip())
        zip_file.writestr("requirements.txt", requirements_txt.strip())
        zip_file.writestr("README.md", readme_md.strip())
            
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
        
    _STATE["model_version"] = f"exp_{experiment_id}"
    return {"status": "success", "message": f"Loaded model from experiment {experiment_id}"}


@app.post("/data/upload", tags=["Data"])
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new CSV dataset for the next experiment."""
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = raw_dir / file.filename
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
        
    import yaml
    # Update config.yaml to use the new CSV file
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f)
            
        yaml_cfg["dataset"]["source"] = "file"
        yaml_cfg["dataset"]["file_path"] = str(file_path)
        
        with open(config_path, "w") as f:
            yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)
            
    return {"status": "success", "file_path": str(file_path)}

