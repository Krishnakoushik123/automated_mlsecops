# automated-mlsecops

> A research-grade **automated MLSecOps** system that extends a baseline MLOps pipeline with **security-by-design** and quantitatively compares **Traditional ML**, **MLOps**, and **MLSecOps** using formal Data Envelopment Analysis (DEA).

---

## Architecture

```
DATA → VALIDATION → FEATURE PIPELINE → TRAINING → SECURITY GATES
     → MODEL REGISTRY → CI/CD → DEPLOYMENT → MONITORING
     → AUTO RESPONSE → DEA EVALUATION
```

### Pipeline Flow Diagram

```
┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌──────────┐
│   DATA   │───▶│ VALIDATION │───▶│   FEATURE    │───▶│ TRAINING │
│ INGESTION│    │  (Pandera) │    │  PIPELINE    │    │(sklearn) │
└──────────┘    └────────────┘    └──────────────┘    └────┬─────┘
                                                          │
          ┌───────────────────────────────────────────────┘
          ▼
┌──────────────────┐    ┌──────────────┐    ┌──────────┐
│  SECURITY GATES  │───▶│   MODEL      │───▶│  CI/CD   │
│ (5 gates: audit, │    │  REGISTRY    │    │ (GitHub  │
│  FGSM, secrets,  │    │  (MLflow)    │    │ Actions) │
│  integrity)      │    └──────────────┘    └────┬─────┘
└──────────────────┘                             │
          ┌──────────────────────────────────────┘
          ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  DEPLOYMENT  │───▶│  MONITORING  │───▶│    AUTO      │
│  (FastAPI +  │    │ (Drift, OOD, │    │  RESPONSE    │
│   Docker)    │    │  Prometheus) │    │(Self-Healing)│
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │     DEA      │
                                        │ EVALUATION   │
                                        │ (LP + Comp.) │
                                        └──────────────┘
```

## Technology Stack

| Layer | Technology |
|---|---|
| ML Framework | scikit-learn (LogReg, RF, GBM) |
| Experiment Tracking | MLflow (SQLite backend) |
| Schema Validation | Pandera |
| Security Scanning | Bandit, pip-audit, FGSM, SHA-256 |
| Drift Monitoring | Custom PSI/JS-divergence |
| Anomaly Detection | IQR-based Out-of-Distribution |
| API | FastAPI + Uvicorn |
| Metrics Export | Prometheus-compatible `/metrics` |
| Containers | Docker + Docker Compose |
| CI/CD | GitHub Actions (3-job pipeline) |
| Testing | Pytest (49 unit + integration tests) |
| DEA Evaluation | Charnes-Cooper LP (scipy.optimize) |

---

## Project Structure

```
automated-mlsecops/
├── src/
│   ├── data/           # Data ingestion, schema & quality validation
│   │   ├── loader.py   # load_dataset(), save_raw() — sklearn/CSV/URL sources
│   │   └── validator.py # Pandera schema, nulls, duplicates, class balance
│   ├── features/       # Feature preprocessing pipeline
│   │   └── preprocessing.py # build_pipeline(), split_dataset(), fit_transform_splits()
│   ├── training/       # Model training, CV, threshold optimization, registry
│   │   ├── trainer.py  # train_model(), train_all(), build_model(), compute_metrics()
│   │   ├── pipeline.py # run_pipeline() — full end-to-end orchestrator
│   │   └── registry.py # MLflow model registry operations
│   ├── security/       # 5 Security Gates
│   │   └── gates.py    # dependency_audit, secrets_scan, data_integrity,
│   │                   # adversarial_robustness (FGSM), model_integrity (SHA-256)
│   ├── monitoring/     # Runtime monitoring & auto-response
│   │   ├── drift_detector.py    # PSI-based feature drift detection
│   │   ├── anomaly_detector.py  # IQR-based out-of-distribution detection
│   │   ├── auto_responder.py    # Automated threat mitigation & fallback
│   │   └── prometheus_exporter.py # Prometheus metrics collection
│   ├── evaluation/     # DEA benchmark evaluation
│   │   └── dea_evaluator.py # Formal Charnes-Cooper LP DEA + composite scoring
│   └── utils/
│       ├── config.py           # YAML config loader with env var overrides
│       ├── experiment_schema.py # Standardized experiment result model
│       └── logger.py           # Structured logging
├── api/
│   └── main.py         # FastAPI inference service with anomaly detection
├── tests/
│   ├── unit/           # 44 unit tests across all modules
│   └── integration/    # 5 API integration tests
├── experiments/
│   ├── traditional/    # Baseline 1: No automation, manual deployment
│   ├── mlops/          # Baseline 2: MLflow + CI/CD, no security gates
│   ├── mlsecops/       # Proposed: Full automated MLSecOps
│   └── run_comparison.py # Master 3-way benchmark & DEA evaluation
├── configs/
│   └── config.yaml     # Central configuration (never put secrets here)
├── .github/workflows/
│   └── mlsecops-ci.yml # 3-job CI: Security → Tests → Benchmark
├── docker/
│   ├── Dockerfile.api  # API container
│   └── docker-compose.yml
├── models/             # Saved model artifacts (.joblib)
├── data/
│   ├── raw/            # Raw ingested data
│   └── processed/      # Transformed train/val/test splits
└── results/
    ├── json/           # Training results, pipeline summaries
    ├── csv/            # Tabular experiment exports
    ├── experiments/    # Standardized experiment result JSONs
    ├── benchmark_report.json  # DEA evaluation results
    └── benchmark_report.md    # Markdown comparison table
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker Desktop (optional, for full stack)

### 1 – Clone & set up environment

```bash
git clone <repo-url>
cd automated-mlsecops

# Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2 – Configure environment

```bash
cp .env.example .env
# Edit .env with your values (never commit .env)
```

### 3 – Run the full pipeline

```bash
# Full end-to-end: Data → Validation → Features → Training → Security Gates → Registry
python -m src.training.pipeline

# With model registration
python -m src.training.pipeline --register --stage Staging
```

### 4 – Run the 3-way comparison experiment

```bash
# Traditional ML vs MLOps vs MLSecOps with DEA evaluation
python -m experiments.run_comparison
```

### 5 – Run tests

```bash
# Full test suite (49 tests)
pytest -o addopts="" -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v
```

### 6 – Start the inference API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs for Swagger UI
```

### 7 – View MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## Implementation Phases

| Phase | Scope | Status |
|---|---|---|
| **1** | Data ingestion, schema validation (Pandera), feature pipeline, multi-model training, MLflow tracking & registry | ✅ Complete |
| **2** | 5 Security Gates: dependency audit, secrets scan, data integrity/poisoning, FGSM adversarial robustness, SHA-256 model integrity | ✅ Complete |
| **3** | FastAPI inference service with input validation, anomaly detection, automated fallback | ✅ Complete |
| **4** | Runtime monitoring: feature drift (PSI), OOD anomaly detection (IQR), Prometheus metrics, automated self-healing response | ✅ Complete |
| **5** | DEA evaluation (formal Charnes-Cooper LP), composite scoring, standardized experiment results, benchmark reports | ✅ Complete |
| **6** | Docker Compose stack, GitHub Actions CI/CD (3-job pipeline), 49 automated tests | ✅ Complete |

---

## ML Quality Features

### Class Imbalance Handling
- `class_weight="balanced"` injected automatically into all classifiers that support it
- Configurable class weights in synthetic data generation

### Threshold Optimization
- Precision-recall curve analysis on validation set
- Finds optimal threshold in [0.05, 0.95] maximizing F1 score
- Both baseline (0.5) and optimized threshold metrics are preserved

### Model Comparison
- 3 algorithms trained & compared: Logistic Regression, Random Forest, Gradient Boosting
- Stratified K-Fold cross-validation (5 folds by default)
- Best model selected by test F1 score

### Evaluation Metrics
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Supports both binary and multiclass classification
- Latency: p50, p95, p99 inference latency
- Security: vulnerability count, compliance score, FGSM robustness score
- Confusion matrix preserved per model

---

## Security Gates

The following 5 checks gate model registration and deployment:

| Gate | Tool | Threshold | Description |
|---|---|---|---|
| Dependency Audit | pip-audit | 0 critical, ≤2 high CVEs | Scans installed packages for known vulnerabilities |
| Secrets Scan | Bandit | No HIGH severity findings | Static analysis for hardcoded credentials/API keys |
| Data Integrity | Custom | Label-flip rate < 5% | Detects training data poisoning attacks |
| Adversarial Robustness | FGSM (numpy) | Accuracy under attack ≥ 70% | Tests model resilience to gradient-based perturbations |
| Model Integrity | SHA-256 | Checksum must match | Verifies model artifact has not been tampered with |

---

## DEA Evaluation

The system uses **formal Data Envelopment Analysis** to quantitatively compare three architectural paradigms:

### Methodology
- **Model**: Charnes-Cooper Linear Programming (CCR Multiplier Model)
- **Solver**: `scipy.optimize.linprog` with HiGHS method
- **Inputs** (costs): Training time, security vulnerabilities, MTTR, deployment lead time
- **Outputs** (benefits): F1 score, precision, recall, security score, robustness score, auto-recovery rate

### Composite Score Tiebreaker
With only 3 DMUs and many variables, formal DEA may rate all paradigms as efficient. A **composite efficiency score** (normalized output/input ratio) serves as tiebreaker.

---

## Standardized Experiment Results

Every pipeline run produces a standardized JSON payload containing:

```json
{
  "experiment_id": "exp_20261004_061500_abc123",
  "experiment_name": "pipeline_experiment",
  "status": "COMPLETED",
  "dataset": { "name", "source", "n_samples", "n_features", "class_distribution" },
  "data_quality": { "passed", "checks" },
  "models": { "<algorithm>": { "metrics", "hyperparams", "run_id" } },
  "best_model": { "algorithm", "optimal_threshold", "metrics" },
  "security": { "overall_passed", "overall_score", "gate_results" },
  "resources": { "pipeline_time", "cpu_count", "ram_usage" },
  "monitoring": { "drift_threshold", "anomaly_detector_active" },
  "deployment": { "stage", "model_version", "api_endpoint" },
  "research_metrics": { "f1", "precision", "recall", "accuracy", "roc_auc", "latency" }
}
```

Results are persisted to:
- `results/experiments/<exp_id>.json` (versioned)
- `results/latest_experiment.json` (symlink to latest)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info & status |
| `GET` | `/health` | Health check (model loaded, detectors active) |
| `POST` | `/predict` | ML inference with anomaly detection & auto-response |
| `GET` | `/metrics` | Prometheus-compatible metrics |
| `GET` | `/security/audit` | Real-time dependency & secrets security scan |

---

## Configuration

All pipeline behaviour is controlled by [`configs/config.yaml`](configs/config.yaml).
Environment variable overrides use the prefix `MLSECOPS_` with double-underscore hierarchy:

```bash
MLSECOPS_PROJECT__RANDOM_SEED=99   # overrides project.random_seed
MLSECOPS_TRAINING__CV_FOLDS=10     # overrides training.cv_folds
```

Secrets (DB passwords, webhook URLs, API keys) must **only** be in `.env`.

---

## Running Individual Stages

Each module is independently executable:

```bash
# Data ingestion
python -m src.data.loader

# Data validation
python -m src.data.validator

# Feature pipeline
python -m src.features.preprocessing

# Training (trains all configured algorithms)
python -m src.training.trainer

# Full pipeline
python -m src.training.pipeline

# 3-way benchmark
python -m experiments.run_comparison
```

---

## Reproducibility

- All random seeds are centralised in `config.yaml` under `project.random_seed`
- Dataset splits are deterministic via stratified seeded splitting
- Feature transformations are fitted only on training data
- Experiment results are versioned in `results/experiments/`
- MLflow tracks all hyperparameters, metrics, and model artifacts

---

## CI/CD Pipeline (GitHub Actions)

```
Job 1: Security Scans & Code Quality
  ├── Bandit static security analysis
  └── pip-audit dependency vulnerability scan

Job 2: Unit & Integration Tests (49 tests)
  └── pytest with full coverage

Job 3: 3-Way Benchmark & DEA Evaluation
  ├── Traditional ML vs MLOps vs MLSecOps
  └── Upload benchmark artifacts
```

---

## Citation

If you use this system for research, please cite:

```bibtex
@software{automated_mlsecops_2026,
  title  = {Automated MLSecOps: A Research-Grade Pipeline with Security-by-Design and DEA Evaluation},
  year   = {2026},
  url    = {https://github.com/<your-repo>}
}
```
