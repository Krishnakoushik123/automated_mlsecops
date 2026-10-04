"""
experiments/traditional/run_experiment.py
------------------------------------------
Baseline Traditional ML experiment:
- No automation, no MLflow tracking, no security gates.
- Manual train/test split, single algorithm, metrics printed to stdout.
- Captures: accuracy, F1, training time, inference latency.

This is the "DMU A" (Decision Making Unit) in the DEA comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# ── Config ────────────────────────────────────────────────────────────────
SEED          = 42
N_SAMPLES     = 5000
N_FEATURES    = 20
N_INFORMATIVE = 10
TEST_SIZE     = 0.20

np.random.seed(SEED)


def run() -> dict:
    print("=" * 55)
    print("TRADITIONAL ML EXPERIMENT")
    print("=" * 55)
    t_wall_start = time.perf_counter()

    # ── Data ──────────────────────────────────────────────────
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=4,
        weights=[0.95, 0.05],
        random_state=SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # ── Feature scaling ───────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Training ──────────────────────────────────────────────
    t_train_start = time.perf_counter()
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t_train_start

    # ── Evaluation ────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Inference latency (single-sample)
    latencies = []
    for _ in range(200):
        idx = np.random.randint(0, len(X_test))
        t0 = time.perf_counter()
        model.predict(X_test[idx : idx + 1])
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies_arr = np.array(latencies)
    wall_time = time.perf_counter() - t_wall_start

    results = {
        "experiment":            "traditional",
        "algorithm":             "random_forest",
        "dataset_size":          N_SAMPLES,
        "training_time_seconds": training_time,
        "wall_time_seconds":     wall_time,
        "accuracy":              float(accuracy_score(y_test, y_pred)),
        "precision":             float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":                float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score":              float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":               float(roc_auc_score(y_test, y_prob)),
        "latency_p50_ms":        float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms":        float(np.percentile(latencies_arr, 95)),
        "latency_p99_ms":        float(np.percentile(latencies_arr, 99)),
        # DEA input proxies (estimated for research comparison)
        "development_time_hours": 8.0,   # manual estimate
        "labor_effort_hours":     8.0,
        "cpu_usage_percent":      None,   # not monitored in traditional
        "memory_usage_mb":        None,
        "robustness_score":       None,   # not measured
        "security_pass_rate":     0.0,    # no security gates
        "mlflow_tracked":         False,
        "security_gated":         False,
        "automated":              False,
    }

    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  F1 Score : {results['f1_score']:.4f}")
    print(f"  ROC-AUC  : {results['roc_auc']:.4f}")
    print(f"  Train    : {results['training_time_seconds']:.3f}s")
    print(f"  Latency  : P99={results['latency_p99_ms']:.2f}ms")

    # ── Persist ───────────────────────────────────────────────
    out_dir = Path("results/json")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_traditional.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n✓  Results saved → {out_path}")
    return results


if __name__ == "__main__":
    run()
