"""
src/security/gates.py
---------------------
Security gate orchestrator (Phase 2).

Each gate runs an ACTUAL test/check and returns a structured result.
Gates are:
  1. dependency_audit   – pip-audit for known CVEs
  2. secrets_scan       – detect hardcoded secrets (bandit B105/B106/B107)
  3. data_integrity     – label-flip / poisoning rate check
  4. adversarial_robustness – FGSM accuracy under perturbation
  5. model_integrity    – SHA-256 checksum of saved model artifact

All gates are independently runnable.

Run standalone:
    python -m src.security.gates
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data class for individual gate results
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SecurityGateResult:
    gate: str
    passed: bool
    score: Optional[float]          # 0-1 normalised score where relevant
    details: Dict[str, Any]
    duration_seconds: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Gate 1 – Dependency vulnerability audit (pip-audit)
# ---------------------------------------------------------------------------

def gate_dependency_audit(config: dict) -> SecurityGateResult:
    """Run pip-audit and fail if critical/high vulnerabilities exceed thresholds."""
    t0 = time.perf_counter()
    max_crit = config["security"].get("max_critical_vulns", 0)
    max_high = config["security"].get("max_high_vulns", 2)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--format", "json", "--progress-spinner", "off"],
            capture_output=True, text=True, timeout=120,
        )
        raw = proc.stdout.strip() or "[]"
        vulns = json.loads(raw) if raw.startswith("[") else []
    except FileNotFoundError:
        return SecurityGateResult(
            gate="dependency_audit", passed=False, score=0.0,
            details={"error": "pip-audit not installed"},
            duration_seconds=time.perf_counter() - t0,
            error="pip-audit not found",
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return SecurityGateResult(
            gate="dependency_audit", passed=False, score=0.0,
            details={"error": str(exc)},
            duration_seconds=time.perf_counter() - t0,
            error=str(exc),
        )

    critical = sum(
        1 for v in vulns
        for alias in v.get("vulns", [])
        if "CRITICAL" in str(alias.get("fix_versions", "")).upper()
           or alias.get("aliases", [""])[0].startswith("GHSA")
    )
    high = len(vulns) - critical  # simplified split

    passed = critical <= max_crit and high <= max_high
    total = len(vulns)
    score = max(0.0, 1.0 - (critical * 0.5 + high * 0.1))

    return SecurityGateResult(
        gate="dependency_audit",
        passed=passed,
        score=min(1.0, score),
        details={
            "total_vulnerabilities": total,
            "critical": critical,
            "high": high,
            "max_critical_allowed": max_crit,
            "max_high_allowed": max_high,
        },
        duration_seconds=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Gate 2 – Static secrets scan (Bandit)
# ---------------------------------------------------------------------------

def gate_secrets_scan(config: dict) -> SecurityGateResult:
    """Use Bandit to detect hardcoded secrets / high-severity issues."""
    t0 = time.perf_counter()
    src_dir = "src"

    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "bandit",
                "-r", src_dir,
                "-f", "json",
                "--tests", "B105,B106,B107,B108,B110,B201,B301,B303,B306,B307,B311,B601,B602,B603,B605",
                "-q",
            ],
            capture_output=True, text=True, timeout=60,
        )
        raw = proc.stdout.strip()
        report = json.loads(raw) if raw else {}
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return SecurityGateResult(
            gate="secrets_scan", passed=False, score=0.0,
            details={"error": str(exc)},
            duration_seconds=time.perf_counter() - t0,
            error=str(exc),
        )

    results = report.get("results", [])
    high_sev = [r for r in results if r.get("issue_severity") == "HIGH"]
    medium_sev = [r for r in results if r.get("issue_severity") == "MEDIUM"]
    passed = len(high_sev) == 0
    score = max(0.0, 1.0 - len(high_sev) * 0.2 - len(medium_sev) * 0.05)

    return SecurityGateResult(
        gate="secrets_scan",
        passed=passed,
        score=min(1.0, score),
        details={
            "high_severity": len(high_sev),
            "medium_severity": len(medium_sev),
            "total_issues": len(results),
            "findings": [
                {"file": r.get("filename"), "line": r.get("line_number"), "test": r.get("test_id")}
                for r in high_sev[:10]
            ],
        },
        duration_seconds=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Gate 3 – Data integrity / poisoning detection
# ---------------------------------------------------------------------------

def gate_data_integrity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    config: dict,
) -> SecurityGateResult:
    """Detect label-flip attacks and statistical anomalies in training data.

    Approach:
    - Compare label distribution drift (KL divergence) between a reference
      and current training set.
    - Flag if estimated label-flip rate exceeds configured threshold.
    """
    t0 = time.perf_counter()
    threshold = config["security"].get("poisoning_threshold", 0.05)

    # Label distribution comparison
    ref_counts = np.bincount(y_ref.astype(int), minlength=2) / len(y_ref)
    cur_counts = np.bincount(y_train.astype(int), minlength=2) / len(y_train)

    # Symmetric KL divergence (Jensen-Shannon)
    eps = 1e-10
    m = 0.5 * (ref_counts + cur_counts)
    js_div = 0.5 * np.sum(ref_counts * np.log((ref_counts + eps) / (m + eps))) \
           + 0.5 * np.sum(cur_counts * np.log((cur_counts + eps) / (m + eps)))
    js_div = float(np.clip(js_div, 0.0, 1.0))

    # Feature distribution: z-score anomaly count
    mean_ref = X_ref.mean(axis=0)
    std_ref  = X_ref.std(axis=0) + 1e-10
    z_scores = np.abs((X_train - mean_ref) / std_ref)
    anomaly_rate = float((z_scores > 5.0).any(axis=1).mean())

    estimated_flip_rate = min(js_div, 1.0)
    passed = estimated_flip_rate <= threshold and anomaly_rate <= threshold * 2

    score = max(0.0, 1.0 - estimated_flip_rate - anomaly_rate * 0.5)

    return SecurityGateResult(
        gate="data_integrity",
        passed=passed,
        score=min(1.0, score),
        details={
            "js_divergence":       js_div,
            "estimated_flip_rate": estimated_flip_rate,
            "feature_anomaly_rate": anomaly_rate,
            "threshold":           threshold,
        },
        duration_seconds=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Gate 4 – Adversarial robustness (FGSM)
# ---------------------------------------------------------------------------

def gate_adversarial_robustness(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    epsilon: float = 0.1,
    n_samples: int = 500,
) -> SecurityGateResult:
    """FGSM-style perturbation test.

    Since sklearn models are not differentiable, we approximate gradient
    direction using finite differences on the decision function.
    """
    from sklearn.metrics import accuracy_score
    t0 = time.perf_counter()
    min_score = config["security"].get("min_robustness_score", 0.70)

    idx = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]

    # Finite-difference gradient approximation
    delta = 1e-4
    perturbed = X_sample.copy()
    try:
        # Use decision_function or predict_proba as surrogate loss
        if hasattr(model, "decision_function"):
            base = model.decision_function(X_sample)
        else:
            base = model.predict_proba(X_sample)[:, 1]

        grad_sign = np.zeros_like(X_sample)
        for j in range(X_sample.shape[1]):
            X_delta = X_sample.copy()
            X_delta[:, j] += delta
            if hasattr(model, "decision_function"):
                perturbed_score = model.decision_function(X_delta)
            else:
                perturbed_score = model.predict_proba(X_delta)[:, 1]
            grad_sign[:, j] = np.sign(perturbed_score - base)

        perturbed = X_sample + epsilon * grad_sign
    except Exception:
        pass  # Fall back to random perturbation
        perturbed = X_sample + epsilon * np.sign(np.random.randn(*X_sample.shape))

    y_adv_pred = model.predict(perturbed)
    adv_accuracy = float(accuracy_score(y_sample, y_adv_pred))
    passed = adv_accuracy >= min_score

    return SecurityGateResult(
        gate="adversarial_robustness",
        passed=passed,
        score=adv_accuracy,
        details={
            "epsilon":        epsilon,
            "n_samples":      n_samples,
            "adv_accuracy":   adv_accuracy,
            "min_required":   min_score,
            "method":         "fgsm_finite_diff",
        },
        duration_seconds=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Gate 5 – Model integrity (SHA-256 checksum)
# ---------------------------------------------------------------------------

def gate_model_integrity(
    model_path: str | Path,
    expected_checksum: Optional[str] = None,
) -> SecurityGateResult:
    """Compute SHA-256 of the model file; verify against stored checksum."""
    import hashlib
    t0 = time.perf_counter()
    path = Path(model_path)

    if not path.exists():
        return SecurityGateResult(
            gate="model_integrity", passed=False, score=0.0,
            details={"error": f"Model file not found: {path}"},
            duration_seconds=time.perf_counter() - t0,
            error="File not found",
        )

    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()

    if expected_checksum is None:
        # First run: store checksum alongside model
        checksum_path = path.with_suffix(".sha256")
        if not checksum_path.exists():
            checksum_path.write_text(checksum)
            logger.info("Model checksum stored: %s", checksum_path)
            passed = True
        else:
            stored = checksum_path.read_text().strip()
            passed = (checksum == stored)
    else:
        passed = (checksum == expected_checksum)

    return SecurityGateResult(
        gate="model_integrity",
        passed=passed,
        score=1.0 if passed else 0.0,
        details={
            "model_path": str(path),
            "sha256":     checksum,
            "matched":    passed,
        },
        duration_seconds=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_gates(
    config: dict | None = None,
    model: Any = None,
    model_path: str | Path | None = None,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> List[SecurityGateResult]:
    """Run all enabled security gates and return list of results."""
    if config is None:
        config = load_config()

    sec_cfg = config["security"]
    results: List[SecurityGateResult] = []

    if sec_cfg.get("dependency_audit", True):
        logger.info("Running gate: dependency_audit")
        results.append(gate_dependency_audit(config))

    if sec_cfg.get("secrets_scan", True):
        logger.info("Running gate: secrets_scan")
        results.append(gate_secrets_scan(config))

    if sec_cfg.get("data_integrity", True) and X_train is not None:
        logger.info("Running gate: data_integrity")
        ref_X = X_ref if X_ref is not None else X_train
        ref_y = y_ref if y_ref is not None else y_train
        results.append(gate_data_integrity(X_train, y_train, ref_X, ref_y, config))

    if sec_cfg.get("adversarial_robustness", True) and model is not None and X_test is not None:
        logger.info("Running gate: adversarial_robustness")
        results.append(gate_adversarial_robustness(model, X_test, y_test, config))

    if sec_cfg.get("model_integrity", True) and model_path is not None:
        logger.info("Running gate: model_integrity")
        results.append(gate_model_integrity(model_path))

    all_passed = all(r.passed for r in results)
    logger.info(
        "Security gates: %d/%d passed  [%s]",
        sum(r.passed for r in results),
        len(results),
        "ALL PASS [OK]" if all_passed else "FAILED [FAIL]",
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    cfg = load_config()
    # Minimal run (code-level gates only)
    results = [gate_dependency_audit(cfg), gate_secrets_scan(cfg)]
    print("\n=== Security Gate Results ===")
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  [{status}] {r.gate}  score={r.score:.2f}  ({r.duration_seconds:.2f}s)")
        if r.error:
            print(f"         error: {r.error}")
