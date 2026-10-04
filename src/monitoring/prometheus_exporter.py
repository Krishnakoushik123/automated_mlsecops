"""
src/monitoring/prometheus_exporter.py
-------------------------------------
Prometheus metrics registry and custom collectors for real-time monitoring of:
  - Inference requests counter
  - Latency histogram (seconds)
  - Predictions distribution
  - Anomaly detection counter
  - Data drift score gauge
  - Security gate status gauge
"""

from __future__ import annotations

from typing import Dict, Any
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Prometheus Counters & Gauges
INFERENCE_REQUESTS = Counter(
    "mlsecops_inference_requests_total",
    "Total number of inference requests processed",
    ["status", "model_version"]
)

INFERENCE_LATENCY = Histogram(
    "mlsecops_inference_latency_seconds",
    "Inference execution latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTIONS_COUNTER = Counter(
    "mlsecops_predictions_total",
    "Total predictions classified by target label",
    ["label"]
)

ANOMALY_COUNTER = Counter(
    "mlsecops_anomalies_detected_total",
    "Total runtime anomalies or OOD inputs detected",
    ["type"]
)

DRIFT_SCORE_GAUGE = Gauge(
    "mlsecops_data_drift_score",
    "Current overall data drift score [0.0 - 1.0]"
)

SECURITY_GATE_STATUS = Gauge(
    "mlsecops_security_gate_status",
    "Status of security gates (1.0 = All Pass, 0.0 = Failed)",
    ["gate_name"]
)

AUTO_RESPONSE_COUNTER = Counter(
    "mlsecops_auto_responses_triggered_total",
    "Total automated response mitigations executed",
    ["action"]
)


def record_inference(status: str, latency: float, prediction: int, model_version: str = "v1"):
    """Record single inference metrics."""
    INFERENCE_REQUESTS.labels(status=status, model_version=model_version).inc()
    INFERENCE_LATENCY.observe(latency)
    PREDICTIONS_COUNTER.labels(label=str(prediction)).inc()


def record_anomaly(anomaly_type: str = "ood"):
    """Record an anomaly detection event."""
    ANOMALY_COUNTER.labels(type=anomaly_type).inc()


def update_drift_score(score: float):
    """Update gauge metric for current feature drift score."""
    DRIFT_SCORE_GAUGE.set(score)


def update_security_gate_status(gate_name: str, passed: bool):
    """Update status of a security gate."""
    SECURITY_GATE_STATUS.labels(gate_name=gate_name).set(1.0 if passed else 0.0)


def record_auto_response(action: str):
    """Record automated mitigation response execution."""
    AUTO_RESPONSE_COUNTER.labels(action=action).inc()


def get_metrics_payload() -> tuple[bytes, str]:
    """Generate Prometheus formatted metrics text."""
    return generate_latest(), CONTENT_TYPE_LATEST
