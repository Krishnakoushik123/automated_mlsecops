"""
src/monitoring/auto_responder.py
--------------------------------
Automated incident response & self-healing engine.

Actions triggered upon security or monitoring alerts:
  - Fallback to safe rule-based heuristic classifier
  - Quarantining / isolation of suspicious input batches
  - Circuit breaking to fallback model version
  - Logging security incident report
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.monitoring.prometheus_exporter import record_auto_response
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class AutoResponseResult:
    triggered: bool
    action_taken: str
    reason: str
    fallback_prediction: Optional[np.ndarray] = None
    incident_id: Optional[str] = None


class AutoResponder:
    """Automated response engine for runtime MLSecOps threats & anomalies."""

    def __init__(self, incident_dir: str = "results/incidents"):
        self.incident_dir = Path(incident_dir)
        self.incident_dir.mkdir(parents=True, exist_ok=True)

    def handle_input_anomaly(
        self,
        X_input: np.ndarray,
        anomaly_score: float,
        threshold: float = 0.75,
    ) -> AutoResponseResult:
        """Handle detected input anomaly or adversarial query."""
        if anomaly_score < threshold:
            return AutoResponseResult(
                triggered=False,
                action_taken="pass",
                reason="Anomaly score within safe threshold",
            )

        incident_id = f"inc_anom_{int(time.time() * 1000)}"
        logger.warning(
            "ALERT: Anomaly threshold exceeded (score=%.3f > %.3f). Triggering auto-response %s",
            anomaly_score, threshold, incident_id
        )

        # Action: Fallback to conservative safe prediction (e.g., flag as fraud / class 1 for safety)
        fallback_preds = np.ones(len(X_input), dtype=int)
        
        self._log_incident(incident_id, {
            "type": "input_anomaly",
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "sample_count": len(X_input),
            "action": "fallback_conservative_prediction",
        })

        record_auto_response("fallback_conservative_prediction")

        return AutoResponseResult(
            triggered=True,
            action_taken="fallback_conservative_prediction",
            reason=f"Anomalous input detected (score={anomaly_score:.2f})",
            fallback_prediction=fallback_preds,
            incident_id=incident_id,
        )

    def handle_data_drift(self, drift_score: float, threshold: float = 0.40) -> AutoResponseResult:
        """Handle severe data drift incident."""
        if drift_score < threshold:
            return AutoResponseResult(
                triggered=False,
                action_taken="pass",
                reason="Drift score within acceptable bounds",
            )

        incident_id = f"inc_drift_{int(time.time() * 1000)}"
        logger.warning("ALERT: Data drift threshold breached (score=%.3f). Triggering model quarantine %s", drift_score, incident_id)

        self._log_incident(incident_id, {
            "type": "data_drift",
            "drift_score": drift_score,
            "threshold": threshold,
            "action": "flag_retrain_quarantine",
        })

        record_auto_response("flag_retrain_quarantine")

        return AutoResponseResult(
            triggered=True,
            action_taken="flag_retrain_quarantine",
            reason=f"Severe data drift detected (drift_score={drift_score:.2f})",
            incident_id=incident_id,
        )

    def _log_incident(self, incident_id: str, data: Dict[str, Any]):
        """Persist incident details to disk."""
        data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data["incident_id"] = incident_id
        file_path = self.incident_dir / f"{incident_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Incident report logged -> %s", file_path)
