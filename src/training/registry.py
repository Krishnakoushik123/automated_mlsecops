"""
src/training/registry.py
------------------------
Wraps MLflow model registry operations:
  * register_model  – promote a run artifact to the registry
  * transition_model_stage – move to Staging / Production / Archived
  * get_latest_model – load the latest Production model
  * list_registered_models – audit helper

Run standalone:
    python -m src.training.registry --help
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import load_config, get_mlflow_uri
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _client(config: dict) -> MlflowClient:
    mlflow.set_tracking_uri(get_mlflow_uri(config))
    return MlflowClient()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_model(
    run_id: str,
    artifact_path: str,
    model_name: Optional[str] = None,
    config: dict | None = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Register a run's artifact in the MLflow Model Registry.

    Returns the new model version string.
    """
    if config is None:
        config = load_config()

    name = model_name or config["mlflow"]["model_name"]
    model_uri = f"runs:/{run_id}/{artifact_path}"

    logger.info("Registering model '%s' from run %s …", name, run_id)
    mv = mlflow.register_model(model_uri=model_uri, name=name, tags=tags)
    logger.info("Registered: %s  version=%s  status=%s", name, mv.version, mv.status)
    return str(mv.version)


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
    config: dict | None = None,
    archive_existing: bool = True,
) -> None:
    """Transition a registered model version to a lifecycle stage.

    stage: 'Staging' | 'Production' | 'Archived' | 'None'
    """
    if config is None:
        config = load_config()

    client = _client(config)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info(
        "Model '%s' version %s -> stage '%s'", model_name, version, stage
    )


def get_latest_model(
    model_name: Optional[str] = None,
    stage: str = "Production",
    config: dict | None = None,
) -> Any:
    """Load the latest model from the given stage."""
    if config is None:
        config = load_config()

    name = model_name or config["mlflow"]["model_name"]
    mlflow.set_tracking_uri(config["paths"]["mlflow_uri"])
    model_uri = f"models:/{name}/{stage}"
    logger.info("Loading model from: %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)


def list_registered_models(config: dict | None = None) -> List[Dict]:
    """Return a list of dicts describing all registered models."""
    if config is None:
        config = load_config()

    client = _client(config)
    models = []
    for rm in client.search_registered_models():
        for mv in rm.latest_versions:
            models.append({
                "name":         rm.name,
                "version":      mv.version,
                "stage":        mv.current_stage,
                "run_id":       mv.run_id,
                "creation_ts":  mv.creation_timestamp,
            })
    return models


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="MLflow Model Registry helper")
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="List registered models")

    p_reg = sub.add_parser("register", help="Register a model version")
    p_reg.add_argument("--run-id",       required=True)
    p_reg.add_argument("--artifact-path", required=True)
    p_reg.add_argument("--model-name",   default=None)

    p_trans = sub.add_parser("transition", help="Transition model to a stage")
    p_trans.add_argument("--model-name", required=True)
    p_trans.add_argument("--version",    required=True)
    p_trans.add_argument("--stage",      required=True, choices=["Staging", "Production", "Archived", "None"])

    args = parser.parse_args()
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])

    if args.cmd == "list":
        models = list_registered_models(cfg)
        print(json.dumps(models, indent=2))

    elif args.cmd == "register":
        version = register_model(
            run_id=args.run_id,
            artifact_path=args.artifact_path,
            model_name=args.model_name,
            config=cfg,
        )
        print(f"Registered version: {version}")

    elif args.cmd == "transition":
        transition_model_stage(
            model_name=args.model_name,
            version=args.version,
            stage=args.stage,
            config=cfg,
        )
    else:
        parser.print_help()
