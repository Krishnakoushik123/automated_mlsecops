"""
src/utils/config.py
-------------------
Load and merge YAML config + .env overrides.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load .env once at import time (no-op if file absent)
load_dotenv(override=False)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load ``configs/config.yaml`` and return as a plain dict.

    Environment variables prefixed with ``MLSECOPS_`` override leaf values
    using double-underscore hierarchy notation, e.g.::

        MLSECOPS_PROJECT__RANDOM_SEED=99  ->  config["project"]["random_seed"] = 99
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    # Apply env-variable overrides
    for key, value in os.environ.items():
        if not key.startswith("MLSECOPS_"):
            continue
        parts = key[len("MLSECOPS_"):].lower().split("__")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)

    return cfg


def get_mlflow_uri(config: Dict[str, Any] | None = None) -> str:
    """Return a resolved, platform-safe MLflow tracking URI."""
    if config is None:
        config = load_config()

    raw_uri = config.get("paths", {}).get("mlflow_uri", "sqlite:///mlflow.db")
    if isinstance(raw_uri, str) and raw_uri.startswith("sqlite:///"):
        rel_path = raw_uri.replace("sqlite:///", "")
        abs_path = (Path.cwd() / rel_path).resolve().as_posix()
        return f"sqlite:///{abs_path}"
    elif isinstance(raw_uri, str) and (raw_uri.startswith("http://") or raw_uri.startswith("https://")):
        return raw_uri
    else:
        # Fallback to SQLite in project root
        abs_path = (Path.cwd() / "mlflow.db").resolve().as_posix()
        return f"sqlite:///{abs_path}"

