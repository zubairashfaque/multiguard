"""File I/O and artifact management utilities."""

import json
import pickle
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"Saved JSON: {path}")


def load_json(path: str | Path) -> Any:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)


def save_yaml(data: Any, path: str | Path) -> None:
    """Save data as YAML file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    logger.debug(f"Saved YAML: {path}")


def load_yaml(path: str | Path) -> Any:
    """Load data from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save object as pickle file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.debug(f"Saved pickle: {path}")


def load_pickle(path: str | Path) -> Any:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
