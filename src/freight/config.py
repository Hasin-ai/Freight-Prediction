"""Paths and YAML configuration. Paths come from environment variables so the same code runs in Docker and locally."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FREIGHT_DATA", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("FREIGHT_OUTPUT", PROJECT_ROOT / "outputs"))
MODEL_DIR = PROJECT_ROOT / "model"
CONFIG_DIR = PROJECT_ROOT / "configs"

EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]  # Dry Van is the reference level


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file; relative paths are resolved against the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as handle:
        return yaml.safe_load(handle)


def data_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "data.yaml")


def feature_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "features.yaml")


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Experiment YAML with `model_from: <file>` resolved to that file's `model` block."""
    config = load_yaml(path)
    if "model_from" in config:
        config["model"] = load_yaml(config["model_from"])["model"]
    return config
