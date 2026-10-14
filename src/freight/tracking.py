"""All MLflow logging goes through this module. The server address comes only from MLFLOW_TRACKING_URI."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from . import data
from .config import PROJECT_ROOT
from .metrics import by_segment, horizon_bucket
from .plots import residual_diagnostics
from .splits import make_folds


def code_version() -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    version = {}
    try:
        version["code.git_commit"] = git("rev-parse", "HEAD")
        changed = git("status", "--porcelain", "--", "src", "scripts", "configs")
        version["code.uncommitted_changes"] = str(bool(changed))
    except (subprocess.CalledProcessError, FileNotFoundError):
        version["code.git_commit"] = "unavailable"
    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        digest.update(path.read_bytes())
    version["code.src_sha256"] = digest.hexdigest()[:16]
    return version


def environment() -> dict[str, str]:
    import lightgbm
    import numpy
    import sklearn

    return {
        "env.python": sys.version.split()[0],
        "env.platform": platform.platform(),
        "env.numpy": numpy.__version__,
        "env.pandas": pd.__version__,
        "env.sklearn": sklearn.__version__,
        "env.lightgbm": lightgbm.__version__,
        "env.mlflow": mlflow.__version__,
    }


def start_run(experiment: str, run_name: str, tags: dict[str, Any] | None = None):
    """Start a run tagged with code version, environment and dataset fingerprints."""
    mlflow.set_experiment(experiment)
    all_tags = {**code_version(), **environment(), **data.fingerprint(), **(tags or {})}
    return mlflow.start_run(run_name=run_name, tags={k: str(v) for k, v in all_tags.items()})


def log_table(frame: pd.DataFrame, file_name: str, folder: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / file_name
        if file_name.endswith(".parquet"):
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        mlflow.log_artifact(str(path), folder)


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    """Nested config -> flat MLflow params ("model.params.alpha": 0.265)."""
    if isinstance(value, dict):
        flat = {}
        for key, item in value.items():
            flat.update(flatten(f"{prefix}.{key}", item))
        return flat
    return {prefix: json.dumps(value) if isinstance(value, (list, tuple)) else value}


def log_cross_validation(
    config: dict[str, Any],
    config_path: str | None,
    role: str,
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    aggregate: dict[str, float],
) -> None:
    """Params, per-fold and mean metrics, fold manifests, out-of-fold predictions, segment tables and a plot."""
    from .cross_validation import PRIMARY_METRIC, labelled_loads

    mlflow.log_params({**flatten("model", config["model"]), "split.role": role, "metric.primary": PRIMARY_METRIC})
    mlflow.log_metrics(aggregate)
    for step, row in fold_metrics.reset_index(drop=True).iterrows():
        mlflow.log_metrics(
            {f"fold.{name}": float(row[name]) for name in fold_metrics.columns if name != "fold"},
            step=int(step),
        )
    if config_path:
        mlflow.log_artifact(str(PROJECT_ROOT / config_path), "config")
    mlflow.log_dict(config, "config/resolved_config.json")
    loads = labelled_loads()
    mlflow.log_dict([fold.manifest(loads) for fold in make_folds(loads, role)], f"split/folds_{role}.json")
    log_table(fold_metrics, f"fold_metrics_{role}.csv", "evaluation")
    log_table(predictions, f"predictions_{role}.parquet", "predictions")
    segments = predictions.assign(
        horizon=horizon_bucket(predictions["days_ahead"]),
        distance_band=pd.cut(predictions["distance"], [0, 250, 500, 1000, 2000, 4000]).astype(str),
    )
    for column in ["equipment", "horizon", "distance_band", "unseen_city"]:
        log_table(by_segment(segments, column).reset_index(), f"segment_{column}_{role}.csv", "evaluation")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"residuals_{role}.png"
        residual_diagnostics(predictions, path, config["name"])
        mlflow.log_artifact(str(path), "plots")


def latest_run(experiment: str, run_name: str) -> mlflow.entities.Run:
    """Most recent finished run with this name; raises if none exists."""
    found = mlflow.get_experiment_by_name(experiment)
    if found is None:
        raise LookupError(f"experiment {experiment!r} does not exist yet")
    runs = mlflow.search_runs(
        [found.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    if not runs:
        raise LookupError(f"no finished run named {run_name!r} in {experiment!r}")
    return runs[0]


def load_predictions(run: mlflow.entities.Run, role: str) -> pd.DataFrame:
    path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=f"predictions/predictions_{role}.parquet"
    )
    return pd.read_parquet(path)
