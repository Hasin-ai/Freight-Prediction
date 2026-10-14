"""Fit a model specification on each fold's training rows and score the fold's assessment rows."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from . import data
from .cleaning import clean_loads
from .label_screen import reporting_clean_flag
from .metrics import summarize
from .models import build_model
from .splits import make_folds

PRIMARY_METRIC = "wMAE_clean"


@lru_cache(maxsize=1)
def labelled_loads() -> pd.DataFrame:
    """Cleaned training loads plus the fixed reporting flag for plausible labels (never used in fitting)."""
    raw = data.load("train")
    problems = data.schema_problems(raw, "train")
    if problems:
        raise ValueError(problems)
    loads = clean_loads(raw)
    loads["clean"] = reporting_clean_flag(loads)
    return loads.reset_index(drop=True)


def run_folds(
    model_spec: dict[str, Any], role: str = "dev", fold_positions: list[int] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Returns (out-of-fold predictions, per-fold metrics, aggregate metrics prefixed with the role)."""
    loads = labelled_loads()
    folds = make_folds(loads, role)
    if fold_positions is not None:
        folds = [folds[i] for i in fold_positions]
    prediction_frames, fold_rows = [], []
    for fold in folds:
        train = loads.iloc[fold.train_rows].reset_index(drop=True)
        assess = loads.iloc[fold.assess_rows].reset_index(drop=True)
        started = time.time()
        model = build_model(model_spec).fit(train, np.log(train["posted_rate"].values))
        fit_seconds = time.time() - started
        predicted = np.exp(model.predict_log(assess))
        frame = pd.DataFrame(
            {
                "fold": fold.name,
                "role": fold.role,
                "load_id": assess["load_id"],
                "date": assess["date"],
                "days_ahead": (assess["date"] - pd.Timestamp(fold.origin)).dt.days,
                "actual": assess["posted_rate"],
                "predicted": predicted,
                "clean": assess["clean"].values,
                "unseen_city": fold.touches_hidden_city,
                "equipment": assess["equipment"].values,
                "distance": assess["distance"].values,
            }
        )
        prediction_frames.append(frame)
        metrics = summarize(frame["actual"], frame["predicted"], frame["clean"], frame["unseen_city"])
        fold_rows.append({"fold": fold.name, **metrics, "fit_seconds": fit_seconds})
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    numeric = fold_metrics.drop(columns="fold")
    aggregate = {f"{role}.mean.{name}": float(value) for name, value in numeric.mean().items()}
    aggregate[f"{role}.worst.{PRIMARY_METRIC}"] = float(fold_metrics[PRIMARY_METRIC].max())
    if len(fold_metrics) > 1:
        aggregate[f"{role}.std.{PRIMARY_METRIC}"] = float(fold_metrics[PRIMARY_METRIC].std(ddof=1))
    return predictions, fold_metrics, aggregate


REPRODUCTION_TOLERANCE = 0.05  # dollars


def check_against_reference(label: str, value: float, reference: float | None) -> str:
    """Compare a fresh result with the value the earlier project logged. Raises if they differ by > $0.05."""
    if reference is None:
        return f"{label}: {value:.2f} (no earlier reference)"
    difference = value - reference
    message = f"{label}: {value:.2f} vs earlier {reference:.2f} (difference {difference:+.3f})"
    if abs(difference) > REPRODUCTION_TOLERANCE:
        raise RuntimeError(f"REPRODUCTION FAILURE - {message}")
    return message
