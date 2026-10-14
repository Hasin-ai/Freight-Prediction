"""Model registry: every model consumes cleaned loads and predicts log(rate)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from .baselines import (
    CatBoostRawDollars,
    CrossSectionPlusNaiveIndex,
    EquipmentRatePerMile,
    LaneRatePerMile,
    LightGBMDirect,
)
from .hybrid import RidgePlusResidualTrees
from .ridge_model import StructuralRidge


class RateModel(Protocol):
    def fit(self, loads: pd.DataFrame, log_rate: np.ndarray) -> RateModel: ...

    def predict_log(self, loads: pd.DataFrame) -> np.ndarray: ...


MODEL_TYPES: dict[str, type] = {
    "equipment_rate_per_mile": EquipmentRatePerMile,
    "lane_rate_per_mile": LaneRatePerMile,
    "cross_section_naive_index": CrossSectionPlusNaiveIndex,
    "catboost_raw_dollars": CatBoostRawDollars,
    "lightgbm_direct": LightGBMDirect,
    "structural_ridge": StructuralRidge,
    "ridge_plus_residual_trees": RidgePlusResidualTrees,
}


def build_model(spec: dict[str, Any]) -> RateModel:
    """spec = {"type": <key of MODEL_TYPES>, "params": {...}} as written in the YAML configs."""
    model_type = spec["type"]
    if model_type not in MODEL_TYPES:
        raise KeyError(f"unknown model type {model_type!r}; choose one of {sorted(MODEL_TYPES)}")
    return MODEL_TYPES[model_type](**(spec.get("params") or {}))
