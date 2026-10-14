"""Champion model: structural ridge + LightGBM on the ridge residuals."""

from __future__ import annotations

from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..cleaning import recency_weights
from ..features import residual_tree_inputs
from .ridge_model import DEFAULT_GROUPS, StructuralRidge

RESIDUAL_TREE_DEFAULTS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": 1,
}


class RidgePlusResidualTrees:
    """Stage 1 (ridge) carries everything that must extrapolate: market, drift and calendar.
    Stage 2 (a small LightGBM) only corrects local, time-free patterns in what the ridge leaves over,
    such as lane geography interacting with distance and weight."""

    def __init__(
        self,
        groups: Sequence[str] = DEFAULT_GROUPS,
        alpha: float = 0.1,
        use_label_screen: bool = True,
        recency_tau_days: float | None = None,
        spline_knots: dict | None = None,
        residual_tree_params: dict | None = None,
        residual_trees_see_market: bool = True,
    ) -> None:
        self.ridge = StructuralRidge(groups, alpha, use_label_screen, recency_tau_days, spline_knots)
        self.residual_tree_params = {**RESIDUAL_TREE_DEFAULTS, **(residual_tree_params or {})}
        self.residual_trees_see_market = residual_trees_see_market

    def fit(self, loads: pd.DataFrame, log_rate: np.ndarray) -> RidgePlusResidualTrees:
        self.ridge.fit(loads, log_rate)
        residual = log_rate - self.ridge.predict_log(loads)
        kept = self.ridge.kept_rows
        weights = recency_weights(loads["days_since_start"].values, self.ridge.recency_tau_days)
        inputs = residual_tree_inputs(loads, self.residual_trees_see_market)
        self.trees = lgb.LGBMRegressor(**self.residual_tree_params).fit(
            inputs[kept], residual[kept], sample_weight=None if weights is None else weights[kept]
        )
        return self

    def predict_log(self, loads: pd.DataFrame) -> np.ndarray:
        inputs = residual_tree_inputs(loads, self.residual_trees_see_market)
        return self.ridge.predict_log(loads) + self.trees.predict(inputs)
