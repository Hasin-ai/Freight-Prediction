"""Reference models for the improvement ladder: lookups, a classical index forecast, CatBoost and LightGBM."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..features import LoadDescription, direct_tree_inputs
from ..label_screen import plausible_label_mask


class EquipmentRatePerMile:
    """rate = median rate per mile of the equipment type x distance. The simplest sensible price."""

    def fit(self, loads: pd.DataFrame, log_rate: np.ndarray) -> EquipmentRatePerMile:
        rate_per_mile = pd.Series(np.exp(log_rate) / loads["distance"].values)
        self.median_rate_per_mile = rate_per_mile.groupby(loads["equipment"].values).median()
        return self

    def predict_log(self, loads: pd.DataFrame) -> np.ndarray:
        rate_per_mile = loads["equipment"].map(self.median_rate_per_mile).values
        return np.log(rate_per_mile * loads["distance"].values)


class LaneRatePerMile:
    """rate = median rate per mile of the (pickup, delivery, equipment) lane x distance; falls back to
    equipment x distance-decile for lanes without history."""

    def fit(self, loads: pd.DataFrame, log_rate: np.ndarray) -> LaneRatePerMile:
        rate_per_mile = pd.Series(np.exp(log_rate) / loads["distance"].values, index=loads.index)
        self.decile_edges = np.quantile(loads["distance"], np.linspace(0, 1, 11))
        self.decile_edges[0], self.decile_edges[-1] = 0, 1e9
        decile = self._decile(loads)
        self.lane_median = rate_per_mile.groupby([loads["pickup"], loads["delivery"], loads["equipment"]]).median()
        self.fallback_median = rate_per_mile.groupby([loads["equipment"].values, decile]).median()
        return self

    def _decile(self, loads: pd.DataFrame) -> np.ndarray:
        return np.clip(np.searchsorted(self.decile_edges, loads["distance"]) - 1, 0, 9)

    def predict_log(self, loads: pd.DataFrame) -> np.ndarray:
        lanes = pd.MultiIndex.from_frame(loads[["pickup", "delivery", "equipment"]])
        lane = self.lane_median.reindex(lanes).values
        fallback_keys = pd.MultiIndex.from_arrays([loads["equipment"].values, self._decile(loads)])
        fallback = self.fallback_median.reindex(fallback_keys).values
        return np.log(np.where(np.isnan(lane), fallback, lane) * loads["distance"].values)


class CrossSectionPlusNaiveIndex:
    """A classical forecasting baseline: log(rate) = load description + market deviation + daily price level,
    where the daily level (day fixed effects on screened rows) is carried forward unchanged from the last day.
    """

    def fit(self, loads: pd.DataFrame, log_rate: np.ndarray) -> CrossSectionPlusNaiveIndex:
        self.description = LoadDescription().fit(loads)
        load_design = self.description.transform(loads).to_numpy(dtype=float)
        design = np.column_stack([load_design, loads["market_deviation_this_load"].values])
        kept = plausible_label_mask(load_design, log_rate)
        days = loads["date"].values[kept]
        # Remove each day's mean (fixed effects) to get load effects net of the daily price level.
        design_within = design[kept] - pd.DataFrame(design[kept]).groupby(days).transform("mean").values
        rate_within = log_rate[kept] - pd.Series(log_rate[kept]).groupby(days).transform("mean").values
        self.ridge = Ridge(alpha=1e-3, fit_intercept=False).fit(design_within, rate_within)
        daily_level = pd.Series(log_rate[kept] - self.ridge.predict(design[kept])).groupby(days).mean()
        daily_level.index = pd.to_datetime(daily_level.index)
        self.last_level = float(daily_level.asfreq("D").interpolate().iloc[-1])
        return self

    def predict_log(self, loads: pd.DataFrame) -> np.ndarray:
        load_design = self.description.transform(loads).to_numpy(dtype=float)
        design = np.column_stack([load_design, loads["market_deviation_this_load"].values])
        return self.ridge.predict(design) + self.last_level


class CatBoostRawDollars:
    """Re-implementation of the original submission script (train_catboost.py): dollar target, RMSE loss,
    city-name categoricals, month and day-of-year features, no label cleaning. 165 trees as in the submission
    (the original chose that count by early stopping on the Sep-Oct holdout)."""

    CATEGORICAL = ["pickup", "delivery", "equipment"]

    def __init__(self, iterations: int = 165, threads: int = 2) -> None:
        self.iterations = iterations
        self.threads = threads

    @staticmethod
    def _inputs(loads: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "pickup",
            "delivery",
            "equipment",
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
        ]
        inputs = loads[columns].copy()
        inputs["weight_abs"] = loads["weight_lbs_raw"].values
        inputs["weight_missing"] = loads["weight_missing"].values
        inputs["weight_was_negative"] = loads["weight_was_negative"].values
        inputs["delta_lat"] = inputs["delivery_lat"] - inputs["pickup_lat"]
        inputs["delta_lon"] = inputs["delivery_lon"] - inputs["pickup_lon"]
        dates = pd.DatetimeIndex(loads["date"])
        inputs["month"] = dates.month
        inputs["day_of_week"] = dates.dayofweek
        inputs["day_of_year"] = dates.dayofyear
        angle = 2 * np.pi * (dates.month - 1) / 12
        inputs["month_sin"] = np.sin(angle)
        inputs["month_cos"] = np.cos(angle)
        return inputs

    def fit(self, loads: pd.DataFrame, log_rate: np.ndarray) -> CatBoostRawDollars:
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=self.iterations,
            depth=8,
            learning_rate=0.04,
            l2_leaf_reg=5,
            random_seed=42,
            thread_count=self.threads,
            verbose=False,
            allow_writing_files=False,
        )
        self.model.fit(self._inputs(loads), np.exp(log_rate), cat_features=self.CATEGORICAL)
        return self

    def predict_log(self, loads: pd.DataFrame) -> np.ndarray:
        return np.log(np.maximum(1.0, self.model.predict(self._inputs(loads))))


LIGHTGBM_DIRECT_DEFAULTS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": 2,
    "objective": "regression",
}


class LightGBMDirect:
    """One LightGBM on log(rate) with the label screen, market, calendar and drift features. It shows what trees
    get wrong here: they cannot extend a trend or a market level beyond the range seen in training."""

    def __init__(self, tree_params: dict | None = None) -> None:
        self.tree_params = {**LIGHTGBM_DIRECT_DEFAULTS, **(tree_params or {})}

    def fit(self, loads: pd.DataFrame, log_rate: np.ndarray) -> LightGBMDirect:
        load_design = LoadDescription().fit(loads).transform(loads).to_numpy(dtype=float)
        kept = plausible_label_mask(load_design, log_rate)
        self.model = lgb.LGBMRegressor(**self.tree_params).fit(direct_tree_inputs(loads)[kept], log_rate[kept])
        return self

    def predict_log(self, loads: pd.DataFrame) -> np.ndarray:
        return self.model.predict(direct_tree_inputs(loads))
