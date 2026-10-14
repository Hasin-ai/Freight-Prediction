"""Label-free cleaning and derived predictors.

Everything here uses only the load's own columns or other loads' predictors on the same date (the day's mean
market index). Nothing is learned and the label is never read, so this step can run before splitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import feature_config


def clean_loads(raw: pd.DataFrame, market_by_day: pd.Series | None = None) -> pd.DataFrame:
    """Return a copy of `raw` with cleaned, human-named predictor columns added.

    market_by_day maps date -> mean market index. When omitted it is the mean over the frame's own loads per
    date, which is right for whole-day files such as validation.csv. Rows without a market_index (the December
    chart rows) need it.
    """
    settings = feature_config()["constants"]
    loads = raw.copy()
    weight = pd.to_numeric(loads["weight"], errors="coerce")
    loads["weight_missing"] = weight.isna().astype(float)
    loads["weight_was_negative"] = (weight < 0).astype(float)  # sign-entry errors; magnitudes look normal
    loads["weight_lbs_raw"] = weight.abs()
    loads["weight_at_cap"] = (loads["weight_lbs_raw"] >= settings["weight_cap_lbs"]).astype(float)
    loads["weight_lbs"] = loads["weight_lbs_raw"].fillna(settings["weight_fill_lbs"])

    if "market_index" not in loads:
        loads["market_index"] = np.nan
    if market_by_day is None:
        market_by_day = loads.groupby("date")["market_index"].mean()
    day_mean = loads["date"].map(market_by_day)
    if day_mean.isna().any():
        raise ValueError("no market index for some dates; pass market_by_day")
    # A missing per-load reading is replaced by the day's mean (within-day SD 0.025 vs between-day SD 0.167).
    loads["market_index_filled"] = loads["market_index"].fillna(day_mean)
    loads["market_level_today"] = np.log(day_mean.values)
    loads["market_deviation_this_load"] = (
        np.log(loads["market_index_filled"].values) - loads["market_level_today"].values
    )

    loads["log_distance"] = np.log(loads["distance"])
    dates = pd.DatetimeIndex(loads["date"])
    loads["day_of_week"] = dates.dayofweek
    loads["day_of_month"] = dates.day
    loads["days_since_start"] = (dates - pd.Timestamp(settings["time_origin"])).days.values.astype(float)
    loads["quarter_end_progress"] = quarter_end_progress(dates)
    return loads


def quarter_end_progress(dates: pd.DatetimeIndex) -> np.ndarray:
    """0 on the 1st and 1 on the last day of Mar/Jun/Sep/Dec; 0 in every other month."""
    in_last_month_of_quarter = dates.month % 3 == 0
    progress = (dates.day - 1) / (dates.days_in_month - 1)
    return np.where(in_last_month_of_quarter, progress, 0.0)


def recency_weights(days_since_start: np.ndarray, tau_days: float | None) -> np.ndarray | None:
    """exp((day - newest day) / tau), normalised to mean 1. A load tau days older than the newest weighs 1/e."""
    if not tau_days:
        return None
    weights = np.exp((days_since_start - days_since_start.max()) / float(tau_days))
    return weights / weights.mean()
