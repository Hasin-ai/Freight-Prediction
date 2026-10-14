"""In-fold screen for corrupted labels.

About 1.4% of rates are off by a factor of 2-5 (or 0.2-0.5) and no feature predicts them. A robust two-pass ridge
fit on the time-free load description flags them. The screen is learned, so it is fitted on training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

LOWER_RATIO, UPPER_RATIO = 0.67, 1.5


def plausible_label_mask(load_design: np.ndarray, log_rate: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """True for rows whose rate is within 0.67x-1.5x of a robust fit; two passes so outliers don't bend the fit."""
    lower, upper = np.log(LOWER_RATIO), np.log(UPPER_RATIO)
    fit = Ridge(alpha=alpha).fit(load_design, log_rate)
    residual = log_rate - fit.predict(load_design)
    residual = residual - np.median(residual)
    keep = (residual > lower) & (residual < upper)

    fit = Ridge(alpha=alpha).fit(load_design[keep], log_rate[keep])
    residual = log_rate - fit.predict(load_design)
    residual = residual - np.median(residual[keep])
    return (residual > lower) & (residual < upper)


def reporting_clean_flag(train: pd.DataFrame) -> np.ndarray:
    """Fixed definition of a plausible label for *reporting only*: rate per mile within 0.67x-1.5x of its
    lane x equipment median. It is the same for every model and never used for fitting."""
    rate_per_mile = train["posted_rate"] / train["distance"]
    lane_median = rate_per_mile.groupby([train["pickup"], train["delivery"], train["equipment"]]).transform("median")
    ratio = rate_per_mile / lane_median
    return ((ratio > LOWER_RATIO) & (ratio < UPPER_RATIO)).to_numpy()
