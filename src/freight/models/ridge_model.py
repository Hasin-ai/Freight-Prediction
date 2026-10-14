"""Stage 1: ridge regression on log(rate) with interpretable, extrapolating terms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..cleaning import recency_weights
from ..features import LoadDescription, as_row_major, ridge_design, standardise
from ..label_screen import plausible_label_mask

DEFAULT_GROUPS = ("load", "market", "trend", "quarter_end", "distance_x_market")


class StructuralRidge:
    """log(rate) = splines(distance, weight, coordinates) + equipment + market elasticities + linear drift
    + quarter-end ramp x equipment + distance x market.

    Ridge keeps these terms linear in time and market, so it extrapolates to dates after the training window,
    which tree models cannot do.
    """

    def __init__(
        self,
        groups: Sequence[str] = DEFAULT_GROUPS,
        alpha: float = 0.1,
        use_label_screen: bool = True,
        recency_tau_days: float | None = None,
        spline_knots: dict | None = None,
    ) -> None:
        self.groups = tuple(groups)
        self.alpha = alpha
        self.use_label_screen = use_label_screen
        self.recency_tau_days = recency_tau_days
        self.spline_knots = dict(spline_knots or {})

    def fit(self, loads: pd.DataFrame, log_rate: np.ndarray) -> StructuralRidge:
        self.description = LoadDescription(**self.spline_knots).fit(loads)
        design = ridge_design(loads, self.description, self.groups)
        self.feature_names = list(design.columns)
        self.means, self.scales = standardise(design)
        standardised = (as_row_major(design) - self.means) / self.scales

        if self.use_label_screen:
            load_only = as_row_major(self.description.transform(loads))
            self.kept_rows = plausible_label_mask(load_only, log_rate)
        else:
            self.kept_rows = np.ones(len(log_rate), dtype=bool)
        weights = recency_weights(loads["days_since_start"].values, self.recency_tau_days)
        self.ridge = Ridge(alpha=self.alpha).fit(
            standardised[self.kept_rows],
            log_rate[self.kept_rows],
            sample_weight=None if weights is None else weights[self.kept_rows],
        )
        return self

    def predict_log(self, loads: pd.DataFrame) -> np.ndarray:
        design = as_row_major(ridge_design(loads, self.description, self.groups))
        return self.ridge.predict((design - self.means) / self.scales)

    def coefficients(self) -> dict[str, float]:
        """Coefficients on the original feature scale (log-rate change per unit of each feature)."""
        raw = self.ridge.coef_ / self.scales
        named = {name: float(value) for name, value in zip(self.feature_names, raw, strict=True)}
        named["intercept"] = float(self.ridge.intercept_ - np.sum(raw * self.means))
        return named
