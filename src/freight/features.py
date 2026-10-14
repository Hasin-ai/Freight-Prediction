"""Feature builders. Learned pieces (spline knots) are fitted on training rows only."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

from .config import EQUIPMENT_TYPES, feature_config

COORDINATES = {
    "pickup_latitude": "pickup_lat",
    "pickup_longitude": "pickup_lon",
    "delivery_latitude": "delivery_lat",
    "delivery_longitude": "delivery_lon",
}
RIDGE_GROUPS = ("load", "market", "trend", "day_of_week", "quarter_end", "distance_x_market")


class LoadDescription:
    """Time-free description of a load: cubic B-splines of log distance, weight and the four coordinates,
    the weight-cap flag and equipment dummies (Dry Van is the reference).

    Splines let price per mile taper smoothly with distance and let geography act smoothly, which carries
    location information to cities that never appear in training.
    """

    def __init__(
        self,
        distance_knots: int = 8,
        weight_knots: int = 5,
        coordinate_knots: int = 6,
        coordinate_extrapolation: str = "linear",
    ) -> None:
        self.distance_knots = distance_knots
        self.weight_knots = weight_knots
        self.coordinate_knots = coordinate_knots
        self.coordinate_extrapolation = coordinate_extrapolation

    def fit(self, loads: pd.DataFrame) -> LoadDescription:
        def spline(knots: int, extrapolation: str, column: str) -> SplineTransformer:
            transformer = SplineTransformer(n_knots=knots, degree=3, extrapolation=extrapolation)
            return transformer.fit(loads[[column]])

        self.splines = {
            "log_distance": spline(self.distance_knots, "linear", "log_distance"),
            "weight_lbs": spline(self.weight_knots, "constant", "weight_lbs"),
        }
        for name, column in COORDINATES.items():
            self.splines[name] = spline(self.coordinate_knots, self.coordinate_extrapolation, column)
        return self

    def transform(self, loads: pd.DataFrame) -> pd.DataFrame:
        blocks = []
        for name, transformer in self.splines.items():
            column = COORDINATES.get(name, name)
            basis = transformer.transform(loads[[column]])
            blocks.append(pd.DataFrame(basis, columns=[f"{name}_basis_{i}" for i in range(basis.shape[1])]))
        flags = pd.DataFrame({"weight_at_cap": loads["weight_at_cap"].values})
        for equipment in EQUIPMENT_TYPES[1:]:
            flags[f"equipment_{equipment.lower().replace(' ', '_')}"] = (loads["equipment"].values == equipment).astype(
                float
            )
        return pd.concat(blocks + [flags], axis=1)


def ridge_design(loads: pd.DataFrame, description: LoadDescription, groups: Sequence[str]) -> pd.DataFrame:
    """Linear design on the log-rate scale, built from named groups so each idea can be switched off alone."""
    unknown = set(groups) - set(RIDGE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups {sorted(unknown)}")
    parts = []
    if "load" in groups:
        parts.append(description.transform(loads))
    if "market" in groups:
        parts.append(loads[["market_level_today", "market_deviation_this_load"]].reset_index(drop=True))
    if "trend" in groups:
        parts.append(loads[["days_since_start"]].reset_index(drop=True))
    if "day_of_week" in groups:
        weekday = loads["day_of_week"].values
        parts.append(pd.DataFrame({f"weekday_{d}": (weekday == d).astype(float) for d in range(1, 7)}))
    if "quarter_end" in groups:
        progress = loads["quarter_end_progress"].values
        equipment = loads["equipment"].values
        parts.append(
            pd.DataFrame(
                {
                    f"quarter_end_progress_x_{e.lower().replace(' ', '_')}": progress * (equipment == e)
                    for e in EQUIPMENT_TYPES
                }
            )
        )
    if "distance_x_market" in groups:
        centre = feature_config()["constants"]["log_distance_centre"]
        interaction = (loads["log_distance"].values - centre) * loads["market_level_today"].values
        parts.append(pd.DataFrame({"distance_x_market": interaction}))
    return pd.concat(parts, axis=1)


def residual_tree_inputs(loads: pd.DataFrame, include_market: bool = True) -> pd.DataFrame:
    """Inputs of the LightGBM residual stage. Column order matters: LightGBM samples columns by position."""
    inputs = pd.DataFrame(
        {
            "distance_miles": loads["distance"].values,
            "weight_lbs_raw": loads["weight_lbs_raw"].values,  # NaN kept: LightGBM handles missing values
            "pickup_latitude": loads["pickup_lat"].values,
            "pickup_longitude": loads["pickup_lon"].values,
            "delivery_latitude": loads["delivery_lat"].values,
            "delivery_longitude": loads["delivery_lon"].values,
            "equipment_code": pd.Categorical(loads["equipment"], categories=EQUIPMENT_TYPES).codes,
        }
    )
    if include_market:
        inputs["market_level_today"] = loads["market_level_today"].values
        inputs["market_deviation_this_load"] = loads["market_deviation_this_load"].values
    return inputs


def direct_tree_inputs(loads: pd.DataFrame) -> pd.DataFrame:
    """Inputs for a single LightGBM that must learn everything itself, calendar and trend included."""
    inputs = residual_tree_inputs(loads, include_market=True)
    inputs["day_of_week"] = loads["day_of_week"].values
    inputs["quarter_end_progress"] = loads["quarter_end_progress"].values
    inputs["day_of_month"] = loads["day_of_month"].values
    inputs["days_since_start"] = loads["days_since_start"].values
    return inputs


def standardise(design: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Column means and standard deviations; the epsilon keeps all-zero columns (e.g. no quarter end yet) finite."""
    values = design.to_numpy(dtype=float)
    return values.mean(axis=0), values.std(axis=0) + 1e-9
