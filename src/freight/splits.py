"""Rolling-origin folds with simulated unseen cities.

The scoring task: on 2025-10-31, with every label so far, price loads 1-61 days ahead while knowing each target
day's market index; 12% of those loads touch cities never seen in training. Every fold reproduces this: train on
loads up to the origin, assess the next 61 days, and hide 8 random cities from training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import data_config


@dataclass
class Fold:
    name: str
    role: str  # "dev" (model selection and tuning) or "test" (locked, evaluated once)
    origin: str
    hidden_cities: list[str]
    train_rows: np.ndarray = field(repr=False)
    assess_rows: np.ndarray = field(repr=False)
    touches_hidden_city: np.ndarray = field(repr=False)

    def manifest(self, loads: pd.DataFrame) -> dict:
        train, assess = loads.iloc[self.train_rows], loads.iloc[self.assess_rows]
        return {
            "name": self.name,
            "role": self.role,
            "origin": self.origin,
            "train_dates": [str(train["date"].min().date()), str(train["date"].max().date())],
            "assess_dates": [str(assess["date"].min().date()), str(assess["date"].max().date())],
            "hidden_cities": self.hidden_cities,
            "train_rows": int(len(self.train_rows)),
            "assess_rows": int(len(self.assess_rows)),
            "assess_rows_touching_hidden_city": int(self.touches_hidden_city.sum()),
        }


def make_folds(loads: pd.DataFrame, role: str = "dev") -> list[Fold]:
    """role: "dev", "test" or "all". Hidden cities use seed 100 + the origin's position in the full list."""
    settings = data_config()["splits"]
    origins = settings["dev_origins"] + [settings["test_origin"]]
    cities = sorted(set(loads["pickup"]) | set(loads["delivery"]))
    folds = []
    for position, origin in enumerate(origins):
        fold_role = "test" if origin == settings["test_origin"] else "dev"
        if role != "all" and fold_role != role:
            continue
        rng = np.random.default_rng(settings["hidden_city_seed_base"] + position)
        hidden = sorted(rng.choice(cities, settings["hidden_cities_per_fold"], replace=False).tolist())
        origin_day = pd.Timestamp(origin)
        horizon_end = origin_day + pd.Timedelta(days=settings["horizon_days"])
        touches_hidden = loads["pickup"].isin(hidden) | loads["delivery"].isin(hidden)
        in_train = (loads["date"] <= origin_day) & ~touches_hidden
        in_assess = (loads["date"] > origin_day) & (loads["date"] <= horizon_end)
        fold = Fold(
            name=f"F{position + 1}_{origin}",
            role=fold_role,
            origin=origin,
            hidden_cities=hidden,
            train_rows=np.flatnonzero(in_train.values),
            assess_rows=np.flatnonzero(in_assess.values),
            touches_hidden_city=touches_hidden.values[in_assess.values],
        )
        assert_no_leakage(loads, fold)
        folds.append(fold)
    return folds


def assert_no_leakage(loads: pd.DataFrame, fold: Fold) -> None:
    train, assess = loads.iloc[fold.train_rows], loads.iloc[fold.assess_rows]
    assert train["date"].max() <= pd.Timestamp(fold.origin) < assess["date"].min(), "time overlap"
    assert not (set(train["pickup"]) | set(train["delivery"])) & set(fold.hidden_cities), "hidden city leaked"
    assert not set(fold.train_rows) & set(fold.assess_rows), "row in both train and assess"
