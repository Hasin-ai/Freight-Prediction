"""MLflow pyfunc wrapper: raw load rows in, predicted rate and 80% / 95% intervals out.

The wrapper applies exactly the training-time cleaning. The day's market level is the mean market_index of the
batch's loads on that date, so score whole days at a time or pass a market_index_day column (the December chart
rows have no market_index and use the validation file's daily mean).
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

INPUT_COLUMNS = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
]
NUMERIC_INPUTS = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "market_index_day",
]

SIGNATURE = ModelSignature(
    inputs=Schema(
        [ColSpec("string", "pickup"), ColSpec("string", "delivery")]
        + [ColSpec("double", c) for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance"]]
        + [
            ColSpec("string", "equipment"),
            ColSpec("double", "weight", required=False),
            ColSpec("string", "date"),
            ColSpec("double", "market_index", required=False),
            ColSpec("double", "market_index_day", required=False),
        ]
    ),
    outputs=Schema([ColSpec("double", c) for c in ["predicted_rate", "lo80", "hi80", "lo95", "hi95"]]),
)


def as_model_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the model's input columns with the dtypes the signature expects."""
    columns = INPUT_COLUMNS + (["market_index_day"] if "market_index_day" in frame else [])
    model_input = frame[columns].copy()
    for column in NUMERIC_INPUTS:
        if column in model_input:
            model_input[column] = model_input[column].astype(float)
    model_input["date"] = pd.to_datetime(model_input["date"]).dt.strftime("%Y-%m-%d")
    return model_input


class FreightRateModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model: Any = None, interval_quantiles: dict | None = None) -> None:
        self.model = model
        self.interval_quantiles = interval_quantiles or {}

    def load_context(self, context) -> None:
        import cloudpickle

        with open(context.artifacts["model"], "rb") as handle:
            bundle = cloudpickle.load(handle)
        self.model, self.interval_quantiles = bundle["model"], bundle["interval_quantiles"]

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from freight.cleaning import clean_loads

        loads = model_input.copy()
        loads["date"] = pd.to_datetime(loads["date"])
        if "market_index_day" in loads and loads["market_index_day"].notna().all():
            market_by_day = loads.groupby("date")["market_index_day"].first()
        else:
            market_by_day = loads.groupby("date")["market_index"].mean()
        cleaned = clean_loads(loads.reset_index(drop=True), market_by_day=market_by_day)
        log_rate = self.model.predict_log(cleaned)
        output = pd.DataFrame({"predicted_rate": np.exp(log_rate)})
        for level in ("80", "95"):
            low = cleaned["equipment"].map(lambda e, lv=level: self.interval_quantiles[e][lv][0]).values
            high = cleaned["equipment"].map(lambda e, lv=level: self.interval_quantiles[e][lv][1]).values
            output[f"lo{level}"] = np.exp(log_rate + low)
            output[f"hi{level}"] = np.exp(log_rate + high)
        return output
