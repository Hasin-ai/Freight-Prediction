"""Pipeline guarantees: schema, label-free cleaning, fold ordering, no leakage, pyfunc round trip, outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight import data
from freight.cleaning import clean_loads
from freight.config import MODEL_DIR, OUTPUT_DIR, load_yaml
from freight.cross_validation import labelled_loads
from freight.models import build_model
from freight.pyfunc import FreightRateModel, as_model_input
from freight.splits import make_folds

FINAL_MODEL = load_yaml("configs/final.yaml")["model"]


@pytest.fixture(scope="module")
def loads() -> pd.DataFrame:
    return labelled_loads()


def test_input_files_match_schema() -> None:
    assert data.schema_problems(data.load("train"), "train") == []
    assert data.schema_problems(data.load("validation"), "validation") == []


def test_cleaning_never_reads_the_label() -> None:
    raw = data.load("train").head(3000)
    changed = raw.assign(posted_rate=raw["posted_rate"] * 7)
    columns = [
        "weight_lbs",
        "market_level_today",
        "market_deviation_this_load",
        "days_since_start",
        "quarter_end_progress",
        "log_distance",
    ]
    pd.testing.assert_frame_equal(clean_loads(raw)[columns], clean_loads(changed)[columns])


def test_negative_weights_become_their_absolute_value(loads: pd.DataFrame) -> None:
    raw = data.load("train")
    negative = raw["weight"] < 0
    assert np.allclose(loads.loc[negative, "weight_lbs_raw"], -raw.loc[negative, "weight"])


def test_folds_are_time_ordered_and_hide_cities(loads: pd.DataFrame) -> None:
    folds = make_folds(loads, "all")
    assert [f.role for f in folds] == ["dev"] * 5 + ["test"]
    for fold in folds:
        train, assess = loads.iloc[fold.train_rows], loads.iloc[fold.assess_rows]
        assert train["date"].max() <= pd.Timestamp(fold.origin) < assess["date"].min()
        assert (assess["date"] - pd.Timestamp(fold.origin)).dt.days.max() <= 61
        assert not (set(train["pickup"]) | set(train["delivery"])) & set(fold.hidden_cities)


@pytest.mark.parametrize(
    "spec",
    [FINAL_MODEL, {"type": "lightgbm_direct"}, {"type": "structural_ridge"}],
    ids=["final", "lgbm", "ridge"],
)
def test_predictions_ignore_test_labels_and_quote_signal(loads: pd.DataFrame, spec: dict) -> None:
    fold = make_folds(loads, "dev")[2]
    train = loads.iloc[fold.train_rows].reset_index(drop=True)
    assess = loads.iloc[fold.assess_rows].reset_index(drop=True)
    model = build_model(spec).fit(train, np.log(train["posted_rate"].values))
    baseline = model.predict_log(assess)
    noise = np.random.default_rng(0).normal(size=len(assess))
    tampered = assess.assign(posted_rate=assess["posted_rate"] * 3, quote_signal=noise)
    assert np.array_equal(baseline, model.predict_log(tampered))
    retrained = build_model(spec).fit(train.assign(quote_signal=0.0), np.log(train["posted_rate"].values))
    assert np.array_equal(baseline, retrained.predict_log(assess))


def test_pyfunc_round_trip(tmp_path, loads: pd.DataFrame) -> None:
    import cloudpickle
    import mlflow.pyfunc

    train = loads[loads["date"] <= "2025-06-30"].reset_index(drop=True)
    model = build_model(FINAL_MODEL).fit(train, np.log(train["posted_rate"].values))
    quantiles = {e: {"80": [-0.02, 0.02], "95": [-0.04, 0.04]} for e in ["Dry Van", "Flatbed", "Reefer"]}
    bundle = tmp_path / "model.pkl"
    bundle.write_bytes(cloudpickle.dumps({"model": model, "interval_quantiles": quantiles}))
    mlflow.pyfunc.save_model(tmp_path / "pyfunc", python_model=FreightRateModel(), artifacts={"model": str(bundle)})
    raw = data.load("train")
    july = raw[(raw["date"] >= "2025-07-01") & (raw["date"] <= "2025-07-03")].reset_index(drop=True)
    output = mlflow.pyfunc.load_model(str(tmp_path / "pyfunc")).predict(as_model_input(july))
    assert np.allclose(output["predicted_rate"], np.exp(model.predict_log(clean_loads(july))))


@pytest.mark.skipif(not (MODEL_DIR / "pyfunc" / "MLmodel").exists(), reason="run `make train predict` first")
def test_exported_model_reproduces_submission() -> None:
    import mlflow.pyfunc

    validation = data.load("validation")
    predicted = mlflow.pyfunc.load_model(str(MODEL_DIR / "pyfunc")).predict(as_model_input(validation))
    submitted = pd.read_csv(OUTPUT_DIR / "validation_predictions.csv")
    by_id = pd.Series(predicted["predicted_rate"].round(2).values, index=validation["load_id"])
    assert submitted["load_id"].tolist() == data.load("template")["load_id"].tolist()
    assert np.array_equal(submitted["predicted_rate"].values, submitted["load_id"].map(by_id).values)
