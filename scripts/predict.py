"""Write the submission files with the registered champion (or the exported model/pyfunc).

    python scripts/predict.py [--model-uri models:/freight-rate-model@champion | --model-dir model/pyfunc]

Outputs: validation_predictions.csv (template order), validation_predictions_with_intervals.csv,
december_chart_inputs.csv (filled copy; the input file is never touched), december_sensitivity.csv and the
chart from the assessment's own score.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import mlflow
import numpy as np
import pandas as pd

from freight import data
from freight.cleaning import clean_loads
from freight.config import OUTPUT_DIR, PROJECT_ROOT, load_yaml
from freight.cross_validation import check_against_reference, labelled_loads
from freight.models import build_model
from freight.pyfunc import as_model_input


def december_rows(validation: pd.DataFrame) -> pd.DataFrame:
    """The chart rows lack coordinates and market index: look up the cities and use validation's daily mean."""
    december = data.load("december").drop(columns="predicted_rate")
    coordinates = data.city_coordinates(data.load("train"), validation)
    for end in ("pickup", "delivery"):
        december[f"{end}_lat"] = coordinates.loc[december[end], "lat"].values
        december[f"{end}_lon"] = coordinates.loc[december[end], "lon"].values
    december["market_index"] = np.nan
    december["market_index_day"] = validation.groupby("date")["market_index"].mean().reindex(december["date"]).values
    return december


def sensitivity(december: pd.DataFrame, final: dict) -> pd.DataFrame:
    """Two what-ifs for the December chart: no quarter-end ramp, and the market held at its October mean."""
    loads = labelled_loads()
    log_rate = np.log(loads["posted_rate"].values)
    params = dict(final["model"]["params"])
    params["groups"] = [g for g in params["groups"] if g != "quarter_end"]
    without_ramp = build_model({"type": final["model"]["type"], "params": params}).fit(loads, log_rate)
    market_by_day = december.groupby("date")["market_index_day"].first()
    champion = build_model(final["model"]).fit(loads, log_rate)
    october = loads[loads["date"].dt.month == 10].groupby("date")["market_index"].mean().mean()
    flat_market = pd.Series(october, index=market_by_day.index)
    return pd.DataFrame(
        {
            "date": december["date"].dt.strftime("%Y-%m-%d"),
            "without_quarter_end_ramp": np.exp(without_ramp.predict_log(clean_loads(december, market_by_day))),
            "market_flat_at_october_mean": np.exp(champion.predict_log(clean_loads(december, flat_market))),
        }
    )


def log_deliverables_to_champion_run(model_name: str) -> None:
    """Attach the submission files to the MLflow run that trained the champion."""
    version = mlflow.MlflowClient().get_model_version_by_alias(model_name, "champion")
    with mlflow.start_run(run_id=version.run_id):
        for name in [
            "validation_predictions.csv",
            "validation_predictions_with_intervals.csv",
            "december_chart_inputs.csv",
            "december_sensitivity.csv",
            "scorer_results/candidate_december.png",
        ]:
            mlflow.log_artifact(str(OUTPUT_DIR / name), "deliverables")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-uri", default="models:/freight-rate-model@champion")
    group.add_argument("--model-dir", default=None)
    args = parser.parse_args()
    model = mlflow.pyfunc.load_model(args.model_dir or args.model_uri)
    final = load_yaml("configs/final.yaml")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation = data.load("validation")
    predictions = model.predict(as_model_input(validation))
    template = data.load("template")
    by_id = pd.Series(predictions["predicted_rate"].round(2).values, index=validation["load_id"])
    submission = template[["load_id"]].assign(predicted_rate=template["load_id"].map(by_id).values)
    assert len(submission) == 12000 and submission["predicted_rate"].gt(0).all()
    submission.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    with_intervals = pd.concat([validation[["load_id", "date", "equipment", "distance"]], predictions.round(2)], axis=1)
    with_intervals.to_csv(OUTPUT_DIR / "validation_predictions_with_intervals.csv", index=False)

    december = december_rows(validation)
    chart = model.predict(as_model_input(december))
    filled = data.load("december")
    filled["date"] = filled["date"].dt.strftime("%Y-%m-%d")
    filled["predicted_rate"] = chart["predicted_rate"].round(2).values
    filled.to_csv(OUTPUT_DIR / "december_chart_inputs.csv", index=False)
    what_if = sensitivity(december, final)
    what_if.insert(1, "predicted_rate", chart["predicted_rate"].values)
    for level in ("80", "95"):
        what_if[f"lo{level}"], what_if[f"hi{level}"] = chart[f"lo{level}"].values, chart[f"hi{level}"].values
    what_if["market_index_day"] = december["market_index_day"].values
    what_if.round(2).to_csv(OUTPUT_DIR / "december_sensitivity.csv", index=False)

    scorer = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scorer" / "score.py"),
            "--predictions",
            str(OUTPUT_DIR / "validation_predictions.csv"),
            "--december-predictions",
            str(OUTPUT_DIR / "december_chart_inputs.csv"),
            "--output-dir",
            str(OUTPUT_DIR / "scorer_results"),
        ],
        capture_output=True,
        text=True,
    )
    print(scorer.stdout, scorer.stderr)
    if scorer.returncode != 0:
        raise SystemExit("score.py rejected the outputs")
    if not args.model_dir:
        log_deliverables_to_champion_run(final["registered_model_name"])
    reference = final["reference"]
    print(
        check_against_reference(
            "December 31 rate", float(filled["predicted_rate"].iloc[-1]), reference["december_31_rate"]
        )
    )
    print(
        check_against_reference(
            "December 31 without ramp",
            float(what_if["without_quarter_end_ramp"].iloc[-1]),
            reference["december_31_without_quarter_end_ramp"],
        )
    )


if __name__ == "__main__":
    main()
