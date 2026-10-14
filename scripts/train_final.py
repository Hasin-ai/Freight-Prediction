"""Fit the frozen champion on all labelled loads, export it to model/, log it and register it as @champion.

Interval widths come from the dev-fold out-of-sample residuals of ladder step 6, and their coverage is measured
on the locked-test predictions already logged by evaluate_locked_test.py. Neither period is re-scored here.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from freight import data
from freight.config import MODEL_DIR, PROJECT_ROOT, load_experiment, load_yaml
from freight.cross_validation import labelled_loads
from freight.models import build_model
from freight.pyfunc import SIGNATURE, FreightRateModel, as_model_input
from freight.tracking import latest_run, load_predictions, start_run

FINAL_STEP = "configs/experiments/06_final_tuned.yaml"


def interval_quantiles(dev_predictions: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Empirical 10/90% and 2.5/97.5% quantiles of log(actual / predicted) per equipment, clean dev rows."""
    clean = dev_predictions[dev_predictions["clean"]]
    log_ratio = np.log(clean["actual"] / clean["predicted"])
    quantiles = {}
    for equipment, values in log_ratio.groupby(clean["equipment"]):
        quantiles[equipment] = {
            "80": values.quantile([0.1, 0.9]).tolist(),
            "95": values.quantile([0.025, 0.975]).tolist(),
        }
    return quantiles


def coverage(predictions: pd.DataFrame, quantiles: dict, clean_only: bool) -> dict[str, float]:
    rows = predictions[predictions["clean"]] if clean_only else predictions
    log_ratio = np.log(rows["actual"] / rows["predicted"])
    result = {}
    for level in ("80", "95"):
        low = rows["equipment"].map(lambda e, lv=level: quantiles[e][lv][0])
        high = rows["equipment"].map(lambda e, lv=level: quantiles[e][lv][1])
        result[level] = float(((log_ratio >= low) & (log_ratio <= high)).mean())
    return result


def export_model_folder(model, quantiles: dict, final: dict) -> Path:
    """model/: parameters, ridge coefficients, LightGBM booster dump, pickled pipeline and a pyfunc export."""
    MODEL_DIR.mkdir(exist_ok=True)
    resolved = {
        "model_type": final["model"]["type"],
        "ridge": {
            "alpha": model.ridge.alpha,
            "groups": list(model.ridge.groups),
            "recency_tau_days": model.ridge.recency_tau_days,
            "spline_knots": {
                "distance_knots": model.ridge.description.distance_knots,
                "weight_knots": model.ridge.description.weight_knots,
                "coordinate_knots": model.ridge.description.coordinate_knots,
                "coordinate_extrapolation": model.ridge.description.coordinate_extrapolation,
            },
            "rows_used_after_label_screen": int(model.ridge.kept_rows.sum()),
            "rows_available": int(len(model.ridge.kept_rows)),
        },
        "residual_lightgbm": model.residual_tree_params,
        "interval_quantiles_log_scale": quantiles,
    }
    (MODEL_DIR / "final_params.json").write_text(json.dumps(resolved, indent=1))
    (MODEL_DIR / "ridge_coefficients.json").write_text(json.dumps(model.ridge.coefficients(), indent=1))
    model.trees.booster_.save_model(str(MODEL_DIR / "lightgbm_residual.txt"))
    bundle = MODEL_DIR / "model.pkl"
    bundle.write_bytes(cloudpickle.dumps({"model": model, "interval_quantiles": quantiles}))
    pyfunc_dir = MODEL_DIR / "pyfunc"
    if pyfunc_dir.exists():
        shutil.rmtree(pyfunc_dir)
    mlflow.pyfunc.save_model(
        pyfunc_dir,
        python_model=FreightRateModel(),
        artifacts={"model": str(bundle)},
        code_paths=[str(PROJECT_ROOT / "src" / "freight")],
        signature=SIGNATURE,
        pip_requirements=str(PROJECT_ROOT / "requirements.txt"),
    )
    return bundle


def main() -> None:
    final = load_yaml("configs/final.yaml")
    step = load_experiment(FINAL_STEP)
    dev_run = latest_run(step["experiment"], step["name"])
    test_run = latest_run("05-locked-test", step["name"])
    quantiles = interval_quantiles(load_predictions(dev_run, "dev"))
    test_predictions = load_predictions(test_run, "test")

    loads = labelled_loads()
    started = time.time()
    model = build_model(final["model"]).fit(loads, np.log(loads["posted_rate"].values))
    fit_seconds = time.time() - started
    bundle = export_model_folder(model, quantiles, final)

    validation = data.load("validation")
    model_input = as_model_input(validation)
    in_memory = FreightRateModel(model, quantiles).predict(None, model_input)

    with start_run(
        "06-final",
        "champion: ridge + residual trees (tuned)",
        {
            "selection_id": final["selection_id"],
            "dev_run_id": dev_run.info.run_id,
            "locked_test_run_id": test_run.info.run_id,
        },
    ) as run:
        mlflow.log_params(
            {
                "training_rows": len(loads),
                "training_period": "2025-01-01..2025-10-31",
                "config": json.dumps(final["model"]["params"]),
            }
        )
        clean_cov, all_cov = coverage(test_predictions, quantiles, True), coverage(test_predictions, quantiles, False)
        mlflow.log_metrics(
            {
                "fit_seconds": fit_seconds,
                "dev.mean.wMAE_clean": dev_run.data.metrics["dev.mean.wMAE_clean"],
                "test.mean.wMAE_clean": test_run.data.metrics["test.mean.wMAE_clean"],
                "locked_test.coverage80_clean": clean_cov["80"],
                "locked_test.coverage95_clean": clean_cov["95"],
                "locked_test.coverage80_all": all_cov["80"],
                "locked_test.coverage95_all": all_cov["95"],
                "rows_used_after_label_screen": int(model.ridge.kept_rows.sum()),
            }
        )
        for name in ["final_params.json", "ridge_coefficients.json", "lightgbm_residual.txt"]:
            mlflow.log_artifact(str(MODEL_DIR / name), "model_files")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=FreightRateModel(),
            artifacts={"model": str(bundle)},
            code_paths=[str(PROJECT_ROOT / "src" / "freight")],
            signature=SIGNATURE,
            input_example=model_input.head(5),
            pip_requirements=str(PROJECT_ROOT / "requirements.txt"),
            registered_model_name=final["registered_model_name"],
        )
        client = MlflowClient()
        version = info.registered_model_version
        client.set_registered_model_alias(final["registered_model_name"], "champion", version)
        client.update_model_version(
            final["registered_model_name"],
            version,
            description=f"{final['selection_id']}; training run {run.info.run_id}",
        )

        reloaded = mlflow.pyfunc.load_model(f"models:/{final['registered_model_name']}@champion")
        started = time.time()
        from_registry = reloaded.predict(model_input)
        latency = (time.time() - started) / len(model_input) * 1e6
        exported = mlflow.pyfunc.load_model(str(MODEL_DIR / "pyfunc")).predict(model_input)
        registry_diff = float(np.abs(from_registry["predicted_rate"] - in_memory["predicted_rate"]).max())
        export_diff = float(np.abs(exported["predicted_rate"] - in_memory["predicted_rate"]).max())
        mlflow.log_metrics(
            {
                "registry_reload_max_abs_diff": registry_diff,
                "export_reload_max_abs_diff": export_diff,
                "inference_microseconds_per_load": latency,
            }
        )
        assert registry_diff == 0 and export_diff == 0, "reloaded model does not reproduce the fitted model"
    record = {
        "run_id": run.info.run_id,
        "registered_model": f"models:/{final['registered_model_name']}/{version}",
        "alias": f"models:/{final['registered_model_name']}@champion",
        "dev_run_id": dev_run.info.run_id,
        "locked_test_run_id": test_run.info.run_id,
        "coverage_locked_test_clean": clean_cov,
        "coverage_locked_test_all": all_cov,
    }
    (MODEL_DIR / "registry_record.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
