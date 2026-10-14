"""Optuna search for the ridge + residual-trees model on the dev folds only (the locked test is never touched).

    python scripts/tune.py --trials 60 --nested-trials 40

1. 60 TPE trials (seed 42) minimising mean dev wMAE_clean; every trial is a nested MLflow run.
2. A forward-nested check of tuning optimism: tune on dev folds 1..k-1 only, then score fold k (k = 4, 5).
3. Confirms that the best trial equals the frozen configs/final.yaml.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import mlflow
import optuna
import pandas as pd

from freight.config import OUTPUT_DIR, load_yaml
from freight.cross_validation import PRIMARY_METRIC, check_against_reference, run_folds
from freight.tracking import log_table, start_run

EXPERIMENT = "04-tuning"
FIXED_GROUPS = ["load", "market", "trend", "quarter_end", "distance_x_market"]
REFERENCE_BEST = 42.30  # best value of the earlier search (run c47fd1cc6dbf)
REFERENCE_NESTED_GAINS = {4: 0.50, 5: 0.48}  # earlier nested check (run 0101abb344f6)


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """The suggestion order is part of the seeded search; changing it changes which trials are drawn."""
    use_recency = trial.suggest_categorical("recency", ["none", "tau"])
    params: dict[str, Any] = {
        "groups": FIXED_GROUPS,
        "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
        "spline_knots": {
            "distance_knots": trial.suggest_int("dist_knots", 5, 12),
            "coordinate_knots": trial.suggest_int("coord_knots", 4, 9),
            "weight_knots": trial.suggest_int("weight_knots", 4, 7),
        },
        "residual_tree_params": {
            "n_estimators": trial.suggest_int("n_estimators", 50, 600, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        },
    }
    if use_recency == "tau":
        params["recency_tau_days"] = trial.suggest_float("recency_tau", 45, 720, log=True)
    return params


def spec(params: dict[str, Any]) -> dict[str, Any]:
    return {"type": "ridge_plus_residual_trees", "params": params}


def run_search(n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
            _, fold_metrics, aggregate = run_folds(spec(params), "dev")
            mlflow.log_params({"trial": trial.number, "params_json": json.dumps(params)})
            mlflow.log_metrics({k: v for k, v in aggregate.items() if k.startswith("dev.mean.")})
        trial.set_user_attr("params", json.dumps(params))
        print(f"trial {trial.number:3d}: {aggregate[f'dev.mean.{PRIMARY_METRIC}']:.3f}", flush=True)
        return aggregate[f"dev.mean.{PRIMARY_METRIC}"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study


def matches_final(best: dict[str, Any]) -> bool:
    frozen = load_yaml("configs/final.yaml")["model"]["params"]
    return json.dumps(best, sort_keys=True) == json.dumps(frozen, sort_keys=True)


def nested_check(n_trials: int) -> pd.DataFrame:
    """Tune on folds 1..k-1, score the tuned and the default settings on fold k."""
    rows = []
    for outer in (3, 4):  # zero-based positions of dev folds 4 and 5
        inner = list(range(outer))

        def objective(trial: optuna.Trial, inner: list[int] = inner) -> float:
            return run_folds(spec(search_space(trial)), "dev", inner)[2][f"dev.mean.{PRIMARY_METRIC}"]

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best = search_space(optuna.trial.FixedTrial(study.best_params))
        tuned = run_folds(spec(best), "dev", [outer])[2][f"dev.mean.{PRIMARY_METRIC}"]
        default = run_folds(spec({}), "dev", [outer])[2][f"dev.mean.{PRIMARY_METRIC}"]
        rows.append(
            {
                "outer_fold": outer + 1,
                "inner_folds": len(inner),
                "inner_best": study.best_value,
                "outer_tuned": tuned,
                "outer_default": default,
                "gain": default - tuned,
            }
        )
        print(
            check_against_reference(
                f"nested gain on fold {outer + 1}", default - tuned, REFERENCE_NESTED_GAINS[outer + 1]
            ),
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--nested-trials", type=int, default=40)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output = OUTPUT_DIR / "tuning"
    output.mkdir(parents=True, exist_ok=True)

    with start_run(EXPERIMENT, "optuna-search", {"sampler": "TPE seed 42", "folds": "5 dev folds"}):
        mlflow.log_params({"n_trials": args.trials, "objective": f"mean dev {PRIMARY_METRIC}"})
        study = run_search(args.trials)
        best = json.loads(study.best_trial.user_attrs["params"])
        trials = study.trials_dataframe()
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=42)  # unseeded fANOVA varies run to run
        importance = {
            k: float(v) for k, v in optuna.importance.get_param_importances(study, evaluator=evaluator).items()
        }
        mlflow.log_metric(f"best.dev.mean.{PRIMARY_METRIC}", study.best_value)
        mlflow.log_dict(best, "tuning/best_params.json")
        mlflow.log_dict(importance, "tuning/param_importance.json")
        log_table(trials, "trials.csv", "tuning")
        mlflow.set_tag("best_matches_final_yaml", str(matches_final(best)))
    trials.to_csv(output / "trials.csv", index=False)
    (output / "best_params.json").write_text(json.dumps(best, indent=1))
    print(check_against_reference("best tuning value", study.best_value, REFERENCE_BEST))
    print("best trial matches configs/final.yaml:", matches_final(best))

    if args.nested_trials:
        with start_run(EXPERIMENT, "nested-forward-check", {"trials_per_outer_fold": args.nested_trials}):
            nested = nested_check(args.nested_trials)
            log_table(nested, "nested_forward_check.csv", "tuning")
            mlflow.log_metric("nested.mean_gain", float(nested["gain"].mean()))
        nested.to_csv(output / "nested_forward_check.csv", index=False)


if __name__ == "__main__":
    main()
