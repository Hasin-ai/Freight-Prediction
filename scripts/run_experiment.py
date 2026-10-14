"""Run one or more experiment configs on the five dev folds and log each as an MLflow run.

    python scripts/run_experiment.py --config configs/experiments/04_structural_ridge.yaml [--config ...]

Each result is compared with the value the earlier project logged; a difference above $0.05 stops the run.
"""

from __future__ import annotations

import argparse

from freight.config import load_experiment
from freight.cross_validation import PRIMARY_METRIC, check_against_reference, run_folds
from freight.tracking import log_cross_validation, start_run


def run(config_path: str) -> None:
    config = load_experiment(config_path)
    tags = {
        "config_file": config_path,
        "step": config.get("step", ""),
        "hypothesis": config.get("hypothesis", ""),
    }
    with start_run(config["experiment"], config["name"], tags):
        predictions, fold_metrics, aggregate = run_folds(config["model"], "dev")
        log_cross_validation(config, config_path, "dev", predictions, fold_metrics, aggregate)
    reference = (config.get("reference") or {}).get("dev_wMAE_clean")
    print(
        check_against_reference(config["name"], aggregate[f"dev.mean.{PRIMARY_METRIC}"], reference),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", required=True)
    for config_path in parser.parse_args().config:
        run(config_path)


if __name__ == "__main__":
    main()
