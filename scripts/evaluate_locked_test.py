"""One-shot evaluation on the locked test period (origin 2025-08-31 -> Sep-Oct 2025, 8 hidden cities).

Runs every config listed under `locked_test` in configs/final.yaml. It refuses to run a second time for the
same selection_id unless --force is given; forced re-runs are tagged so they stay visible.
"""

from __future__ import annotations

import argparse
import sys

import mlflow

from freight.config import load_experiment, load_yaml
from freight.cross_validation import PRIMARY_METRIC, check_against_reference, run_folds
from freight.tracking import log_cross_validation, start_run

EXPERIMENT = "05-locked-test"


def already_evaluated(selection_id: str) -> bool:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return False
    runs = mlflow.search_runs([experiment.experiment_id], filter_string=f"tags.selection_id = '{selection_id}'")
    return len(runs) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="evaluate again (tagged forced_rerun=true)")
    args = parser.parse_args()
    final = load_yaml("configs/final.yaml")
    if already_evaluated(final["selection_id"]) and not args.force:
        print(
            f"The locked test was already evaluated for {final['selection_id']}; not running it again. "
            "Use --force to override (the re-run will be tagged)."
        )
        return 0
    for config_path in final["locked_test"]:
        config = load_experiment(config_path)
        tags = {
            "selection_id": final["selection_id"],
            "config_file": config_path,
            "forced_rerun": str(args.force),
        }
        with start_run(EXPERIMENT, config["name"], tags):
            predictions, fold_metrics, aggregate = run_folds(config["model"], "test")
            log_cross_validation(config, config_path, "test", predictions, fold_metrics, aggregate)
        reference = (config.get("reference") or {}).get("test_wMAE_clean")
        print(
            check_against_reference(config["name"], aggregate[f"test.mean.{PRIMARY_METRIC}"], reference),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
