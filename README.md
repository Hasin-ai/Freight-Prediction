# Freight rate prediction

Predicts the posted rate of 12,000 truckload loads (Nov–Dec 2025) from 48,000 labelled loads (Jan–Oct 2025), and
fills the December chart for one lane. Everything runs in Docker and every run is tracked in MLflow.

**Model.** Ridge regression on log(rate) — splines of distance, weight and location, equipment, the daily market
index, a slow drift and a quarter-end ramp — plus a small LightGBM on the ridge residuals. Recent loads weigh more.
Registered in MLflow as `freight-rate-model@champion`; weights and parameters are also exported to `model/`.

**Report.** [`report/freight_report.pdf`](report/freight_report.pdf) explains the data problems, the validation
design, every feature, the improvement ladder, tuning, the locked test and the December forecast.

## Run it (3 commands)

Needs Docker (Compose v2), `make`, and the four CSVs in `data/` (see `data/README.md`).

```bash
cp .env.example .env     # set a database password
make all                 # ~15 min: experiments, tuning, locked test, final model, outputs, report, tests
open http://localhost:5050   # MLflow UI (Windows: start http://localhost:5050)
```

Single steps: `make up | data-check | experiments | tune | test-lock | train | predict | report | test | lint`.
`make restore` loads the shipped `mlflow_backup/` instead (the exact runs whose IDs the report cites);
`make backup` saves your own.
Each run checks its result against the value logged by the earlier project and stops if they differ by more than
$0.05. The locked test refuses to run twice (`make test-lock ARGS=--force` overrides; forced runs are tagged).

## Results

<!-- results:start -->
| Model | Dev wMAE ($) | Locked-test wMAE ($) | Locked-test MAPE |
|---|---:|---:|---:|
| Step 6: final model (tuned, recency-weighted) | 42.30 | 38.14 | 1.56% |
| Step 5: ridge + LightGBM on residuals (default settings) | 42.67 | 37.88 | 1.55% |
| Step 4: structural ridge (splines, coordinates, market, trend, quarter-end) | 44.01 | 39.40 | 1.60% |
| Step 3: LightGBM, log target + label screen + market + calendar | 63.77 | 48.94 | 2.08% |
| Step 2: CatBoost on raw dollars (original submission) | 111.25 | 64.31 | 3.74% |
| Step 1: equipment rate per mile x distance | 157.55 | 175.52 | 8.13% |
| Baseline: lane rate per mile x distance | 144.84 | 90.66 | 3.80% |
| Baseline: load description + naive daily price level | 74.61 | 73.98 | 3.07% |

wMAE = clean-row MAE re-weighted to the scoring mix of new and known cities. December 31 chart value: $877.62. Champion run `f51bb0ec64524e3eb1d170f3ee1765a4`; locked-test run `95fb9814916747019528b05e5b28c031`.
<!-- results:end -->

## Layout

```
configs/      data.yaml (files, folds, seeds), features.yaml, final.yaml (frozen champion),
              experiments/ (baselines + ladder steps 1-6), ablations/
src/freight/  data, cleaning, features, label_screen, splits, cross_validation, metrics, plots, tracking, pyfunc,
              models/ (baselines, ridge_model, hybrid)
scripts/      check_data, run_experiment, tune, evaluate_locked_test, train_final, predict, make_report_assets
model/        final_params.json, ridge_coefficients.json, lightgbm_residual.txt, model.pkl, pyfunc/ (MLmodel)
outputs/      validation_predictions.csv, december_chart_inputs.csv, validation_predictions_with_intervals.csv,
              december_sensitivity.csv, scorer_results/candidate_december.png
report/       main.tex, sections/, generated tables/ and figures/, freight_report.pdf
scorer/       the assessment's score.py, unchanged
requirements.txt (direct pins) · requirements.lock (full freeze of the trainer image)
tests/        pytest: schema, label-free cleaning, folds, leakage, pyfunc round trip, outputs
```

## Using the model

```python
import mlflow, pandas as pd
from freight.pyfunc import as_model_input            # PYTHONPATH=src
model = mlflow.pyfunc.load_model("model/pyfunc")      # or "models:/freight-rate-model@champion"
loads = pd.read_csv("data/validation.csv")
predictions = model.predict(as_model_input(loads))    # predicted_rate, lo80, hi80, lo95, hi95
```

Score whole days at a time (the day's market level is the mean over that day's loads), or pass a
`market_index_day` column, as `scripts/predict.py` does for the December chart rows.

## Notes

* `quote_signal`, `load_id` and city names are never model inputs (a test checks it).
* Behind a TLS-intercepting proxy, set `EXTRA_CA_CERT` in `.env` before `make up`.
* On Windows, run `make` from Git Bash or WSL.
