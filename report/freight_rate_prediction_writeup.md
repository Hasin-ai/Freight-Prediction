# Freight rate prediction: a step-by-step project write-up

## 1. Exploratory data analysis

I started with 48,000 labelled loads from January–October 2025 and 12,000 unlabelled loads from November–December. Distance explained most of the rate, but the data also had several traps: 677 labels (1.41%) sat in separate bands at roughly 0.2–0.5× or 2–5× the lane-and-equipment norm; 292 training weights and 145 validation weights were negative; and weights piled up at 47,500 lb. Weight was missing in 0.63% / 1.4% of train / validation rows, and `market_index` in 0.78% / 2.1%.

`market_index` turned out to be a real daily index, not just a date proxy. Its weekly swing ran from −9% on Sunday to +9% on Thursday, while composition-adjusted prices moved in the same direction by about −1.1% to +1.0%. By contrast, `quote_signal` copied or mirrored the target in different training months, then behaved like noise in validation: its daily-mean standard deviation was 0.016 and its lag-1 autocorrelation was 0.03. I left it out of the final model.

I also found a rate ramp in March, June and September that grew toward quarter-end, plus an upward drift beyond the market index. Eight cities were new in validation, touching 12.06% of its rows; Laredo was south of every training city. Those findings shaped the time split and the location features.

## 2. Preprocessing

I corrected negative weights with their absolute values (292 training rows; 145 validation rows) and marked weights at the 47,500 lb cap (1,204 training rows; 301 validation rows). I filled the 300 missing training weights and 165 missing validation weights with 31,000 lb for ridge, while LightGBM kept them as missing. For the 374 and 249 missing `market_index` values, I used that day's mean across the file. The 48 training and 21 validation distances at the 70-mile floor were genuine short hauls, so I left them alone.

For labels, I used a two-pass screen based on a ridge fit and removed training rows outside 0.67–1.5× of the fitted lane-and-equipment rate. It flags about 1.4% of each training fold. The screen is fit inside each fold and never applied to assessment rows; those labels remain in the all-row metrics. Removing the screen raises dev `wMAE_clean` by $14.18.

## 3. Feature engineering

I kept B-splines for log distance, weight and all four coordinates, plus the cap flag and equipment indicators. Replacing the splines with straight lines cost $10.4. Coordinates carried across to unseen cities: novel-city dev MAE was $40.9 versus $42.5 for seen cities. City categories did worse on novel cities in the feature study (MAE 186 versus 124), so I used coordinates instead of city IDs.

I split `market_index` into its daily mean and per-load deviation. Removing the market features cost $34.11; removing the time trend cost $13.34; removing the quarter-end ramp by equipment cost $28.99. By month-end, the ramp added 2.3% for Dry Van, 5.0% for Reefer and 7.8% for Flatbed. The distance-by-market interaction added a smaller $0.28 gain (95% CI 0.19–0.33, 5/5 folds). Ridge alone was $1.30 worse than the hybrid, and removing market features from the residual trees cost $0.86, so I kept both stages.

I dropped weekday dummies because adding them cost $3.00 once the weekly market swing was present. I also dropped per-load `quote_signal` (the feature study found removal helped in 2/3 folds, with clean MAE 114 versus 120), `load_id`, city names, and month or day-of-year features in the trees. The last group had no useful support for forecasting beyond October.

## 4. Model selection and training: the improvement journey

The first feature-study lookup was simple: equipment, distance and a rate-per-mile average. On its 3-fold development run, clean weighted MAE was $119.7. Corrupted labels pulled raw-target LightGBM to $161.5. A log target alone reached $122.3; adding the in-fold label screen brought it to $72.2.

Then I gave ridge a better description of the problem. In that same study, clean weighted MAE moved from $75.7 with the basic linear features to $71.7 after splines, $59.5 after coordinate splines, $53.3 after splitting the market index, $51.4 after adding a daily quote-signal mean and trend, $49.4 after distance-by-market, and $47.5 after residual LightGBM. Those early numbers use the feature study's 3 folds; after the validation shift showed quote signal was noise, I left it out of the final model. The later five-fold comparisons are below.

The earlier CatBoost submission and PDF-report model scored $111.25 and $85.18 on the final dev folds. The log-and-screen LightGBM with market, calendar and trend reached $63.77; structural ridge reached $44.01; and the default ridge-plus-LightGBM hybrid reached $42.67. A MARS follow-up looked promising against an older, weaker ridge comparison ($40.96 versus $43.71 on its 3 folds), but on the common five folds its best version reached $46.06. It did not beat the selected hybrid.

| Final dev leaderboard | `wMAE_clean` |
|---|---:|
| CatBoost submission | $111.25 |
| PDF-report model | $85.18 |
| Log-and-screen LightGBM, market + calendar + trend | $63.77 |
| MARS + LightGBM follow-up (best M3) | $46.06 |
| Structural ridge | $44.01 |
| Default hybrid | $42.67 |
| Tuned hybrid | **$42.30** |

## 5. Hyperparameter tuning

I used Optuna's TPE sampler for 60 trials on the five dev folds, minimizing mean `wMAE_clean`. I searched ridge strength and spline knot counts, residual-tree count, learning rate, leaves, minimum child size, and whether to use recency weighting. The best result improved from $42.67 to $42.30. In the forward-nested check, the gain was $0.50 on fold F4 and $0.48 on F5. That is a small difference: the tuned and default models are practically equivalent.

## 6. Final evaluation

I used five expanding-window dev folds with 61-day horizons and hid 8 cities in each fold. The locked test trained through August 31 and assessed September 1–October 31, covering 9,523 loads.

| Locked Sep–Oct test | `wMAE_clean` | `MAE_clean` | `MAPE_clean` | `MAE_all` |
|---|---:|---:|---:|---:|
| Tuned hybrid | **$38.14** | $38.27 | 1.56% | $91.86 |
| Default hybrid | $37.88 | $37.85 | 1.55% | $91.45 |
| Structural ridge | $39.40 | $39.31 | 1.60% | $92.88 |
| PDF-report model | $49.69 | $49.71 | 2.08% | $103.13 |
| CatBoost submission | $64.31 | $65.62 | 3.74% | $118.79 |
| Cross-section + naive index | $73.98 | $73.72 | 3.07% | $126.82 |

On paired test comparisons, the CatBoost model was $27.35 worse (95% CI $22.60–$32.21), the PDF model $11.44 worse ($6.56–$19.34), and the naive-index baseline $35.45 worse ($28.68–$44.51). The untuned hybrid was $0.42 better than the tuned one ($0.05–$0.81), another sign that tuning made little practical difference.

Dev-derived intervals covered 84.7% of clean rows at the 80% level and 96.9% at the 95% level; coverage on all rows was 83.5% and 95.5%. The 80% interval averaged ±2.6%.

## 7. December forecast

The December chart for Lexington–Fort Wayne, 360 miles, Dry Van and 32,000 lb ranges from $829 to $878. The chart rows do not include coordinates or a market index, so I filled coordinates from the city lookup and each date's market index from that day's validation mean. I assumed the quarter-end ramp would return: by December 31 it adds 2.3% for Dry Van, giving $878 instead of $858 without the ramp. There are no December labels to confirm that assumption.

## 8. Setting up MLflow and viewing results

Clone the project, create `.env` for local secrets, then run `docker compose up`. Open the MLflow UI; its port is not stated, so check in `docker-compose.yml`.

The reports identify `freight-07-locked-test` as the Sep–Oct evaluation and `freight-08-final` as the final training run. They name `freight-00` through `freight-08` as the experiment range, but do not describe `freight-00` through `freight-06` individually; check their descriptions in MLflow. In the model registry, look for `models:/freight-rate-model@champion` (v2; v1 is superseded).

Use `scripts/predict.py` to create `outputs/validation_predictions.csv`, then use `score.py` with that prediction file and the filled `december_chart_inputs.csv`. `score.py` checks format and fixed December inputs, not accuracy. The project reports do not give the command-line syntax for either script.
