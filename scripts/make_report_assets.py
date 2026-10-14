"""Build every table, figure and number used by report/ from MLflow runs and the raw data.

Writes report/tables/*.tex (booktabs tables), report/tables/numbers.tex (one LaTeX macro per number quoted in the
text) and report/figures/*.png. Nothing in the report is typed by hand.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freight import data  # noqa: E402
from freight.config import PROJECT_ROOT, load_experiment, load_yaml  # noqa: E402
from freight.cross_validation import labelled_loads  # noqa: E402
from freight.metrics import by_segment, horizon_bucket, paired_block_bootstrap  # noqa: E402
from freight.plots import BLUE, GREEN, GREY, INK, ORANGE  # noqa: E402
from freight.tracking import latest_run, load_predictions  # noqa: E402

REPORT = PROJECT_ROOT / "report"
TABLES, FIGURES = REPORT / "tables", REPORT / "figures"
LADDER = sorted((PROJECT_ROOT / "configs/experiments").glob("0*.yaml"))
ABLATIONS = sorted((PROJECT_ROOT / "configs/ablations").glob("*.yaml"))
BASELINES = sorted((PROJECT_ROOT / "configs/experiments").glob("baseline_*.yaml"))
macros: dict[str, str] = {}


def define(name: str, value: str | float, decimals: int = 2) -> None:
    """Register a LaTeX macro; numbers are formatted with thousands separators."""
    macros[name] = f"{value:,.{decimals}f}" if isinstance(value, (int, float, np.floating)) else str(value)


def escape(text: str) -> str:
    for char, replacement in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("~", r"\textasciitilde{}"),
    ]:
        text = text.replace(char, replacement)
    return text


def write_table(name: str, frame: pd.DataFrame, align: str, caption_note: str = "") -> None:
    """A booktabs tabular; the caption lives in the .tex section so the prose and the table stay together."""
    header = " & ".join(escape(str(c)) for c in frame.columns) + r" \\"
    body = "\n".join(" & ".join(escape(str(v)) for v in row) + r" \\" for row in frame.itertuples(index=False))
    note = f"\n\\par\\smallskip{{\\footnotesize {caption_note}}}" if caption_note else ""
    content = (
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}{note}\n"
    )
    (TABLES / f"{name}.tex").write_text(content)


def macro_key(path: Path) -> str:
    """LaTeX macro names cannot contain digits: '06_final_tuned' -> 'FinalTuned', 'A1_no_market' -> 'NoMarket'."""
    words = [w for w in path.stem.split("_") if not any(ch.isdigit() for ch in w)]
    return "".join(w.capitalize() for w in words)


def money(value: float) -> str:
    return f"{value:,.2f}"


def config_run(path: Path, experiment: str | None = None):
    config = load_experiment(path)
    return config, latest_run(experiment or config["experiment"], config["name"])


def short_id(run) -> str:
    return run.info.run_id[:12]


def ci_text(result: dict) -> str:
    return f"{result['difference']:+.2f} [{result['ci_low']:+.2f}, {result['ci_high']:+.2f}]"


# ---------------------------------------------------------------- data facts
def data_quality_facts() -> None:
    raw_train, raw_validation = data.load("train"), data.load("validation")
    loads = labelled_loads()
    corrupt = ~loads["clean"]
    define("NTrain", len(raw_train), 0)
    define("NValidation", len(raw_validation), 0)
    define("CorruptRows", int(corrupt.sum()), 0)
    define("CorruptPct", 100 * corrupt.mean())
    define("NegativeWeights", int((raw_train["weight"] < 0).sum()), 0)
    define("CappedWeights", int((raw_train["weight"].abs() == 47500).sum()), 0)
    define("MissingWeights", int(raw_train["weight"].isna().sum()), 0)
    define("MissingMarket", int(raw_train["market_index"].isna().sum()), 0)
    seen = set(raw_train["pickup"]) | set(raw_train["delivery"])
    new_cities = sorted((set(raw_validation["pickup"]) | set(raw_validation["delivery"])) - seen)
    touches = raw_validation["pickup"].isin(new_cities) | raw_validation["delivery"].isin(new_cities)
    define("UnseenCities", len(new_cities), 0)
    define("UnseenCityList", ", ".join(new_cities))
    define("UnseenShare", 100 * touches.mean())

    clean_rows = loads[loads["clean"]]
    rate_per_mile = clean_rows["posted_rate"] / clean_rows["distance"]
    monthly = clean_rows.assign(rpm=rate_per_mile).groupby(clean_rows["date"].dt.month)
    correlations = monthly.apply(lambda g: np.corrcoef(g["quote_signal"], g["rpm"])[0, 1], include_groups=False)
    define("QuoteCopyMin", correlations[correlations > 0.5].min())
    define("QuoteCopyMax", correlations[correlations > 0.5].max())
    define("QuoteMirrorMin", correlations[correlations < -0.5].max())
    define("QuoteMirrorMax", correlations[correlations < -0.5].min())
    validation_correlation = np.corrcoef(raw_validation["quote_signal"], np.log(raw_validation["distance"]))[0, 1]
    train_correlation = monthly.apply(
        lambda g: abs(np.corrcoef(g["quote_signal"], np.log(g["distance"]))[0, 1]), include_groups=False
    ).median()
    define("QuoteDistanceCorrTrain", train_correlation)
    define("QuoteDistanceCorrValidation", validation_correlation)
    daily_market = raw_train.groupby("date")["market_index"]
    define("MarketWithinDaySD", daily_market.std().mean(), 3)
    define("MarketBetweenDaySD", daily_market.mean().std(), 3)

    ratio = loads["posted_rate"] / loads["distance"]
    lane = ratio / ratio.groupby([loads["pickup"], loads["delivery"], loads["equipment"]]).transform("median")
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.hist(np.log10(lane), bins=200, color=BLUE)
    axis.set_yscale("log")
    for bound in (0.67, 1.5):
        axis.axvline(np.log10(bound), color=ORANGE, ls="--", lw=1)
    axis.set_xlabel("rate per mile / lane median (log10 scale)")
    axis.set_ylabel("loads (log scale)")
    axis.set_title("Corrupted labels form separate bands far from the clean peak")
    figure.tight_layout()
    figure.savefig(FIGURES / "label_noise.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 2.8))
    axis.bar(correlations.index, correlations.values, color=[BLUE if v > 0 else ORANGE for v in correlations])
    axis.set_xticks(range(1, 11))
    axis.set_xlabel("month of 2025")
    axis.set_ylabel("correlation")
    axis.set_title("quote_signal vs rate per mile: copy, mirror or noise depending on the month")
    figure.tight_layout()
    figure.savefig(FIGURES / "quote_signal.png", dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------- experiments
def ladder() -> dict[str, pd.DataFrame]:
    rows, fold_values, predictions = [], [], {}
    previous = None
    for path in LADDER:
        config, run = config_run(path)
        metrics = run.data.metrics
        value = metrics["dev.mean.wMAE_clean"]
        folds = pd.read_csv(
            mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="evaluation/fold_metrics_dev.csv")
        )
        fold_values.append(folds["wMAE_clean"].values)
        predictions[config["step"]] = load_predictions(run, "dev")
        rows.append(
            {
                "Step": config["step"],
                "Model": config["name"].split(": ", 1)[1],
                "Dev wMAE ($)": money(value),
                "MAPE": f"{metrics['dev.mean.MAPE_clean']:.2f}%",
                "Worst fold ($)": money(metrics["dev.worst.wMAE_clean"]),
                "Change ($)": "" if previous is None else f"{value - previous:+.2f}",
                "Run": short_id(run),
            }
        )
        define(f"DevStep{'ABCDEF'[config['step'] - 1]}", value)
        define(f"MapeDevStep{'ABCDEF'[config['step'] - 1]}", metrics["dev.mean.MAPE_clean"])
        previous = value
    table = pd.DataFrame(rows)
    write_table("ladder", table, "rlrrrrl")

    figure, axis = plt.subplots(figsize=(7.5, 3.6))
    steps = table["Step"].values
    means = [float(v.replace(",", "")) for v in table["Dev wMAE ($)"]]
    for step, values in zip(steps, fold_values, strict=True):
        axis.scatter(np.full(len(values), step), values, s=12, color=GREY, zorder=2)
    axis.plot(steps, means, marker="o", color=BLUE, lw=2, zorder=3)
    for step, value in zip(steps, means, strict=True):
        axis.annotate(f"${value:.2f}", (step, value), textcoords="offset points", xytext=(6, 6), fontsize=8)
    axis.set_xticks(steps)
    axis.set_xlabel("ladder step")
    axis.set_ylabel("dev wMAE_clean ($)")
    axis.set_yscale("log")
    axis.set_title("Each step fixes what the previous one got wrong (dots = the five dev folds)")
    figure.tight_layout()
    figure.savefig(FIGURES / "ladder.png", dpi=150)
    plt.close(figure)

    final, default = predictions[6], predictions[5]
    comparisons = [("Final vs step 5 (untuned)", final, default)]
    comparisons += [(f"Final vs step {s}", final, predictions[s]) for s in (4, 3, 2, 1)]
    rows = []
    for label, candidate, reference in comparisons:
        result = paired_block_bootstrap(candidate, reference)
        rows.append(
            {
                "Comparison": label,
                "Difference in clean MAE, 95% CI ($)": ci_text(result),
                "Folds final better": f"{result['folds_candidate_better']}/{result['n_folds']}",
            }
        )
        if "step 5" in label:
            define("DevTunedGain", -result["difference"])
            define("DevTunedGainLow", -result["ci_high"])
            define("DevTunedGainHigh", -result["ci_low"])
    write_table("dev_comparisons", pd.DataFrame(rows), "lrr")
    return predictions


def baselines() -> None:
    rows = []
    for path in BASELINES:
        config, run = config_run(path)
        rows.append(
            {
                "Baseline": config["name"].split(": ", 1)[1],
                "Dev wMAE ($)": money(run.data.metrics["dev.mean.wMAE_clean"]),
                "MAPE": f"{run.data.metrics['dev.mean.MAPE_clean']:.2f}%",
                "Run": short_id(run),
            }
        )
        if "naive daily" in config["name"]:
            define("DevNaiveIndex", run.data.metrics["dev.mean.wMAE_clean"])
    write_table("baselines", pd.DataFrame(rows), "lrrl")


def ablations(ladder_predictions: dict) -> None:
    reference_value = latest_run("02-improvement-ladder", load_experiment(LADDER[4])["name"]).data.metrics[
        "dev.mean.wMAE_clean"
    ]
    rows, labels, deltas = [], [], []
    for path in ABLATIONS:
        config, run = config_run(path)
        value = run.data.metrics["dev.mean.wMAE_clean"]
        result = paired_block_bootstrap(load_predictions(run, "dev"), ladder_predictions[5])
        label = config["name"].replace("Ablation: ", "")
        rows.append(
            {
                "Change to the step-5 model": label,
                "Dev wMAE ($)": money(value),
                "Change ($)": f"{value - reference_value:+.2f}",
                "Paired 95% CI ($)": f"[{result['ci_low']:+.2f}, {result['ci_high']:+.2f}]",
                "Run": short_id(run),
            }
        )
        labels.append(label)
        deltas.append(value - reference_value)
        define("Abl" + macro_key(path), value - reference_value)
    write_table("ablations", pd.DataFrame(rows), "lrrrl")
    order = np.argsort(deltas)
    figure, axis = plt.subplots(figsize=(7.5, 3.4))
    axis.barh(range(len(order)), np.array(deltas)[order], color=[ORANGE if deltas[i] > 0 else GREEN for i in order])
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([labels[i] for i in order], fontsize=8)
    axis.axvline(0, color=INK, lw=0.8)
    axis.set_xlabel("change in dev wMAE_clean vs step 5 ($, positive = worse)")
    axis.set_title("What each idea is worth")
    figure.tight_layout()
    figure.savefig(FIGURES / "ablations.png", dpi=150)
    plt.close(figure)


def tuning() -> None:
    run = latest_run("04-tuning", "optuna-search")
    best = json.loads(
        Path(
            mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="tuning/best_params.json")
        ).read_text()
    )
    importance = json.loads(
        Path(
            mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="tuning/param_importance.json")
        ).read_text()
    )
    trials = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="tuning/trials.csv"))
    define("TuneTrials", len(trials), 0)
    define("TuneBest", run.data.metrics["best.dev.mean.wMAE_clean"])
    define("TuneMatchesFinal", "yes" if run.data.tags.get("best_matches_final_yaml") == "True" else "no")
    define("TuningRun", short_id(run))
    trees = best["residual_tree_params"]
    rows = [
        ("Ridge penalty alpha", f"{best['alpha']:.3f}"),
        (
            "Spline knots: distance / weight / coordinates",
            f"{best['spline_knots']['distance_knots']} / {best['spline_knots']['weight_knots']} / "
            f"{best['spline_knots']['coordinate_knots']}",
        ),
        ("Recency time constant (days)", f"{best.get('recency_tau_days', float('nan')):.1f}"),
        ("Residual trees", str(trees["n_estimators"])),
        ("Learning rate", f"{trees['learning_rate']:.4f}"),
        ("Leaves per tree", str(trees["num_leaves"])),
        ("Minimum loads per leaf", str(trees["min_child_samples"])),
        ("L2 penalty lambda", f"{trees['reg_lambda']:.2f}"),
        ("Share of columns per tree", f"{trees['colsample_bytree']:.2f}"),
    ]
    write_table("tuned_params", pd.DataFrame(rows, columns=["Setting", "Tuned value"]), "lr")
    top = sorted(importance.items(), key=lambda kv: -kv[1])[:5]
    write_table(
        "tuning_importance",
        pd.DataFrame(
            [(k.replace("_", " "), f"{v:.2f}") for k, v in top],
            columns=["Hyperparameter", "Importance (fANOVA)"],
        ),
        "lr",
    )
    figure, axis = plt.subplots(figsize=(7, 3))
    values = trials["value"].values
    axis.scatter(trials["number"], values, s=10, color=GREY, label="trial")
    axis.plot(trials["number"], np.minimum.accumulate(values), color=BLUE, lw=2, label="best so far")
    axis.set_ylim(top=np.nanpercentile(values, 90))
    axis.set_xlabel("trial")
    axis.set_ylabel("dev wMAE_clean ($)")
    axis.legend(frameon=False)
    axis.set_title("Optuna search on the dev folds")
    figure.tight_layout()
    figure.savefig(FIGURES / "tuning.png", dpi=150)
    plt.close(figure)

    nested_run = latest_run("04-tuning", "nested-forward-check")
    nested = pd.read_csv(
        mlflow.artifacts.download_artifacts(
            run_id=nested_run.info.run_id, artifact_path="tuning/nested_forward_check.csv"
        )
    )
    write_table(
        "nested",
        pd.DataFrame(
            {
                "Scored fold": nested["outer_fold"],
                "Tuned on folds": [f"1--{n}" for n in nested["inner_folds"]],
                "Default ($)": nested["outer_default"].map(money),
                "Tuned ($)": nested["outer_tuned"].map(money),
                "Gain ($)": nested["gain"].map(lambda g: f"{g:+.2f}"),
            }
        ),
        "rlrrr",
    )
    define("NestedGainFour", float(nested.loc[nested["outer_fold"] == 4, "gain"].iloc[0]))
    define("NestedGainFive", float(nested.loc[nested["outer_fold"] == 5, "gain"].iloc[0]))
    define("NestedRun", short_id(nested_run))


def locked_test() -> pd.DataFrame:
    final = load_yaml("configs/final.yaml")
    runs = {}
    for path in final["locked_test"]:
        config = load_experiment(path)
        runs[path] = (config, latest_run("05-locked-test", config["name"]))
    final_config, final_run = runs[final["locked_test"][0]]
    final_predictions = load_predictions(final_run, "test")
    rows = []
    for path, (config, run) in runs.items():
        metrics = run.data.metrics
        is_final = path == final["locked_test"][0]
        comparison = (
            "" if is_final else ci_text(paired_block_bootstrap(load_predictions(run, "test"), final_predictions))
        )
        rows.append(
            {
                "Model": config["name"].split(": ", 1)[1],
                "wMAE ($)": money(metrics["test.mean.wMAE_clean"]),
                "MAPE": f"{metrics['test.mean.MAPE_clean']:.2f}%",
                "All rows MAE ($)": money(metrics["test.mean.MAE_all"]),
                "RMSE ($)": money(metrics["test.mean.RMSE_all"]),
                "Bias": f"{metrics['test.mean.median_bias_pct_clean']:+.2f}%",
                "vs final, 95% CI ($)": comparison,
                "Run": short_id(run),
            }
        )
        key = macro_key(Path(path))
        define(f"Test{key}", metrics["test.mean.wMAE_clean"])
        if not is_final:
            result = paired_block_bootstrap(load_predictions(run, "test"), final_predictions)
            define(f"TestGap{key}", result["difference"])
            define(f"TestGap{key}Low", result["ci_low"])
            define(f"TestGap{key}High", result["ci_high"])
    write_table("locked_test", pd.DataFrame(rows), "lrrrrrrl")
    define("MapeTestFinal", final_run.data.metrics["test.mean.MAPE_clean"])
    define("TestFinalAllMAE", final_run.data.metrics["test.mean.MAE_all"])
    define("LockedTestRun", final_run.info.run_id)
    corrupt = ~final_predictions["clean"]
    absolute = (final_predictions["predicted"] - final_predictions["actual"]).abs()
    define("CorruptShareOfError", 100 * absolute[corrupt].sum() / absolute.sum(), 0)

    segments = final_predictions.assign(
        horizon=horizon_bucket(final_predictions["days_ahead"]),
        unseen=final_predictions["unseen_city"].map({True: "unseen city", False: "seen cities"}),
    )
    parts = []
    for column, title in [("equipment", "Equipment"), ("horizon", "Days ahead"), ("unseen", "City")]:
        table = by_segment(segments, column).reset_index()
        parts.append(
            pd.DataFrame(
                {
                    "Segment": [f"{title}: {v}" for v in table[column]],
                    "Loads": table["n"],
                    "MAE ($)": table["MAE"].map(money),
                    "MAPE": table["MAPE"].map(lambda v: f"{v:.2f}%"),
                    "Bias": table["median_bias_pct"].map(lambda v: f"{v:+.2f}%"),
                }
            )
        )
    write_table("segments", pd.concat(parts), "lrrrr")
    shutil.copy(
        mlflow.artifacts.download_artifacts(run_id=final_run.info.run_id, artifact_path="plots/residuals_test.png"),
        FIGURES / "residuals_test.png",
    )
    return final_predictions


def final_model() -> None:
    run = latest_run("06-final", "champion: ridge + residual trees (tuned)")
    metrics = run.data.metrics
    define("ChampionRun", run.info.run_id)
    define("CoverEighty", 100 * metrics["locked_test.coverage80_clean"], 1)
    define("CoverNinetyFive", 100 * metrics["locked_test.coverage95_clean"], 1)
    define("CoverEightyAll", 100 * metrics["locked_test.coverage80_all"], 1)
    define("CoverNinetyFiveAll", 100 * metrics["locked_test.coverage95_all"], 1)
    define("RowsAfterScreen", int(metrics["rows_used_after_label_screen"]), 0)
    define("FitSeconds", metrics["fit_seconds"], 1)
    define("InferenceMicroseconds", metrics["inference_microseconds_per_load"], 0)
    coefficients = json.loads(
        Path(
            mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path="model_files/ridge_coefficients.json"
            )
        ).read_text()
    )
    define("ElasticityDay", coefficients["market_level_today"], 3)
    define("ElasticityLoad", coefficients["market_deviation_this_load"], 3)
    define("DriftPerMonth", 100 * 30 / 100 * coefficients["hundreds_of_days_since_start"], 2)
    for equipment, key in [("dry_van", "DryVan"), ("flatbed", "Flatbed"), ("reefer", "Reefer")]:
        define(f"Ramp{key}", 100 * coefficients[f"quarter_end_progress_x_{equipment}"], 1)
    define("DistanceMarket", coefficients["distance_x_market"], 3)
    shown = {k: v for k, v in coefficients.items() if "basis" not in k}
    write_table(
        "ridge_coefficients",
        pd.DataFrame({"Feature": list(shown), "Coefficient": [f"{v:+.4f}" for v in shown.values()]}),
        "lr",
    )
    params = json.loads(
        Path(
            mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="model_files/final_params.json")
        ).read_text()
    )
    flat = [("ridge." + k, json.dumps(v)) for k, v in params["ridge"].items()]
    flat += [("residual_lightgbm." + k, json.dumps(v)) for k, v in params["residual_lightgbm"].items()]
    write_table("final_params", pd.DataFrame(flat, columns=["Parameter", "Value"]), "ll")

    for name in ["december_sensitivity.csv", "scorer_results/candidate_december.png"]:
        source = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path=f"deliverables/{Path(name).name}"
        )
        shutil.copy(source, FIGURES / Path(name).name)
    december = pd.read_csv(FIGURES / "december_sensitivity.csv")
    define("DecFirst", december["predicted_rate"].iloc[0])
    define("DecLast", december["predicted_rate"].iloc[-1])
    define("DecLastNoRamp", december["without_quarter_end_ramp"].iloc[-1])
    define("DecMin", december["predicted_rate"].min())
    define("DecMax", december["predicted_rate"].max())
    define("DecFlatMin", december["market_flat_at_october_mean"].min())
    define("DecFlatMax", december["market_flat_at_october_mean"].max())
    dates = pd.to_datetime(december["date"])
    figure, axis = plt.subplots(figsize=(7.5, 3.4))
    axis.fill_between(dates, december["lo80"], december["hi80"], color=BLUE, alpha=0.12, label="80% interval")
    axis.plot(dates, december["predicted_rate"], color=BLUE, lw=2, marker="o", ms=3, label="final model")
    axis.plot(
        dates,
        december["without_quarter_end_ramp"],
        color=ORANGE,
        lw=1.5,
        ls="--",
        label="without quarter-end ramp",
    )
    axis.plot(
        dates,
        december["market_flat_at_october_mean"],
        color=GREY,
        lw=1.5,
        ls=":",
        label="market held at October mean",
    )
    axis.set_ylabel("predicted rate ($)")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.set_title("December 2025, Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(FIGURES / "december_sensitivity.png", dpi=150)
    plt.close(figure)


def run_inventory() -> None:
    client = mlflow.MlflowClient()
    rows = []
    for experiment in [
        "01-baselines",
        "02-improvement-ladder",
        "03-ablations",
        "04-tuning",
        "05-locked-test",
        "06-final",
    ]:
        found = client.get_experiment_by_name(experiment)
        for run in client.search_runs([found.experiment_id], order_by=["attributes.start_time ASC"], max_results=1000):
            if run.data.tags.get("mlflow.parentRunId"):
                continue
            rows.append({"Experiment": experiment, "Run name": run.info.run_name, "Run ID": run.info.run_id})
    write_table("run_ids", pd.DataFrame(rows), "lll")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data_quality_facts()
    ladder_predictions = ladder()
    baselines()
    ablations(ladder_predictions)
    tuning()
    locked_test()
    final_model()
    run_inventory()
    lines = [
        f"\\newcommand{{\\{name}}}{{{escape(value) if name.endswith(('Run', 'List')) else value}}}"
        for name, value in sorted(macros.items())
    ]
    (TABLES / "numbers.tex").write_text(
        "% Generated by scripts/make_report_assets.py - do not edit.\n" + "\n".join(lines) + "\n"
    )
    print(
        f"wrote {len(macros)} numbers, {len(list(TABLES.glob('*.tex')))} tables, "
        f"{len(list(FIGURES.glob('*.png')))} figures"
    )


if __name__ == "__main__":
    main()
