"""Dollar-scale metrics and the paired block bootstrap.

Primary metric, fixed before any experiment: wMAE_clean. It is the mean absolute error on rows with plausible
labels, with seen-city and unseen-city rows re-weighted to the validation mix (87.94% / 12.06%). MAE is in dollars,
is robust to the 1.4% corrupted labels (which make RMSE uninformative), and matches the median forecast exp(log).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNSEEN_CITY_SHARE_IN_VALIDATION = 0.1206


def summarize(actual, predicted, clean, unseen, prefix: str = "") -> dict[str, float]:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    clean, unseen = np.asarray(clean, bool), np.asarray(unseen, bool)
    error = predicted - actual
    absolute = np.abs(error)
    seen_clean, unseen_clean = clean & ~unseen, clean & unseen
    mae_seen = absolute[seen_clean].mean() if seen_clean.any() else np.nan
    mae_unseen = absolute[unseen_clean].mean() if unseen_clean.any() else mae_seen
    share = UNSEEN_CITY_SHARE_IN_VALIDATION
    values = {
        "wMAE_clean": (1 - share) * mae_seen + share * mae_unseen,
        "MAE_clean": absolute[clean].mean(),
        "MAE_clean_seen": mae_seen,
        "MAE_clean_unseen": mae_unseen,
        "MAPE_clean": 100 * np.mean(absolute[clean] / actual[clean]),
        "MAE_all": absolute.mean(),
        "RMSE_all": np.sqrt(np.mean(error**2)),
        "MAPE_all": 100 * np.mean(absolute / actual),
        "median_bias_pct_clean": 100 * np.median(np.log(predicted[clean] / actual[clean])),
        "n": len(actual),
        "n_clean": int(clean.sum()),
    }
    return {prefix + name: float(value) for name, value in values.items()}


def by_segment(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    """Clean-row MAE, MAPE and median bias per segment value."""
    clean = predictions[predictions["clean"]]
    frame = clean.assign(
        absolute_error=(clean["predicted"] - clean["actual"]).abs(),
        percent_error=(clean["predicted"] - clean["actual"]).abs() / clean["actual"] * 100,
        log_bias_pct=np.log(clean["predicted"] / clean["actual"]) * 100,
    )
    grouped = frame.groupby(column, observed=True)
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "MAE": grouped["absolute_error"].mean(),
            "MAPE": grouped["percent_error"].mean(),
            "median_bias_pct": grouped["log_bias_pct"].median(),
        }
    )


def horizon_bucket(days_ahead: pd.Series) -> pd.Series:
    return pd.cut(days_ahead, [0, 7, 14, 28, 42, 61], labels=["1-7", "8-14", "15-28", "29-42", "43-61"])


def paired_block_bootstrap(
    candidate: pd.DataFrame, reference: pd.DataFrame, block_days: int = 7, n_boot: int = 2000, seed: int = 7
) -> dict[str, float]:
    """Difference in clean-row MAE (candidate - reference; negative = candidate better) with a 95% interval.

    Loads on the same day share the market, and neighbouring days are correlated, so whole 7-day blocks of days
    are resampled within each fold instead of individual loads.
    """
    key = ["fold", "load_id"]
    merged = candidate.merge(reference[key + ["predicted"]], on=key, suffixes=("_a", "_b"))
    assert len(merged) == len(candidate) == len(reference), "models were not scored on identical rows"
    merged = merged[merged["clean"]]
    merged["difference"] = (merged["predicted_a"] - merged["actual"]).abs() - (
        merged["predicted_b"] - merged["actual"]
    ).abs()
    daily = merged.groupby(["fold", "date"])["difference"].agg(["sum", "size"]).reset_index()
    per_fold = [group.sort_values("date").reset_index(drop=True) for _, group in daily.groupby("fold")]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        total = count = 0.0
        for group in per_fold:
            length = len(group)
            n_blocks = int(np.ceil(length / block_days))
            starts = rng.integers(0, max(1, length - block_days + 1), n_blocks)
            rows = np.concatenate([np.arange(s, min(s + block_days, length)) for s in starts])[:length]
            total += group["sum"].values[rows].sum()
            count += group["size"].values[rows].sum()
        samples.append(total / count)
    low, high = np.quantile(samples, [0.025, 0.975])
    fold_differences = merged.groupby("fold")["difference"].mean()
    return {
        "difference": float(daily["sum"].sum() / daily["size"].sum()),
        "ci_low": float(low),
        "ci_high": float(high),
        "folds_candidate_better": int((fold_differences < 0).sum()),
        "n_folds": int(len(fold_differences)),
    }
