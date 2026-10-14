"""Diagnostic figures saved to files, then logged to MLflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

BLUE, ORANGE, GREEN, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#33322f"
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 9,
    }
)


def residual_diagnostics(predictions: pd.DataFrame, path: Path, title: str) -> None:
    """Daily mean residual over time, error by horizon, residual histogram and daily-residual autocorrelation."""
    clean = predictions[predictions["clean"]].copy()
    clean["log_residual_pct"] = np.log(clean["actual"] / clean["predicted"]) * 100
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    for fold, group in clean.groupby("fold"):
        daily = group.groupby("date")["log_residual_pct"].mean()
        axes[0, 0].plot(daily.index, daily.values, lw=1, label=fold)
    axes[0, 0].axhline(0, color=INK, lw=0.8)
    axes[0, 0].set_title("Daily mean residual (log %, actual vs forecast)")
    axes[0, 0].legend(fontsize=6)
    axes[0, 0].tick_params(axis="x", rotation=30, labelsize=7)

    horizon = pd.cut(clean["days_ahead"], [0, 7, 14, 28, 42, 61])
    by_horizon = (clean["predicted"] - clean["actual"]).abs().groupby(horizon, observed=True).mean()
    axes[0, 1].bar([str(b) for b in by_horizon.index], by_horizon.values, color=BLUE)
    axes[0, 1].set_title("Clean MAE ($) by days ahead")

    axes[1, 0].hist(np.clip(clean["log_residual_pct"], -15, 15), bins=120, color=BLUE)
    axes[1, 0].set_title("Residual distribution (log %, clipped at ±15)")

    autocorrelations = [
        acf(group.groupby("date")["log_residual_pct"].mean().values, nlags=14)
        for _, group in clean.groupby("fold")
        if group["date"].nunique() > 20
    ]
    if autocorrelations:
        mean_acf = np.mean(autocorrelations, axis=0)
        axes[1, 1].bar(range(len(mean_acf)), mean_acf, color=ORANGE)
        band = 1.96 / np.sqrt(61)
        axes[1, 1].axhline(band, color=GREY, ls="--")
        axes[1, 1].axhline(-band, color=GREY, ls="--")
    axes[1, 1].set_title("Autocorrelation of the daily mean residual")
    figure.suptitle(title, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
