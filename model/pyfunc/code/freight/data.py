"""Loading the raw CSVs, schema checks and dataset fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, EQUIPMENT_TYPES, data_config

PREDICTOR_COLUMNS = [
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]
TARGET = "posted_rate"


def file_path(kind: str) -> Path:
    """kind is one of: train, validation, template, december (names come from configs/data.yaml)."""
    return DATA_DIR / data_config()["files"][kind]


def load(kind: str) -> pd.DataFrame:
    frame = pd.read_csv(file_path(kind))
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def schema_problems(frame: pd.DataFrame, kind: str) -> list[str]:
    """Hard schema violations. Known data-quality issues (label noise, weight caps) are not violations."""
    required = PREDICTOR_COLUMNS + ([TARGET] if kind == "train" else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return [f"{kind}: missing columns {missing}"]
    problems = []
    if frame["load_id"].duplicated().any():
        problems.append(f"{kind}: duplicate load_id")
    unknown = set(frame["equipment"].unique()) - set(EQUIPMENT_TYPES)
    if unknown:
        problems.append(f"{kind}: unknown equipment {sorted(unknown)}")
    if frame["distance"].isna().any() or (frame["distance"] <= 0).any():
        problems.append(f"{kind}: missing or non-positive distance")
    if frame["date"].isna().any():
        problems.append(f"{kind}: unparseable dates")
    for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        if frame[column].isna().any():
            problems.append(f"{kind}: missing {column}")
    if kind == "train" and (frame[TARGET].isna() | (frame[TARGET] <= 0)).any():
        problems.append("train: missing or non-positive posted_rate")
    return problems


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint() -> dict[str, str]:
    """SHA-256 of every input file; logged with each MLflow run instead of copying the data."""
    return {f"data.{kind}.sha256": sha256(file_path(kind)) for kind in data_config()["files"]}


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon). Each city has exactly one coordinate pair in the data, which is asserted here."""
    parts = []
    for frame in frames:
        for end in ("pickup", "delivery"):
            part = frame[[end, f"{end}_lat", f"{end}_lon"]]
            parts.append(part.set_axis(["city", "lat", "lon"], axis=1))
    coordinates = pd.concat(parts).drop_duplicates()
    assert not coordinates["city"].duplicated().any(), "a city has more than one coordinate pair"
    return coordinates.set_index("city")
