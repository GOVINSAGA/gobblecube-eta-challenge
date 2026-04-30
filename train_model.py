#!/usr/bin/env python
"""Train the ETA lookup model.

The model is deliberately tabular: it stores robust medians for route/time
buckets and blends dense recent buckets back to stable full-year priors. This
beats the starter GBT locally while keeping inference fast and Docker small.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

ZONE_FACTOR = 266
PAIR_SIZE = ZONE_FACTOR * ZONE_FACTOR
HODOW_FACTOR = 168
HOUR_FACTOR = 24

FULL_PAIR_HODOW_K = 2.0
RECENT_PAIR_HODOW_K = 50.0
RECENT_PAIR_HOUR_K = 200.0
WEEK_ADJUST_SCALE = 0.75

ENSEMBLE_WEIGHTS = {
    "recent_pair_hodow": 0.50,
    "recent_pair_hour": 0.20,
    "week_adjusted_full": 0.30,
}

RECENT_HODOW_START = pd.Timestamp("2023-12-01")
RECENT_HOUR_START = pd.Timestamp("2023-12-08")

REQUEST_COLUMNS = [
    "pickup_zone",
    "dropoff_zone",
    "requested_at",
    "passenger_count",
    "duration_seconds",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact integer keys used by the lookup model."""
    ts = pd.to_datetime(df["requested_at"])
    pickup = df["pickup_zone"].astype("int32")
    dropoff = df["dropoff_zone"].astype("int32")
    pair = (pickup * ZONE_FACTOR + dropoff).astype("int32")
    hour = ts.dt.hour.astype("int16")
    dow = ts.dt.dayofweek.astype("int16")
    hodow = (dow * 24 + hour).astype("int16")

    out = pd.DataFrame(
        {
            "pickup_zone": pickup,
            "dropoff_zone": dropoff,
            "duration_seconds": df["duration_seconds"].astype("float64"),
            "ts": ts,
            "pair": pair,
            "hour": hour,
            "dow": dow,
            "hodow": hodow,
            "week": ts.dt.isocalendar().week.astype("int16"),
        }
    )
    out["pair_hodow"] = (pair.astype("int64") * HODOW_FACTOR + hodow).astype("int64")
    out["pair_hour"] = (pair.astype("int64") * HOUR_FACTOR + hour).astype("int64")
    return out


def dense_median(df: pd.DataFrame, key: str, size: int) -> np.ndarray:
    values = np.full(size, np.nan, dtype=np.float32)
    medians = df.groupby(key, observed=True)["duration_seconds"].median()
    index = medians.index.to_numpy(dtype=np.int64)
    valid = (0 <= index) & (index < size)
    values[index[valid]] = medians.to_numpy(dtype=np.float32)[valid]
    return values


def sparse_table(df: pd.DataFrame, key: str) -> dict[str, np.ndarray]:
    grouped = (
        df.groupby(key, observed=True)["duration_seconds"]
        .agg(["median", "count"])
        .reset_index()
        .sort_values(key)
    )
    return {
        "keys": grouped[key].to_numpy(dtype=np.int64),
        "median": grouped["median"].to_numpy(dtype=np.float32),
        "count": grouped["count"].to_numpy(dtype=np.float32),
    }


def lookup_sparse(table: dict[str, np.ndarray], query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = table["keys"]
    pos = np.searchsorted(keys, query)
    in_bounds = pos < len(keys)

    medians = np.full(len(query), np.nan, dtype=np.float64)
    counts = np.zeros(len(query), dtype=np.float64)
    if not np.any(in_bounds):
        return medians, counts

    candidate_rows = np.flatnonzero(in_bounds)
    candidate_pos = pos[in_bounds]
    found_rows = candidate_rows[keys[candidate_pos] == query[candidate_rows]]
    found_pos = pos[found_rows]
    medians[found_rows] = table["median"][found_pos].astype(np.float64)
    counts[found_rows] = table["count"][found_pos].astype(np.float64)
    return medians, counts


def shrink(median: np.ndarray, count: np.ndarray, fallback: np.ndarray, k: float) -> np.ndarray:
    weight = count / (count + k)
    return np.where(np.isfinite(median), weight * median + (1.0 - weight) * fallback, fallback)


def predict_frame(features: pd.DataFrame, model: dict) -> np.ndarray:
    pair = features["pair"].to_numpy(dtype=np.int64)
    pickup = features["pickup_zone"].to_numpy(dtype=np.int64)
    dropoff = features["dropoff_zone"].to_numpy(dtype=np.int64)
    pair_hodow = features["pair_hodow"].to_numpy(dtype=np.int64)
    pair_hour = features["pair_hour"].to_numpy(dtype=np.int64)
    week = features["week"].to_numpy(dtype=np.int64)

    global_median = float(model["global_median"])

    pair_pred = model["pair_median"][pair].astype(np.float64)
    pickup_pred = model["pickup_median"][pickup].astype(np.float64)
    dropoff_pred = model["dropoff_median"][dropoff].astype(np.float64)
    zone_fallback = np.where(
        np.isfinite(pickup_pred) & np.isfinite(dropoff_pred),
        0.5 * (pickup_pred + dropoff_pred),
        global_median,
    )
    route_pred = np.where(np.isfinite(pair_pred), pair_pred, zone_fallback)

    full_med, full_count = lookup_sparse(model["tables"]["full_pair_hodow"], pair_hodow)
    full_pred = shrink(
        full_med,
        full_count,
        route_pred,
        float(model["shrinkage"]["full_pair_hodow"]),
    )

    recent_med, recent_count = lookup_sparse(model["tables"]["recent_pair_hodow"], pair_hodow)
    recent_hodow_pred = shrink(
        recent_med,
        recent_count,
        full_pred,
        float(model["shrinkage"]["recent_pair_hodow"]),
    )

    recent_hour_med, recent_hour_count = lookup_sparse(
        model["tables"]["recent_pair_hour"], pair_hour
    )
    recent_hour_pred = shrink(
        recent_hour_med,
        recent_hour_count,
        full_pred,
        float(model["shrinkage"]["recent_pair_hour"]),
    )

    week_adjust = np.zeros(len(features), dtype=np.float64)
    valid_week = (0 <= week) & (week < len(model["week_adjust"]))
    week_adjust[valid_week] = model["week_adjust"][week[valid_week]]
    week_pred = full_pred + float(model["week_adjust_scale"]) * week_adjust

    weights = model["ensemble_weights"]
    pred = (
        float(weights["recent_pair_hodow"]) * recent_hodow_pred
        + float(weights["recent_pair_hour"]) * recent_hour_pred
        + float(weights["week_adjusted_full"]) * week_pred
    )
    return np.clip(pred, 30.0, 3.0 * 3600.0)


def build_model(features: pd.DataFrame, trained_on: str) -> dict:
    global_median = float(features["duration_seconds"].median())

    pair_median = dense_median(features, "pair", PAIR_SIZE)
    pickup_median = dense_median(features, "pickup_zone", ZONE_FACTOR)
    dropoff_median = dense_median(features, "dropoff_zone", ZONE_FACTOR)

    full_pair_hodow = sparse_table(features, "pair_hodow")
    recent_pair_hodow = sparse_table(features[features["ts"] >= RECENT_HODOW_START], "pair_hodow")
    recent_pair_hour = sparse_table(features[features["ts"] >= RECENT_HOUR_START], "pair_hour")

    bootstrap_model = {
        "global_median": global_median,
        "pair_median": pair_median,
        "pickup_median": pickup_median,
        "dropoff_median": dropoff_median,
        "tables": {
            "full_pair_hodow": full_pair_hodow,
            "recent_pair_hodow": recent_pair_hodow,
            "recent_pair_hour": recent_pair_hour,
        },
        "shrinkage": {
            "full_pair_hodow": FULL_PAIR_HODOW_K,
            "recent_pair_hodow": RECENT_PAIR_HODOW_K,
            "recent_pair_hour": RECENT_PAIR_HOUR_K,
        },
        "week_adjust": np.zeros(54, dtype=np.float32),
        "week_adjust_scale": WEEK_ADJUST_SCALE,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
    }

    full_pred = _full_component(features, bootstrap_model)
    residual = features["duration_seconds"].to_numpy(dtype=np.float64) - full_pred
    week_adjust = np.zeros(54, dtype=np.float32)
    weekly = pd.Series(residual).groupby(features["week"].to_numpy()).median()
    week_index = weekly.index.to_numpy(dtype=np.int64)
    valid = (0 <= week_index) & (week_index < len(week_adjust))
    week_adjust[week_index[valid]] = weekly.to_numpy(dtype=np.float32)[valid]

    return {
        "version": 2,
        "trained_on": trained_on,
        "global_median": global_median,
        "pair_median": pair_median,
        "pickup_median": pickup_median,
        "dropoff_median": dropoff_median,
        "tables": {
            "full_pair_hodow": full_pair_hodow,
            "recent_pair_hodow": recent_pair_hodow,
            "recent_pair_hour": recent_pair_hour,
        },
        "shrinkage": {
            "full_pair_hodow": FULL_PAIR_HODOW_K,
            "recent_pair_hodow": RECENT_PAIR_HODOW_K,
            "recent_pair_hour": RECENT_PAIR_HOUR_K,
        },
        "week_adjust": week_adjust,
        "week_adjust_scale": WEEK_ADJUST_SCALE,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "metadata": {
            "recent_hodow_start": str(RECENT_HODOW_START.date()),
            "recent_hour_start": str(RECENT_HOUR_START.date()),
        },
    }


def _full_component(features: pd.DataFrame, model: dict) -> np.ndarray:
    pair = features["pair"].to_numpy(dtype=np.int64)
    pickup = features["pickup_zone"].to_numpy(dtype=np.int64)
    dropoff = features["dropoff_zone"].to_numpy(dtype=np.int64)

    pair_pred = model["pair_median"][pair].astype(np.float64)
    pickup_pred = model["pickup_median"][pickup].astype(np.float64)
    dropoff_pred = model["dropoff_median"][dropoff].astype(np.float64)
    fallback = np.where(
        np.isfinite(pickup_pred) & np.isfinite(dropoff_pred),
        0.5 * (pickup_pred + dropoff_pred),
        float(model["global_median"]),
    )
    route_pred = np.where(np.isfinite(pair_pred), pair_pred, fallback)

    med, count = lookup_sparse(
        model["tables"]["full_pair_hodow"],
        features["pair_hodow"].to_numpy(dtype=np.int64),
    )
    return shrink(med, count, route_pred, float(model["shrinkage"]["full_pair_hodow"]))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for path in (train_path, dev_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run `python data/download_data.py` first.")

    print("Loading train/dev parquet...")
    train = add_features(pd.read_parquet(train_path, columns=REQUEST_COLUMNS))
    dev = add_features(pd.read_parquet(dev_path, columns=REQUEST_COLUMNS))
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")
    return train, dev


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--honest-only",
        action="store_true",
        help="save the train-only model instead of refitting on train+dev",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    train, dev = load_split()

    print("\nBuilding train-only model for honest Dev scoring...")
    train_model = build_model(train, trained_on="train")
    dev_pred = predict_frame(dev, train_model)
    dev_mae = mae(dev_pred, dev["duration_seconds"].to_numpy(dtype=np.float64))
    print(f"Honest full-Dev MAE: {dev_mae:.3f} seconds")

    if args.honest_only:
        model = train_model
    else:
        print("\nRefitting final artifact on train+dev (all distributed 2023 labels)...")
        all_2023 = pd.concat([train, dev], ignore_index=True)
        model = build_model(all_2023, trained_on="train+dev")
        model["metadata"]["honest_dev_mae_before_refit"] = dev_mae

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {MODEL_PATH}")
    print(f"Model trained_on: {model['trained_on']}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
