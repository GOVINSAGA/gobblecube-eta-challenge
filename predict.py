"""Submission interface for the Gobblecube ETA Challenge.

The artifact in ``model.pkl`` is a plain dictionary of numpy arrays produced by
``train_model.py``. Keeping the runtime as simple data lookups makes per-request
latency predictable and avoids shipping a heavyweight ML library in Docker.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

_ZONE_FACTOR = 266
_HODOW_FACTOR = 168
_HOUR_FACTOR = 24


with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)


def _table_lookup(table: dict, key: int) -> tuple[float, float]:
    """Return (median, count) from a sorted sparse table, or (nan, 0)."""
    keys = table["keys"]
    pos = int(np.searchsorted(keys, key))
    if pos < len(keys) and int(keys[pos]) == key:
        return float(table["median"][pos]), float(table["count"][pos])
    return float("nan"), 0.0


def _shrink(median: float, count: float, fallback: float, k: float) -> float:
    if not np.isfinite(median) or count <= 0:
        return fallback
    weight = count / (count + k)
    return weight * median + (1.0 - weight) * fallback


def _safe_zone(value: object) -> int:
    try:
        zone = int(value)
    except (TypeError, ValueError):
        return 0
    if 1 <= zone <= 265:
        return zone
    return 0


def _zone_median(values: np.ndarray, zone: int) -> float:
    if 0 <= zone < len(values):
        return float(values[zone])
    return float("nan")


def predict(request: dict) -> float:
    """Predict trip duration in seconds for one ETA request."""
    pickup_zone = _safe_zone(request.get("pickup_zone"))
    dropoff_zone = _safe_zone(request.get("dropoff_zone"))

    ts = datetime.fromisoformat(str(request["requested_at"]))
    hour = ts.hour
    dow = ts.weekday()
    hodow = dow * 24 + hour
    week = int(ts.isocalendar().week)

    pair = pickup_zone * _ZONE_FACTOR + dropoff_zone
    pair_hodow = pair * _HODOW_FACTOR + hodow
    pair_hour = pair * _HOUR_FACTOR + hour

    global_median = float(_MODEL["global_median"])

    pair_median = _zone_median(_MODEL["pair_median"], pair)
    pickup_median = _zone_median(_MODEL["pickup_median"], pickup_zone)
    dropoff_median = _zone_median(_MODEL["dropoff_median"], dropoff_zone)
    if np.isfinite(pickup_median) and np.isfinite(dropoff_median):
        zone_fallback = 0.5 * (pickup_median + dropoff_median)
    else:
        zone_fallback = global_median
    route_pred = pair_median if np.isfinite(pair_median) else zone_fallback

    full_med, full_count = _table_lookup(_MODEL["tables"]["full_pair_hodow"], pair_hodow)
    full_pred = _shrink(full_med, full_count, route_pred, _MODEL["shrinkage"]["full_pair_hodow"])

    recent_med, recent_count = _table_lookup(
        _MODEL["tables"]["recent_pair_hodow"], pair_hodow
    )
    recent_hodow_pred = _shrink(
        recent_med,
        recent_count,
        full_pred,
        _MODEL["shrinkage"]["recent_pair_hodow"],
    )

    recent_hour_med, recent_hour_count = _table_lookup(
        _MODEL["tables"]["recent_pair_hour"], pair_hour
    )
    recent_hour_pred = _shrink(
        recent_hour_med,
        recent_hour_count,
        full_pred,
        _MODEL["shrinkage"]["recent_pair_hour"],
    )

    week_adjust = 0.0
    if 0 <= week < len(_MODEL["week_adjust"]):
        week_adjust = float(_MODEL["week_adjust"][week])
    week_pred = full_pred + _MODEL["week_adjust_scale"] * week_adjust

    weights = _MODEL["ensemble_weights"]
    prediction = (
        weights["recent_pair_hodow"] * recent_hodow_pred
        + weights["recent_pair_hour"] * recent_hour_pred
        + weights["week_adjusted_full"] * week_pred
    )

    return float(np.clip(prediction, 30.0, 3.0 * 3600.0))
