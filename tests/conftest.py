"""Deterministic synthetic datasets, exposed as factory fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Fixed reference point (downtown San Francisco) for the regression target.
_REF_LAT, _REF_LON = 37.7749, -122.4194


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _binary(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f_signal1 = rng.normal(size=n)
    f_signal2 = rng.normal(size=n)
    noise = rng.normal(size=(n, 3))
    cat_levels = np.array(["v", "w", "x", "y", "z"])
    cat_idx = rng.integers(0, len(cat_levels), size=n)
    cat_effect = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])[cat_idx]
    n_groups = 100
    group_id = rng.integers(0, n_groups, size=n)
    group_effect = rng.normal(scale=0.25, size=n_groups)[group_id]  # mild
    # Target depends on the signals, cat and group only: never on ts.
    logit = 1.5 * f_signal1 + 1.0 * f_signal2 + cat_effect + group_effect
    target = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    flip = rng.random(n) < 0.02
    leak = np.where(flip, 1 - target, target)
    ts = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n), unit="h")
    return pd.DataFrame(
        {
            "f_signal1": f_signal1,
            "f_signal2": f_signal2,
            "f_noise1": noise[:, 0],
            "f_noise2": noise[:, 1],
            "f_noise3": noise[:, 2],
            "cat": pd.Series(cat_levels[cat_idx], dtype=object),
            "group_id": group_id,
            "ts": ts,
            "leak": leak,
            "target": target,
        }
    )


def _regression(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    lat = rng.uniform(37.0, 39.0, size=n)
    lon = rng.uniform(-123.0, -121.0, size=n)
    noise1 = rng.normal(size=n)
    dist = _haversine_km(lat, lon, _REF_LAT, _REF_LON)
    target = x1 * x2 + 0.05 * dist + rng.normal(scale=0.1, size=n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "lat": lat,
            "lon": lon,
            "noise1": noise1,
            "target": target,
        }
    )


def _multiclass(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    noise1 = rng.normal(size=n)
    scores = np.column_stack([f1, f2, -f1 - f2]) * 1.5 + rng.gumbel(size=(n, 3))
    target = np.array(["a", "b", "c"])[scores.argmax(axis=1)]
    return pd.DataFrame(
        {"f1": f1, "f2": f2, "noise1": noise1, "target": pd.Series(target, dtype=object)}
    )


@pytest.fixture
def make_binary():
    return _binary


@pytest.fixture
def make_regression():
    return _regression


@pytest.fixture
def make_multiclass():
    return _multiclass
