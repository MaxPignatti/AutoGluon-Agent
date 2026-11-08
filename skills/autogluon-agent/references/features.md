# Feature-engineering playbook

These are patterns, not answers. The features that beat plain AutoGluon come from the domain: ask what physically, economically or behaviourally causes the target, then compute the quantities that measure those causes. Use the patterns below to implement those ideas correctly.

## Principles

- **Trees do not need monotonic transforms.** A log, square root, rank or binning of a single column does not change the splits a tree can make. Value comes from combining columns (ratios, differences, distances, angles), parsing hidden structure (dates, nested fields, codes) and bringing in knowledge the data does not contain (landmark positions, holiday dates, physical constants).
- **Learn from the training fold only.** Anything estimated from data (frequencies, encodings, clusters, scalers, top-k lists) is computed in `fit()` and applied in `transform()`.
- **Batch-independent `transform()`.** A row's features must not depend on which other rows are in `X`.
- **Small, fast, vectorised.** Modules run on every CV fold. Avoid per-row Python loops on large data; prefer numpy and pandas column operations.
- **One coherent family per module**, 2–8 features, with a stated hypothesis.

## Geo (latitude/longitude)

- **Haversine distance** between two points (km), and to fixed landmarks you know from domain knowledge (city centre, port, station, border crossing). Embed landmark coordinates as constants.
- **Bearing** (direction of travel), and its sin/cos.
- **Clusters** fit inside `fit()` on training coordinates; output the cluster id and distance to the centroid.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(x, y)) % 360

class Features:
    def fit(self, X, y):
        self.fill_ = X[["lat", "lon"]].median()
        pts = X[["lat", "lon"]].dropna().to_numpy()
        self.km_ = KMeans(n_clusters=20, n_init=4, random_state=0).fit(pts)
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        pts = X[["lat", "lon"]].fillna(self.fill_).to_numpy()  # fill values learned in fit()
        out["geo_cluster"] = self.km_.predict(pts).astype(str)
        out["geo_cluster_dist"] = self.km_.transform(pts).min(axis=1)
        return out
```

## Datetime

- Only the `--time` column arrives parsed; other date columns arrive as strings: `pd.to_datetime(X[col], errors="coerce")`.
- **Calendar parts:** hour, day of week, day of month, month, day of year, week of year, weekend flag.
- **Cyclical encodings** for periodic parts: `sin(2π·h/24)`, `cos(2π·h/24)`; same for day of week (7), month (12), day of year (365.25).
- **Elapsed time** between two events in the same row (signup → first order, admission → procedure), in sensible units.
- **Time to/from known dates:** holidays, paydays, season starts, policy changes. Hard-code dates for the relevant country and years, or use `pandas.tseries.holiday.USFederalHolidayCalendar` for the US.
- **Local time:** convert UTC to the local time zone when behaviour follows the local clock.
- **Trend:** a numeric timestamp lets trees model level shifts but they cannot extrapolate beyond the training range; prefer calendar parts in temporal splits.

```python
def cyc(values, period):
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)

# in transform():
t = pd.to_datetime(X["event_time"], errors="coerce")
out["hour_sin"], out["hour_cos"] = cyc(t.dt.hour + t.dt.minute / 60, 24)
out["dow_sin"], out["dow_cos"] = cyc(t.dt.dayofweek, 7)
out["event_time"] = t  # replaces the raw string; the proxy expands datetimes into parts
```

## Grouped and temporal aggregates

- **Entity statistics** (per customer, store, device, product): counts, means, spreads of features, and smoothed target means. Compute them in `fit()` from the training fold and map them in `transform()`; unseen entities get the global value.
- **History in temporal data:** only strictly earlier events count. For lags of a row's own features within an entity, prepend the training history kept in `fit()`, sort by time, `groupby(entity).shift(1)`, and return only the rows of `X`. Anything involving the target comes from `fit()` only; `transform()` never sees `y`.
- **Relative position within an entity:** the row's value minus the entity's training mean, or its ratio to it.
- With `--group`, the group column is removed from `X` on purpose: prediction-time entities are new, so per-group statistics cannot exist. Aggregate by other columns instead.

```python
class Features:
    def fit(self, X, y):
        # y is the raw target (labels may be strings). Binary: factorize(sort=True) maps the
        # positive class (last label in sorted order, as the harness uses) to 1. Regression: y as is.
        t = pd.factorize(y, sort=True)[0] if y.nunique() == 2 else y
        df = X[["store"]].assign(t=np.asarray(t, dtype=float), qty=X["qty"])
        g = df.groupby("store")
        self.prior_ = df["t"].mean()
        m = 20.0  # smoothing strength
        self.te_ = (g["t"].sum() + m * self.prior_) / (g["t"].count() + m)
        self.qty_mean_ = g["qty"].mean()
        self.count_ = g.size()
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        out["store_te"] = X["store"].map(self.te_).astype(float).fillna(self.prior_)
        out["store_count"] = X["store"].map(self.count_).astype(float).fillna(0.0)
        out["qty_vs_store"] = X["qty"] / X["store"].map(self.qty_mean_).astype(float)
        return out
```

## Categorical

- **Frequency/count encoding** learned in `fit()`.
- **Crosses** of two categoricals with a joint meaning (`region + "_" + product_type`).
- **Rare-level grouping:** levels seen fewer than k times in training become `"__rare__"`.
- **Structured codes:** split codes into parts (`"A12-B"` → prefix, number, suffix); extract the meaningful prefix of postal codes, product codes or IDs that encode a hierarchy.
- **Ordinal mapping** when the domain defines an order (sizes, grades, severity levels).
- **Target encoding** (multiclass: one column per class) fit in `fit()` with smoothing `(sum + m·prior) / (count + m)`, m ≈ 10–50. It is applied in-sample to the training rows, so weak smoothing overfits and `aga try` flags `target_leak_suspect`. Increase m, or drop the idea. Try encodings last: AutoGluon handles raw categoricals well.

## Numeric

- **Ratios with meaning:** per unit, per person, per area, per hour; price per weight; load per capacity.
- **Differences:** actual vs expected, current vs previous, two measurements of the same quantity.
- **Physics and economics:** speed = distance / time, density = mass / volume, energy ∝ mass·speed², margin = price − cost, utilisation = used / available.
- **Row-wise summaries** over a block of similar columns: sum, mean, max, std, count of zeros, count of missing.
- **Data artefacts:** fractional part or round-number flags (prices ending in .99, rounded estimates).
- Logs and bins help only inside combinations; on their own they rarely pass the keep rule.

## Nested JSON and list-in-a-string columns

These are often the richest columns and the proxy cannot use them raw. Parse them in a module.

- `json.loads` for JSON, `ast.literal_eval` for Python-literal lists and dicts; wrap in try/except and return NaN on failure.
- Summaries: number of items, total and mean of numeric fields, presence flags for the top-k keys or items (choose top-k in `fit()` from the training fold), first/last item, specific nested fields.
- Replace the raw column under its own name (for example with its item count) so the unparsed string stops acting as a high-cardinality category.

```python
import ast
import json
from collections import Counter

def parse(s):
    if not isinstance(s, str):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(s)
            return v if isinstance(v, list) else [v]
        except Exception:
            pass
    return []

class Features:
    def fit(self, X, y):
        counts = Counter(str(i) for s in X["tags"] for i in parse(s))
        self.top_ = [k for k, _ in counts.most_common(15)]
        return self

    def transform(self, X):
        items = X["tags"].map(parse)
        out = pd.DataFrame(index=X.index)
        out["tags"] = items.map(len).astype(float)  # replaces the raw string column
        for k in self.top_:
            out[f"tag_{k}"] = items.map(lambda v, k=k: float(k in map(str, v)))
        return out
```

## Geometry (points, angles, shapes)

- **Distance** between two points in the same coordinate system (Euclidean for planar coordinates, haversine for lat/lon).
- **Angle between vectors:** `arccos(clip(u·v / (|u||v|), −1, 1))`, for example movement direction vs facing direction.
- **Position in a local frame:** coordinates relative to a reference point or axis, signed offset from a centre line, distance to the nearest boundary.
- **Shape descriptors:** area, perimeter, aspect ratio, surface-to-volume ratio, from width/height/depth columns.

```python
def angle_between(ux, uy, vx, vy):
    dot = ux * vx + uy * vy
    norm = np.hypot(ux, uy) * np.hypot(vx, vy)
    return np.arccos(np.clip(dot / np.where(norm == 0, np.nan, norm), -1, 1))
```

## Free text inside tabular data (basics)

Long free-text columns are ignored by the proxy. Cheap summaries help: character and word counts, digit and uppercase ratios, presence of keywords chosen in `fit()` from the most frequent training-fold tokens, or domain keywords you know matter.
