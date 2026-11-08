"""LightGBM proxy evaluator: fast, leakage-safe CV scoring used for baselines and feature trials."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from .metrics import oriented, score

PROXY_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
    n_jobs=-1,
)

_CLASSIFICATION = ("binary", "multiclass")
_TEXT_MEAN_LEN = 30
_TEXT_MIN_UNIQUE = 1000


@dataclass
class CVResult:
    fold_scores: list[float]  # oriented: higher is better
    mean: float
    std: float
    flags: list[str] = field(default_factory=list)


def _is_text_like(s: pd.Series) -> bool:
    if s.dtype != object:
        return False
    vals = s.dropna()
    if vals.empty or vals.nunique() <= _TEXT_MIN_UNIQUE:
        return False
    return vals.astype(str).str.len().mean() > _TEXT_MEAN_LEN


def _datetime_parts(s: pd.Series, name: str) -> dict[str, pd.Series]:
    dt = s.dt
    return {
        f"{name}_year": dt.year.astype(float),
        f"{name}_month": dt.month.astype(float),
        f"{name}_day": dt.day.astype(float),
        f"{name}_dow": dt.dayofweek.astype(float),
        f"{name}_hour": dt.hour.astype(float),
    }


def prepare_pair(Xtr: pd.DataFrame, Xva: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make a train/validation pair LightGBM-ready; every learned choice comes from ``Xtr`` only."""
    tr_cols: dict[str, pd.Series] = {}
    va_cols: dict[str, pd.Series] = {}
    for col in Xtr.columns:
        a, b = Xtr[col], Xva[col]
        if pd.api.types.is_datetime64_any_dtype(a):
            tr_cols.update(_datetime_parts(a, col))
            va_cols.update(_datetime_parts(pd.to_datetime(b), col))
        elif isinstance(a.dtype, pd.CategoricalDtype) or a.dtype == object or pd.api.types.is_bool_dtype(a):
            if _is_text_like(a):
                continue
            cats = pd.Index(a.astype(object).dropna().unique())
            dtype = pd.CategoricalDtype(categories=cats)
            tr_cols[col] = a.astype(object).astype(dtype)
            va_cols[col] = b.astype(object).astype(dtype)
        elif pd.api.types.is_numeric_dtype(a):
            tr_cols[col] = a.astype(float).replace([np.inf, -np.inf], np.nan)
            va_cols[col] = b.astype(float).replace([np.inf, -np.inf], np.nan)
        # other dtypes (timedelta, complex, ...) are dropped
    return pd.DataFrame(tr_cols, index=Xtr.index), pd.DataFrame(va_cols, index=Xva.index)


def encode_target(y: pd.Series, problem) -> np.ndarray:
    if problem.task == "binary":
        return (y == problem.positive_class).to_numpy().astype(int)
    if problem.task == "multiclass":
        index = {label: i for i, label in enumerate(problem.labels)}
        return y.map(index).to_numpy().astype(int)
    return y.to_numpy().astype(float)


def _full_proba(proba: np.ndarray, classes, k: int) -> np.ndarray:
    """Place model columns into a (n, k) matrix; classes absent from the fold get zeros."""
    out = np.zeros((proba.shape[0], k))
    out[:, np.asarray(classes, dtype=int)] = proba
    return out


def _predict(model, Xva, problem):
    if problem.task == "regression":
        return model.predict(Xva)
    k = 2 if problem.task == "binary" else len(problem.labels)
    proba = _full_proba(model.predict_proba(Xva), model.classes_, k)
    return proba[:, 1] if problem.task == "binary" else proba


def _score_labels(problem):
    return list(range(len(problem.labels))) if problem.task == "multiclass" else None


def _result(scores: list[float], flags: list[str] | None = None) -> CVResult:
    arr = np.asarray(scores, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return CVResult(list(map(float, arr)), float(arr.mean()), std, flags or [])


def _safe_names(Xtr: pd.DataFrame, Xva: pd.DataFrame):
    """LightGBM rejects some characters in feature names; use positional names."""
    names = [f"f{i}" for i in range(Xtr.shape[1])]
    return Xtr.set_axis(names, axis=1), Xva.set_axis(names, axis=1)


def cv_score(X: pd.DataFrame, y: pd.Series, problem, folds, fold_transform=None) -> CVResult:
    """Oriented per-fold LightGBM scores. ``folds`` are positional pairs into ``X``."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_enc = encode_target(y, problem)
    labels = _score_labels(problem)
    estimator = LGBMClassifier if problem.task in _CLASSIFICATION else LGBMRegressor
    scores: list[float] = []
    flags: list[str] = []
    for tr_idx, va_idx in folds:
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        if fold_transform is not None:
            Xtr, Xva, fold_flags = fold_transform(Xtr, y.iloc[tr_idx], Xva, yva=y.iloc[va_idx])
            flags.extend(f for f in fold_flags if f not in flags)
        Xtr, Xva = prepare_pair(Xtr, Xva)
        Xtr, Xva = _safe_names(Xtr, Xva)
        model = estimator(**PROXY_PARAMS, random_state=problem.seed)
        model.fit(Xtr, y_enc[tr_idx])
        pred = _predict(model, Xva, problem)
        scores.append(oriented(problem.metric, score(problem.metric, y_enc[va_idx], pred, labels)))
    return _result(scores, flags)


def dummy_cv_score(y: pd.Series, problem, folds) -> CVResult:
    """Oriented per-fold scores of a no-information baseline (class prior / mean)."""
    y = y.reset_index(drop=True)
    y_enc = encode_target(y, problem)
    labels = _score_labels(problem)
    scores: list[float] = []
    for tr_idx, va_idx in folds:
        Xtr = np.zeros((len(tr_idx), 1))
        Xva = np.zeros((len(va_idx), 1))
        if problem.task in _CLASSIFICATION:
            model = DummyClassifier(strategy="prior").fit(Xtr, y_enc[tr_idx])
        else:
            model = DummyRegressor().fit(Xtr, y_enc[tr_idx])
        pred = _predict(model, Xva, problem)
        scores.append(oriented(problem.metric, score(problem.metric, y_enc[va_idx], pred, labels)))
    return _result(scores)
