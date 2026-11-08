"""Metric registry. Predictions: binary -> 1-D P(positive); multiclass -> (n, k) in `labels` order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm

from .errors import AgaError

_EPS = 1e-15


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable
    greater_is_better: bool
    needs_proba: bool
    ag_name: str


def _hard_labels(pred, labels):
    """Hard labels from probabilities: binary threshold 0.5, multiclass argmax over labels."""
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return (pred >= 0.5).astype(int), [0, 1]
    labels = list(range(pred.shape[1])) if labels is None else list(labels)
    return np.asarray(labels)[pred.argmax(axis=1)], labels


def _roc_auc(y_true, pred, labels=None):
    return float(skm.roc_auc_score(y_true, pred))


def _pr_auc(y_true, pred, labels=None):
    return float(skm.average_precision_score(y_true, pred))


def _log_loss(y_true, pred, labels=None):
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 1:
        return float(skm.log_loss(y_true, np.clip(pred, _EPS, 1 - _EPS), labels=[0, 1]))
    if labels is None:
        labels = list(range(pred.shape[1]))
    pred = np.clip(pred, _EPS, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return float(skm.log_loss(y_true, pred, labels=list(labels)))


def _accuracy(y_true, pred, labels=None):
    hat, _ = _hard_labels(pred, labels)
    return float(skm.accuracy_score(y_true, hat))


def _f1_macro(y_true, pred, labels=None):
    hat, labs = _hard_labels(pred, labels)
    return float(skm.f1_score(y_true, hat, labels=labs, average="macro", zero_division=0))


def _rmse(y_true, pred, labels=None):
    return float(np.sqrt(skm.mean_squared_error(y_true, pred)))


def _mae(y_true, pred, labels=None):
    return float(skm.mean_absolute_error(y_true, pred))


def _r2(y_true, pred, labels=None):
    return float(skm.r2_score(y_true, pred))


METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        Metric("roc_auc", _roc_auc, True, True, "roc_auc"),
        Metric("pr_auc", _pr_auc, True, True, "average_precision"),
        Metric("log_loss", _log_loss, False, True, "log_loss"),
        Metric("accuracy", _accuracy, True, True, "accuracy"),
        Metric("f1_macro", _f1_macro, True, True, "f1_macro"),
        Metric("rmse", _rmse, False, False, "root_mean_squared_error"),
        Metric("mae", _mae, False, False, "mean_absolute_error"),
        Metric("r2", _r2, True, False, "r2"),
    )
}

_DEFAULTS = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
_SECONDARY = {
    "binary": ["log_loss", "accuracy"],
    "multiclass": ["accuracy", "f1_macro"],
    "regression": ["mae", "r2"],
}


def default_metric(task: str) -> str:
    try:
        return _DEFAULTS[task]
    except KeyError:
        raise AgaError(f"Unknown task: {task!r}") from None


def secondary_metrics(task: str) -> list[str]:
    try:
        return list(_SECONDARY[task])
    except KeyError:
        raise AgaError(f"Unknown task: {task!r}") from None


def _get(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise AgaError(f"Unknown metric: {name!r}. Available: {', '.join(METRICS)}") from None


def score(name: str, y_true, pred, labels=None) -> float:
    return _get(name).fn(np.asarray(y_true), np.asarray(pred), labels)


def oriented(name: str, value: float) -> float:
    """Return value such that larger is always better."""
    return value if _get(name).greater_is_better else -value
