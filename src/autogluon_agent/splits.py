"""Holdout and cross-validation splits for iid, grouped and temporal data.

Invariants the rest of the harness relies on:
- grouped: no group appears on both sides of any split;
- temporal: every training time is <= every validation time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from .errors import AgaError

N_FOLDS = 5
_CLASSIFICATION = ("binary", "multiclass")
_SPLITS = ("iid", "grouped", "temporal")


def _check(df: pd.DataFrame, problem) -> None:
    if problem.split not in _SPLITS:
        raise AgaError(f"unknown split {problem.split!r}; expected one of {', '.join(_SPLITS)}")
    need = {"grouped": "group_col", "temporal": "time_col"}.get(problem.split)
    if need:
        col = getattr(problem, need)
        if not col:
            raise AgaError(f"split={problem.split!r} requires {need}")
        if col not in df.columns:
            raise AgaError(f"column {col!r} not found in data")


def _time_order(df: pd.DataFrame, time_col: str) -> np.ndarray:
    """Positional order of rows by time (stable, so ties keep input order)."""
    return np.argsort(df[time_col].to_numpy(), kind="stable")


def holdout_split(df: pd.DataFrame, problem, frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, holdout) with reset indices."""
    _check(df, problem)
    if problem.split == "iid":
        stratify = df[problem.target] if problem.task in _CLASSIFICATION else None
        tr, ho = train_test_split(
            df, test_size=frac, random_state=problem.seed, stratify=stratify
        )
    elif problem.split == "grouped":
        splitter = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=problem.seed)
        a, b = next(splitter.split(df, groups=df[problem.group_col]))
        tr, ho = df.iloc[a], df.iloc[b]
    else:
        order = _time_order(df, problem.time_col)
        n_ho = int(np.ceil(len(df) * frac))
        tr, ho = df.iloc[order[: len(df) - n_ho]], df.iloc[order[len(df) - n_ho :]]
    return tr.reset_index(drop=True), ho.reset_index(drop=True)


def cv_folds(
    df: pd.DataFrame, problem, repeats: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, val_idx) pairs of positional indices into ``df``.

    Default repeats: 3 when len(df) < 50_000 and the split is not temporal, else 1.
    Temporal splits are deterministic, so repeats never apply to them.
    """
    _check(df, problem)
    if problem.split == "temporal":
        order = _time_order(df, problem.time_col)
        splitter = TimeSeriesSplit(n_splits=N_FOLDS)
        return [(order[a], order[b]) for a, b in splitter.split(order)]

    if repeats is None:
        repeats = 3 if len(df) < 50_000 else 1

    if problem.split == "grouped":
        groups = df[problem.group_col]
        folds = []
        for r in range(repeats):
            gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=problem.seed + r)
            folds.extend(gkf.split(df, groups=groups))
        return folds

    if problem.task in _CLASSIFICATION:
        rkf = RepeatedStratifiedKFold(
            n_splits=N_FOLDS, n_repeats=repeats, random_state=problem.seed
        )
        return list(rkf.split(df, df[problem.target]))
    rkf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=repeats, random_state=problem.seed)
    return list(rkf.split(df))
