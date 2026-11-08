"""Data profiling and task inference."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api import types as pdt

Task = Literal["binary", "multiclass", "regression"]

_GROUP_HINTS = ("group", "user", "customer", "match", "player", "session", "patient", "store")


def infer_task(y: pd.Series) -> Task:
    y = y.dropna()
    n_unique = y.nunique()
    if pdt.is_bool_dtype(y) or not pdt.is_numeric_dtype(y):
        return "binary" if n_unique == 2 else "multiclass"
    if n_unique == 2:
        return "binary"
    if n_unique <= 20 and bool((y == np.round(y)).all()):
        return "multiclass"
    return "regression"


def _parses_as_datetime(s: pd.Series) -> bool:
    sample = s.dropna()
    if sample.empty:
        return False
    sample = sample.iloc[:500].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


def _looks_like_text(s: pd.Series) -> bool:
    """Long strings, mostly with whitespace or JSON-like: not opaque keys (UUIDs, hashes)."""
    v = s.dropna().astype(str)
    if v.empty or v.str.len().mean() <= 30:
        return False
    texty = v.str.contains(r"\s", regex=True) | v.str.startswith(("{", "["))
    return bool(texty.mean() >= 0.5)


def _kind(s: pd.Series) -> str:
    n = len(s)
    n_unique = s.nunique()
    if n_unique <= 1:
        return "constant"
    is_obj = pdt.is_object_dtype(s) or pdt.is_string_dtype(s)
    if pdt.is_datetime64_any_dtype(s) or (is_obj and _parses_as_datetime(s)):
        return "datetime"
    ratio = n_unique / n if n else 0.0
    if is_obj and ratio > 0.5 and _looks_like_text(s):
        return "text"
    if not pdt.is_float_dtype(s) and ratio >= 0.95:
        return "id_like"
    if is_obj or pdt.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    return "numeric"


def profile(df: pd.DataFrame, target: str, task: Task | None = None) -> dict:
    n = len(df)
    y = df[target]
    task = task or infer_task(y)
    tinfo: dict = {"task": task, "n_unique": int(y.nunique())}
    if task == "regression":
        tinfo.update(mean=float(y.mean()), std=float(y.std()), skew=float(y.skew()))
    else:
        tinfo["class_balance"] = {
            str(k): float(v) for k, v in y.value_counts(normalize=True).items()
        }

    columns = []
    group_candidates, time_candidates = [], []
    for col in df.columns:
        s = df[col]
        kind = _kind(s)
        n_unique = int(s.nunique())
        columns.append(
            {
                "name": col,
                "dtype": str(s.dtype),
                "kind": kind,
                "n_unique": n_unique,
                "missing_frac": float(s.isna().mean()),
            }
        )
        if col == target:
            continue
        if kind == "datetime":
            time_candidates.append(col)
        low = str(col).lower()
        ratio = n_unique / n if n else 0.0
        if (
            not pdt.is_float_dtype(s)
            and 0.001 <= ratio <= 0.5
            and (low.endswith("id") or any(h in low for h in _GROUP_HINTS))
        ):
            group_candidates.append(col)

    return {
        "n_rows": n,
        "n_cols": int(df.shape[1]),
        "target": tinfo,
        "columns": columns,
        "suggestions": {
            "group_candidates": group_candidates,
            "time_candidates": time_candidates,
        },
    }


def format_profile(p: dict) -> str:
    t = p["target"]
    lines = [f"Rows: {p['n_rows']}  Columns: {p['n_cols']}"]
    lines.append(f"Target task: {t['task']} ({t['n_unique']} unique values)")
    if "class_balance" in t:
        bal = ", ".join(f"{k}: {v:.1%}" for k, v in t["class_balance"].items())
        lines.append(f"Class balance: {bal}")
    else:
        lines.append(f"Target mean={t['mean']:.4g} std={t['std']:.4g} skew={t['skew']:.3g}")
    lines.append("")
    lines.append(f"{'column':<24}{'kind':<13}{'dtype':<16}{'unique':>8}{'missing':>9}")
    for c in p["columns"]:
        lines.append(
            f"{str(c['name']):<24}{c['kind']:<13}{c['dtype']:<16}"
            f"{c['n_unique']:>8}{c['missing_frac']:>9.1%}"
        )
    sug = p["suggestions"]
    lines.append("")
    lines.append(f"Group candidates: {', '.join(map(str, sug['group_candidates'])) or '-'}")
    lines.append(f"Time candidates: {', '.join(map(str, sug['time_candidates'])) or '-'}")
    return "\n".join(lines)
