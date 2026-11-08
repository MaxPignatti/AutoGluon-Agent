"""Feature modules: loading, fold-safe application, contract and leak checks, keep/reject rule.

A feature module is a Python file defining ``class Features`` with
``fit(self, X, y) -> self`` and ``transform(self, X) -> DataFrame`` (new or replaced
columns, same index as ``X``). The harness fits it on training folds only.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import ContractError, FeatureModuleError
from .proxy import encode_target

LEAK_TRAIN_CORR = 0.9  # |spearman| on the training fold at or above this ...
LEAK_DROP = 0.3  # ... that falls by at least this much on the validation fold is suspect


def load_module(path: Path) -> type:
    """Import ``path`` under a unique module name and return its class ``Features``."""
    path = Path(path)
    name = f"aga_feature_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FeatureModuleError(f"cannot import feature module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    cls = getattr(module, "Features", None)
    if not isinstance(cls, type):
        raise FeatureModuleError(f"{path.name} does not define a class 'Features'")
    return cls


def merge(X: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add the columns of ``new`` to ``X``; same-named columns are replaced in place."""
    out = X.copy()
    added = []
    for col in new.columns:
        if col in out.columns:
            out[col] = new[col].array
        else:
            added.append(col)
    if added:
        out = pd.concat([out, new[added].set_axis(out.index, axis=0)], axis=1)
    return out


def accept(candidate: list[float], best: list[float]) -> tuple[bool, float, float]:
    """Keep rule: mean paired improvement must exceed one standard error of the paired deltas."""
    d = np.asarray(candidate, dtype=float) - np.asarray(best, dtype=float)
    if d.size == 0:
        return False, 0.0, 0.0
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
    mean = float(d.mean())
    return bool(mean > se), mean, se


def _numeric(col: pd.Series) -> np.ndarray | None:
    """The column as floats, or None when it is not numeric (such columns skip the leak check)."""
    if pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
        return col.to_numpy(dtype=float, na_value=np.nan)
    if col.dtype == object:
        coerced = pd.to_numeric(col, errors="coerce")
        if coerced.notna().any():
            return coerced.to_numpy(dtype=float)
    return None


def _spearman(values: np.ndarray, y: np.ndarray) -> float:
    if pd.Series(values).nunique() < 2 or pd.Series(y).nunique() < 2:
        return 0.0  # constant input: correlation undefined, counts as 0
    r = pd.Series(values).corr(pd.Series(y), method="spearman")
    return 0.0 if r is None or not np.isfinite(r) else float(r)


def check_contract(out, X: pd.DataFrame, cls: type) -> pd.DataFrame:
    who = cls.__module__
    if not isinstance(out, pd.DataFrame):
        raise ContractError(f"{who}: transform must return a DataFrame, got {type(out).__name__}")
    if len(out) != len(X):
        raise ContractError(f"{who}: transform returned {len(out)} rows for {len(X)} input rows")
    if not out.index.equals(X.index):
        raise ContractError(f"{who}: transform must return the same index as its input")
    return out


class _FoldTransform:
    """Callable ``(Xtr, ytr, Xva, yva=None) -> (Xtr, Xva, flags)`` for ``cv_score``."""

    def __init__(self, sources: list, problem):
        self.sources = list(sources)  # classes, or file paths re-imported fresh for every fold
        self.problem = problem
        self.produced: list[list[str]] = [[] for _ in self.sources]  # columns per module
        self.current: int | None = None  # index of the module being applied; set if it raised

    def __call__(self, Xtr, ytr, Xva, yva=None):
        p = self.problem
        check_leak = p is not None and p.task in ("binary", "regression") and yva is not None
        if check_leak:
            ytr_enc, yva_enc = encode_target(ytr, p), encode_target(yva, p)
        flags: list[str] = []
        for i, source in enumerate(self.sources):
            self.current = i
            cls = source if isinstance(source, type) else load_module(source)
            module = cls()
            fitted = module.fit(Xtr, ytr)  # modules see training labels only, never yva
            if fitted is not None:
                module = fitted
            new_tr = check_contract(module.transform(Xtr), Xtr, cls)
            new_va = check_contract(module.transform(Xva), Xva, cls)
            self.produced[i] = [str(c) for c in new_tr.columns]
            if check_leak:
                for col in new_tr.columns:
                    if col not in new_va.columns:
                        continue
                    a, b = _numeric(new_tr[col]), _numeric(new_va[col])
                    if a is None or b is None:
                        continue
                    r_tr, r_va = abs(_spearman(a, ytr_enc)), abs(_spearman(b, yva_enc))
                    flag = f"target_leak_suspect:{col}"
                    if r_tr >= LEAK_TRAIN_CORR and r_va <= r_tr - LEAK_DROP and flag not in flags:
                        flags.append(flag)
            Xtr, Xva = merge(Xtr, new_tr), merge(Xva, new_va)
        self.current = None
        return Xtr, Xva, flags


def make_fold_transform(classes: list, problem) -> _FoldTransform:
    """Build the ``fold_transform`` for ``cv_score``, applying modules in order.

    Each item is a ``Features`` class or a path to a module file; a path is imported
    afresh for every fold so module-level state cannot leak between folds.
    """
    return _FoldTransform(classes, problem)
