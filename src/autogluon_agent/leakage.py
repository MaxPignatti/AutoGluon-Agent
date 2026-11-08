"""Automated leakage checks. The target column is removed from every frame the checks see
(training y is passed separately), so holdout targets cannot be used; the holdout is read
only by the duplicates and adversarial checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from .proxy import PROXY_PARAMS, _predict, encode_target, prepare_pair

Severity = Literal["critical", "warning", "info"]
_CLASSIFICATION = ("binary", "multiclass")
_SINGLE_PARAMS = dict(PROXY_PARAMS, num_leaves=8, n_estimators=50)
_ADVERSARIAL_PARAMS = dict(PROXY_PARAMS, n_estimators=100)
_NAME_HINTS = ("result", "outcome", "status", "final", "post_", "after_")
_CRITICAL_AUC, _CRITICAL_R2 = 0.95, 0.9
_STRONG_AUC, _STRONG_R2 = 0.75, 0.3
_ID_AUC, _ID_R2 = 0.6, 0.1
_MISSING_AUC, _MISSING_R2 = 0.7, 0.1


@dataclass
class Finding:
    column: str | None
    check: str
    severity: Severity
    evidence: str


def _is_cls(problem) -> bool:
    return problem.task in _CLASSIFICATION


def _chance(problem) -> float:
    return 0.5 if _is_cls(problem) else 0.0


def _thresholds(problem, auc: float, r2: float) -> float:
    return auc if _is_cls(problem) else r2


def _fmt(problem, power: float) -> str:
    return f"AUC {power:.3f}" if _is_cls(problem) else f"R² {power:.3f}"


def _lift(problem, power: float) -> float:
    return max(power - _chance(problem), 0.0) if _is_cls(problem) else power


def _fit_predict(Xtr, ytr, Xva, problem):
    Xtr, Xva = prepare_pair(Xtr, Xva)
    if Xtr.shape[1] == 0:
        return None
    names = [f"f{i}" for i in range(Xtr.shape[1])]
    Xtr, Xva = Xtr.set_axis(names, axis=1), Xva.set_axis(names, axis=1)
    est = LGBMClassifier if _is_cls(problem) else LGBMRegressor
    model = est(**_SINGLE_PARAMS, random_state=problem.seed).fit(Xtr, ytr)
    return _predict(model, Xva, problem)


def _score_oof(y_enc: np.ndarray, oof: np.ndarray, problem) -> float:
    if problem.task == "binary":
        return float(roc_auc_score(y_enc, oof))
    if problem.task == "multiclass":
        labels = list(range(len(problem.labels)))
        return float(roc_auc_score(y_enc, oof, multi_class="ovr", average="macro", labels=labels))
    return max(float(r2_score(y_enc, oof)), 0.0)


def _single_power(X: pd.DataFrame, y_enc: np.ndarray, problem) -> float:
    """Out-of-fold power of ``X`` (one column) alone: 3-fold screening, independent of the run's split."""
    X = X.reset_index(drop=True)
    if _is_cls(problem):
        folds = StratifiedKFold(3, shuffle=True, random_state=problem.seed).split(X, y_enc)
    else:
        folds = KFold(3, shuffle=True, random_state=problem.seed).split(X)
    k = len(problem.labels) if problem.task == "multiclass" else 0
    oof = np.zeros((len(X), k)) if k else np.zeros(len(X))
    for tr, va in folds:
        pred = _fit_predict(X.iloc[tr], y_enc[tr], X.iloc[va], problem)
        if pred is None:
            return _chance(problem)
        oof[va] = pred
    return _score_oof(y_enc, oof, problem)


class Powers:
    """Single-feature powers on the training rows, cached per key."""

    def __init__(self, train: pd.DataFrame, y_enc: np.ndarray, problem):
        self.train, self.y, self.problem = train, y_enc, problem
        self._cache: dict = {}

    def of(self, key, frame: pd.DataFrame) -> float:
        if key not in self._cache:
            self._cache[key] = _single_power(frame, self.y, self.problem)
        return self._cache[key]

    def __call__(self, col: str) -> float:
        return self.of(col, self.train[[col]])


def check_single_feature_power(train, holdout, problem, powers) -> list[Finding]:
    cols = [c for c in train.columns if c != problem.target]
    vals = {c: powers(c) for c in cols}
    crit = _thresholds(problem, _CRITICAL_AUC, _CRITICAL_R2)
    out = [
        Finding(c, "single_feature_power", "critical", f"alone it scores {_fmt(problem, v)} (> {crit})")
        for c, v in vals.items()
        if v > crit
    ]
    ranked = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
    if ranked and ranked[0][1] <= crit:
        top, top_v = ranked[0]
        second = _lift(problem, ranked[1][1]) if len(ranked) > 1 else 0.0
        strong = _thresholds(problem, _STRONG_AUC, _STRONG_R2)
        if top_v >= strong and _lift(problem, top_v) >= 2 * second:
            out.append(
                Finding(top, "single_feature_power", "warning",
                        f"alone it scores {_fmt(problem, top_v)}, at least twice the lift of the next best column")
            )
    return out


def _tokens(low: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-z]+", low) if t]


def check_names(train, holdout, problem, powers) -> list[Finding]:
    out = []
    target = problem.target.lower()
    for c in train.columns:
        low = str(c).lower()
        hit = next((h for h in _NAME_HINTS if h in low), None)
        if hit is None and target and (len(target) >= 4 and target in low or target in _tokens(low)):
            hit = target
        if hit:
            out.append(Finding(c, "names", "warning", f"name contains {hit!r}"))
    return out


def _monotonic(s: pd.Series) -> bool:
    try:
        return bool(s.is_monotonic_increasing or s.is_monotonic_decreasing)
    except TypeError:
        return False


def check_id_like(train, holdout, problem, powers) -> list[Finding]:
    out = []
    floor = _thresholds(problem, _ID_AUC, _ID_R2)
    for c in train.columns:
        s = train[c]
        if c == problem.target or pd.api.types.is_float_dtype(s):
            continue
        ratio = s.nunique(dropna=True) / max(len(s), 1)
        if ratio >= 0.9 or _monotonic(s):
            power = powers(c)
            if power >= floor:
                out.append(
                    Finding(c, "id_like", "warning",
                            f"identifier-like (unique ratio {ratio:.2f}) yet alone it scores {_fmt(problem, power)}")
                )
    return out


def check_duplicates(train, holdout, problem, powers) -> list[Finding]:
    out = []
    cols = [c for c in train.columns if c not in (problem.target, problem.group_col)]
    if cols:
        seen = set(pd.util.hash_pandas_object(train[cols], index=False))
        share = float(pd.util.hash_pandas_object(holdout[cols], index=False).isin(seen).mean())
        if share > 0.01:
            out.append(
                Finding(None, "duplicates", "warning",
                        f"{share:.1%} of holdout rows have an exact feature match in train")
            )
    if problem.split == "grouped":
        overlap = set(train[problem.group_col]) & set(holdout[problem.group_col])
        if overlap:
            out.append(
                Finding(problem.group_col, "duplicates", "critical",
                        f"{len(overlap)} groups appear in both train and holdout")
            )
    return out


def check_post_timestamps(train, holdout, problem, powers) -> list[Finding]:
    if problem.split != "temporal":
        return []
    out = []
    t = pd.to_datetime(train[problem.time_col])
    for c in train.columns:
        if c in (problem.target, problem.time_col) or not pd.api.types.is_datetime64_any_dtype(train[c]):
            continue
        share = float((train[c] > t).mean())
        if share > 0.5:
            out.append(Finding(c, "post_timestamps", "warning",
                               f"later than {problem.time_col} in {share:.0%} of rows"))
    return out


def check_missingness(train, holdout, problem, powers) -> list[Finding]:
    out = []
    floor = _thresholds(problem, _MISSING_AUC, _MISSING_R2)
    for c in train.columns:
        miss = train[c].isna()
        if c == problem.target or miss.mean() < 0.05:
            continue
        power = powers.of(("missing", c), miss.astype(float).to_frame("m"))
        if power > floor:
            out.append(Finding(c, "missingness", "warning",
                               f"missing indicator alone scores {_fmt(problem, power)} ({miss.mean():.0%} missing)"))
    return out


def check_adversarial(train, holdout, problem, powers) -> list[Finding]:
    cols = [c for c in train.columns if c != problem.target]
    X = pd.concat([train[cols], holdout[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(train), dtype=int), np.ones(len(holdout), dtype=int)]
    oof = np.zeros(len(X))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=problem.seed).split(X, y):
        a, b = prepare_pair(X.iloc[tr], X.iloc[va])
        if a.shape[1] == 0:
            return [Finding(None, "adversarial", "info", "skipped: no usable feature columns")]
        names = [f"f{i}" for i in range(a.shape[1])]
        model = LGBMClassifier(**_ADVERSARIAL_PARAMS, random_state=problem.seed)
        model.fit(a.set_axis(names, axis=1), y[tr])
        oof[va] = model.predict_proba(b.set_axis(names, axis=1))[:, 1]
    auc = float(roc_auc_score(y, oof))
    note = " (drift: holdout differs from train)" if auc > 0.7 else ""
    return [Finding(None, "adversarial", "info", f"train-vs-holdout AUC {auc:.3f}{note}")]


_CHECKS = (
    check_single_feature_power,
    check_names,
    check_id_like,
    check_duplicates,
    check_post_timestamps,
    check_missingness,
    check_adversarial,
)


def run_checks(train: pd.DataFrame, holdout: pd.DataFrame, problem, dropped: Iterable[str] = ()) -> list[Finding]:
    """Run every check on the base features (target, group column and dropped columns excluded).

    The target never reaches a check; the group column reaches only ``check_duplicates``."""
    dropped = set(dropped)
    group = problem.group_col
    cols = [c for c in train.columns if c not in dropped or c == group]
    cols = [c for c in cols if c != problem.target]
    feats = [c for c in cols if c != group]
    f_train, f_holdout = train[feats], holdout[[c for c in feats if c in holdout.columns]]
    powers = Powers(f_train, encode_target(train[problem.target], problem), problem)
    findings: list[Finding] = []
    for check in _CHECKS:
        if check is check_duplicates:  # the only check that needs the group column itself
            findings.extend(check(train[cols], holdout[[c for c in cols if c in holdout.columns]], problem, powers))
        else:
            findings.extend(check(f_train, f_holdout, problem, powers))
    return findings
