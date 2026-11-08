"""One-shot evaluation on the locked holdout, with calibration and permutation importance."""

from __future__ import annotations

import json

import numpy as np

from .errors import HoldoutReusedError, AgaError
from .fit import _align, load_final, predict, transform_with
from .metrics import METRICS, score, secondary_metrics
from .proxy import encode_target
from .run import Run

IMPORTANCE_TOP = 15


def _top_label_view(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, outcome) pairs: P(positive) vs label for binary, top-label confidence vs correctness otherwise."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    if proba.ndim == 1:
        return proba, y_true.astype(float)
    return proba.max(axis=1), (proba.argmax(axis=1) == y_true).astype(float)


def _calibration_bins(conf: np.ndarray, outcome: np.ndarray, n_bins: int) -> list[tuple[float, float, int]]:
    """Per non-empty equal-width bin: (mean confidence, mean outcome, count)."""
    idx = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)  # confidence 1.0 falls in the last bin
    out = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out.append((float(conf[mask].mean()), float(outcome[mask].mean()), int(mask.sum())))
    return out


def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins.

    Binary: ``proba`` is P(positive) and ``y_true`` is 0/1. Multiclass: ``proba`` is (n, k) and
    ``y_true`` holds class indices; top-label confidence is compared with correctness.
    """
    conf, outcome = _top_label_view(y_true, proba)
    n = len(conf)
    return float(sum(c / n * abs(acc - mean_conf) for mean_conf, acc, c in _calibration_bins(conf, outcome, n_bins)))


def _save_calibration_plot(path, y_true: np.ndarray, proba: np.ndarray, multiclass: bool, n_bins: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf, outcome = _top_label_view(y_true, proba)
    bins = _calibration_bins(conf, outcome, n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        ax.plot([b[0] for b in bins], [b[1] for b in bins], "o-", label="model")
        ax.set_xlabel("Top-label confidence" if multiclass else "Predicted probability")
        ax.set_ylabel("Observed accuracy" if multiclass else "Observed frequency")
        ax.set_title("Reliability diagram (holdout)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def _importance(run: Run, holdout) -> tuple[list[dict] | None, str | None]:
    """Permutation importance on the transformed holdout; a failure is recorded, not raised."""
    p = run.problem
    try:
        modules, predictor = load_final(run)
        base, _ = _align(run, holdout.reset_index(drop=True))
        X = transform_with(modules, base)
        X[p.target] = holdout[p.target].to_numpy()  # AutoGluon needs the labels to score permutations
        fi = predictor.feature_importance(X, subsample_size=5000, num_shuffle_sets=3, time_limit=120)
        fi = fi.sort_values("importance", ascending=False).head(IMPORTANCE_TOP)
        rows = []
        for feature, r in fi.iterrows():
            rows.append(
                {
                    "feature": str(feature),
                    "importance": float(r["importance"]),
                    "stddev": float(r["stddev"]) if "stddev" in r else None,
                    "p_value": float(r["p_value"]) if "p_value" in r else None,
                }
            )
        return rows, None
    except Exception as exc:  # the metrics are the essential output; importance is best effort
        return None, f"{type(exc).__name__}: {exc}"


def _natural(metric: str, oriented_value: float) -> float:
    return oriented_value if METRICS[metric].greater_is_better else -oriented_value


def evaluate(run: Run, force: bool = False) -> dict:
    """Score the fitted model once on the holdout. A second evaluation needs ``force=True``;
    after a production refit no evaluation is possible."""
    s, p = run.state, run.problem
    if not s.fitted:
        raise AgaError("No fitted model; run 'aga fit' first.")
    if s.refit:  # not even with force: the holdout is part of the production model's training data
        raise AgaError("the production model was refit on the holdout rows; it cannot be evaluated on them")
    if s.eval_count >= 1 and not force:
        raise HoldoutReusedError(
            f"the holdout was already evaluated {s.eval_count} time(s); scoring it again makes it "
            "a tuning set. Re-run with --force only if you accept a compromised estimate."
        )

    holdout = run.holdout_df()
    y = encode_target(holdout[p.target], p)
    pred = predict(run, holdout)
    if p.task == "multiclass":
        yhat = pred[[str(label) for label in p.labels]].to_numpy()
        labels = list(range(len(p.labels)))
    else:
        yhat, labels = pred["prediction"].to_numpy(), None

    primary = score(p.metric, y, yhat, labels)
    secondary = {m: score(m, y, yhat, labels) for m in secondary_metrics(p.task)}

    cv_best = None
    if s.best_scores:
        cv_best = _natural(p.metric, float(np.mean(s.best_scores)))
    fits = run.journal.entries("fit")
    ag_val = fits[-1].get("score_val_natural") if fits else None
    gap = primary - ag_val if ag_val is not None else None

    result: dict = {
        "metric": p.metric,
        "holdout_score": primary,
        "secondary": secondary,
        "cv_proxy_best": cv_best,
        "ag_score_val": ag_val,
        "gap": gap,
        "n_holdout": int(len(holdout)),
        "holdout_reused": bool(force and s.eval_count >= 1),
    }
    if p.task in ("binary", "multiclass"):
        if p.task == "binary":
            result["brier"] = float(np.mean((yhat - y) ** 2))
        result["ece"] = ece(y, yhat)
        _save_calibration_plot(run.root / "calibration.png", y, yhat, p.task == "multiclass")
    importance, importance_error = _importance(run, holdout)
    result["importance"] = importance
    if importance_error is not None:
        result["importance_error"] = importance_error

    (run.root / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    s.eval_count += 1
    s.evaluated = True
    run.save()
    run.journal.append("evaluate", **result, eval_count=s.eval_count)
    return result
