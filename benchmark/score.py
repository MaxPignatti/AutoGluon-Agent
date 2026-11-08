"""Score raw benchmark predictions against the hidden test labels.

Reads ``benchmark/results/raw/<dataset>/<system>/{preds_naive,preds_production}.csv`` and
``meta.json`` (written by run.py) and the labels in ``benchmark/.hidden/<dataset>/test.csv``.
Writes under ``benchmark/results/``:

    results.csv   one row per dataset x system x variant (``abs_diff`` on aga rows: aga - AutoGluon)
    RESULTS.md    one table per variant, with aga's absolute difference and relative error reduction vs AutoGluon
    chart.png     relative error reduction per dataset, production variant

and replaces the block between ``<!-- BENCHMARK:START -->`` and ``<!-- BENCHMARK:END -->`` in the
repository README with the production-variant table.

Every predictions file has one ``prediction`` column (the format ``aga predict`` writes):
the value for regression, P(positive class) for binary tasks.

Usage: python benchmark/score.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SYSTEMS = ("autogluon", "aga")
VARIANTS = ("naive", "production")
COLUMNS = [
    "dataset", "system", "variant", "metric", "score", "abs_diff", "cv_score", "cv_test_gap",
    "wall_minutes", "cost_usd", "status",
]
HIGHER_IS_BETTER = {"roc_auc"}
ERROR_METRICS = {"rmse", "mae", "log_loss"}
README_START, README_END = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"


def relative_error_reduction(metric: str, base: float, ours: float) -> float:
    """Fraction of the baseline's error removed by ``ours`` (negative when ours is worse)."""
    if metric in HIGHER_IS_BETTER:  # AUC-type: the error is 1 - score
        return ((1 - base) - (1 - ours)) / (1 - base)
    if metric in ERROR_METRICS:
        return (base - ours) / base
    raise ValueError(f"unknown metric {metric!r}")


def score_predictions(entry: dict, truth: pd.DataFrame, preds: pd.DataFrame) -> float:
    """Metric value of ``preds['prediction']`` against ``truth[entry['target']]``."""
    from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

    y = truth[entry["target"]]
    p = preds["prediction"].to_numpy(dtype=float)
    if len(p) != len(y):
        raise ValueError(f"{len(p)} predictions for {len(y)} test rows")
    if not np.isfinite(p).all():
        raise ValueError("predictions contain missing or non-finite values")
    metric = entry["metric"]
    if entry["task"] == "binary":
        positive = (y == entry["positive_class"]).to_numpy()
        if metric == "roc_auc":
            return float(roc_auc_score(positive, p))
        if metric == "log_loss":
            return float(log_loss(positive, np.clip(p, 1e-15, 1 - 1e-15), labels=[False, True]))
    elif metric == "rmse":
        return float(math.sqrt(mean_squared_error(y, p)))
    elif metric == "mae":
        return float(mean_absolute_error(y, p))
    raise ValueError(f"unsupported metric {metric!r} for a {entry['task']} task")


def _read_meta(folder: Path) -> dict:
    path = folder / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def score_all(entries: list[dict], raw_dir: Path, hidden_dir: Path) -> list[dict]:
    """One row per dataset x system x variant for every system that has a raw results folder.

    A missing predictions file or ``status: failed`` yields a row with ``status`` failed and
    an empty score, never an exception."""
    rows = []
    for entry in entries:
        name = entry["name"]
        system_dirs = {s: Path(raw_dir) / name / s for s in SYSTEMS if (Path(raw_dir) / name / s).is_dir()}
        if not system_dirs:
            continue
        truth = pd.read_csv(Path(hidden_dir) / name / "test.csv", low_memory=False)
        for system, folder in system_dirs.items():
            meta = _read_meta(folder)
            cv = meta.get("cv_score")
            wall = meta.get("wall_seconds")
            for variant in VARIANTS:
                row = {
                    "dataset": name, "system": system, "variant": variant, "metric": entry["metric"],
                    "score": None, "abs_diff": None, "cv_score": cv, "cv_test_gap": None,
                    "wall_minutes": None if wall is None else wall / 60,
                    "cost_usd": meta.get("total_cost_usd"), "status": "failed",
                }
                preds_path = folder / f"preds_{variant}.csv"
                if meta.get("status") != "failed" and preds_path.is_file():
                    try:
                        row["score"] = score_predictions(entry, truth, pd.read_csv(preds_path))
                        row["status"] = "ok"
                        if cv is not None:
                            row["cv_test_gap"] = row["score"] - cv  # test minus CV, in the metric's units
                    except (ValueError, KeyError) as exc:
                        print(f"{name}/{system}/{variant}: unscorable predictions ({exc})", file=sys.stderr)
                rows.append(row)
    _fill_abs_diff(rows)
    return rows


def _fill_abs_diff(rows: list[dict]) -> None:
    """``abs_diff`` on each scored aga row: aga minus AutoGluon, in the metric's units."""
    ag = {(r["dataset"], r["variant"]): r["score"] for r in rows if r["system"] == "autogluon"}
    for r in rows:
        base = ag.get((r["dataset"], r["variant"]))
        if r["system"] == "aga" and r["score"] is not None and base is not None:
            r["abs_diff"] = r["score"] - base


def replace_block(text: str, body: str) -> str:
    """Replace what lies between the README markers; everything else stays byte-for-byte."""
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError("README has no BENCHMARK:START / BENCHMARK:END markers")
    eol = "\r\n" if "\r\n" in text else "\n"
    block = eol.join([README_START, *body.split("\n"), README_END])
    return pattern.sub(lambda _m: block, text, count=1)


def _fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def comparison(entries: list[dict], rows: list[dict], variant: str) -> list[dict]:
    """Per-dataset side by side of the two systems for one variant."""
    by = {(r["dataset"], r["system"]): r for r in rows if r["variant"] == variant}
    out = []
    for entry in entries:
        ag, aga = by.get((entry["name"], "autogluon")), by.get((entry["name"], "aga"))
        if ag is None and aga is None:
            continue
        reduction = None
        if ag and aga and ag["score"] is not None and aga["score"] is not None:
            reduction = relative_error_reduction(entry["metric"], ag["score"], aga["score"])
        out.append({"entry": entry, "ag": ag, "aga": aga, "reduction": reduction})
    return out


def markdown_table(entries: list[dict], rows: list[dict], variant: str) -> str:
    header = (
        "| Dataset | Task | Metric | AutoGluon (test) | AutoGluon-Agent (test) | Absolute difference "
        "| Relative error reduction "
        "| CV-vs-test gap (AutoGluon / AutoGluon-Agent) | Wall-clock min (AutoGluon / AutoGluon-Agent) "
        "| Claude cost (USD) |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for c in comparison(entries, rows, variant):
        e, ag, aga = c["entry"], c["ag"] or {}, c["aga"] or {}
        score = lambda r: _fmt(r.get("score")) if r.get("status") == "ok" else "failed" if r else "n/a"  # noqa: E731
        diff = "n/a" if aga.get("abs_diff") is None else f"{aga['abs_diff']:+.4f}"
        reduction = "n/a" if c["reduction"] is None else f"{c['reduction'] * 100:+.1f}%"
        lines.append(
            f"| {e['name']} | {e['task']} | {e['metric']} | {score(ag)} | {score(aga)} | {diff} | {reduction} "
            f"| {_fmt(ag.get('cv_test_gap'))} / {_fmt(aga.get('cv_test_gap'))} "
            f"| {_fmt(ag.get('wall_minutes'), 1)} / {_fmt(aga.get('wall_minutes'), 1)} "
            f"| {_fmt(aga.get('cost_usd'), 2)} |"
        )
    if len(lines) == 1:
        lines.append("| _pending_ | | | | | | | | | |")
    return "\n".join(lines)


def write_results_md(path: Path, entries: list[dict], rows: list[dict]) -> None:
    parts = [
        "# Benchmark results\n",
        "Absolute difference is AutoGluon-Agent's test score minus AutoGluon's, in the metric's units "
        "(for `roc_auc` positive is better, for `rmse`, `mae` and `log_loss` negative is better). "
        "Relative error reduction is AutoGluon-Agent against AutoGluon `best_quality`: positive means "
        "AutoGluon-Agent's error is lower. For `roc_auc` the error is 1 - AUC; for `rmse`, `mae` and "
        "`log_loss` it is the metric itself. The CV-vs-test gap is test score minus the system's own "
        "validation score, in the metric's units.\n",
    ]
    notes = {
        "production": "Production variant: leaky columns are blank in the test features, as at prediction time.",
        "naive": "Naive variant: test features still contain the leaky columns.",
    }
    for variant in ("production", "naive"):
        parts += [f"## {variant.capitalize()}\n", notes[variant] + "\n", markdown_table(entries, rows, variant) + "\n"]
    path.write_text("\n".join(parts), encoding="utf-8")


def write_chart(path: Path, entries: list[dict], rows: list[dict]) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [(c["entry"]["name"], c["reduction"] * 100) for c in comparison(entries, rows, "production")
              if c["reduction"] is not None]
    if not points:
        return False
    names, values = zip(*points)
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(names) + 2), 4.5))
    ax.bar(names, values, color=["#2a7f62" if v >= 0 else "#b5483a" for v in values])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Relative error reduction vs AutoGluon (%)")
    ax.set_title("AutoGluon-Agent vs AutoGluon best_quality (production variant)")
    ax.tick_params(axis="x", labelrotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    import yaml

    entries = yaml.safe_load((HERE / "registry.yaml").read_text(encoding="utf-8"))
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    rows = score_all(entries, results / "raw", HERE / ".hidden")
    pd.DataFrame(rows, columns=COLUMNS).to_csv(results / "results.csv", index=False)
    write_results_md(results / "RESULTS.md", entries, rows)
    charted = write_chart(results / "chart.png", entries, rows)
    readme = HERE.parent / "README.md"
    if readme.is_file():
        text = readme.read_bytes().decode("utf-8")  # bytes: keep the file's line endings
        readme.write_bytes(replace_block(text, markdown_table(entries, rows, "production")).encode("utf-8"))
    print(f"Wrote {len(rows)} rows to {results / 'results.csv'}" + ("" if charted else " (no chart: no paired scores)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
