"""REPORT.md generation from the run's artefacts and journal."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from .metrics import METRICS
from .run import Run

NOT_RUN = "Not run."
NOT_WRITTEN = "Not written."
NOTE_PHASES = ("frame", "leakage", "features", "fit", "evaluate", "refit")


def _cell(value) -> str:
    """Make a value safe for a markdown table cell."""
    text = "-" if value is None or value == "" else str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _num(value, fmt: str = ".4f") -> str:
    return "-" if value is None else format(value, fmt)


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _sign(run: Run) -> float:
    """Multiplier from the oriented (higher-is-better) value to the metric's natural unit."""
    return 1.0 if METRICS[run.problem.metric].greater_is_better else -1.0


def _fmt_ts(ts: float | None) -> str:
    return "-" if ts is None else datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")


def _problem(run: Run) -> str:
    p = run.problem
    init = (run.journal.entries("init") or [{}])[0]
    split = p.split + (f" (group={p.group_col})" if p.group_col else "") + (f" (time={p.time_col})" if p.time_col else "")
    rows = f"{init.get('n_rows', '-')} total ({init.get('n_train', '-')} training, {init.get('n_holdout', '-')} holdout)"
    if init.get("dropped_missing_target"):
        rows += f"; {init['dropped_missing_target']} rows with a missing target were dropped"
    return "\n".join(
        [
            f"- Goal: {p.goal}",
            f"- Target: {p.target}",
            f"- Task: {p.task}",
            f"- Metric: {p.metric}",
            f"- Split: {split}",
            f"- Rows: {rows}",
        ]
    )


def _leakage(run: Run) -> str:
    findings = _read_json(run.root / "leakage.json")
    if findings is None:
        parts = [NOT_RUN]
    elif not findings:
        parts = ["No findings."]
    else:
        parts = [
            _table(
                ["Severity", "Column", "Check", "Evidence"],
                [[f.get("severity"), f.get("column"), f.get("check"), f.get("evidence")] for f in findings],
            )
        ]
    s = run.state
    decisions = [["dropped", c, r] for c, r in s.dropped.items()] + [["kept", c, r] for c, r in s.kept.items()]
    if decisions:
        parts.append("Decisions:\n\n" + _table(["Decision", "Column", "Reason"], decisions))
    else:
        parts.append("No drop/keep decisions.")
    if s.unresolved:
        parts.append("Unresolved critical findings: " + ", ".join(s.unresolved))
    return "\n\n".join(parts)


def _baseline(run: Run) -> str:
    entries = run.journal.entries("baseline")
    if not entries:
        return NOT_RUN
    b, sign = entries[-1], _sign(run)
    rows = [
        ["dummy", f"{sign * b['dummy_mean']:.4f} ± {b['dummy_std']:.4f}"],
        ["lightgbm", f"{sign * b['lightgbm_mean']:.4f} ± {b['lightgbm_std']:.4f}"],
    ]
    head = f"{run.problem.metric}, {b.get('n_folds', '-')} CV folds on training rows (mean ± std, natural units):"
    return head + "\n\n" + _table(["Model", run.problem.metric], rows)


def _experiments(run: Run) -> str:
    events = [e for e in run.journal.entries() if e["kind"] in ("trial", "rescore")]
    if not events:
        return NOT_RUN
    sign = _sign(run)
    rows, n, accepted = [], 0, 0
    for e in events:
        if e["kind"] == "rescore":
            note = f"reference re-scored: {e.get('reason')} ({sign * e['old_mean']:.4f} -> {sign * e['new_mean']:.4f})"
            rows.append(["-", "(rescore)", note, "-", "-", "-", "-"])
            continue
        n += 1
        accepted += bool(e.get("accepted"))
        flags = list(e.get("flags") or [])
        if e.get("error"):
            flags.append("error: " + str(e["error"]).strip().splitlines()[-1])
        elif e.get("reason") and not e.get("accepted"):
            flags.append(str(e["reason"]))
        rows.append(
            [
                n,
                e.get("module"),
                e.get("note"),
                f"{e['delta_natural']:+.4f}",
                f"{abs(e['se']):.4f}",
                "yes" if e.get("accepted") else "no",
                "; ".join(flags),
            ]
        )
    table = _table(["Trial", "Module", "Note", f"Δ {run.problem.metric}", "SE", "Accepted", "Flags"], rows)
    return f"{table}\n\n{n} trial{'s' if n != 1 else ''}, {accepted} accepted"


def _final_model(run: Run) -> str:
    fits = run.journal.entries("fit")
    if not fits:
        return NOT_RUN
    f = fits[-1]
    table = _table(
        ["Model", "Validation score (AutoGluon sign)"],
        [[r["model"], f"{r['score_val']:.4f}"] for r in f.get("leaderboard_top5", [])],
    )
    lines = [
        "Top 5 AutoGluon models:",
        "",
        table,
        "",
        f"- Presets: {f.get('presets')}",
        f"- Fit time: {_num(f.get('time_used'), '.0f')} s (limit {_num(f.get('time_limit'), '.0f')} s)",
        f"- Feature modules: {', '.join(f.get('modules') or []) or '-'}",
    ]
    lines += [f"- Note: {n}" for n in f.get("notes") or []]
    lines += ["", _delivered_model(run, f)]
    return "\n".join(lines)


def _delivered_model(run: Run, fit: dict) -> str:
    """Which rows the delivered model was trained on."""
    train_rows = fit.get("n_rows") or (run.journal.entries("init") or [{}])[0].get("n_train", "-")
    s = run.state
    if not s.refit:
        return f"The delivered model was not refit; it is trained on the train portion ({train_rows} rows)."
    head = f"The delivered model was refit on all {s.refit_rows} labelled rows (train + holdout)"
    refits = run.journal.entries("refit")
    if not refits:
        return (
            f"{head}; its journal entry is missing, so its fit time and recipe cannot be compared with the "
            f"holdout estimate below, which describes the evaluated model trained on {train_rows} rows."
        )
    r = refits[-1]
    head += f" in {_num(r.get('time_used'), '.0f')} s"
    from .fit import drift_notes

    drift = drift_notes(r)
    if r.get("fit_presets") and r.get("presets") != r["fit_presets"]:
        drift.append(f"the refit used presets {r.get('presets')} instead of {r['fit_presets']}")
    if not drift:
        return f"{head}; the holdout estimate below describes the same recipe trained on {train_rows} rows."
    return (
        f"{head}; the holdout estimate below describes the evaluated recipe trained on {train_rows} rows, "
        f"and the delivered model differs from it: {'; '.join(drift)}."
    )


def _holdout(run: Run, evaluation: dict | None) -> str:
    if evaluation is None:
        return NOT_RUN
    metric = evaluation.get("metric", run.problem.metric)
    evals = run.journal.entries("evaluate")
    lines = []
    if len(evals) > 1:
        first = evals[0]["holdout_score"]
        lines.append(
            f"- {metric} (first evaluation, unbiased estimate): {first:.4f}\n"
            f"- {metric} (latest evaluation, evaluation {len(evals)}): {evaluation['holdout_score']:.4f}"
        )
    else:
        lines.append(f"- {metric}: {evaluation['holdout_score']:.4f}")
    lines.append(f"- Holdout rows: {evaluation.get('n_holdout', '-')}")
    for m, v in (evaluation.get("secondary") or {}).items():
        lines.append(f"- {m}: {v:.4f}")
    if evaluation.get("cv_proxy_best") is not None:
        lines.append(f"- CV proxy best ({metric}): {evaluation['cv_proxy_best']:.4f}")
    if evaluation.get("ag_score_val") is not None:
        lines.append(f"- AutoGluon validation ({metric}): {evaluation['ag_score_val']:.4f}")
        lines.append(f"- CV-vs-holdout gap (holdout - validation): {evaluation['gap']:+.4f}")
    else:
        lines.append("- CV-vs-holdout gap: not available.")
    out = "\n".join(lines)
    if evaluation.get("holdout_reused") or len(evals) > 1:
        out += (
            "\n\n**Warning: holdout reused.** The holdout was evaluated more than once, so the latest "
            "numbers are no longer an unbiased measure of generalisation; use the first evaluation."
        )
    if run.state.evaluated is False:  # None: state from before the flag existed
        out += (
            "\n\n**Warning: stale holdout result.** A new `aga fit` or a feature-recipe change came "
            "after this evaluation, so these numbers describe an earlier model, not the current one."
        )
    return out


def _calibration(run: Run, evaluation: dict | None, image_link: str) -> str:
    if evaluation is None:
        return NOT_RUN
    if run.problem.task == "regression":
        return "Not applicable (regression task)."
    lines = []
    if evaluation.get("brier") is not None:
        lines.append(f"- Brier score: {evaluation['brier']:.4f}")
    if evaluation.get("ece") is not None:
        lines.append(f"- ECE: {evaluation['ece']:.4f}")
    lines.append("")
    lines.append(f"![Reliability diagram]({image_link})")
    return "\n".join(lines)


def _importance(evaluation: dict | None) -> str:
    if evaluation is None:
        return NOT_RUN
    rows = evaluation.get("importance")
    if not rows:
        return f"Unavailable: {evaluation.get('importance_error') or 'no importance computed'}"
    return _table(
        ["Feature", "Importance", "Std dev"],
        [[r["feature"], f"{r['importance']:.4f}", _num(r.get("stddev"))] for r in rows[:15]],
    )


def _decision_log(run: Run) -> str:
    """Every note, grouped by phase in workflow order; other phases next, untagged notes last."""
    notes = run.journal.entries("note")
    if not notes:
        return "No notes recorded."
    phases = list(NOTE_PHASES)
    phases += [ph for ph in dict.fromkeys(n.get("phase") for n in notes) if ph and ph not in phases]
    parts = []
    for phase in [*phases, None]:
        group = [n for n in notes if (n.get("phase") or None) == phase]
        if group:
            items = "\n".join(f"- {n['ts'].replace('T', ' ')}: {_cell(n.get('text'))}" for n in group)
            parts.append(f"### {phase or 'untagged'}\n\n{items}")
    return "\n\n".join(parts)


def _stats(run: Run) -> str:
    s = run.state
    total = s.budget_minutes * 60
    used = max(0.0, total - run.remaining_seconds())
    entries = run.journal.entries()
    end = entries[-1]["ts"].replace("T", " ") if entries else "-"
    return "\n".join(
        [
            f"- Budget: {s.budget_minutes:g} min, {used / 60:.1f} min used ({min(100, round(used / total * 100))}%)",
            f"- Trials: {s.trial_count}",
            f"- Holdout evaluations: {s.eval_count}",
            f"- Started: {_fmt_ts(s.started)}",
            f"- Last activity: {end}",
        ]
    )


def _render(run: Run, workdir: Path, image_link: str) -> str:
    evaluation = _read_json(run.root / "evaluation.json")
    sections = [
        ("## Problem", _problem(run)),
        ("## Framing", _read_text(Path(workdir) / "FRAMING.md") or NOT_WRITTEN),
        ("## Leakage", _leakage(run)),
        ("## Baseline", _baseline(run)),
        ("## Feature experiments", _experiments(run)),
        ("## Final model", _final_model(run)),
        ("## Holdout results", _holdout(run, evaluation)),
        ("## Calibration", _calibration(run, evaluation, image_link)),
        ("## Feature importance", _importance(evaluation)),
        ("## Analysis", _read_text(Path(workdir) / "NARRATIVE.md") or NOT_WRITTEN),
        ("## Decision log", _decision_log(run)),
        ("## Run stats", _stats(run)),
    ]
    parts = [f"# AutoGluon-Agent Report\n\nRun `{run.root.name}`"]
    parts += [f"{heading}\n\n{body}" for heading, body in sections]
    return "\n\n".join(parts) + "\n"


def _link(target: Path, start: Path) -> str:
    try:
        rel = Path(os.path.relpath(target, start)).as_posix()
    except ValueError:  # different drives on Windows
        rel = target.resolve().as_posix()
    return quote(rel)


def build_report(run: Run, workdir: Path) -> str:
    """Write REPORT.md to the run directory and to ``workdir``; return the workdir copy's text."""
    workdir = Path(workdir)
    image = run.root / "calibration.png"
    text = _render(run, workdir, _link(image, workdir))
    (run.root / "REPORT.md").write_text(_render(run, workdir, "calibration.png"), encoding="utf-8")
    (workdir / "REPORT.md").write_text(text, encoding="utf-8")
    return text
