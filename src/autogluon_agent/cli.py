"""Command-line entry point for ``aga``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autogluon_agent import __version__
from autogluon_agent.errors import BlockedError, AgaError
from autogluon_agent.run import Run, best_label


TRY_DESCRIPTION = """\
Evaluate a feature module with fold-safe CV (active modules + this one) and keep it
only if the mean paired per-fold improvement exceeds one standard error.

Module contract: the file defines `class Features` with
  fit(self, X, y) -> self           # fitted on each training fold only
  transform(self, X) -> DataFrame   # new or replaced columns, same index as X
"""

TEMPORAL_SPLIT_HELP = "; temporal split: 60%% for the fit, 40%% for refit_full on the tuning tail"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aga", description="Leakage-safe harness for tabular ML."
    )
    parser.add_argument("--version", action="version", version=f"aga {__version__}")
    parser.add_argument("--run", metavar="RUN_ID", default=None, help="run id (default: current run)")
    # Subparsers accept --run too; SUPPRESS keeps a value given before the command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", metavar="RUN_ID", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("init", parents=[common], help="start a run from a data file")
    p.add_argument("data", help="data file (.csv, .txt or .parquet)")
    p.add_argument("--target", required=True)
    p.add_argument("--goal", required=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--group", default=None, help="group column (grouped split)")
    p.add_argument("--time", default=None, help="time column (temporal split)")
    p.add_argument("--budget", type=float, default=60.0, help="time budget in minutes")
    p.add_argument(
        "--task", choices=("binary", "multiclass", "regression"), default=None,
        help="override the inferred task type",
    )
    p.add_argument(
        "--fresh-budget", action="store_true",
        help="start a new budget clock instead of inheriting the active run's remaining budget",
    )

    sub.add_parser("profile", parents=[common], help="show the data profile")

    p = sub.add_parser("note", parents=[common], help="add a note to the journal")
    p.add_argument("text")
    p.add_argument("--phase", default=None)

    sub.add_parser("status", parents=[common], help="show run status")

    for name, text in (("drop", "drop a column from modelling"), ("keep", "keep a flagged column")):
        p = sub.add_parser(name, parents=[common], help=text)
        p.add_argument("column")
        p.add_argument("--reason", required=True)

    sub.add_parser("leakage", parents=[common], help="run automated leakage checks")

    sub.add_parser("baseline", parents=[common], help="score a dummy and a LightGBM baseline with CV")

    p = sub.add_parser(
        "try",
        parents=[common],
        help="evaluate a feature module against the current best",
        description=TRY_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("features", metavar="features.py", help="feature module file")
    p.add_argument("--note", default=None, help="what this trial tests")

    p = sub.add_parser("fit", parents=[common], help="final AutoGluon fit on the full training portion")
    p.add_argument(
        "--time-limit", type=float, default=None,
        help=f"seconds (default: (remaining budget - 180) / 2, min 60){TEMPORAL_SPLIT_HELP}",
    )
    p.add_argument("--presets", default="best_quality", help="AutoGluon presets (default: best_quality)")

    p = sub.add_parser(
        "refit", parents=[common],
        help="after 'aga evaluate': refit the production model on all labelled rows (train + holdout)",
    )
    p.add_argument(
        "--time-limit", type=float, default=None,
        help=f"seconds (default: remaining budget - 180, min 60){TEMPORAL_SPLIT_HELP}",
    )
    p.add_argument(
        "--presets", default=None,
        help="AutoGluon presets (default: those of the last 'aga fit'; a different value is recorded as recipe drift)",
    )

    p = sub.add_parser("predict", parents=[common], help="predict on new data with the fitted model")
    p.add_argument("data", help="data file (.csv, .txt or .parquet); no target column needed")
    p.add_argument("--out", required=True, help="output CSV")

    p = sub.add_parser("evaluate", parents=[common], help="score the fitted model once on the locked holdout")
    p.add_argument("--force", action="store_true", help="re-evaluate the holdout (marks it as reused)")
    sub.add_parser("report", parents=[common], help="write REPORT.md from the run's artefacts")
    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    run = Run.create(
        Path.cwd(),
        Path(args.data),
        args.target,
        args.goal,
        args.metric,
        args.group,
        args.time,
        args.budget,
        task=args.task,
        fresh_budget=args.fresh_budget,
    )
    entry = run.journal.entries("init")[0]
    dropped = entry["dropped_missing_target"]
    print(run.describe_profile())
    print()
    print(run.describe_problem())
    if dropped:
        print(f"Dropped {dropped} rows with a missing target.")
    if entry.get("budget_inherited_from"):
        print(
            f"Budget inherited from run {entry['budget_inherited_from']} (the clock keeps running; "
            "use --fresh-budget to start a new one)."
        )
    for line in _date_hints(run):
        print(line)
    print(run.budget_line())
    return 0


def _date_hints(run: Run) -> list[str]:
    """One hint per datetime-kind feature column when the run has no time column (no behaviour change)."""
    import warnings

    import pandas as pd

    p = run.problem
    if p.time_col is not None:
        return []
    cols = [c for c in run.profile_dict()["suggestions"]["time_candidates"] if c not in (p.target, p.group_col)]
    if not cols:
        return []
    train = run.train_df()  # the profile's rows: the holdout stays unexamined
    lines = []
    for col in cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(train[col], errors="coerce").dropna()
        if parsed.empty:
            continue
        lo, hi = _fmt_date(parsed.min()), _fmt_date(parsed.max())
        lines.append(
            f"Hint: column '{col}' holds dates ({lo} to {hi}). "
            f"If predictions are about the future, re-run init with --time {col}."
        )
    return lines


def _fmt_date(ts) -> str:
    return ts.date().isoformat() if ts == ts.normalize() else ts.isoformat(sep=" ")


def _cmd_profile(args: argparse.Namespace) -> int:
    run = Run.load(Path.cwd(), args.run)
    print(run.describe_profile())
    print(run.budget_line())
    return 0


def _cmd_note(args: argparse.Namespace) -> int:
    run = Run.load(Path.cwd(), args.run)
    run.journal.append("note", text=args.text, phase=args.phase)
    print("Note recorded.")
    print(run.budget_line())
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    run = Run.load(Path.cwd(), args.run)
    s = run.state
    print(run.describe_problem())
    print(f"Active feature modules: {', '.join(s.active_modules) or '-'}")
    print(f"Dropped: {', '.join(f'{c} ({r})' for c, r in s.dropped.items()) or '-'}")
    print(f"Kept: {', '.join(f'{c} ({r})' for c, r in s.kept.items()) or '-'}")
    print(f"Unresolved: {', '.join(s.unresolved) or '-'}")
    print(f"Trials: {s.trial_count}  Evaluations: {s.eval_count}  Fitted: {'yes' if s.fitted else 'no'}")
    print(f"Refit: yes ({s.refit_rows} rows)" if s.refit else "Refit: no")
    if s.best_label is not None:
        print(f"Best so far: {s.best_label}")
    print(run.budget_line())
    return 0


def _cmd_drop_keep(args: argparse.Namespace) -> int:
    run = Run.load(Path.cwd(), args.run)
    col = args.column
    if col == run.problem.target:
        raise AgaError(f"{col!r} is the target and cannot be dropped or kept as a feature")
    s = run.state
    known = {c["name"] for c in run.profile_dict()["columns"]}
    if col not in known and col not in s.unresolved:
        print(f"Warning: {col!r} is not a training column nor an unresolved finding.")
    if args.command == "drop":
        if col not in s.dropped:
            s.evaluated = False  # the feature recipe changed since the holdout evaluation
        s.dropped[col] = args.reason
        s.kept.pop(col, None)
    else:
        if col in s.dropped:
            s.evaluated = False  # re-admitting a dropped column changes the feature recipe
        s.kept[col] = args.reason
        s.dropped.pop(col, None)
    if col in s.unresolved:
        s.unresolved.remove(col)
    run.save()
    run.journal.append(args.command, column=col, reason=args.reason)
    print(f"{col}: {'dropped' if args.command == 'drop' else 'kept'} ({args.reason})")
    print(run.budget_line())
    return 0


def _cmd_leakage(args: argparse.Namespace) -> int:
    import json
    from dataclasses import asdict

    from autogluon_agent.leakage import run_checks

    run = Run.load(Path.cwd(), args.run)
    s = run.state
    found = run_checks(run.train_df(), run.holdout_df(), run.problem, dropped=s.dropped)
    order = {"critical": 0, "warning": 1, "info": 2}
    found.sort(key=lambda f: order[f.severity])
    (run.root / "leakage.json").write_text(json.dumps([asdict(f) for f in found], indent=2), encoding="utf-8")
    resolved = set(s.dropped) | set(s.kept)
    s.unresolved = sorted({f.column for f in found if f.severity == "critical" and f.column is not None} - resolved)
    run.save()
    counts = {sev: sum(f.severity == sev for f in found) for sev in order}
    run.journal.append("leakage", counts=counts, unresolved=list(s.unresolved))
    rows = [(f.severity, f.column or "-", f.check, f.evidence) for f in found]
    w1 = max([len(r[1]) for r in rows] + [6])
    w2 = max([len(r[2]) for r in rows] + [5])
    print(f"{'SEVERITY':<9} {'COLUMN':<{w1}} {'CHECK':<{w2}} EVIDENCE")
    for sev, col, check, evidence in rows:
        print(f"{sev:<9} {col:<{w1}} {check:<{w2}} {evidence}")
    if s.unresolved:
        print("BLOCKED: resolve with 'aga drop <col> --reason ...' or 'aga keep <col> --reason ...'")
    print(run.budget_line())
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    from autogluon_agent.metrics import METRICS
    from autogluon_agent.proxy import cv_score, dummy_cv_score
    from autogluon_agent.splits import cv_folds

    run = Run.load(Path.cwd(), args.run)
    run.require_unblocked()
    p = run.problem
    train = run.train_df()
    folds = cv_folds(train, p)
    X, y = run.base_features(train), train[p.target]
    dummy = dummy_cv_score(y, p, folds)
    lgbm = cv_score(X, y, p, folds)
    s = run.state
    s.baseline_scores = {"dummy": dummy.fold_scores, "lightgbm": lgbm.fold_scores}
    # With accepted modules the current best is base features + modules: keep it as the
    # keep/reject reference, otherwise the next try would be credited with their gain.
    # (If dropped columns changed, the next try re-scores the active modules anyway.)
    if not s.active_modules:
        s.best_label = "baseline"
        s.best_scores = list(lgbm.fold_scores)
        s.best_context = sorted(s.dropped)
    run.save()
    run.journal.append(
        "baseline",
        metric=p.metric,
        n_folds=len(folds),
        dummy_mean=dummy.mean,
        dummy_std=dummy.std,
        lightgbm_mean=lgbm.mean,
        lightgbm_std=lgbm.std,
    )
    sign = 1.0 if METRICS[p.metric].greater_is_better else -1.0  # back to the metric's natural sign
    print(f"Metric: {p.metric}  ({len(folds)} CV folds, training rows only)")
    print(f"dummy:    {sign * dummy.mean:.4f} ± {dummy.std:.4f}")
    print(f"lightgbm: {sign * lgbm.mean:.4f} ± {lgbm.std:.4f}")
    if s.active_modules:
        print(f"Reference only: the keep/reject reference stays {s.best_label} "
              f"(accepted modules: {', '.join(s.active_modules)}).")
    print(run.budget_line())
    return 0


def _require_module_files(features_dir: Path, modules: list[str]) -> None:
    missing = [m for m in modules if not (features_dir / m).is_file()]
    if missing:
        raise AgaError(f"accepted feature module file(s) missing from {features_dir}: {', '.join(missing)}")


def _cmd_try(args: argparse.Namespace) -> int:
    import shutil
    import traceback

    from autogluon_agent.errors import FeatureModuleError
    from autogluon_agent.features import accept, make_fold_transform
    from autogluon_agent.metrics import METRICS
    from autogluon_agent.proxy import cv_score
    from autogluon_agent.splits import cv_folds

    run = Run.load(Path.cwd(), args.run)
    run.require_unblocked()
    s, p = run.state, run.problem
    if not s.baseline_scores or s.best_scores is None:
        raise AgaError("Run 'aga baseline' first.")
    src = Path(args.features)
    if not src.is_file():
        raise AgaError(f"feature file not found: {src}")

    # Harness steps run outside the trial's try block: their errors are real errors.
    train = run.train_df()
    folds = cv_folds(train, p)
    X, y = run.base_features(train), train[p.target]
    features_dir = run.root / "features"
    features_dir.mkdir(exist_ok=True)

    if s.best_context != sorted(s.dropped):
        # Columns were dropped/kept since best_scores were computed: re-score the active
        # modules on the current base features so the candidate is compared like for like.
        # An accepted module that now fails (e.g. it reads a dropped column) is deactivated.
        old_mean = sum(s.best_scores) / len(s.best_scores)
        _require_module_files(features_dir, s.active_modules)
        while True:
            transform = make_fold_transform([features_dir / m for m in s.active_modules], p)
            try:
                s.best_scores = list(cv_score(X, y, p, folds, fold_transform=transform).fold_scores)
                break
            except Exception as exc:
                if transform.current is None:  # not a module failure: a real error
                    raise
                print(run.deactivate_module(s.active_modules[transform.current], exc, "rescore"))
        s.best_context = sorted(s.dropped)
        run.save()
        new_mean = sum(s.best_scores) / len(s.best_scores)
        run.journal.append(
            "rescore",
            reason="dropped columns changed since best scores were computed",
            old_mean=old_mean,
            new_mean=new_mean,
            dropped=sorted(s.dropped),
            modules=list(s.active_modules),
        )
        print(f"Re-scored current best on the current feature set (dropped columns changed): {new_mean:.4f} (oriented mean)")

    n = s.trial_count + 1
    name = f"{n:02d}_{src.stem}.py"
    shutil.copyfile(src, features_dir / name)

    accepted, mean_delta, se = False, 0.0, 0.0
    flags: list[str] = []
    new_columns: list[str] = []
    error = reason = None
    new_scores: list[float] = []
    leaks: list[str] = []
    try:
        sources = [features_dir / m for m in s.active_modules] + [features_dir / name]
        transform = make_fold_transform(sources, p)
        res = cv_score(X, y, p, folds, fold_transform=transform)
        new_scores = res.fold_scores
        flags = list(res.flags)
        new_columns = transform.produced[-1]
        accepted, mean_delta, se = accept(new_scores, s.best_scores)
        leaks = [f.split(":", 1)[1] for f in flags if f.startswith("target_leak_suspect")]
        if leaks:
            accepted = False
            reason = "target leakage suspected: " + ", ".join(leaks)
        elif not accepted:
            reason = "improvement not larger than one standard error"
    except Exception as exc:  # a broken module is a rejected trial, never a crash
        accepted = False
        error = str(exc) if isinstance(exc, FeatureModuleError) else traceback.format_exc()
        reason = "module raised an error"

    s.trial_count = n
    if accepted:
        s.active_modules.append(name)
        s.evaluated = False  # the feature recipe changed since the holdout evaluation
        s.best_scores = list(new_scores)
        s.best_context = sorted(s.dropped)
        s.best_label = best_label(s.active_modules)
    run.save()
    sign = 1.0 if METRICS[p.metric].greater_is_better else -1.0  # natural units of the metric
    run.journal.append(
        "trial",
        module=name,
        note=args.note,
        mean_delta=mean_delta,  # oriented: positive is better
        se=se,
        delta_natural=sign * mean_delta,
        accepted=accepted,
        flags=flags,
        new_columns=new_columns,
        reason=reason,
        error=error,
        # the module raised or was flagged target_leak_suspect: a fix-and-retry, not a finished round
        errored=error is not None or bool(leaks),
    )
    if error is not None:
        print(f"REJECTED {name}: {reason}")
        print(error.rstrip())
    else:
        print(f"{'ACCEPTED' if accepted else 'REJECTED'} {name}: {', '.join(new_columns) or 'no columns'}")
        if not accepted:
            print(f"Reason: {reason}")
        print(f"Delta {p.metric}: {sign * mean_delta:+.4f} ± {se:.4f} (SE)")
    print(f"Trials: {s.trial_count}")
    print(_feature_loop_line(run))
    print(run.budget_line())
    return 0


LOOP_MAX_REJECTED_IN_A_ROW = 3
LOOP_MAX_BUDGET_PCT = 40


def _feature_loop_line(run: Run) -> str:
    """Where the feature loop stands against the skill's stop rule (guidance only, not enforced).

    Trials that errored (module raised, contract violation, or target_leak_suspect) are fix-and-retry
    attempts: they neither count toward the rejection streak nor reset it."""
    streak = 0
    for t in reversed(run.journal.entries("trial")):
        if t.get("accepted"):
            break
        if not t.get("errored", t.get("error") is not None):
            streak += 1
    used = run.budget_used_pct()
    if streak >= LOOP_MAX_REJECTED_IN_A_ROW:
        return f"Feature loop: stop condition met ({streak} rejected in a row). Run 'aga fit' next."
    if used >= LOOP_MAX_BUDGET_PCT:
        return f"Feature loop: stop condition met ({used}% of the budget used). Run 'aga fit' next."
    return (
        f"Feature loop: {streak} rejected in a row, {used}% of the budget used. "
        f"Keep going until {LOOP_MAX_REJECTED_IN_A_ROW} rejected in a row or {LOOP_MAX_BUDGET_PCT}% used."
    )


MIN_FIT_SECONDS = 60.0
FIT_RESERVE_SECONDS = 180.0


def _time_limit(run: Run, args: argparse.Namespace, share: float) -> float:
    """``--time-limit`` if given, else ``share`` of (remaining budget - reserve), at least the minimum."""
    if args.time_limit is not None:
        if not args.time_limit > 0:
            raise AgaError("--time-limit must be positive")
        return args.time_limit
    wanted = (run.remaining_seconds() - FIT_RESERVE_SECONDS) * share
    if wanted < MIN_FIT_SECONDS:
        print(
            f"Warning: little budget left; using the {MIN_FIT_SECONDS:.0f} s minimum "
            "time limit, which may overrun the budget."
        )
    return max(MIN_FIT_SECONDS, wanted)


def _print_fit_result(run: Run, res: dict) -> None:
    for note in res["notes"]:
        print(f"Note: {note}")
    print(f"Top models by validation score ({run.problem.metric}, AutoGluon sign):")
    for row in res["leaderboard_top5"]:
        print(f"  {row['model']}: {row['score_val']:.4f}")
    print(run.budget_line())


def _cmd_fit(args: argparse.Namespace) -> int:
    from autogluon_agent.fit import fit_final

    run = Run.load(Path.cwd(), args.run)
    run.require_unblocked()
    time_limit = _time_limit(run, args, share=0.5)  # the other half is left for evaluate and refit
    print(f"Fitting AutoGluon (presets={args.presets}, time limit {time_limit:.0f} s) ...")
    res = fit_final(run, time_limit, args.presets)
    print(f"Fitted in {res['time_used']:.0f} s on {res['n_rows']} training rows.")
    _print_fit_result(run, res)
    return 0


def _cmd_refit(args: argparse.Namespace) -> int:
    from autogluon_agent.fit import fit_presets, refit_final, require_refit_allowed

    run = Run.load(Path.cwd(), args.run)
    run.require_unblocked()
    require_refit_allowed(run)  # before printing a time limit; refit_final checks again
    time_limit = _time_limit(run, args, share=1.0)
    presets = args.presets or fit_presets(run)
    print(f"Refitting AutoGluon on all labelled rows (presets={presets}, time limit {time_limit:.0f} s) ...")
    res = refit_final(run, time_limit, presets)
    print(
        f"Refit in {res['time_used']:.0f} s on {res['n_rows']} labelled rows (train + holdout); "
        "this is now the delivered model."
    )
    _print_fit_result(run, res)
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from autogluon_agent.fit import predict
    from autogluon_agent.io import read_table

    run = Run.load(Path.cwd(), args.run)
    if not run.state.fitted:
        raise AgaError("No fitted model; run 'aga fit' first.")
    src = Path(args.data)
    if not src.is_file():
        raise AgaError(f"data file not found: {src}")
    if not run.state.refit:
        print(
            "Warning: this model was trained on the train portion only (80%); run 'aga refit' to train "
            "the delivered model on all labelled rows."
        )
    out = predict(run, read_table(src))
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    run.journal.append("predict", data=str(src), out=str(dest), n_rows=len(out))
    print(f"Wrote {len(out)} predictions to {dest}")
    print(run.budget_line())
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from autogluon_agent.evaluate import evaluate

    run = Run.load(Path.cwd(), args.run)
    if not run.state.fitted:
        raise AgaError("No fitted model; run 'aga fit' first.")
    res = evaluate(run, force=args.force)
    p = run.problem
    if res["holdout_reused"]:
        print("HOLDOUT REUSED: this estimate is no longer an unbiased measure of generalisation.")
    print(f"Holdout {p.metric}: {res['holdout_score']:.4f}  ({res['n_holdout']} rows)")
    print("Secondary: " + ", ".join(f"{m}={v:.4f}" for m, v in res["secondary"].items()))
    if res["cv_proxy_best"] is not None:
        print(f"CV proxy best: {res['cv_proxy_best']:.4f}")
    if res["ag_score_val"] is not None:
        print(f"AutoGluon validation: {res['ag_score_val']:.4f}  gap (holdout - validation): {res['gap']:+.4f}")
    if "ece" in res:
        brier = f"Brier: {res['brier']:.4f}  " if "brier" in res else ""
        print(f"Calibration: {brier}ECE: {res['ece']:.4f}  (plot: {run.root / 'calibration.png'})")
    if res["importance"]:
        top = ", ".join(f"{r['feature']} ({r['importance']:.4f})" for r in res["importance"][:5])
        print(f"Top features: {top}")
    elif res.get("importance_error"):
        print(f"Feature importance unavailable: {res['importance_error']}")
    print(run.budget_line())
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from autogluon_agent.report import build_report

    run = Run.load(Path.cwd(), args.run)
    build_report(run, Path.cwd())
    print(f"Wrote {Path.cwd() / 'REPORT.md'} and {run.root / 'REPORT.md'}")
    print(run.budget_line())
    return 0


_COMMANDS = {
    "report": _cmd_report,
    "evaluate": _cmd_evaluate,
    "fit": _cmd_fit,
    "refit": _cmd_refit,
    "predict": _cmd_predict,
    "try": _cmd_try,
    "leakage": _cmd_leakage,
    "baseline": _cmd_baseline,
    "init": _cmd_init,
    "profile": _cmd_profile,
    "note": _cmd_note,
    "status": _cmd_status,
    "drop": _cmd_drop_keep,
    "keep": _cmd_drop_keep,
}


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Dispatch to a subcommand."""
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


def _utf8_output() -> None:
    # On Windows a piped stdout defaults to the ANSI code page, so "±" reaches the agent garbled.
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _utf8_output()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 0 for --help/--version and 2 for usage errors; exit 2 is
        # reserved for BlockedError, so any usage error maps to 1.
        return 0 if exc.code in (0, None) else 1
    try:
        return _run(parser, args)
    except BlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    except AgaError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
