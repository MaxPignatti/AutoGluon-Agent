"""Run state: the ``.aga/<run_id>/`` directory, problem definition, state and budget."""

from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from .errors import BlockedError, AgaError
from .io import read_table
from .journal import Journal
from .metrics import METRICS, default_metric
from .profile import format_profile, infer_task, profile
from .splits import holdout_split

MIN_ROWS = 200
MIN_GROUPS = 10
MIN_CLASS_ROWS = 5

_CLASSIFICATION = ("binary", "multiclass")
TASKS = ("binary", "multiclass", "regression")
_ALLOWED_METRICS = {
    "binary": ("roc_auc", "pr_auc", "log_loss", "accuracy", "f1_macro"),
    "multiclass": ("log_loss", "accuracy", "f1_macro"),
    "regression": ("rmse", "mae", "r2"),
}
_NO_RUN = "No run found; run 'aga init' first."


def _py(value):
    """Convert numpy scalars to plain Python so YAML/JSON stay clean."""
    return value.item() if hasattr(value, "item") else value


@dataclass
class Problem:
    target: str
    task: str
    metric: str
    split: str
    group_col: str | None
    time_col: str | None
    goal: str
    positive_class: object | None
    labels: list | None
    seed: int = 42

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> "Problem":
        return cls(**yaml.safe_load(text))


@dataclass
class State:
    budget_minutes: float
    started: float
    deadline: float
    active_modules: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    kept: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    best_label: str | None = None
    best_scores: list[float] | None = None
    best_context: list[str] | None = None  # sorted dropped columns best_scores was computed under
    baseline_scores: dict[str, list[float]] = field(default_factory=dict)
    trial_count: int = 0
    eval_count: int = 0
    fitted: bool = False
    # the current model and feature recipe were scored on the holdout; any fit or recipe change
    # (accepted try, drop, re-admitting keep, deactivation) clears it. None: state.json before the key
    evaluated: bool | None = None
    refit: bool = False  # the production model was refit on all labelled rows after the evaluation
    refit_rows: int | None = None


def _new_run_id(aga_dir: Path) -> str:
    stamp = datetime.now()
    while True:
        run_id = stamp.strftime("%Y%m%d-%H%M%S")
        if not (aga_dir / run_id).exists():
            return run_id
        stamp += timedelta(seconds=1)  # re-init within the same second


def _check_task(task: str, y: pd.Series) -> None:
    """Refuse a ``--task`` override that the target cannot support."""
    if task not in TASKS:
        raise AgaError(f"unknown task {task!r}; use one of: {', '.join(TASKS)}")
    n_unique = y.nunique()
    if task == "binary" and n_unique != 2:
        raise AgaError(f"task 'binary' needs exactly 2 target classes; the target has {n_unique}")
    if task == "multiclass" and n_unique < 3:
        raise AgaError(f"task 'multiclass' needs at least 3 target classes; the target has {n_unique}")
    if task == "regression" and (pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y)):
        raise AgaError(f"task 'regression' needs a numeric target; the target has dtype {y.dtype}")


def _validate_and_prepare(df, target, group, time_col, metric, task=None):
    """Check the data against the init refusals; return (df, n_dropped, task, metric, split).

    ``task`` overrides the inferred task type after checking the target supports it."""
    if target not in df.columns:
        raise AgaError(f"target column {target!r} not found in data")
    if group and time_col:
        raise AgaError("give either --group or --time, not both")
    for flag, col in (("group", group), ("time", time_col)):
        if col and col == target:
            raise AgaError(f"{flag} column cannot be the target column {target!r}")
    for flag, col in (("group", group), ("time", time_col)):
        if col and col not in df.columns:
            raise AgaError(f"{flag} column {col!r} not found in data")

    n_before = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    n_dropped = n_before - len(df)

    y = df[target]
    if y.nunique() < 2:
        raise AgaError(f"target {target!r} has a single class; nothing to predict")
    if len(df) < MIN_ROWS:
        raise AgaError(
            f"need at least {MIN_ROWS} rows after dropping missing-target rows; got {len(df)}"
        )
    if task is None:
        task = infer_task(y)
    else:
        _check_task(task, y)
    if task in _CLASSIFICATION:
        counts = y.value_counts()
        rare = counts[counts < MIN_CLASS_ROWS]
        if len(rare):
            names = ", ".join(f"{k!r} ({v})" for k, v in rare.items())
            raise AgaError(f"every class needs at least {MIN_CLASS_ROWS} rows; too few for {names}")

    if metric is not None:
        if metric not in METRICS:
            raise AgaError(f"unknown metric {metric!r}; available: {', '.join(METRICS)}")
        if metric not in _ALLOWED_METRICS[task]:
            raise AgaError(
                f"metric {metric!r} does not suit a {task} task; "
                f"use one of: {', '.join(_ALLOWED_METRICS[task])}"
            )
    else:
        metric = default_metric(task)

    if group:
        if df[group].isna().any():
            raise AgaError(
                f"group column {group!r} has {int(df[group].isna().sum())} missing values; "
                "fix or drop those rows first"
            )
        n_groups = df[group].nunique()
        if n_groups < MIN_GROUPS:
            raise AgaError(f"grouped split needs at least {MIN_GROUPS} distinct groups; {group!r} has {n_groups}")
    if time_col:
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        if parsed.isna().any():
            raise AgaError(
                f"time column {time_col!r} has {int(parsed.isna().sum())} missing or "
                "unparseable values; fix or drop those rows first"
            )
        df[time_col] = parsed

    split = "grouped" if group else "temporal" if time_col else "iid"
    return df, n_dropped, task, metric, split


class Run:
    def __init__(self, root: Path, problem: Problem, state: State):
        self.root = Path(root)
        self.problem = problem
        self.state = state
        self.journal = Journal(self.root / "journal.jsonl")

    # ---- creation / loading -------------------------------------------------
    @classmethod
    def create(
        cls,
        base: Path,
        data: Path,
        target: str,
        goal: str,
        metric: str | None,
        group: str | None,
        time: str | None,
        budget_minutes: float,
        *,
        task: str | None = None,
        fresh_budget: bool = False,
    ) -> "Run":
        """Create a run. Unless ``fresh_budget``, a re-init while the current run's budget is
        still running inherits that run's budget clock (start, deadline and total)."""
        if not (math.isfinite(budget_minutes) and budget_minutes > 0):
            raise AgaError("budget must be a finite, positive number of minutes")
        base = Path(base)
        df = read_table(Path(data))
        df, n_dropped, task, metric, split = _validate_and_prepare(df, target, group, time, metric, task)

        labels = positive = None
        if task in _CLASSIFICATION:
            labels = [_py(v) for v in sorted(df[target].unique())]
            positive = labels[-1] if task == "binary" else None
        problem = Problem(
            target=target,
            task=task,
            metric=metric,
            split=split,
            group_col=group,
            time_col=time,
            goal=goal,
            positive_class=positive,
            labels=labels,
        )
        train, holdout = holdout_split(df, problem)
        prof = profile(train, target, task)  # training rows only: the holdout stays unexamined

        aga_dir = base / ".aga"
        aga_dir.mkdir(parents=True, exist_ok=True)
        run_id = _new_run_id(aga_dir)
        root = aga_dir / run_id
        root.mkdir()

        now = time_now()
        state = State(budget_minutes=budget_minutes, started=now, deadline=now + budget_minutes * 60)
        inherited_from = None
        if not fresh_budget:
            previous = _current_run(base)
            if previous is not None and previous.state.deadline > now:
                inherited_from = previous.root.name
                state.budget_minutes = previous.state.budget_minutes
                state.started = previous.state.started
                state.deadline = previous.state.deadline
        run = cls(root, problem, state)
        (root / "problem.yaml").write_text(problem.to_yaml(), encoding="utf-8")
        (root / "profile.json").write_text(json.dumps(prof, indent=2, default=str), encoding="utf-8")
        train.to_parquet(root / "train.parquet", index=False)
        holdout.to_parquet(root / "holdout.parquet", index=False)
        run.save()
        run.journal.append(
            "init",
            data=str(data),
            goal=goal,
            n_rows=len(df),
            n_train=len(train),
            n_holdout=len(holdout),
            dropped_missing_target=n_dropped,
            problem=asdict(problem),
            budget_minutes=state.budget_minutes,
            budget_inherited_from=inherited_from,
        )
        (aga_dir / "CURRENT").write_text(run_id, encoding="utf-8")
        return run

    @classmethod
    def load(cls, base: Path, run_id: str | None = None) -> "Run":
        aga_dir = Path(base) / ".aga"
        if run_id is None:
            current = aga_dir / "CURRENT"
            if not current.exists():
                raise AgaError(_NO_RUN)
            run_id = current.read_text(encoding="utf-8").strip()
        root = aga_dir / run_id
        if not (root / "state.json").exists():
            raise AgaError(f"Run {run_id!r} not found in {aga_dir}")
        problem = Problem.from_yaml((root / "problem.yaml").read_text(encoding="utf-8"))
        raw = json.loads((root / "state.json").read_text(encoding="utf-8"))
        known = {f.name for f in fields(State)}
        state = State(**{k: v for k, v in raw.items() if k in known})
        return cls(root, problem, state)

    def save(self) -> None:
        tmp = self.root / "state.json.tmp"
        tmp.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")
        tmp.replace(self.root / "state.json")

    # ---- data ---------------------------------------------------------------
    def train_df(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "train.parquet")

    def holdout_df(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "holdout.parquet")

    def profile_dict(self) -> dict:
        return json.loads((self.root / "profile.json").read_text(encoding="utf-8"))

    def base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = [self.problem.target, *self.state.dropped]
        if self.problem.group_col:
            drop.append(self.problem.group_col)
        return df.drop(columns=drop, errors="ignore")

    # ---- budget / blocking --------------------------------------------------
    def remaining_seconds(self) -> float:
        return max(0.0, self.state.deadline - time_now())

    def budget_used_pct(self, remaining: float | None = None) -> int:
        total = self.state.budget_minutes * 60
        if remaining is None:
            remaining = self.remaining_seconds()
        return min(100, max(0, round((total - remaining) / total * 100)))

    def budget_line(self) -> str:
        remaining = self.remaining_seconds()
        return f"Budget: {remaining / 60:.1f} min left ({self.budget_used_pct(remaining)}% used)"

    def deactivate_module(self, name: str, exc: BaseException, during: str) -> str:
        """Remove a failing accepted module from the active set and journal it.

        Returns the line to print. The module stays on disk and in the trial history."""
        s = self.state
        s.active_modules.remove(name)
        s.best_label = best_label(s.active_modules)
        s.evaluated = False  # the recipe changed
        self.save()
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.journal.append("deactivate", module=name, error=error, during=during)
        return (
            f"DEACTIVATED {name}: this accepted module failed during {during} "
            f"({type(exc).__name__}: {exc}); continuing without it."
        )

    def require_unblocked(self) -> None:
        if self.state.unresolved:
            cols = ", ".join(self.state.unresolved)
            raise BlockedError(
                f"unresolved leakage findings for: {cols}. "
                "Resolve each with 'aga drop <col> --reason ...' or 'aga keep <col> --reason ...'."
            )

    # ---- summaries ----------------------------------------------------------
    def describe_problem(self) -> str:
        p = self.problem
        lines = [
            f"Run: {self.root.name}",
            f"Goal: {p.goal}",
            f"Target: {p.target}  Task: {p.task}  Metric: {p.metric}",
            f"Split: {p.split}"
            + (f" (group={p.group_col})" if p.group_col else "")
            + (f" (time={p.time_col})" if p.time_col else ""),
        ]
        if p.task in _CLASSIFICATION:
            lines.append(f"Classes: {', '.join(map(str, p.labels))}")
            if p.positive_class is not None:
                lines.append(f"Positive class: {p.positive_class}")
        return "\n".join(lines)

    def describe_profile(self) -> str:
        return format_profile(self.profile_dict())


def best_label(modules: list[str]) -> str:
    """Label of the current best: the baseline plus the active modules, in order."""
    return "baseline + " + ", ".join(modules) if modules else "baseline"


def _current_run(base: Path) -> "Run | None":
    """The active run in ``base``, or None when there is none (or it cannot be loaded)."""
    try:
        return Run.load(base)
    except AgaError:
        return None


def time_now() -> float:
    return time.time()
