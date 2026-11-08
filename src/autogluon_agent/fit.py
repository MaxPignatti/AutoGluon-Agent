"""Final AutoGluon fit (train portion), production refit (all labelled rows) and prediction."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .errors import AgaError
from .features import check_contract, load_module, merge
from .metrics import METRICS
from .run import Run

TEMPORAL_TAIL = 0.15  # newest share of the training rows held out as AutoGluon's tuning data
TEMPORAL_FIT_SHARE = 0.6  # temporal: share of the time limit for the fit; refit_full gets the rest
N_FOLDS = 5
FOLD_COL = "_aga_fold"  # fold assignment column handed to AutoGluon's `groups=`
EVALUATED = "models"  # trained on the train portion and scored once on the holdout
PRODUCTION = "models_full"  # refit on all labelled rows after the evaluation


def _models_dir(run: Run) -> Path:
    """The model in use: the production refit when there is one, else the evaluated model."""
    return run.root / (PRODUCTION if run.state.refit else EVALUATED)


def _discard(path: Path) -> None:
    """Delete a directory. It is renamed aside first so its name is free at once, even when
    rmtree leaves files behind (OneDrive, open handles on Windows)."""
    if not path.exists():
        return
    trash = path.with_name(f"{path.name}.trash-{time.time_ns()}")
    try:
        path.rename(trash)
    except OSError:
        trash = path
    shutil.rmtree(trash, ignore_errors=True)


def _swap_in(staging: Path, final: Path) -> None:
    """Make ``staging`` the content of ``final``, replacing what was there."""
    _discard(final)
    try:
        staging.rename(final)
    except OSError:  # leftovers of ``final`` could not be cleared: copy over them instead
        shutil.copytree(staging, final, dirs_exist_ok=True)
        _discard(staging)


def _pickle_by_value(modules: list) -> bytes:
    """Pickle fitted instances so they load in a fresh process (their module file is not importable)."""
    mods = [sys.modules[type(m).__module__] for m in modules if type(m).__module__ in sys.modules]
    for mod in mods:
        cloudpickle.register_pickle_by_value(mod)
    try:
        return cloudpickle.dumps(modules)
    finally:
        for mod in mods:
            cloudpickle.unregister_pickle_by_value(mod)


def group_folds(train: pd.DataFrame, problem, n_splits: int = N_FOLDS) -> np.ndarray:
    """Fold index (0..n_splits-1) per row with every group inside a single fold.

    AutoGluon's ``groups=`` column is the fold assignment itself (one fold per distinct
    value), so raw group ids must be mapped to a few folds first.
    """
    groups = train[problem.group_col].to_numpy()
    if problem.task in ("binary", "multiclass"):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=problem.seed)
        splits = splitter.split(train, train[problem.target], groups)
    else:
        splits = GroupKFold(n_splits=n_splits, shuffle=True, random_state=problem.seed).split(train, groups=groups)
    folds = np.empty(len(train), dtype=int)
    for k, (_, val_idx) in enumerate(splits):
        folds[val_idx] = k
    return folds


def _fit_modules(
    run: Run, X: pd.DataFrame, y: pd.Series, during: str = "fit"
) -> tuple[list, pd.DataFrame, list[str]]:
    """Fit each active module in order on all of (X, y); return the instances, the merged frame and
    the names of the modules used.

    An accepted module that raises (e.g. it reads a column dropped after it was accepted) is
    deactivated and the fit continues without it. During a refit it is only skipped: the evaluated
    recipe keeps it. A missing module file is a harness error."""
    features_dir = run.root / "features"
    missing = [m for m in run.state.active_modules if not (features_dir / m).is_file()]
    if missing:
        raise AgaError(f"accepted feature module file(s) missing from {features_dir}: {', '.join(missing)}")
    instances, used = [], []
    for name in list(run.state.active_modules):
        try:
            cls = load_module(features_dir / name)
            module = cls()
            fitted = module.fit(X.copy(), y)  # copies: a module mutating its input cannot alter X
            if fitted is not None:
                module = fitted
            new = check_contract(module.transform(X.copy()), X, cls)
        except Exception as exc:
            if during == "refit":
                print(f"skipped for the refit: {name} (failed on the full data); the evaluated recipe keeps it")
                print(f"  {type(exc).__name__}: {exc}")
                continue
            print(run.deactivate_module(name, exc, during))
            run.state.best_context = None  # the next 'aga try' re-scores the reduced feature set
            run.save()
            continue
        X = merge(X, new)
        instances.append(module)
        used.append(name)
    return instances, X, used


def transform_with(modules: list, X: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted feature modules in order, exactly as at fit time."""
    for module in modules:
        X = merge(X, check_contract(module.transform(X.copy()), X, type(module)))
    return X


def _refit_full(predictor, n_rows: int, budget: float) -> list[str]:
    """Refit the best model (and its ensemble members) on train + tuning rows and make it the one that
    predicts; return the notes describing what happened. A failure here never loses the tuned model.

    AutoGluon applies a refit ``time_limit`` to each model it refits, not to the whole step, so the
    budget is shared out over the models of the best ensemble."""
    best = predictor.model_best
    notes = []
    try:  # private AutoGluon API: best and its ancestors
        n_models = max(1, len(predictor._trainer.get_minimum_model_set(best)))
    except Exception:
        n_models = 1
        notes.append("could not count ensemble members; refit_full time limit applied per model unchanged")
    error = None
    try:
        refit_map = predictor.refit_full(model="best", time_limit=budget / n_models)
    except Exception as exc:  # the tuned model still works: keep it rather than lose the fit
        refit_map, error = {}, f"{type(exc).__name__}: {exc}"
    full = refit_map.get(best) or (best if best in refit_map.values() else None)
    if full is None:
        return notes + [
            f"the refit_full step failed ({error or 'no refit model was produced'}); the model trained "
            f"without the newest 15% tuning rows ({best}) makes the predictions"
        ]
    predictor.set_model_best(full, save_trainer=True)
    return notes + [
        f"then the best model was refit on all {n_rows} rows including that tail (refit_full, time limit "
        f"{budget:.0f} s, the remaining {1 - TEMPORAL_FIT_SHARE:.0%}) and the refit model ({full}) makes "
        "the predictions"
    ]


def _train(run: Run, data: pd.DataFrame, out: Path, time_limit: float, presets: str, during: str = "fit") -> dict:
    """The AutoGluon recipe shared by the evaluated fit and the production refit.

    Fits the active feature modules and AutoGluon on ``data`` and writes ``features.pkl``,
    ``schema.json`` and the predictor (``ag/``) into ``out``. Temporal: rows sorted by time,
    the newest 15% as tuning data (60% of the time limit), then the best model refit on all rows
    (``refit_full``, the other 40%). Grouped: group-aware folds over ``data``."""
    from autogluon.tabular import TabularPredictor

    p = run.problem
    if p.split == "temporal":
        data = data.iloc[pd.Series(data[p.time_col]).argsort(kind="stable").to_numpy()]
    data = data.reset_index(drop=True)
    y = data[p.target]
    base = run.base_features(data)

    started = time.time()
    modules, X, used_modules = _fit_modules(run, base, y, during)
    out.mkdir(parents=True, exist_ok=True)
    (out / "features.pkl").write_bytes(_pickle_by_value(modules))
    (out / "schema.json").write_text(
        json.dumps({str(c): str(t) for c, t in base.dtypes.items()}, indent=2), encoding="utf-8"
    )

    train_X = X.assign(**{p.target: y.to_numpy()})
    grouped = p.split == "grouped"
    if grouped:
        train_X[FOLD_COL] = group_folds(data, p)  # fold assignment only, never a feature

    ag_path = out / "ag"
    _discard(ag_path)  # a new fit replaces the previous predictor
    predictor = TabularPredictor(
        label=p.target,
        problem_type=p.task,
        eval_metric=METRICS[p.metric].ag_name,
        path=str(ag_path),
        groups=FOLD_COL if grouped else None,
        positive_class=p.positive_class if p.task == "binary" else None,
        verbosity=1,
    )
    notes: list[str] = []
    if p.split == "temporal":
        cut = int(len(train_X) * (1 - TEMPORAL_TAIL))
        fit_limit = time_limit * TEMPORAL_FIT_SHARE
        predictor.fit(
            train_X.iloc[:cut],
            tuning_data=train_X.iloc[cut:],
            presets=presets,
            time_limit=fit_limit,
            num_bag_folds=0,
            num_stack_levels=0,
            dynamic_stacking=False,
        )
        lb = predictor.leaderboard(silent=True)  # before refit_full: _FULL models have no validation score
        notes.append(
            "temporal split: bagging and stacking disabled; the newest 15% of rows were used as tuning data "
            f"(fit time limit {fit_limit:.0f} s, {TEMPORAL_FIT_SHARE:.0%} of {time_limit:.0f} s)"
        )
        notes += _refit_full(predictor, len(train_X), time_limit - fit_limit)
    else:
        predictor.fit(train_X, presets=presets, time_limit=time_limit)
        lb = predictor.leaderboard(silent=True)

    lb = lb[lb["score_val"].notna()]
    top5 = [
        {"model": str(r["model"]), "score_val": float(r["score_val"])}
        for _, r in lb.head(5).iterrows()
    ]
    score_val = top5[0]["score_val"]
    return {
        "score_val": score_val,
        # AutoGluon reports every metric higher-is-better, so error metrics come back negated.
        "score_val_natural": score_val if METRICS[p.metric].greater_is_better else -score_val,
        "leaderboard_top5": top5,
        "time_used": time.time() - started,
        "presets": presets,
        "notes": notes,
        "n_rows": len(train_X),
        "modules": used_modules,
    }


def fit_final(run: Run, time_limit: float, presets: str = "best_quality") -> dict:
    """Fit the evaluated model on the training portion. It replaces any production refit."""
    s = run.state
    # Mark unfitted first: a failed fit must not leave a state that claims a usable model.
    s.fitted = False
    s.evaluated = False  # this model has not seen the holdout yet
    s.refit, s.refit_rows = False, None  # the new evaluated model replaces everything
    run.save()
    _discard(run.root / PRODUCTION)
    res = _train(run, run.train_df(), run.root / EVALUATED, time_limit, presets)
    s.fitted = True
    run.save()
    run.journal.append(
        "fit",
        score_val=res["score_val"],
        score_val_natural=res["score_val_natural"],
        leaderboard_top5=res["leaderboard_top5"],
        time_used=res["time_used"],
        time_limit=time_limit,
        presets=presets,
        modules=list(s.active_modules),
        dropped=sorted(s.dropped),
        notes=res["notes"],
        n_rows=res["n_rows"],
    )
    return res


def require_refit_allowed(run: Run) -> None:
    """A refit needs a fitted model whose holdout evaluation has already happened."""
    s = run.state
    if not s.fitted:
        raise AgaError("No fitted model; run 'aga fit' first.")
    if s.eval_count < 1:
        raise AgaError(
            "run 'aga evaluate' first: refitting before the one-shot evaluation would train on the holdout"
        )
    changed = drift_notes(recipe_drift(run, list(s.active_modules)))
    if changed:  # 'evaluate --force' would re-score the old model, not this recipe
        raise AgaError(
            f"the feature recipe changed since the last fit ({'; '.join(changed)}); "
            "run 'aga fit', then 'aga evaluate --force'"
        )
    evaluated = s.evaluated if s.evaluated is not None else s.eval_count >= 1  # state.json before the key
    if not evaluated:
        raise AgaError(
            "the current model has not been evaluated; run 'aga evaluate --force' first "
            "(the holdout will be marked reused)"
        )


def fit_presets(run: Run) -> str:
    """Presets of the evaluated model: those of the last ``fit`` journal entry."""
    fits = run.journal.entries("fit")
    return (fits[-1].get("presets") if fits else None) or "best_quality"


def refit_final(run: Run, time_limit: float, presets: str | None = None) -> dict:
    """Refit the evaluated recipe on all labelled rows (train + holdout) as the production model.

    ``presets`` defaults to those of the evaluated fit. Trains into a staging directory and swaps it
    in only after success, so a failed or interrupted refit leaves the evaluated model in charge and
    no refit claimed. The evaluated recipe's record (active modules) is never changed: a module that
    fails on the full data is left out of the production model only, and the notes say so."""
    require_refit_allowed(run)
    s = run.state
    evaluated_presets = fit_presets(run)
    presets = presets or evaluated_presets
    had_refit = s.refit
    s.refit, s.refit_rows = False, None
    run.save()
    data = pd.concat([run.train_df(), run.holdout_df()], ignore_index=True)
    staging = run.root / f"{PRODUCTION}.tmp"
    _discard(staging)
    recipe = (list(s.active_modules), s.best_label, s.best_context)
    try:
        res = _train(run, data, staging, time_limit, presets, during="refit")
        _swap_in(staging, run.root / PRODUCTION)
    except BaseException:
        _discard(staging)
        if had_refit:
            print("Refit failed: the earlier production refit is no longer used; predictions now come "
                  "from the evaluated train-only model until 'aga refit' succeeds.", file=sys.stderr)
        raise
    finally:  # the refit never changes the evaluated recipe's record
        s.active_modules, s.best_label, s.best_context = recipe
        run.save()
    drift = recipe_drift(run, res["modules"])
    notes = [f"production model trained on all {res['n_rows']} labelled rows (train + holdout)"]
    notes += drift_notes(drift)
    if presets != evaluated_presets:
        notes.append(f"presets differ from the evaluated fit: refit used {presets}, the evaluated model {evaluated_presets}")
    res["notes"] = notes + res["notes"]
    s.refit, s.refit_rows = True, res["n_rows"]
    run.save()
    run.journal.append(
        "refit",
        n_rows=res["n_rows"],
        time_used=res["time_used"],
        time_limit=time_limit,
        presets=presets,
        fit_presets=evaluated_presets,
        modules=res["modules"],
        **drift,
        notes=res["notes"],
        leaderboard_top5=res["leaderboard_top5"],
    )
    return res


def recipe_drift(run: Run, used: list[str]) -> dict[str, list[str]]:
    """How the refit's recipe differs from the evaluated one, recorded in the last ``fit`` entry.

    ``used`` are the modules the refit applied; the evaluated modules and dropped columns come from
    the fit entry (older entries without them fall back to the current state)."""
    fits = run.journal.entries("fit")
    fit = fits[-1] if fits else {}
    active = list(run.state.active_modules)
    evaluated = fit.get("modules")
    evaluated = active if evaluated is None else evaluated
    dropped_then = fit.get("dropped")
    dropped_now = sorted(run.state.dropped)
    dropped_then = dropped_now if dropped_then is None else dropped_then
    return {
        "skipped_modules": [m for m in active if m not in used],  # failed on the full data
        "added_modules": [m for m in used if m not in evaluated],
        "removed_modules": [m for m in evaluated if m not in active],
        "dropped_after_fit": [c for c in dropped_now if c not in dropped_then],
        "readmitted_after_fit": [c for c in dropped_then if c not in dropped_now],
    }


DRIFT_TEXT = {
    "skipped_modules": "module {}, which failed on the full data, is left out",
    "added_modules": "module {}, accepted after the evaluated fit, is added",
    "removed_modules": "module {}, deactivated after the evaluated fit, is left out",
    "dropped_after_fit": "column {}, dropped after the evaluated fit, is left out",
    "readmitted_after_fit": "column {}, re-admitted after the evaluated fit, is used",
}


def drift_notes(drift: dict) -> list[str]:
    """One phrase per difference between the refit's recipe and the evaluated one."""
    return [DRIFT_TEXT[key].format(item) for key in DRIFT_TEXT for item in drift.get(key) or []]


def load_final(run: Run):
    """Return ``(fitted feature modules, TabularPredictor)`` of the model in use."""
    from autogluon.tabular import TabularPredictor

    models = _models_dir(run)
    if not run.state.fitted or not (models / "features.pkl").exists():
        raise AgaError("No fitted model; run 'aga fit' first.")
    modules = cloudpickle.loads((models / "features.pkl").read_bytes())
    return modules, TabularPredictor.load(str(models / "ag"))


def _align(run: Run, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select the base feature columns the model expects, in training order and dtype."""
    schema = json.loads((_models_dir(run) / "schema.json").read_text(encoding="utf-8"))
    cols: dict[str, pd.Series] = {}
    missing = []
    for col, dtype in schema.items():
        if col not in df.columns:
            cols[col] = pd.Series(np.nan, index=df.index)
            missing.append(col)
            continue
        s = df[col]
        if dtype.startswith("datetime"):
            s = pd.to_datetime(s, errors="coerce")
        elif dtype.startswith(("int", "float", "uint")):
            s = pd.to_numeric(s, errors="coerce")
        cols[col] = s
    return pd.DataFrame(cols, index=df.index), missing


def predict(run: Run, df: pd.DataFrame) -> pd.DataFrame:
    """Predictions for ``df`` (no target needed). Binary: P(positive); multiclass: one column per class."""
    p = run.problem
    modules, predictor = load_final(run)
    base, missing = _align(run, df.reset_index(drop=True))
    if missing:
        print(f"Warning: input lacks columns the model expects (filled with NaN): {', '.join(missing)}")
    X = transform_with(modules, base)
    if p.task == "regression":
        return pd.DataFrame({"prediction": predictor.predict(X).to_numpy()})
    proba = predictor.predict_proba(X)
    by_name = {str(c): c for c in proba.columns}
    if p.task == "binary":
        return pd.DataFrame({"prediction": proba[by_name[str(p.positive_class)]].to_numpy()})
    return pd.DataFrame({str(label): proba[by_name[str(label)]].to_numpy() for label in p.labels})
