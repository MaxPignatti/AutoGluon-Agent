# ML Anything Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mlanything` (a Python harness with an `mla` CLI) plus a Claude Code plugin whose skill drives it, and a benchmark comparing ML Anything against AutoGluon alone on six public tabular datasets.

**Architecture:** "Brain over engine." The harness owns the data splits, the locked holdout, scoring, a fast LightGBM proxy for feature trials, leakage checks, an append-only journal and the final AutoGluon fit. The skill (SKILL.md + playbooks) tells Claude how to frame the problem, hunt leakage and engineer domain features *through* the harness, so it cannot score itself on leaked or holdout data.

**Tech Stack:** Python 3.10–3.13, uv, pandas, scikit-learn, LightGBM, AutoGluon-Tabular, cloudpickle, matplotlib, PyYAML, pytest; Claude Code plugin format; `claude -p` headless for the benchmark.

**Spec:** `docs/superpowers/specs/2026-10-29-ml-anything-design.md`

**Additions to the spec made while planning** (each needed to make the spec work):
- `mla predict` — the benchmark must score the agent's final model on a hidden test set.
- `mla keep` — the counterpart of `mla drop`, to resolve a critical leakage finding by keeping the column with a reason (the block in spec §4 needs a way to be lifted).
- `mla note` — how the agent logs reasoning into the journal ("every decision logged", spec §4).
- Benchmark: NYC taxi → Ames housing (no coordinate-bearing taxi data without a Kaggle login); bike sharing drops `casual`/`registered` (they sum to the target) so it stays a clean temporal test; football-xg excludes `statsbomb_xg`, `end_location` and `outcome`; StatsBomb data is downloaded at prepare time, never committed.

## Global Constraints

- Python `>=3.10,<3.14` (AutoGluon does not support 3.14). The developer machine has 3.14 globally; use a uv-managed 3.13 venv.
- Dependencies: `autogluon.tabular>=1.4,<2`, `lightgbm>=4`, `pandas>=2.2`, `scikit-learn>=1.6`, `cloudpickle`, `matplotlib`, `pyyaml`. Dev: `pytest`. No other runtime dependencies (CLI uses stdlib `argparse`).
- Package name `mlanything`, CLI `mla`, plugin `ml-anything`, slash command `/mla`. License Apache-2.0. All content in English.
- Run directory: `.mla/<run_id>/` in the current working directory; `.mla/CURRENT` holds the active run id. `run_id` = `YYYYmmdd-HHMMSS`.
- Holdout fraction 0.2; CV 5 folds; 3×5 repeated CV when training rows < 50,000 (iid and grouped only); seed 42.
- Default metrics: binary `roc_auc`, multiclass `log_loss`, regression `rmse`.
- Keep rule: accept iff mean paired per-fold improvement > one standard error of the paired improvements.
- Leakage thresholds: single-feature AUC > 0.95 or R² > 0.9 = critical.
- `mla fit` default time limit = remaining budget − 180 s, minimum 60 s. Agent stops feature engineering at ~40% budget or after 3 rounds with no accepted module.
- Init refuses: missing target column, single-class target, fewer than 200 rows (after dropping missing-target rows).
- Exit codes: 0 success, 1 user/usage error (`MlaError`), 2 blocked (`BlockedError`: unresolved critical leakage, holdout reuse without `--force`).
- Every command prints a final line `Budget: <m> min left (<p>% used)`.
- All subprocess calls use argument lists (never shell strings); the repo lives under a path with spaces (`OneDrive - Example Ltd`).
- The shell sandbox in this environment blocks network access; installs and dataset downloads need the user's approval to run outside the sandbox.

## Review Focus

1. **Paths with spaces** (OneDrive, Windows): every command must work when data and cwd live under `.../My Data Dir/`. → test in Task 4.
2. **Missing target values**: rows with NaN target are dropped at init with a printed count, never silently scored. → test in Task 4.
3. **Unseen categories** in validation folds, holdout or `mla predict` input must not crash the proxy or prediction. → tests in Tasks 5 and 8.
4. **European CSVs** (semicolon delimiter, comma decimals) and Parquet input must load correctly. → tests in Task 2.
5. **Re-running `mla init`** in the same directory creates a new run, updates CURRENT and preserves the old run. → test in Task 4.

---

## File Structure

```
pyproject.toml                       # package metadata, deps, mla entry point, pytest markers
LICENSE                              # Apache-2.0
README.md
.gitignore                           # .venv, .mla/, benchmark/.cache, benchmark/.hidden, benchmark/workspace
.github/workflows/ci.yml
.claude-plugin/plugin.json
.claude-plugin/marketplace.json
skills/ml-anything/SKILL.md
skills/ml-anything/references/{leakage,features,metrics}.md
commands/mla.md
src/mlanything/__init__.py           # __version__ = "0.1.0"
src/mlanything/errors.py             # MlaError, BlockedError, HoldoutReusedError, FeatureModuleError, ContractError
src/mlanything/io.py                 # read_table(path) -> DataFrame
src/mlanything/metrics.py            # Metric registry, score/oriented
src/mlanything/profile.py            # infer_task, profile, format_profile
src/mlanything/splits.py             # holdout_split, cv_folds
src/mlanything/journal.py            # Journal (append-only jsonl)
src/mlanything/run.py                # Problem, State, Run (run dir, budget)
src/mlanything/proxy.py              # matrix prep, LightGBM CV, dummy CV
src/mlanything/features.py           # module loading, fold application, contract checks, accept rule
src/mlanything/leakage.py            # Finding + 7 checks
src/mlanything/fit.py                # final AutoGluon fit, load, predict
src/mlanything/evaluate.py           # holdout evaluation, calibration, importance
src/mlanything/report.py             # REPORT.md builder
src/mlanything/cli.py                # argparse commands
tests/conftest.py                    # synthetic dataset factories
tests/test_*.py                      # one per module + test_e2e.py + test_plugin.py
benchmark/registry.yaml
benchmark/datasets.py                # one builder per dataset
benchmark/prepare.py
benchmark/run.py
benchmark/score.py
benchmark/results/
examples/football-xg/
```

---

### Task 1: Project scaffold and environment

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/mlanything/__init__.py`, `src/mlanything/errors.py`, `src/mlanything/cli.py` (skeleton), `tests/conftest.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `mlanything.__version__ = "0.1.0"`; `errors.MlaError(Exception)` (exit 1), `errors.BlockedError(MlaError)` (exit 2), `errors.HoldoutReusedError(BlockedError)`, `errors.FeatureModuleError(MlaError)`, `errors.ContractError(FeatureModuleError)`; `cli.main(argv: list[str] | None = None) -> int` — catches `BlockedError` → prints message, returns 2; `MlaError` → returns 1.
- Produces (fixtures in `tests/conftest.py`): `make_binary(n=3000, seed=0) -> DataFrame` with columns `f_signal1`, `f_signal2` (informative), `f_noise1..3`, `cat` (object, 5 levels, mildly informative), `group_id` (int, ~100 groups), `ts` (datetime, increasing), `leak` (= target with 2% label noise), `target` (0/1); `make_regression(n=3000, seed=0) -> DataFrame` with `x1`, `x2`, `lat`, `lon`, `noise1`, `target` where target depends on `x1*x2` and on haversine distance from (`lat`,`lon`) to a fixed point; `make_multiclass(n=3000, seed=0) -> DataFrame` with 3 classes `"a","b","c"`.

- [ ] **Step 1: Install uv and create the 3.13 environment**

Run: `python -m pip install uv` then `uv python install 3.13` then `uv venv --python 3.13`
Expected: `.venv/` created; `.venv/Scripts/python --version` prints `Python 3.13.x`.

- [ ] **Step 2: Write `pyproject.toml`**

hatchling build backend, `src/` layout, `requires-python = ">=3.10,<3.14"`, the dependencies from Global Constraints, `[project.scripts] mla = "mlanything.cli:main"`, `[dependency-groups] dev = ["pytest>=8"]`, pytest config with `markers = ["slow: runs AutoGluon"]` and `testpaths = ["tests"]`. `.gitignore` per File Structure.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_cli.py
from mlanything import __version__
from mlanything.cli import main

def test_version(capsys):
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out
```

- [ ] **Step 4: Run `uv sync` then the test to verify it fails**

Run: `uv sync && uv run pytest tests/test_cli.py -v`
Expected: FAIL (ImportError or missing `main`). Also confirms AutoGluon installs on Windows / Python 3.13; record the resolved `autogluon.tabular` version in the commit message.

- [ ] **Step 5: Implement `__init__.py`, `errors.py`, `cli.main` skeleton with `--version`, and the three fixture factories**

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src tests
git commit -m "chore: scaffold mlanything package and test fixtures"
```

---

### Task 2: Metrics registry and data profiling

**Files:**
- Create: `src/mlanything/metrics.py`, `src/mlanything/profile.py`, `src/mlanything/io.py`
- Test: `tests/test_metrics.py`, `tests/test_profile.py`

**Interfaces:**
- Produces: `Metric` frozen dataclass `(name: str, fn: Callable, greater_is_better: bool, needs_proba: bool, ag_name: str)`; `METRICS: dict[str, Metric]` with keys and AutoGluon names: `roc_auc→"roc_auc"`, `pr_auc→"average_precision"`, `log_loss→"log_loss"`, `accuracy→"accuracy"`, `f1_macro→"f1_macro"`, `rmse→"root_mean_squared_error"`, `mae→"mean_absolute_error"`, `r2→"r2"`.
- Produces: `default_metric(task: str) -> str`; `secondary_metrics(task: str) -> list[str]` (binary `["log_loss","accuracy"]` — Brier is computed in evaluate; multiclass `["accuracy","f1_macro"]`; regression `["mae","r2"]`); `score(name: str, y_true, pred, labels=None) -> float`; `oriented(name: str, value: float) -> float` (negates when `greater_is_better` is False).
- Prediction format convention used everywhere: binary → 1-D probability of the positive class; multiclass → `(n, k)` probability matrix with columns in `labels` order; regression → 1-D values. Hard-label metrics (`accuracy`, `f1_macro`) derive labels from probabilities inside `score`.
- Produces: `infer_task(y: Series) -> Literal["binary","multiclass","regression"]` — object/bool/category: 2 unique → binary else multiclass; numeric: 2 unique → binary; integer-valued with ≤ 20 unique → multiclass; else regression.
- Produces: `profile(df: DataFrame, target: str) -> dict` with keys `n_rows`, `n_cols`, `target` (`task`, `n_unique`, and `class_balance` dict or `mean/std/skew`), `columns` (list of `{name, dtype, kind, n_unique, missing_frac}`), `suggestions` (`group_candidates`, `time_candidates`). `kind` ∈ `numeric|categorical|datetime|text|id_like|constant`: constant if ≤ 1 unique; datetime if datetime dtype or ≥ 90% of a 500-row sample of an object column parses; id_like if non-float and unique ratio ≥ 0.95; text if object, mean length > 30 and unique ratio > 0.5; categorical if object/bool/category; else numeric. Group candidates: non-float, non-target columns with unique ratio in [0.001, 0.5] whose lowercase name ends with `id` or contains one of `group, user, customer, match, player, session, patient, store`. Time candidates: datetime-kind columns.
- Produces: `format_profile(p: dict) -> str`.
- Produces: `io.read_table(path: Path) -> DataFrame` — `.parquet` via `read_parquet`; `.csv`/`.txt` via `read_csv(sep=None, engine="python")`; if the result has comma-decimal strings in otherwise-numeric columns (semicolon file), re-read with `sep=";", decimal=","`. Other extensions → `MlaError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import numpy as np
from mlanything.metrics import METRICS, score, oriented, default_metric

def test_defaults():
    assert default_metric("binary") == "roc_auc"
    assert default_metric("multiclass") == "log_loss"
    assert default_metric("regression") == "rmse"

def test_rmse_oriented_is_negative():
    v = score("rmse", np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    assert np.isclose(v, np.sqrt(2.0))
    assert oriented("rmse", v) == -v

def test_accuracy_from_binary_proba():
    assert score("accuracy", np.array([0, 1, 1]), np.array([0.2, 0.9, 0.4])) == 2 / 3

def test_multiclass_log_loss_with_missing_class_in_fold():
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    assert score("log_loss", np.array(["a", "b"]), p, labels=["a", "b", "c"]) > 0

def test_ag_names():
    assert METRICS["rmse"].ag_name == "root_mean_squared_error"
    assert METRICS["pr_auc"].ag_name == "average_precision"
```

```python
# tests/test_profile.py
import pandas as pd
from mlanything.profile import infer_task, profile
from mlanything.io import read_table

def test_infer_task(make_binary, make_regression, make_multiclass):
    assert infer_task(make_binary()["target"]) == "binary"
    assert infer_task(make_regression()["target"]) == "regression"
    assert infer_task(make_multiclass()["target"]) == "multiclass"

def test_profile_kinds_and_suggestions(make_binary):
    df = make_binary()
    df["row_id"] = range(len(df))
    p = profile(df, "target")
    kinds = {c["name"]: c["kind"] for c in p["columns"]}
    assert kinds["ts"] == "datetime"
    assert kinds["row_id"] == "id_like"
    assert kinds["cat"] == "categorical"
    assert "group_id" in p["suggestions"]["group_candidates"]
    assert "ts" in p["suggestions"]["time_candidates"]

def test_read_semicolon_comma_decimal(tmp_path):
    f = tmp_path / "eu.csv"
    f.write_text("a;b\n1,5;x\n2,25;y\n", encoding="utf-8")
    df = read_table(f)
    assert list(df.columns) == ["a", "b"] and df["a"].tolist() == [1.5, 2.25]

def test_read_parquet(tmp_path, make_binary):
    f = tmp_path / "d.parquet"; make_binary().to_parquet(f)
    assert read_table(f).shape == make_binary().shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py tests/test_profile.py -v`
Expected: FAIL (modules not found)

- [ ] **Step 3: Implement `metrics.py`, `profile.py`, `io.py` per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py tests/test_profile.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlanything/metrics.py src/mlanything/profile.py src/mlanything/io.py tests/test_metrics.py tests/test_profile.py
git commit -m "feat: metric registry, task inference, data profiling and table reader"
```

---

### Task 3: Holdout and CV splits

**Files:**
- Create: `src/mlanything/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `Problem` fields `task`, `split`, `group_col`, `time_col`, `target`, `seed` (defined in Task 4; tests here build a `types.SimpleNamespace` with those attributes).
- Produces: `holdout_split(df: DataFrame, problem, frac: float = 0.2) -> tuple[DataFrame, DataFrame]` — iid: `train_test_split` stratified for classification; grouped: `GroupShuffleSplit(test_size=frac)`; temporal: sort by `time_col`, last `frac` of rows. Returns `(train, holdout)` with reset indices.
- Produces: `cv_folds(df: DataFrame, problem, repeats: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]` — positional indices into `df` as given. `repeats` default: 3 if `len(df) < 50_000` and split ≠ temporal, else 1. iid: `RepeatedStratifiedKFold`/`RepeatedKFold(n_splits=5)`; grouped: `GroupKFold(n_splits=5, shuffle=True, random_state=seed + r)` per repeat; temporal: `TimeSeriesSplit(n_splits=5)` over rows ordered by `time_col`, mapped back to original positions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_splits.py
from types import SimpleNamespace as NS
from mlanything.splits import holdout_split, cv_folds

def P(**kw):
    base = dict(task="binary", split="iid", group_col=None, time_col=None, target="target", seed=42)
    return NS(**{**base, **kw})

def test_iid_holdout_is_20pct_stratified(make_binary):
    df = make_binary()
    tr, ho = holdout_split(df, P())
    assert len(ho) == 600 and abs(tr.target.mean() - ho.target.mean()) < 0.02

def test_grouped_never_shares_groups(make_binary):
    df = make_binary()
    p = P(split="grouped", group_col="group_id")
    tr, ho = holdout_split(df, p)
    assert not set(tr.group_id) & set(ho.group_id)
    for a, b in cv_folds(tr, p):
        assert not set(tr.group_id.iloc[a]) & set(tr.group_id.iloc[b])

def test_temporal_holdout_is_latest_and_folds_look_forward(make_binary):
    df = make_binary().sample(frac=1, random_state=1)  # shuffled input
    p = P(split="temporal", time_col="ts")
    tr, ho = holdout_split(df, p)
    assert tr.ts.max() <= ho.ts.min()
    folds = cv_folds(tr, p)
    assert len(folds) == 5
    for a, b in folds:
        assert tr.ts.iloc[a].max() <= tr.ts.iloc[b].min()

def test_repeats_default(make_binary):
    assert len(cv_folds(make_binary(), P())) == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_splits.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `splits.py` per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_splits.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlanything/splits.py tests/test_splits.py
git commit -m "feat: holdout and CV splits for iid, grouped and temporal data"
```

---

### Task 4: Run state, journal, budget and the bookkeeping commands

**Files:**
- Create: `src/mlanything/journal.py`, `src/mlanything/run.py`
- Modify: `src/mlanything/cli.py`
- Test: `tests/test_run.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `read_table`, `infer_task`, `profile`, `format_profile`, `default_metric`, `METRICS`, `holdout_split`.
- Produces: `Journal(path: Path)` with `append(kind: str, **payload) -> dict` (adds ISO `ts`, writes one JSON line, flushes) and `entries(kind: str | None = None) -> list[dict]`. Kinds used across the plan: `init, note, leakage, drop, keep, baseline, trial, fit, evaluate, report`.
- Produces: `Problem` dataclass `(target, task, metric, split, group_col, time_col, goal, positive_class, labels: list | None, seed=42)` with `to_yaml()/from_yaml()`. `positive_class` = last of the sorted unique labels for binary; `labels` = sorted unique labels for classification.
- Produces: `State` dataclass `(budget_minutes, started: float, deadline: float, active_modules: list[str], dropped: dict[str, str], kept: dict[str, str], unresolved: list[str], best_label: str | None, best_scores: list[float] | None, baseline_scores: dict[str, list[float]], trial_count: int, eval_count: int, fitted: bool)` persisted as `state.json`.
- Produces: `Run` with attributes `root: Path`, `problem`, `state`, `journal`; `Run.create(base: Path, data: Path, target: str, goal: str, metric: str | None, group: str | None, time: str | None, budget_minutes: float) -> Run`; `Run.load(base: Path, run_id: str | None = None) -> Run`; `save()`; `train_df()`; `holdout_df()`; `base_features(df) -> DataFrame` (drops target, `dropped` columns and `group_col`); `remaining_seconds() -> float`; `budget_line() -> str`; `require_unblocked()` raises `BlockedError` listing `unresolved` columns.
- `Run.create` behaviour: reads data; errors (`MlaError`) on missing target / group / time column; drops NaN-target rows and records the count; errors on single-class target or < 200 rows; parses `time_col` to datetime; computes problem; writes `problem.yaml`, `profile.json`, `train.parquet`, `holdout.parquet`, `state.json`, journal `init` entry; updates `.mla/CURRENT`.
- CLI commands added: `mla init <data> --target T --goal G [--metric M] [--group C] [--time C] [--budget 60]` (prints formatted profile + problem); `mla profile`; `mla note <text> [--phase P]`; `mla status`; `mla drop <col> --reason R`; `mla keep <col> --reason R`. `drop`/`keep` remove the column from `state.unresolved`. Global option `--run RUN_ID`. Every command ends with `run.budget_line()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run.py
import json, pytest
from mlanything.run import Run
from mlanything.errors import MlaError, BlockedError

def write(df, d, name="data.csv"):
    d.mkdir(parents=True, exist_ok=True); p = d / name; df.to_csv(p, index=False); return p

def test_create_in_dir_with_spaces(tmp_path, make_binary):
    base = tmp_path / "My Data Dir"
    data = write(make_binary(), base)
    run = Run.create(base, data, "target", "predict target", None, None, None, 60)
    assert run.problem.task == "binary" and run.problem.metric == "roc_auc"
    assert len(run.train_df()) == 2400 and len(run.holdout_df()) == 600
    assert (base / ".mla" / "CURRENT").read_text().strip() == run.root.name

def test_missing_target_rows_dropped_and_logged(tmp_path, make_binary):
    df = make_binary(); df.loc[:9, "target"] = None
    run = Run.create(tmp_path, write(df, tmp_path), "target", "g", None, None, None, 60)
    assert run.journal.entries("init")[0]["dropped_missing_target"] == 10

@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.drop(columns="target"), "target"),
    (lambda d: d.assign(target=1), "single class"),
    (lambda d: d.head(150), "200"),
])
def test_init_refuses(tmp_path, make_binary, mutate, msg):
    with pytest.raises(MlaError, match=msg):
        Run.create(tmp_path, write(mutate(make_binary()), tmp_path), "target", "g", None, None, None, 60)

def test_reinit_preserves_old_run(tmp_path, make_binary):
    data = write(make_binary(), tmp_path)
    r1 = Run.create(tmp_path, data, "target", "g", None, None, None, 60)
    r2 = Run.create(tmp_path, data, "target", "g", None, "group_id", None, 60)
    assert r1.root.exists() and r1.root != r2.root
    assert Run.load(tmp_path).root == r2.root

def test_blocked_until_resolved(tmp_path, make_binary):
    run = Run.create(tmp_path, write(make_binary(), tmp_path), "target", "g", None, None, None, 60)
    run.state.unresolved = ["leak"]; run.save()
    with pytest.raises(BlockedError, match="leak"):
        Run.load(tmp_path).require_unblocked()
```

Add to `tests/test_cli.py`: `test_cli_init_drop_status` — runs `main(["init", str(data), "--target", "target", "--goal", "g"])` with `monkeypatch.chdir(tmp_path)`, then `main(["drop", "leak", "--reason", "post-outcome"])`, then `main(["status"])`; asserts exit codes 0, that `"leak"` appears in `state.json`'s `dropped`, and that the last line of stdout starts with `"Budget:"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run.py tests/test_cli.py -v`
Expected: FAIL (modules / commands missing)

- [ ] **Step 3: Implement `journal.py`, `run.py` and the CLI commands per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlanything/journal.py src/mlanything/run.py src/mlanything/cli.py tests/test_run.py tests/test_cli.py
git commit -m "feat: run directory, state, journal, budget and init/profile/note/status/drop/keep commands"
```

---

### Task 5: Proxy evaluator and `mla baseline`

**Files:**
- Create: `src/mlanything/proxy.py`
- Modify: `src/mlanything/cli.py`
- Test: `tests/test_proxy.py`

**Interfaces:**
- Consumes: `Run`, `cv_folds`, `score`, `oriented`.
- Produces: `PROXY_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)` (`random_state` = problem seed).
- Produces: `prepare_pair(Xtr: DataFrame, Xva: DataFrame) -> tuple[DataFrame, DataFrame]` — datetime columns → `<col>_year/_month/_day/_dow/_hour` numeric columns, original dropped; object/bool/category → pandas `Categorical` with categories taken from `Xtr` (unseen values in `Xva` become NaN); text-like object columns (mean length > 30 and > 1000 unique) dropped; `±inf` → NaN.
- Produces: `encode_target(y: Series, problem) -> np.ndarray` (binary → 0/1 with `positive_class` = 1; multiclass → index into `problem.labels`; regression → float).
- Produces: `CVResult` dataclass `(fold_scores: list[float], mean: float, std: float, flags: list[str])`; fold scores are **oriented** (higher is better).
- Produces: `cv_score(X: DataFrame, y: Series, problem, folds, fold_transform=None) -> CVResult` — for each fold, if `fold_transform` is given call `fold_transform(Xtr, ytr, Xva, yva=yva) -> (Xtr, Xva, flags)` first (`yva` is used only for the leak check in Task 6), then `prepare_pair`, fit `LGBMClassifier`/`LGBMRegressor(**PROXY_PARAMS)`, predict per the prediction-format convention, `oriented(score(...))`.
- Produces: `dummy_cv_score(y: Series, problem, folds) -> CVResult` (`DummyClassifier(strategy="prior")` / `DummyRegressor()`).
- CLI: `mla baseline` — `require_unblocked()`; computes dummy and LightGBM CV on `base_features(train)`; stores both in `state.baseline_scores`; sets `best_label="baseline"`, `best_scores` = LightGBM fold scores; journal `baseline`; prints both means ± std in the metric's natural sign.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_proxy.py
import numpy as np, pandas as pd
from types import SimpleNamespace as NS
from mlanything.proxy import cv_score, dummy_cv_score, prepare_pair
from mlanything.splits import cv_folds

def P(task="binary", metric="roc_auc", labels=(0, 1)):
    return NS(task=task, metric=metric, split="iid", group_col=None, time_col=None,
              target="target", seed=42, positive_class=labels[-1] if task == "binary" else None,
              labels=list(labels) if task != "regression" else None)

def test_lgbm_beats_dummy_binary(make_binary):
    df = make_binary().drop(columns=["leak", "group_id"])
    p = P(); folds = cv_folds(df, p)
    X, y = df.drop(columns="target"), df.target
    assert cv_score(X, y, p, folds).mean > dummy_cv_score(y, p, folds).mean + 0.1

def test_regression_scores_are_oriented(make_regression):
    df = make_regression(); p = P("regression", "rmse")
    r = cv_score(df.drop(columns="target"), df.target, p, cv_folds(df, p))
    assert r.mean < 0 and len(r.fold_scores) == 15

def test_unseen_category_in_validation():
    tr = pd.DataFrame({"c": ["a", "b"] * 50}); va = pd.DataFrame({"c": ["z"] * 10})
    _, va2 = prepare_pair(tr, va)
    assert va2["c"].isna().all()

def test_multiclass_runs(make_multiclass):
    df = make_multiclass(); p = P("multiclass", "log_loss", ("a", "b", "c"))
    assert cv_score(df.drop(columns="target"), df.target, p, cv_folds(df, p)).mean < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_proxy.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `proxy.py` and the `baseline` command per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlanything/proxy.py src/mlanything/cli.py tests/test_proxy.py
git commit -m "feat: LightGBM proxy CV evaluator and baseline command"
```

---

### Task 6: Feature modules and `mla try`

**Files:**
- Create: `src/mlanything/features.py`
- Modify: `src/mlanything/cli.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `cv_score`, `encode_target`, `Run`.
- Produces: `load_module(path: Path) -> type` — imports the file under a unique module name and returns its class `Features`; raises `FeatureModuleError` if absent.
- Produces: `merge(X: DataFrame, new: DataFrame) -> DataFrame` — adds new columns, replaces same-named ones.
- Produces: `make_fold_transform(classes: list[type], problem) -> Callable` — the `fold_transform` for `cv_score`: for each class in order, instantiate, `fit(Xtr, ytr)`, then `Xtr = merge(Xtr, m.transform(Xtr))` and `Xva = merge(Xva, m.transform(Xva))`. Contract checks after each `transform`: result is a DataFrame with identical index and length, else `ContractError`. Leak check (binary and regression only; skipped when `problem` is `None`): for each new column, if `|spearman(col, y)|` ≥ 0.9 on the training fold and ≤ that value − 0.3 on the validation fold (a NaN correlation counts as 0), append flag `"target_leak_suspect:<col>"`. `make_fold_transform` receives `y` for the validation fold through an optional keyword argument `yva=None` (leak check skipped when `None`) that `cv_score` passes when calling `fold_transform(Xtr, ytr, Xva, yva=yva)`; the transform never exposes `yva` to modules.
- Produces: `accept(candidate: list[float], best: list[float]) -> tuple[bool, float, float]` returning `(accepted, mean_delta, se)` with `d = candidate − best`, `se = d.std(ddof=1) / sqrt(len(d))`, `accepted = d.mean() > se`.
- CLI: `mla try <features.py> [--note TEXT]` — requires baseline done and `require_unblocked()`; copies the file to `run/features/NN_<stem>.py` (NN = zero-padded trial number); evaluates active modules + candidate; any exception during the trial → journal `trial` with `accepted: false` and the traceback, exit 0, print the error; any `target_leak_suspect` flag → rejected with reason. Accepted → append to `state.active_modules`, update `best_scores`/`best_label`. Always increment `trial_count`; journal `trial` with `module, note, mean_delta, se, accepted, flags, new_columns`. Prints `ACCEPTED`/`REJECTED`, delta ± SE in natural units, and trial count.
- Module contract documented in the CLI help: a file defining `class Features` with `fit(self, X, y) -> self` and `transform(self, X) -> DataFrame` returning new or replaced columns, same index.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
import textwrap, pytest
from mlanything.features import load_module, accept, make_fold_transform
from mlanything.errors import ContractError
from mlanything.cli import main

GOOD = """
import pandas as pd
class Features:
    def fit(self, X, y): return self
    def transform(self, X): return pd.DataFrame({"x1x2": X["x1"] * X["x2"]}, index=X.index)
"""
NOISE = """
import numpy as np, pandas as pd
class Features:
    def fit(self, X, y): return self
    def transform(self, X):
        return pd.DataFrame({"rnd": np.random.default_rng(0).normal(size=len(X))}, index=X.index)
"""
DROPS_ROWS = """
class Features:
    def fit(self, X, y): return self
    def transform(self, X): return X.iloc[:-1][["x1"]].rename(columns={"x1": "y"})
"""
LEAKY_TE = """
import pandas as pd
class Features:
    def fit(self, X, y): self.y = pd.Series(y, index=X.index); return self
    def transform(self, X): return pd.DataFrame({"te": self.y.reindex(X.index)}, index=X.index)
"""

def mod(tmp_path, src, name):
    p = tmp_path / f"{name}.py"; p.write_text(textwrap.dedent(src)); return p

def test_accept_rule():
    assert accept([0.82, 0.83, 0.81], [0.80, 0.80, 0.80])[0]
    assert not accept([0.81, 0.79, 0.80], [0.80, 0.80, 0.80])[0]

def test_contract_violation(tmp_path, make_regression):
    df = make_regression(); X, y = df.drop(columns="target"), df.target
    ft = make_fold_transform([load_module(mod(tmp_path, DROPS_ROWS, "d"))], problem=None)
    with pytest.raises(ContractError):
        ft(X.iloc[:2000], y.iloc[:2000], X.iloc[2000:])

def test_try_accepts_signal_rejects_noise_and_leak(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path); make_regression().to_csv("d.csv", index=False)
    assert main(["init", "d.csv", "--target", "target", "--goal", "g"]) == 0
    assert main(["baseline"]) == 0
    for name, src, expect in [("good", GOOD, True), ("noise", NOISE, False), ("leaky", LEAKY_TE, False)]:
        assert main(["try", str(mod(tmp_path, src, name))]) == 0
    from mlanything.run import Run
    trials = Run.load(tmp_path).journal.entries("trial")
    assert [t["accepted"] for t in trials] == [True, False, False]
    assert any(f.startswith("target_leak_suspect") for f in trials[2]["flags"])

def test_try_logs_module_exception(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path); make_regression().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"])
    bad = mod(tmp_path, "class Features:\n def fit(self,X,y): raise ValueError('boom')\n def transform(self,X): pass\n", "bad")
    assert main(["try", str(bad)]) == 0
    from mlanything.run import Run
    t = Run.load(tmp_path).journal.entries("trial")[-1]
    assert not t["accepted"] and "boom" in t["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `features.py` and the `try` command per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlanything/features.py src/mlanything/cli.py tests/test_features.py
git commit -m "feat: fold-safe feature modules with contract checks and keep/reject rule"
```

---

### Task 7: Leakage checks and `mla leakage`

**Files:**
- Create: `src/mlanything/leakage.py`
- Modify: `src/mlanything/cli.py`
- Test: `tests/test_leakage.py`

**Interfaces:**
- Consumes: `Run`, `cv_folds`, `cv_score` (single-feature power uses a shallow LightGBM: `num_leaves=8, n_estimators=50`, 3-fold, no repeats).
- Produces: `Finding` dataclass `(column: str | None, check: str, severity: Literal["critical","warning","info"], evidence: str)`.
- Produces: `run_checks(train: DataFrame, holdout: DataFrame, problem, dropped: Iterable[str] = ()) -> list[Finding]` calling, in order:
  1. `check_single_feature_power` — power = AUC (binary), macro one-vs-rest AUC (multiclass) or R² (regression) of each column alone. Critical if AUC > 0.95 or R² > 0.9. Warning if top column has AUC ≥ 0.75 (R² ≥ 0.3) and its lift over chance (AUC − 0.5, or R²) is ≥ 2× the second-best column's.
  2. `check_names` — warning for names (lowercased) containing `result, outcome, status, final, post_, after_` or the target name.
  3. `check_id_like` — warning for non-float columns with unique ratio ≥ 0.9, or monotonic in row order, whose single-feature power is ≥ 0.6 AUC / ≥ 0.1 R².
  4. `check_duplicates` — warning if > 1% of holdout rows have an exact feature-row match in train; critical on any group overlap in grouped mode.
  5. `check_post_timestamps` — temporal mode only: warning for datetime columns whose value exceeds `time_col` in > 50% of rows.
  6. `check_missingness` — for columns with ≥ 5% missing: warning if the missing indicator alone has AUC > 0.7 (classification) or R² > 0.1.
  7. `check_adversarial` — LightGBM train-vs-holdout classifier AUC (3-fold); info always, evidence includes the AUC and "drift" when > 0.7.
- CLI: `mla leakage` — runs checks on base features; writes `leakage.json`; sets `state.unresolved` to the columns with critical findings not already dropped/kept; journal `leakage`; prints a severity-sorted table and, if unresolved is non-empty, `BLOCKED: resolve with 'mla drop <col> --reason ...' or 'mla keep <col> --reason ...'` (exit 0; later commands enforce the block).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_leakage.py
from types import SimpleNamespace as NS
from mlanything.leakage import run_checks
from mlanything.splits import holdout_split
from mlanything.cli import main

def P(**kw):
    base = dict(task="binary", metric="roc_auc", split="iid", group_col=None, time_col=None,
                target="target", seed=42, positive_class=1, labels=[0, 1])
    return NS(**{**base, **kw})

def findings(df, p):
    tr, ho = holdout_split(df, p); return run_checks(tr, ho, p)

def test_planted_leak_is_critical(make_binary):
    f = findings(make_binary().drop(columns=["ts"]), P())
    assert any(x.column == "leak" and x.severity == "critical" for x in f)
    assert not any(x.column == "f_noise1" and x.severity != "info" for x in f)

def test_suspicious_name_and_missingness(make_binary):
    df = make_binary().drop(columns=["ts", "leak"])
    df["final_status"] = 0
    df["opt"] = df["f_signal1"].where(df.target == 0)  # missing iff positive
    f = findings(df, P())
    assert any(x.column == "final_status" and x.check == "names" for x in f)
    assert any(x.column == "opt" and x.check == "missingness" for x in f)

def test_post_timestamp(make_binary):
    df = make_binary().drop(columns=["leak"])
    df["closed_at"] = df["ts"] + __import__("pandas").Timedelta(days=3)
    f = findings(df, P(split="temporal", time_col="ts"))
    assert any(x.column == "closed_at" and x.check == "post_timestamps" for x in f)

def test_cli_blocks_until_drop(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    assert main(["leakage"]) == 0
    assert main(["baseline"]) == 2
    assert main(["drop", "leak", "--reason", "copy of target"]) == 0
    assert main(["baseline"]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_leakage.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `leakage.py` and the `leakage` command per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlanything/leakage.py src/mlanything/cli.py tests/test_leakage.py
git commit -m "feat: automated leakage checks with blocking on critical findings"
```

---

### Task 8: Final AutoGluon fit and `mla predict`

**Files:**
- Create: `src/mlanything/fit.py`
- Modify: `src/mlanything/cli.py`
- Test: `tests/test_fit.py` (all tests `@pytest.mark.slow`)

**Interfaces:**
- Consumes: `Run`, `load_module`, `merge`, `METRICS`.
- Produces: `fit_final(run: Run, time_limit: float, presets: str = "best_quality") -> dict` — instantiates and fits each active module on the full training portion (sequentially, as in `make_fold_transform`), saves the fitted instances to `models/features.pkl` with `cloudpickle`, builds `train_X = merge(...)` plus target (and `group_col` in grouped mode), then `TabularPredictor(label=target, problem_type=task, eval_metric=METRICS[metric].ag_name, path=run.root/"models"/"ag", groups=group_col if grouped)`. Temporal mode: sort by time, `fit(train[:-15%], tuning_data=train[-15%:], num_bag_folds=0, num_stack_levels=0, dynamic_stacking=False)`. Others: `fit(train, presets=presets, time_limit=time_limit)`. Returns `{"score_val", "leaderboard_top5": list[dict], "time_used", "presets", "notes": list[str]}` (temporal adds a note that bagging was disabled). Sets `state.fitted = True`; journal `fit`.
- Produces: `load_final(run: Run) -> tuple[list, TabularPredictor]`; `transform_with(modules: list, X: DataFrame) -> DataFrame`; `predict(run: Run, df: DataFrame) -> DataFrame` — applies `base_features` + modules, returns column `prediction` (binary: positive-class probability; regression: value) or one column per class (multiclass).
- CLI: `mla fit [--time-limit S] [--presets P]` — `require_unblocked()`; default time limit = `max(60, remaining_seconds() − 180)`, warn if the floor was used. `mla predict <data> --out <csv>` — writes predictions; no target needed in input.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fit.py
import pandas as pd, pytest, textwrap
from mlanything.cli import main
from mlanything.run import Run

pytestmark = pytest.mark.slow
GOOD = textwrap.dedent("""
    import pandas as pd
    class Features:
        def fit(self, X, y): return self
        def transform(self, X): return pd.DataFrame({"x1x2": X["x1"] * X["x2"]}, index=X.index)
""")

def test_fit_and_predict_with_module(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path); df = make_regression(n=1500)
    df.to_csv("d.csv", index=False); (tmp_path / "f.py").write_text(GOOD)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"]); main(["try", "f.py"])
    assert main(["fit", "--time-limit", "60", "--presets", "medium_quality"]) == 0
    new = df.drop(columns="target").head(20)
    new.to_csv("new.csv", index=False)
    assert main(["predict", "new.csv", "--out", "p.csv"]) == 0
    assert len(pd.read_csv("p.csv")) == 20

def test_grouped_fit_excludes_group_column(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_binary(n=1500).drop(columns=["leak"]).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g", "--group", "group_id"]); main(["baseline"])
    assert main(["fit", "--time-limit", "60", "--presets", "medium_quality"]) == 0
    from mlanything.fit import load_final
    _, pred = load_final(Run.load(tmp_path))
    assert "group_id" not in pred.features()

def test_predict_handles_unseen_category(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_binary(n=1500).drop(columns=["leak"]).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"])
    main(["fit", "--time-limit", "60", "--presets", "medium_quality"])
    new = make_binary(n=10).drop(columns=["leak", "target"]).assign(cat="never_seen")
    new.to_csv("new.csv", index=False)
    assert main(["predict", "new.csv", "--out", "p.csv"]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fit.py -v -m slow`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `fit.py` and the `fit` / `predict` commands per the Interfaces block**

If the pinned AutoGluon version names a `fit` argument differently (e.g. `dynamic_stacking`), adapt to it and note it in the commit.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v -m slow tests/test_fit.py`
Expected: PASS (each test ≲ 2 min)

- [ ] **Step 5: Commit**

```bash
git add src/mlanything/fit.py src/mlanything/cli.py tests/test_fit.py
git commit -m "feat: final AutoGluon fit with fold-consistent features, and predict command"
```

---

### Task 9: Holdout evaluation, calibration and protection

**Files:**
- Create: `src/mlanything/evaluate.py`
- Modify: `src/mlanything/cli.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Run`, `predict`, `load_final`, `transform_with`, `score`, `secondary_metrics`.
- Produces: `ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float` — equal-width bins; binary uses positive-class probability; multiclass uses top-label confidence vs correctness.
- Produces: `evaluate(run: Run, force: bool = False) -> dict` — raises `HoldoutReusedError` when `state.eval_count ≥ 1` and not `force`; computes primary + secondary metrics on `holdout_df()`; `cv_proxy_best` (mean of `best_scores` in natural units), `ag_score_val` (from the `fit` journal entry; AutoGluon reports error metrics negated, so convert to natural units), `gap = holdout − ag_score_val` (natural units); classification adds `brier` (binary) and `ece`, and saves `calibration.png` (reliability diagram, matplotlib Agg backend); `importance` = top 15 of `predictor.feature_importance(transformed_holdout, subsample_size=5000, num_shuffle_sets=3, time_limit=120)`; sets `holdout_reused: true` when forced. Writes `evaluation.json`, increments `eval_count`, journal `evaluate`.
- CLI: `mla evaluate [--force]` — requires `state.fitted`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
import json, numpy as np, pytest
from mlanything.evaluate import ece
from mlanything.cli import main

def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0); p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert ece(y, p) < 0.02

def test_ece_overconfident_is_large():
    y = np.array([0, 1] * 500); p = np.where(y == 1, 0.99, 0.98)
    assert ece(y, p) > 0.4

@pytest.mark.slow
def test_evaluate_once_then_blocked(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_binary(n=1500).drop(columns=["leak"]).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"])
    main(["fit", "--time-limit", "60", "--presets", "medium_quality"])
    assert main(["evaluate"]) == 0
    assert main(["evaluate"]) == 2
    assert main(["evaluate", "--force"]) == 0
    run_dir = next((tmp_path / ".mla").glob("2*"))
    ev = json.loads((run_dir / "evaluation.json").read_text())
    assert ev["holdout_reused"] is True and "ece" in ev and (run_dir / "calibration.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py -v -m "slow or not slow"`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `evaluate.py` and the `evaluate` command per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v -m "slow or not slow"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlanything/evaluate.py src/mlanything/cli.py tests/test_evaluate.py
git commit -m "feat: one-shot holdout evaluation with calibration and importance"
```

---

### Task 10: Report generation and end-to-end smoke test

**Files:**
- Create: `src/mlanything/report.py`, `tests/test_e2e.py`
- Modify: `src/mlanything/cli.py`

**Interfaces:**
- Consumes: `Run`, journal entries of all kinds, `profile.json`, `leakage.json`, `evaluation.json`, optional `FRAMING.md` and `NARRATIVE.md` in the run's working directory (cwd).
- Produces: `build_report(run: Run, workdir: Path) -> str` and writes `run.root/"REPORT.md"` and a copy at `workdir/"REPORT.md"`. Sections, in order, each with this exact heading: `# ML Anything Report`, `## Problem` (goal, target, task, metric, split, rows), `## Framing` (FRAMING.md contents or "Not written."), `## Leakage` (findings table + drop/keep decisions with reasons), `## Baseline` (dummy and LightGBM CV), `## Feature experiments` (table: trial, module, note, Δ, SE, accepted, flags; then "N trials, M accepted"), `## Final model` (AutoGluon top-5 leaderboard, notes), `## Holdout results` (primary + secondary metrics, CV-vs-holdout gap, "holdout reused" warning when applicable), `## Calibration` (Brier, ECE, image link), `## Feature importance` (top 15), `## Analysis` (NARRATIVE.md or "Not written."), `## Run stats` (budget used, trial count, start/end times). Missing artefacts render as "Not run." rather than failing.
- CLI: `mla report`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_e2e.py
import textwrap, pytest
from mlanything.cli import main

HEADINGS = ["## Problem", "## Framing", "## Leakage", "## Baseline", "## Feature experiments",
            "## Final model", "## Holdout results", "## Calibration", "## Feature importance",
            "## Analysis", "## Run stats"]

def test_report_with_nothing_run(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    assert main(["report"]) == 0
    text = (tmp_path / "REPORT.md").read_text()
    assert all(h in text for h in HEADINGS) and "Not run." in text

@pytest.mark.slow
def test_full_pipeline(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path); make_regression(n=1500).to_csv("d.csv", index=False)
    (tmp_path / "f.py").write_text(textwrap.dedent("""
        import pandas as pd
        class Features:
            def fit(self, X, y): return self
            def transform(self, X): return pd.DataFrame({"x1x2": X["x1"] * X["x2"]}, index=X.index)
    """))
    (tmp_path / "NARRATIVE.md").write_text("Interaction x1*x2 drives the target.")
    for cmd in (["init", "d.csv", "--target", "target", "--goal", "g", "--budget", "10"],
                ["leakage"], ["baseline"], ["try", "f.py", "--note", "interaction"],
                ["fit", "--time-limit", "60", "--presets", "medium_quality"],
                ["evaluate"], ["report"]):
        assert main(cmd) == 0, cmd
    text = (tmp_path / "REPORT.md").read_text()
    assert "x1x2" in text and "Interaction x1*x2" in text and "1 accepted" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_e2e.py -v -m "slow or not slow"`
Expected: FAIL (`report` command missing)

- [ ] **Step 3: Implement `report.py` and the `report` command per the Interfaces block**

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -v -m "slow or not slow"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlanything/report.py src/mlanything/cli.py tests/test_e2e.py
git commit -m "feat: REPORT.md generation and end-to-end smoke test"
```

---

### Task 11: Claude Code plugin — skill, playbooks and slash command

**Files:**
- Create: `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `skills/ml-anything/SKILL.md`, `skills/ml-anything/references/leakage.md`, `skills/ml-anything/references/features.md`, `skills/ml-anything/references/metrics.md`, `commands/mla.md`
- Test: `tests/test_plugin.py`

**Interfaces:**
- Consumes: the CLI surface from Tasks 4–10: `init, profile, note, status, drop, keep, leakage, baseline, try, fit, predict, evaluate, report`.
- `plugin.json`: `{"name": "ml-anything", "version": "0.1.0", "description": "...", "author": {"name": "Massimo Pignatti"}, "license": "Apache-2.0", "keywords": ["machine-learning","automl","tabular","autogluon","kaggle"]}`.
- `marketplace.json`: `{"name": "ml-anything", "owner": {"name": "Massimo Pignatti"}, "plugins": [{"name": "ml-anything", "source": "./", "description": "..."}]}`.
- `SKILL.md` frontmatter: `name: ml-anything`; `description:` starting "Use when the user provides a tabular dataset (CSV or Parquet) and a prediction goal..." Body sections, in order: **Setup** (check `mla --version`; if missing install with `uv tool install --python 3.13 mlanything`, or from git before the PyPI release); **Hard rules** (never read `holdout.parquet` or any file under `.mla/` except via `mla`; never fit models outside `mla`; log reasoning with `mla note` at every decision; obey the printed budget); **Phase 1 Frame** … **Phase 6 Verify and report** exactly as spec §4, including the stop rules (3 rounds without acceptance or ~40% budget), the AUC > 0.95 block, the feature-round order (domain → interactions/ratios → group/time aggregates → encodings), writing `FRAMING.md` in phase 1 and `NARRATIVE.md` before `mla report`; **Feature module template** (a complete minimal `class Features` example); **When things go wrong** (rejected trial with error → read traceback, fix, retry once; blocked → resolve with drop/keep). Keep SKILL.md under 300 lines; detail lives in references.
- `references/leakage.md`: generic leakage patterns with one example each — post-outcome columns, columns measured during the event being predicted, target-encoded IDs, global aggregates/normalisation over the full dataset, duplicates across splits, temporal look-ahead, missingness caused by the outcome, another model's score of the same target. Must not name any benchmark dataset or its specific leaky column.
- `references/features.md`: playbook by data type — geo (haversine, bearing, distance to landmarks, clusters fit inside `fit`), datetime (cyclical encodings, elapsed time, holidays), grouped/temporal aggregates (computed from the training fold only, lagged within groups), categorical (frequency, target encoding fit in `fit` with smoothing), numeric (ratios, differences, logs, binning), nested JSON/list columns (parse and summarise), geometry (angles/distances between points).
- `references/metrics.md`: when to override defaults (PR-AUC for heavy imbalance, MAE for heavy-tailed regression, log loss when probabilities are consumed), and how to pass `--metric`.
- `commands/mla.md`: frontmatter `description: Build a leakage-checked tabular model from a dataset and a goal` and `argument-hint: <data-path> "<goal>" [budget-minutes]`; body instructs Claude to use the ml-anything skill with `$ARGUMENTS`, default budget 60.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plugin.py
import json, re
from pathlib import Path
import yaml
from mlanything.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/ml-anything/SKILL.md"

def frontmatter(p):
    m = re.match(r"^---\n(.*?)\n---\n", p.read_text(encoding="utf-8"), re.S)
    return yaml.safe_load(m.group(1))

def test_manifests_parse():
    pj = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
    mj = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    assert pj["name"] == "ml-anything" and mj["plugins"][0]["name"] == "ml-anything"

def test_skill_frontmatter_and_size():
    fm = frontmatter(SKILL)
    assert fm["name"] == "ml-anything" and fm["description"].startswith("Use when")
    assert len(SKILL.read_text(encoding="utf-8").splitlines()) < 300

def test_every_mla_command_in_skill_exists():
    used = set(re.findall(r"`mla (\w+)", SKILL.read_text(encoding="utf-8")))
    known = set(build_parser()._subparsers._group_actions[0].choices)
    assert used and used <= known

def test_references_exist_and_do_not_name_benchmark_leaks():
    text = (ROOT / "skills/ml-anything/references/leakage.md").read_text(encoding="utf-8").lower()
    for banned in ["duration", "bank", "statsbomb", "adult", "casual", "registered"]:
        assert banned not in text
```

(`cli.build_parser() -> argparse.ArgumentParser` is exposed for this test; refactor `main` to use it if not already.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plugin.py -v`
Expected: FAIL (files missing)

- [ ] **Step 3: Write the manifests, SKILL.md, three references and the command per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plugin.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .claude-plugin skills commands src/mlanything/cli.py tests/test_plugin.py
git commit -m "feat: Claude Code plugin with ml-anything skill, playbooks and /mla command"
```

---

### Task 12: CI, README skeleton and license

**Files:**
- Create: `.github/workflows/ci.yml`, `README.md`, `LICENSE`

**Interfaces:**
- CI: on push and pull request; matrix `ubuntu-latest` × Python `3.11`, `3.13`; steps: checkout, `astral-sh/setup-uv`, `uv sync --python ${{ matrix.python }}`, `uv run pytest -v -m "slow or not slow"`.
- README sections: title + one-line pitch; **Why** (three bullets from spec §1); **Install** (`/plugin marketplace add <owner>/ml-anything`, `/plugin install ml-anything@ml-anything`, `uv tool install --python 3.13 mlanything`); **Quickstart** (`/mla data.csv "predict whether the shot is a goal"`); **How it works** (six phases, harness guarantees); **Benchmark** (placeholder table with an HTML comment `<!-- BENCHMARK:START -->`/`<!-- BENCHMARK:END -->` that Task 14's `score.py` fills); **Honesty notes** (single seed; ML Anything trains on 80% because of its holdout while the AutoGluon baseline trains on 100%; leakage-track scoring); **Data attribution** (StatsBomb attribution and logo requirement, UCI CC BY 4.0); **License**.
- LICENSE: standard Apache-2.0 text, copyright "2026 Massimo Pignatti".

- [ ] **Step 1: Write the three files per the Interfaces block**

- [ ] **Step 2: Validate the workflow YAML and run the suite locally**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` then `uv run pytest -q`
Expected: no error; all tests PASS

- [ ] **Step 3: Commit**

```bash
git add .github README.md LICENSE
git commit -m "docs: README skeleton, CI workflow and Apache-2.0 license"
```

---

### Task 13: Benchmark dataset preparation

**Files:**
- Create: `benchmark/registry.yaml`, `benchmark/datasets.py`, `benchmark/prepare.py`
- Test: `tests/test_benchmark_prepare.py`

**Interfaces:**
- `registry.yaml` entry fields: `name, builder, target, task, metric, goal, test_split: {kind: random|group|time, column}, leak_columns: [], license, attribution`. Entries (substitution from spec §6.2 recorded here: NYC taxi trip duration → **Ames housing**, because a coordinate-bearing taxi dataset is not available without a Kaggle login; Ames tests domain feature engineering instead of geo, which California already covers):

| name | builder | target | task / metric | test split | leak_columns |
|---|---|---|---|---|---|
| football-xg | `build_football_xg` | `is_goal` | binary / roc_auc | group `match_id` | — |
| ames-housing | `build_ames` | `SalePrice` | regression / rmse | random | — |
| california-housing | `build_california` | `MedHouseVal` | regression / rmse | random | — |
| bike-sharing | `build_bike` | `cnt` | regression / rmse | time `dteday_hour` | — |
| bank-marketing | `build_bank` | `y` | binary / roc_auc | random | `[duration]` |
| adult-planted-leak | `build_adult_leak` | `income` | binary / roc_auc | random | `[tax_bracket_following_year]` |

- `datasets.py` builders, each `build_<name>(cache: Path) -> DataFrame`, downloading into `benchmark/.cache/` once:
  - `build_football_xg`: StatsBomb open data from `https://raw.githubusercontent.com/statsbomb/open-data/master/data/`; iterate `competitions.json` → matches → events, capped at 800 matches (deterministic order); one row per `Shot` event with `match_id, competition, period, minute, second, team, player, position, play_pattern, x, y` (from `location`), `body_part, technique, shot_type, first_time, under_pressure, one_on_one, open_goal, aerial_won, follows_dribble, freeze_frame_json` (the shot's `freeze_frame` serialised as JSON, or empty). Target `is_goal = outcome == "Goal"`. Exclude `statsbomb_xg`, `end_location` and `outcome`.
  - `build_ames`: OpenML "house_prices" via `sklearn.datasets.fetch_openml(data_id=42165, as_frame=True)`; drop `Id`.
  - `build_california`: `sklearn.datasets.fetch_california_housing(as_frame=True)`.
  - `build_bike`: UCI Bike Sharing (`https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip`, `hour.csv`); drop `casual`, `registered` (components of the target) and `instant`; add `dteday_hour` datetime = `dteday + hr`.
  - `build_bank`: UCI Bank Marketing (`https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`, `bank-additional-full.csv`, `sep=";"`), all columns kept including `duration`.
  - `build_adult_leak`: UCI Adult (`https://archive.ics.uci.edu/static/public/2/adult.zip`, `adult.data` + `adult.test`, cleaned labels to `<=50K`/`>50K`); add `tax_bracket_following_year`: for `>50K` rows `"high"` with probability 0.85 else uniformly one of `"low","mid"`; for `<=50K` rows `"high"` with probability 0.05 else `"low"`/`"mid"` (seed 42).
  - If a source is unreachable or materially different from this description, replace the dataset with one testing the same capability, update the registry and record the reason in `benchmark/README.md`.
- `prepare.py`: `prepare(entry: dict, root: Path, df: DataFrame | None = None) -> None` — splits 80/20 by `test_split` (random: stratified for classification; group: `GroupShuffleSplit`; time: last 20% by column), seed 42; writes `root/workspace/<name>/train.csv`, `root/workspace/<name>/task.md` (goal, target, metric — nothing else), `root/.hidden/<name>/test.csv` (with target), `test_features_naive.csv` (no target) and `test_features_production.csv` (no target; `leak_columns` set to empty). CLI: `python benchmark/prepare.py [--only NAME]` (builds and prepares).

- [ ] **Step 1: Write the failing tests** (offline: inject a DataFrame instead of building)

```python
# tests/test_benchmark_prepare.py
import pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from prepare import prepare

def test_prepare_hides_test_and_blanks_leaks(tmp_path, make_binary):
    df = make_binary()
    entry = dict(name="toy", target="target", task="binary", metric="roc_auc", goal="predict target",
                 test_split={"kind": "group", "column": "group_id"}, leak_columns=["leak"])
    prepare(entry, tmp_path, df=df)
    train = pd.read_csv(tmp_path / "workspace/toy/train.csv")
    hidden = tmp_path / ".hidden/toy"
    prod = pd.read_csv(hidden / "test_features_production.csv")
    naive = pd.read_csv(hidden / "test_features_naive.csv")
    assert not set(train.group_id) & set(pd.read_csv(hidden / "test.csv").group_id)
    assert prod["leak"].isna().all() and naive["leak"].notna().all()
    assert "target" not in prod.columns
    assert not any(p.name.startswith("test") for p in (tmp_path / "workspace/toy").iterdir())
    assert "roc_auc" in (tmp_path / "workspace/toy/task.md").read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_prepare.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `registry.yaml`, `datasets.py` and `prepare.py` per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_prepare.py -v`
Expected: PASS

- [ ] **Step 5: Build all six datasets for real (needs network — ask the user to approve running outside the sandbox)**

Run: `uv run python benchmark/prepare.py`
Expected: six `benchmark/workspace/<name>/train.csv` files; print row/column counts per dataset; football-xg has ≥ 10,000 shots and a non-empty `freeze_frame_json` for most rows. Record counts in `benchmark/README.md`.

- [ ] **Step 6: Commit**

```bash
git add benchmark/registry.yaml benchmark/datasets.py benchmark/prepare.py benchmark/README.md tests/test_benchmark_prepare.py
git commit -m "feat(benchmark): dataset registry, builders and hidden-test preparation"
```

---

### Task 14: Benchmark runner and scorer

**Files:**
- Create: `benchmark/run.py`, `benchmark/score.py`
- Test: `tests/test_benchmark_score.py`

**Interfaces:**
- Consumes: registry, `prepare` outputs, `mla predict`.
- `run.py`: `python benchmark/run.py --dataset NAME --system {autogluon,mla} [--minutes 60]`. Output dir `benchmark/results/raw/<name>/<system>/` with `preds_naive.csv`, `preds_production.csv`, `meta.json` (`wall_seconds`, `cv_score` — AutoGluon `score_val` or the mla `fit` journal's `score_val` — and for mla the `usage`/`total_cost_usd` from claude's JSON output).
  - `autogluon`: `TabularPredictor(label, problem_type, eval_metric=ag_name).fit(train, presets="best_quality", time_limit=minutes*60 - 120)` on full `train.csv`; predict both test feature files in the prediction-format convention.
  - `mla`: in a fresh copy of the workspace dir, run `["claude", "-p", PROMPT, "--plugin-dir", REPO_ROOT, "--output-format", "json", "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep"]` with `timeout = minutes*60 + 300` where `PROMPT` = "Use the ml-anything skill. Data: train.csv. Goal: <goal>. Target: <target>. Metric: <metric>. Budget: <minutes> minutes. Work fully autonomously; do not ask questions." Then `mla predict` on both test feature files (with `--run` = the workspace's CURRENT). On timeout or missing fitted model, record `"status": "failed"` in `meta.json` and skip predictions.
- `score.py`: `relative_error_reduction(metric: str, base: float, ours: float) -> float` — AUC-type: `((1−base) − (1−ours)) / (1−base)`; error-type (rmse, mae, log_loss): `(base − ours) / base`. `python benchmark/score.py` reads hidden test labels + raw preds, writes `benchmark/results/results.csv` (columns: `dataset, system, variant, metric, score, cv_score, cv_test_gap, wall_minutes, cost_usd, status`), `RESULTS.md` (table per variant + relative error reduction column for mla vs autogluon), `chart.png` (bar chart of relative error reduction per dataset, production variant), and replaces the README block between `<!-- BENCHMARK:START -->` and `<!-- BENCHMARK:END -->` with the production-variant table.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_benchmark_score.py
import sys, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from score import relative_error_reduction

def test_auc_error_reduction():
    assert relative_error_reduction("roc_auc", 0.80, 0.85) == pytest.approx(0.25)

def test_rmse_error_reduction():
    assert relative_error_reduction("rmse", 10.0, 9.0) == pytest.approx(0.10)

def test_worse_is_negative():
    assert relative_error_reduction("rmse", 10.0, 11.0) < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_score.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `run.py` and `score.py` per the Interfaces block**

- [ ] **Step 4: Run tests to verify they pass, then dry-run the AutoGluon system for 3 minutes**

Run: `uv run pytest tests/test_benchmark_score.py -v` then `uv run python benchmark/run.py --dataset california-housing --system autogluon --minutes 3` then `uv run python benchmark/score.py`
Expected: tests PASS; `results.csv` has two rows (naive/production) for california-housing autogluon with a plausible RMSE (< 0.7).

- [ ] **Step 5: Commit**

```bash
git add benchmark/run.py benchmark/score.py tests/test_benchmark_score.py
git commit -m "feat(benchmark): runner for autogluon and headless mla, and scorer"
```

---

### Task 15: Skill check — one real headless run

**Files:**
- Modify: `skills/ml-anything/SKILL.md`, references, CLI messages (only as findings require)
- Create: `docs/skill-check-2026-10-29.md`

- [ ] **Step 1: Run ML Anything headless on California housing with a 15-minute budget**

Run: `uv run python benchmark/run.py --dataset california-housing --system mla --minutes 15`
Expected: `meta.json` status not `failed`; workspace contains `FRAMING.md`, `NARRATIVE.md`, `REPORT.md`; journal shows ≥ 1 `trial` and exactly one `evaluate`.

- [ ] **Step 2: Review the transcript and journal against SKILL.md**

Check each: phases run in order; reasoning logged with `mla note`; no direct read of `.mla/` files; budget respected (finished within 15 min + 5 min grace); at least one domain feature tried (e.g. distance-based geo feature). Write findings to `docs/skill-check-2026-10-29.md`.

- [ ] **Step 3: Fix SKILL.md / CLI messages for every deviation, then re-run Step 1 once**

Expected: the deviations recorded in Step 2 no longer occur.

- [ ] **Step 4: Run the suite and commit**

Run: `uv run pytest -q`
Expected: PASS

```bash
git add skills docs/skill-check-2026-10-29.md src
git commit -m "fix(skill): address findings from first headless skill check"
```

---

### Task 16: Full benchmark, football example and README results

**Files:**
- Create: `benchmark/results/{results.csv,RESULTS.md,chart.png}`, `examples/football-xg/{REPORT.md,FRAMING.md,NARRATIVE.md,features/}`
- Modify: `README.md`

This task consumes ~12 hours of CPU and significant Claude usage; confirm with the user before starting, and run batches in the background.

- [ ] **Step 1: Run AutoGluon baselines for all six datasets (60 min each, background)**

Run: `for d in football-xg ames-housing california-housing bike-sharing bank-marketing adult-planted-leak; do uv run python benchmark/run.py --dataset $d --system autogluon --minutes 60; done`
Expected: six `meta.json` files without `failed`.

- [ ] **Step 2: Run ML Anything for all six datasets (60 min each, background)**

Run: same loop with `--system mla`
Expected: six `meta.json` files; any `failed` run is investigated and re-run once, and the failure is reported in RESULTS.md if it persists.

- [ ] **Step 3: Score and update README**

Run: `uv run python benchmark/score.py`
Expected: `results.csv`, `RESULTS.md`, `chart.png` written; README benchmark block filled with real numbers, whatever they are.

- [ ] **Step 4: Copy the football-xg run's FRAMING.md, NARRATIVE.md, REPORT.md, calibration.png and accepted feature modules to `examples/football-xg/`**

- [ ] **Step 5: Commit**

```bash
git add benchmark/results examples README.md
git commit -m "docs: benchmark results and football xG example run"
```
