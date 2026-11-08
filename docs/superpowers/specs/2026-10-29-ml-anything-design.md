# ML Anything — Design Spec

- **Date:** 2026-10-29
- **Status:** Draft, awaiting review
- **Author:** Massimo Pignatti (with Claude)

## 1. Intent

### What we are building
An open-source Claude Code plugin + Python package that turns "here is a CSV and a one-line goal" into a competitive, leakage-checked tabular model with a written report explaining what worked, what did not, and why.

### Why it should exist
Strong AutoML engines (AutoGluon, FLAML, H2O) already do model search, tuning and ensembling well. Re-implementing that loses. What AutoML cannot do is:

1. **Reason about the domain** — invent features such as shot distance and angle to goal.
2. **Hunt leakage** — notice that a column is only known after the outcome.
3. **Explain** — say why the model works instead of shipping a black box.

ML Anything uses AutoGluon as the engine and adds an agent "brain" on top for framing, leakage control, domain feature engineering and reporting.

### Prior art (checked 2026-10-29)
- `nvidia-kaggle` — Claude Code/Codex skill running a Kaggle workflow; no evidence of leakage hunting or an AutoGluon engine.
- `autoresearch-claude-code` — generic autonomous experiment loop skill, not tabular-specific.
- MLE-STAR, AIDE, AutoMind (MLE-bench) — research agent systems, not one-command installable skills, not benchmarked against plain AutoGluon.
- AutoGluon's own Claude skills — for benchmarking/TabArena integration, not modeling.

The open niche is the combination: AutoGluon engine + agent-side domain features and leakage hunting + an honest benchmark against AutoGluon alone.

### Success criteria
1. One-command install as a Claude Code plugin; `pip install mlanything` for the harness.
2. `/mla data.csv "predict X"` produces a model, `REPORT.md`, and a journal of every experiment, within a user-given time budget.
3. A reproducible benchmark on 5–6 public datasets showing test-set results of ML Anything vs. AutoGluon alone under the same 1h wall-clock budget, including a leakage track.
4. The README headline is a real number from that benchmark, whatever it turns out to be.

### Non-goals (v1)
- Time-series forecasting, text, images, multimodal.
- Pluggable engines other than AutoGluon.
- Agent-written model code, custom stacking, or hyper-parameter search outside AutoGluon.
- GPU support, distributed training, model serving/deployment.
- Multi-seed benchmark (planned for v1.1).

## 2. Constraints

- Python **3.10–3.13** (AutoGluon does not support 3.14). The developer machine has 3.14 globally; the project uses a `uv`-managed 3.13 environment.
- CPU-only laptop for development and benchmark.
- Benchmark budget: ~1h per system per dataset, 5–6 datasets (~12h total, run in overnight batches).
- License: Apache-2.0.
- All repo content in English.

## 3. Architecture

Approach: **brain over AutoGluon engine**. Three parts in one repository.

### 3.1 `mlanything` Python package (`mla` CLI)

Small deterministic commands that the agent calls. The harness owns splitting, scoring and the holdout, so the agent cannot win by accident or by leakage.

| Command | Responsibility |
|---|---|
| `mla init <data> --target <col> --goal "<text>" [--group <col>] [--time <col>] [--budget <minutes>]` | Create run dir `.mla/<run_id>/`; profile data; infer task type, metric and split strategy into `problem.yaml`; split off and store the locked holdout; start the budget clock. |
| `mla profile` | Print/refresh dataset profile (dtypes, cardinality, missingness, target stats, ID-like and datetime columns). |
| `mla leakage` | Run automated leakage checks (§5.4); print suspects with evidence; log to journal. |
| `mla baseline` | Score a dummy model and default LightGBM in CV; set them as the initial "current best". |
| `mla try <features.py> [--note "<hypothesis>"]` | Evaluate a feature module with the fast LightGBM proxy in CV against the current best; apply the keep/reject rule (§5.3); log result. Accepted modules join the active feature set. |
| `mla drop <col> --reason "<text>"` | Remove a column from the active feature set (e.g. leakage) with a recorded reason. |
| `mla fit [--time-limit <s>]` | Final AutoGluon `best_quality` fit on the training portion with the active feature set, using the run's split strategy; default time limit = (remaining budget − 180 s) / 2, minimum 60 s, so the other half is left for evaluate and refit. This is the evaluated model. |
| `mla evaluate [--force]` | Score the holdout once: primary metric, secondary metrics, calibration (Brier, ECE, reliability curve for classifiers), permutation importance. A second call requires `--force` and is labelled "holdout reused". Refused after `mla refit`, even with `--force`. |
| `mla refit [--time-limit <s>] [--presets <p>]` | After the current model was evaluated: refit the same recipe (feature modules, presets of the last fit, metric, split logic) on all labelled rows (train + holdout) as the production model; default time limit = remaining budget − 180 s, minimum 60 s. Trains into a staging directory and swaps in only on success. Different presets, or a module that fails on the full data, are recorded as recipe drift in the journal and the report. |
| `mla predict <data> --out <csv>` | Predict with the production model after a refit, otherwise with the evaluated model (and a warning). |
| `mla report` | Generate `REPORT.md` from the journal and artefacts. |
| `mla status` | Print current best, accepted features, trial count, remaining budget. |

Every command prints remaining budget and appends to the journal.

**Modules** (`src/mlanything/`): `cli`, `run` (run dir + config + budget clock), `profile`, `splits`, `metrics`, `leakage`, `proxy` (LightGBM CV evaluator), `features` (module loader + contract checks), `fit` (AutoGluon wrapper), `evaluate`, `report`, `journal`. Each module has one responsibility and is testable on its own.

### 3.2 Claude Code plugin

```
.claude-plugin/plugin.json
.claude-plugin/marketplace.json
skills/ml-anything/SKILL.md
skills/ml-anything/references/leakage.md
skills/ml-anything/references/features.md
skills/ml-anything/references/metrics.md
commands/mla.md
```

- `SKILL.md` — the phased workflow (§4), hard rules, and time-budget rules. Written in the open Agent Skills format so other agents (Codex etc.) can use it.
- `references/leakage.md` — leakage playbook: post-outcome columns, target-encoding IDs, duplicates, global aggregates, temporal leakage, with examples.
- `references/features.md` — feature-engineering playbook by data type: geo (haversine, bearing, clusters), datetime (cyclical, holidays, lags within groups), categorical (frequency, target encoding inside folds), numeric (ratios, interactions, binning), grouped aggregates, text-in-tabular basics.
- `references/metrics.md` — metric choice guide and when to override defaults.
- `commands/mla.md` — `/mla <data> "<goal>" [budget]` slash command that invokes the skill.

### 3.3 Benchmark (`benchmark/`)

- `registry.yaml` — datasets, source, license, target, goal sentence, test-split definition, leakage-track settings.
- `prepare.py` — downloads/builds each dataset, creates the hidden test split, writes `workspace/<dataset>/train.csv` + `task.md`; test stays in `benchmark/.hidden/` outside the workspace.
- `run.py --dataset <name> --system {autogluon,mla}` — runs one system with a 1h wall-clock budget; `mla` system runs `claude -p "/mla train.csv \"<goal>\" 60"` headless inside the workspace and records token usage.
- `score.py` — scores both systems' predictions on hidden test; writes `results/results.csv`, `results/RESULTS.md`, and `results/chart.png`.

## 4. Agent workflow (SKILL.md)

Each phase ends with a journal entry containing the agent's reasoning.

1. **Frame.** Run `mla init`, read the profile, write `FRAMING.md`: what a row is; when a prediction is made in reality; which columns are known at that moment; group/time structure; why the metric fits. Override inferred metric/split with a recorded reason if wrong (re-run `mla init` with flags).
2. **Hunt leakage.** Run `mla leakage`; review manually using `references/leakage.md`. Each suspect is dropped (`mla drop`) or explicitly kept with justification. Any single feature with AUC > 0.95 (or R² > 0.9) blocks progress until explained.
3. **Baseline.** Run `mla baseline`.
4. **Feature-engineering loop.** Write a hypothesis ("distance and angle to goal should matter because…"), implement it as a feature module, run `mla try`. Rounds in order: domain features → interactions/ratios → group/time aggregates → encodings. Stop after 3 consecutive rounds with no accepted module, or at ~40% of budget consumed.
5. **Final fit.** `mla fit` with half the remaining budget. AutoGluon builds its own stacked ensemble; we do not re-implement stacking.
6. **Verify and report.** `mla evaluate` (holdout once), `mla refit` (production model on all labelled rows), `mla report`, then add a narrative section: top features and why they make sense, CV-vs-holdout gap, calibration, residual risks, things that failed.

**Hard rules:** never read the holdout file; never fit models outside the harness; every decision logged with reasoning; respect the budget printed by every command.

## 5. Validation and leakage rules

### 5.1 Split strategies

| Data structure | Holdout | CV |
|---|---|---|
| i.i.d. (default) | 20% random, stratified for classification | 5-fold, stratified for classification |
| grouped (`--group`) | 20% of groups | GroupKFold, 5 folds |
| temporal (`--time`) | most recent 20% by time | expanding-window time-series CV, 5 splits |

Temporal mode in `mla fit` and `mla refit`: AutoGluon bagging uses random folds, so bagging is disabled and the most recent 15% of the rows is passed as `tuning_data`. The fit gets 60% of the time limit. Then the best model is refit on all rows including that tail (`refit_full`, the remaining 40%, shared over the models of the best ensemble because AutoGluon applies a refit time limit per model), and the refit model makes the predictions; the validation scores come from the leaderboard before the refit. If the refit step fails, the tuned model is kept and the notes say so. The report states this.

**Production refit.** The honest estimate comes from the model trained on the 80% training portion and scored once on the locked holdout. The delivered model is then trained on 100% of the labelled rows by `mla refit`, with the same recipe. After a refit the holdout is part of the production model's training data, so it can no longer be evaluated: `mla evaluate` is refused, even with `--force`. A new `mla fit` replaces the evaluated model and clears the refit; `mla refit` then requires that new model to be evaluated first (`mla evaluate --force`, which marks the holdout reused). The same applies to any change of the feature recipe after the evaluation: an accepted `mla try`, `mla drop`, a `mla keep` that re-admits a dropped column, or a module deactivation. The refit compares its recipe with the last fit's recorded modules and dropped columns and reports every difference.

Data with fewer than 50k training rows uses 3×5 repeated CV in the proxy (i.i.d. and grouped modes; temporal uses a single expanding-window pass).

### 5.2 Default metrics
- Binary classification: ROC-AUC (log loss and Brier always reported).
- Multiclass: log loss (accuracy and macro-F1 reported).
- Regression: RMSE (MAE and R² reported).

The agent may override (e.g. PR-AUC under heavy imbalance, MAE for robust targets) with a recorded reason.

### 5.3 Keep/reject rule (`mla try`)
Candidate and current best are evaluated on identical folds and seeds. Accept iff the mean paired per-fold improvement is greater than one standard error of the paired improvements. The total trial count is logged and shown in the report to keep search-overfitting visible.

### 5.4 Automated leakage checks (`mla leakage`)
1. **Single-feature power** — each column alone in a shallow GBM in CV; flag AUC > 0.95 or R² > 0.9, or a column far stronger than the next best.
2. **Suspicious names** — `result`, `outcome`, `status`, `final`, `post_`, `after_`, or containing the target name.
3. **ID-like columns** — high-cardinality, monotonic or row-order columns correlated with the target.
4. **Duplicates** — identical feature rows across train and holdout; group overlap in grouped mode.
5. **Post-prediction timestamps** — datetime columns later than the time column (temporal mode).
6. **Missingness as signal** — a missing indicator alone strongly predicting the target.
7. **Adversarial validation** — train vs. holdout classifier; reports drift (informational).

### 5.5 Feature-module contract
A feature module defines a class with `fit(X, y) -> self` and `transform(X) -> DataFrame`. The harness:
- fits it inside each CV fold (the target is visible only to `fit` on the training fold);
- checks row count and index are preserved;
- flags any output column that is near-perfectly correlated with the target on the training fold but not on the validation fold (target-encoding bug);
- records exceptions as a rejected trial with traceback instead of crashing.

### 5.6 Holdout protection
- Holdout stored at `.mla/<run_id>/holdout.parquet`; only `mla evaluate` reads it.
- Evaluation count persisted; re-evaluation requires `--force` and is labelled in the report.
- In the benchmark, the real test set is never inside the agent's workspace.

## 6. Benchmark design

### 6.1 Comparison
Per dataset: **AutoGluon `best_quality`, 1h, raw features** vs. **ML Anything, 1h total wall-clock** (agent reasoning + proxy trials + its final AutoGluon fit). The agent receives only `train.csv`, the target name and a one-line goal, as a human analyst would. It must discover group/time structure itself.

### 6.2 Datasets (candidates; availability and licenses verified during planning)

| Dataset | Task | Tests |
|---|---|---|
| Football shots → goal (StatsBomb open data) | binary, grouped by match | Geometry features (distance, angle to goal) — flagship demo |
| NYC taxi trip duration | regression | Haversine distance, bearing from coordinates |
| California housing | regression | Geo features (clusters, distance to coast) |
| Bike sharing demand | regression, temporal | Correct temporal split; hour × weekday features |
| Bank marketing | binary | Real documented leak: `duration` |
| Adult + planted leaked column | binary | Controlled leakage trap |

Any dataset that turns out unavailable or unsuitable is replaced by one testing the same capability, and the substitution is recorded.

### 6.3 Leakage-track scoring
For the two leakage datasets, the test set simulates prediction-time availability: the leaky column is missing in test, as it would be in production. The README explains this and also reports the "naive" scores (leaky column present in test).

### 6.4 Reported numbers
Per dataset and system: test primary metric, relative error reduction vs. AutoGluon (for AUC: reduction in 1−AUC; for RMSE: reduction in RMSE), CV-vs-test gap, wall-clock time, Claude token usage (ML Anything). Single seed, stated explicitly.

## 7. Error handling

- Clear error messages; non-zero exit codes on failure.
- Append-only journal (`journal.jsonl`) so an interrupted run can resume.
- Feature module exceptions → logged as rejected trial with traceback.
- Budget exhausted → `fit` uses remaining time with a 60s minimum and warns.
- Mixed/unknown column types → reported in the profile, coerced or excluded with a note.
- Single-class target or fewer than 200 rows → stop with an explanatory error.

## 8. Testing

- **Unit tests** per module on small synthetic datasets with known answers: a planted leak must be flagged; a useful feature accepted and a pure-noise feature rejected; holdout counter blocks re-evaluation; temporal/grouped splits never mix time periods or groups.
- **Feature-contract tests:** modules that leak via `transform` or drop rows must be caught.
- **End-to-end smoke test:** `init → leakage → baseline → try → fit (60s) → evaluate → report` on a small dataset; CI on Python 3.11 and 3.13.
- **Skill check:** one real headless `claude -p` run on a small dataset before the full benchmark.
- Development follows TDD.

## 9. Repository layout

```
ml-anything/
├── .claude-plugin/{plugin.json, marketplace.json}
├── skills/ml-anything/{SKILL.md, references/{leakage,features,metrics}.md}
├── commands/mla.md
├── src/mlanything/{cli,run,profile,splits,metrics,leakage,proxy,features,fit,evaluate,report,journal}.py
├── tests/
├── benchmark/{registry.yaml, prepare.py, run.py, score.py, results/}
├── examples/football-xg/        # demo run with full REPORT.md
├── docs/
├── README.md
├── pyproject.toml               # uv, Python >=3.10,<3.14
└── LICENSE                      # Apache-2.0
```

## 10. Open items (resolved during planning)
- Verify each benchmark dataset's availability, license and size; build the StatsBomb shot table.
- Check name availability (`mlanything` on PyPI, `ml-anything` on GitHub).
- Exact AutoGluon version pin compatible with Python 3.13 on Windows.
