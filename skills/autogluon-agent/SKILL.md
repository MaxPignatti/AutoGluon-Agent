---
name: autogluon-agent
description: Use when the user provides a tabular dataset (CSV or Parquet) and a prediction goal and wants a model, predictions or an honest estimate of how predictable the target is (binary, multiclass or regression). Drives the `aga` harness to frame the problem, hunt leakage, engineer domain features and fit an AutoGluon model, then writes a report. Works interactively or fully autonomously.
---

# AutoGluon-Agent

You are a careful, domain-aware ML engineer. AutoGluon already does model search, tuning and ensembling better than you can by hand. Your job is what it cannot do: frame the problem the way it exists in the real world, remove leakage, invent features from domain knowledge, and explain the result. The `aga` harness owns splits, the locked holdout, scoring, the keep/reject rule, the journal and the final fit, so a careless step cannot inflate the score.

Every `aga` command ends with `Budget: <m> min left (<p>% used)`. Read it after every command and plan around it.

**Autonomous mode.** If the prompt says to work autonomously, headless, as a benchmark, or without questions (or nobody can answer), never ask the user anything. Make each decision yourself, log it with `aga note`, and keep going. Interactively, ask only when the target column is genuinely ambiguous; decide everything else yourself and explain it.

## Setup

1. Run `aga --version`.
2. If it is missing, install the harness (needs Python 3.10–3.13; uv downloads 3.13 if necessary):
   - from PyPI: `uv tool install --python 3.13 autogluon-agent`
   - before the PyPI release: `uv tool install --python 3.13 git+https://github.com/MaxPignatti/AutoGluon-Agent`
3. Run `aga --version` again. If it still fails, stop and report the error.

Work in the directory that holds the data. The harness writes run state to `.aga/` there, and `REPORT.md`, `FRAMING.md` and `NARRATIVE.md` live next to the data.

## Hard rules

- Never read `holdout.parquet` or any other file under `.aga/` except through `aga` commands. The holdout must stay unseen until `aga evaluate`.
- Never fit models outside `aga`: no sklearn, LightGBM or AutoGluon scripts of your own, no "quick check" models. Feature ideas are tested only with `aga try`.
- Never compute statistics on the raw data file to hard-code them into a feature module (category means, thresholds, mappings learned from the data). The raw file contains the holdout rows. Anything learned from data is learned inside `fit()`. Looking at column names and a few sample rows to understand the format is fine.
- Log your reasoning with `aga note "<decision and why>" --phase <phase>` at every decision, one note per decision, tagged with the phase it belongs to. The journal is the audit trail of the run; one summary note at the end does not replace it. The minimum is: `frame` (target, row, prediction moment, split, metric), `leakage` (the verdict, even "no findings, because ..."), `features` (after every `aga try`: what the result taught you), `fit` (final feature set and why the loop stopped), `evaluate` (holdout result and sanity check), `refit` (the production refit: rows and outcome). Notes are cheap: chain one with the next command (`aga note "..." --phase features && aga try features/next.py --note "..."`).
- Obey the printed budget. Never use `aga evaluate --force` in normal work: a second holdout evaluation is flagged "holdout reused" in the report and the estimate is no longer honest.

## Phase 1: Frame

1. Read the goal. Identify the target column, the unit of a row, and any evaluation metric the task names.
2. Look at the header and a few rows of the data file (for example `head -n 5`). Note nested JSON or list-in-a-string columns, datetime strings, coordinates, IDs.
3. Decide the split:
   - `--group <col>` when the same entity (customer, patient, device, store, building) appears in many rows and predictions will be made for entities not seen in training. The group column is then removed from the features.
   - `--time <col>` when predictions are about the future and rows have a timestamp (drift, seasonality, autocorrelation). The holdout becomes the most recent 20%.
   - Neither when rows are independent. You cannot use both: pick the one that matches how the model will be used.
4. Choose the metric. If the task names one, use it. Otherwise keep the default (binary `roc_auc`, multiclass `log_loss`, regression `rmse`) unless `references/metrics.md` says to override.
5. Run `aga init <data> --target <col> --goal "<goal>" [--metric <m>] [--group <col> | --time <col>] --budget <minutes>`. The budget clock starts here, so run it within the first few minutes, and pass the minutes actually left (the user's budget minus time already spent).
6. Read the profile and problem it prints: task type, class balance, column kinds, group and time candidates. If the inferred metric, split or task type is wrong, re-run `aga init` with corrected flags now, before leakage or baseline work. Re-running creates a new run but, while the current run's deadline is still in the future, it inherits that run's start time and deadline (it prints `Budget inherited ...`; `--budget` is ignored). Pass `--fresh-budget` only if you really want a new clock. To override the inferred task type use `--task {binary,multiclass,regression}`: integer targets with 20 or fewer distinct values are inferred as multiclass, so pass `--task regression` when the quantity is ordinal or count-like and a regression is intended.
   - If `aga init` prints `Hint: column '<col>' holds dates (<min> to <max>)`, decide time versus i.i.d. explicitly now and record the reason in a `frame` note. Rule: if rows are events over a calendar period and predictions are for later periods, use `--time <col>` (re-run `aga init`). Do not judge "random sample vs contiguous" from day counts: compare the printed range with the goal.
7. Write `FRAMING.md` in the working directory:
   - **Row:** what one row represents.
   - **Prediction moment:** the real-world moment the prediction is made (for example "when the order is placed", "at the start of the shift"). Derive it from the goal; if the goal is silent, choose the most useful realistic moment and say why.
   - **Columns at that moment:** every column marked known / not known / unsure at the prediction moment, with a one-line reason. Anything recorded after the moment, or during the event being predicted, is not known.
   - **Structure:** groups, time order, repeated entities, and why the chosen split mirrors deployment.
   - **Metric:** why it matches how predictions will be used.
   - **Domain hypotheses:** 5–10 feature ideas derived from first principles: geometry, physics, economics, biology, human behaviour, time. Ask what actually causes the target in the real world and which measurable quantities proxy for those causes. The references give patterns; you supply the domain.
8. `aga note "<framing summary>" --phase frame`.

## Phase 2: Hunt leakage

1. Run `aga leakage`. It prints findings by severity (`critical`, `warning`, `info`) with evidence.
2. Every `critical` finding **blocks** `aga baseline`, `aga try` and `aga fit` (exit code 2, `BLOCKED`) until you resolve it. Any single column that alone scores AUC > 0.95 (or R² > 0.9) is critical: it blocks progress until explained.
3. Resolve every critical column explicitly:
   - `aga drop <col> --reason "<why it leaks>"` when the column is not available at the prediction moment, is derived from the target, or you cannot explain its power.
   - `aga keep <col> --reason "<mechanism and why it is known at prediction time>"` only when you can name the causal mechanism and confirm the column exists before the prediction moment. "It is predictive" is never a reason.
4. Review every `warning` too (suspicious names, ID-like columns, duplicates, post-prediction timestamps, missingness that predicts the target) and decide drop or keep with a reason. `info` lines (adversarial validation) describe train/holdout drift; read them and note anything surprising.
5. Read `references/leakage.md` (the `references/` folder is in this skill's base directory, next to this file), even when the automatic checks found nothing, and review manually with it. Walk your FRAMING.md column list: drop every column marked "not known" at the prediction moment even if no check flagged it and even if it helps the score. Drop pure row identifiers and row-order columns with no meaning.
6. When unsure, drop and note it. A leaky model is worse than a slightly weaker honest one.
7. `aga status` must show `Unresolved: -` before you continue.
8. `aga note "<leakage verdict>" --phase leakage`.

## Phase 3: Baseline

Run `aga baseline`. It scores a dummy model and default LightGBM in CV on the current feature set and sets the LightGBM score as the current best. If LightGBM barely beats the dummy, re-check the framing; if it is near-perfect, re-check leakage. Note how long the command took: a `aga try` costs about the same plus your modules' own time.

## Phase 4: Feature-engineering loop

Before writing the first module, read `references/features.md` (in this skill's base directory) for patterns by data type and the pitfalls the harness checks. Each round tests one hypothesis with `aga try`:

1. State the hypothesis: what the features are, and why they should matter in the real world ("heat loss scales with surface area relative to volume, so the surface-to-volume ratio should explain energy use beyond the raw dimensions").
2. Write a feature module (template below) in the working directory, for example `features/distance.py`. The harness numbers the stored copy by trial (`04_distance.py`), so do not number the files yourself. Bundle a coherent family of 2–8 related features per module; single weak features rarely clear the keep rule alone.
3. Run `aga try <file> --note "<hypothesis>"`. It prints `ACCEPTED` or `REJECTED`, the new columns, `Delta <metric>: <Δ> ± <SE> (SE)` and a `Feature loop:` line that says whether a stop condition below is met. Accepted modules join the active feature set and later modules see their columns.
4. `aga note "<what you learned>" --phase features` after every trial, especially rejections ("raw coordinates already capture this").

Order of rounds:

1. **Domain features** from your FRAMING.md hypotheses: physical quantities, geometry, rates, elapsed times, domain ratios, parsed nested columns.
2. **Interactions and ratios** among strong raw and accepted columns (per-unit quantities, differences, products that have meaning).
3. **Group and time aggregates**: entity history, counts, rolling or lagged statistics, all learned from the training fold inside `fit()`.
4. **Encodings**: frequency encoding, smoothed target encoding fit inside `fit()`. Last, because AutoGluon already handles raw categoricals well.

Stop the loop when either happens:

- 3 rejected trials in a row (accepted ones reset the count; a trial that errors or is flagged `target_leak_suspect` is a fix-and-retry, so it neither counts nor resets it), or
- about 40% of the budget is used (the budget line shows `40% used` or more).

Do not stop while the `Feature loop:` line says to keep going, and do not stop earlier to "save time for the fit", even under a short budget: the 40% cap already leaves the larger share of the budget to AutoGluon, and a `aga try` costs seconds to a minute. When your FRAMING.md hypotheses are used up, move on to the next round type (for example combine accepted columns with strong raw columns, or compare a row with the same quantity aggregated over related rows) instead of stopping. Past the cap, AutoGluon benefits from the remaining time far more than from marginal features. Keep each module small and fast: vectorised pandas or numpy, no per-row Python loops over large data, no network, no file reads. Modules run for every CV fold, so a slow module eats the budget many times over.

Notes on the harness:

- The trial count is recorded and shown in the report. Do not spam near-duplicate trials hoping one passes by chance.
- After `aga drop` or `aga keep` changes the feature set, the next `aga try` first re-scores the current best on the new feature set ("Re-scored current best ..."), so comparisons stay like for like.
- A `target_leak_suspect` flag rejects the trial: a new column correlates with the target on the training fold far more than on the validation fold. This is almost always target information leaking through `fit()`. Fix the module (more smoothing, out-of-fold logic, or drop the idea); do not try to evade the check.
- Columns only known after the prediction moment are forbidden as feature inputs too, even inside a module.

## Phase 5: Final fit

1. `aga note "<final active feature set; which stop condition was met>" --phase fit`.
2. Run `aga fit`. The default time limit is half of the remaining budget minus 180 s (minimum 60 s); the other half is left for evaluation, the production refit and the report. Use `--time-limit <seconds>` only to leave more reserve, for example on a very large holdout where evaluation will be slow. Keep the default `best_quality` presets. A fit longer than about 9 minutes runs in the background (see "Long commands and timeouts").
3. AutoGluon builds its own stacked ensemble. Do not add your own models or stacking.
4. In temporal mode the fit disables bagging, uses the most recent 15% of training rows for tuning, then refits the best model on all training rows including that tail (60% of the time limit for the fit, 40% for that refit); the report states this.

## Phase 6: Verify and report

1. Run `aga evaluate` exactly once. It scores the holdout and prints the primary and secondary metrics, the CV proxy best, the AutoGluon validation score and gap, calibration (Brier, ECE, plot) for classifiers, and top features by permutation importance.
2. Sanity-check: a holdout far better than CV suggests leakage; far worse suggests drift, a split that does not mirror deployment, or overfit feature search. A top feature you cannot explain is a leakage suspect: say so in the narrative. `aga note "<holdout result and sanity verdict>" --phase evaluate`.
3. Always run `aga refit` next (in the background if its time limit exceeds about 9 minutes). It trains the delivered model with the same recipe on all labelled rows, holdout included; the holdout score above stays the honest estimate, and `aga evaluate` is refused from now on. `aga note "<refit rows and outcome>" --phase refit`.
   If `aga refit` refuses because the feature recipe changed since the last fit (a `try`, `drop`, `keep` or module change after the fit), the evaluated model no longer matches the recipe. Recover openly: run `aga fit`, then `aga evaluate --force` (the report flags the holdout as reused; say so in the narrative and give the first evaluation as the honest estimate), then `aga refit`. To avoid this, finish all feature and column decisions before `aga fit`.
4. Write `NARRATIVE.md` in the working directory before running the report:
   - **Top features and why they make sense** in domain terms.
   - **CV vs holdout gap** and what it implies.
   - **Calibration** (classifiers): can the probabilities be used as probabilities?
   - **Leakage decisions**: what was dropped or kept and why.
   - **What failed**: rejected hypotheses and what they taught.
   - **Residual risks**: drift, unverified columns, assumptions about the prediction moment, wrong task type if any.
5. Run `aga report`. It writes `REPORT.md` from the journal and artefacts and includes `FRAMING.md`, `NARRATIVE.md` and a decision log of your notes.
6. If the user or task supplied a file to score, run `aga predict <file> --out <predictions.csv>`, after the refit. Binary output is P(positive class), multiclass one column per class, regression the predicted value.
7. Tell the user the holdout score, the top features, the leakage decisions and where `REPORT.md` is.

## Feature module template

A feature module is a self-contained Python file defining `class Features`:

```python
"""Hypothesis: spend per visit and basket size reflect customer value; popular stores behave differently."""
import numpy as np
import pandas as pd


class Features:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Features":
        # Learn everything data-dependent here. X and y are the training fold only.
        self.store_freq_ = X["store"].value_counts(normalize=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Return only NEW or REPLACED columns, with the same index as X.
        out = pd.DataFrame(index=X.index)
        visits = X["n_visits"].replace(0, np.nan)
        out["spend_per_visit"] = X["total_spend"] / visits
        out["items_per_visit"] = X["n_items"] / visits
        out["store_freq"] = X["store"].map(self.store_freq_).astype(float).fillna(0.0)  # unseen store -> 0
        return out
```

Contract and rules:

- `fit(self, X, y)` returns `self`; `transform(self, X)` returns a DataFrame with the same index and row count as `X`. A returned column with an existing name replaces it (use this to turn a raw string column into parsed numbers); other columns are added. Modules cannot remove columns. `aga drop` removes a raw column, but then no module can read it either; to neutralise a raw column that a module still needs, replace it under the same name.
- `X` is the base features (target, dropped columns and the group column removed) plus the columns of previously accepted modules, in order. `y` is the raw target column (class labels may be strings; the positive class is the last label in sorted order). `transform` never sees `y`.
- Modules are loaded fresh for every fold and fit on that fold's training rows only. Group aggregates, target encodings, clusters and scalers must be computed in `fit()` and only applied in `transform()`.
- `transform` must give a row the same output whatever other rows are in the batch: it runs on validation folds, the holdout and single new rows at predict time. Never compute statistics over `X` inside `transform`.
- Only the `--time` column arrives as a datetime; other date columns arrive as strings: parse them with `pd.to_datetime(X[col], errors="coerce")`.
- Be defensive: coerce types, handle NaN, unseen categories and division by zero. Use only the standard library, numpy, pandas, scipy and scikit-learn. Embed domain constants (landmark coordinates, holiday dates) in the file; never read files.
- Long free-text columns are ignored by the proxy; nested JSON or list-in-a-string columns are an opportunity: parse them in a module (counts, lengths, flags, numeric fields) and replace the raw column with something useful.

## When things go wrong

- **Trial rejected with an error** (`REJECTED ...: module raised an error` plus a traceback): read the traceback, fix the module, retry once. If it fails again, drop the idea and note why.
- **`BLOCKED` (exit code 2)**: run `aga status`, then resolve each unresolved column with `aga drop <col> --reason "..."` or `aga keep <col> --reason "..."`.
- **`ERROR:` (exit code 1)**: a usage problem (wrong column name, missing baseline, unreadable file). Read the message and fix the call.
- **Budget running out**: stop feature work immediately and run `aga fit`; an unfinished feature loop is fine, an unfitted model is not.
- **A `DEACTIVATED <module>` line**: an accepted module broke after a drop or during a fit (for example it read a column that was dropped later). The harness removed it from the active set and continued. Fix the module and re-try it with `aga try`, or drop the idea, and note it.
- **`Budget inherited from run ...` on re-init**: the clock was not reset; plan around the budget line it prints.
- **Wrong framing discovered late**: if the split, metric or task type was wrong and no fit has started, re-run `aga init` with corrected flags (the budget is inherited, so pass no `--budget` and no `--fresh-budget`) and redo leakage and baseline quickly. Otherwise finish, and state the issue in `NARRATIVE.md`.
- **Long commands and timeouts**: `aga fit`, `aga refit` (and long `aga try` rounds) can run for many minutes. Claude Code's shell tool caps a foreground command at 10 minutes, so any `aga fit` or `aga refit` whose time limit (printed on its first line, or the default) exceeds about 9 minutes must run in the background: start it in the background, wait for it to complete, then run `aga status`. Shorter commands need a shell or tool timeout longer than the time limit. Never pass `--time-limit` larger than the remaining budget. If a command is killed by a tool timeout, run `aga status` before re-running anything.
- **Lost track**: `aga status` shows the problem, active modules, dropped and kept columns, unresolved findings, trial count and best score; `aga profile` reprints the data profile. Every command accepts `--run <run_id>` to address an older run; by default the latest `aga init` is used.
