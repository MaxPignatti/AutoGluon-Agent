# Skill check: headless runs on California housing (2026-10-29 plan, run 2026-10-30)

Task 15 of the v1 plan: run the `autogluon-agent` skill for real, headless, review what the agent did against `SKILL.md`, fix the deviations, and run once more.

## What ran

```
uv run python benchmark/prepare.py --only california-housing    # 16,512 train / 4,128 hidden test rows
uv run python benchmark/run.py --dataset california-housing --system aga --minutes 15
```

`run.py` copies the workspace (`train.csv`, `task.md`) to a fresh temp directory and runs `claude -p "<prompt>" --plugin-dir <repo> --output-format json --allowedTools Bash,Read,Write,Edit,Glob,Grep` there. Machine: Windows 11, `claude` 2.1.303, `aga` 0.1.0 from the repo venv.

Evidence used:

- the agent's full session transcript. `run.py` does not save it, but Claude Code keeps it at `~/.claude/projects/<encoded workspace path>/<session_id>.jsonl`. `claude_output.json` (saved by `run.py`) holds only the final result text, usage and cost.
- `benchmark/results/raw/california-housing/aga/aga_run/` (`journal.jsonl`, `problem.yaml`, `REPORT.md`, copied by `run.py`).
- `FRAMING.md`, `NARRATIVE.md`, `REPORT.md` and `features/*.py` in the workspace. `run.py` deletes the temp workspace at the end, so these were mirrored out while the run was in progress. `REPORT.md` in `aga_run/` includes FRAMING.md and NARRATIVE.md anyway.
- holdout RMSE on the benchmark's hidden test set, computed directly from `preds_*.csv` and `benchmark/.hidden/california-housing/test.csv`. `score.py` was not run because it rewrites the repository README.

## Results

| | Run 1 (skill as of 55f7788) | Run 2 (after fixes 1, 2a, 3, 4, 5) |
|---|---|---|
| `meta.json` status | ok | ok |
| Wall time (`run.py`) | 870 s (14.5 min) | 793 s (13.2 min) |
| Cost | $0.57 | $0.22 |
| `aga init --budget` | 14 | 14 |
| Trials (accepted) | 3 (2) | 2 (1) |
| `aga note` entries | 3 | 6 |
| Fit time | 545 s (default limit) | ~480 s (`--time-limit 480`) |
| CV proxy best RMSE | 0.4165 | 0.4440 |
| AutoGluon validation RMSE | 0.4083 | 0.4253 |
| **aga holdout RMSE** (internal 20% of train.csv) | **0.3853** | **0.4017** |
| **Hidden test RMSE** (benchmark test set, naive = production) | **0.3942** | **0.4110** |
| Budget left at the end | 1.0 min | 1.9 min |

Both runs met Step 1's expectations: FRAMING.md, NARRATIVE.md and REPORT.md were all written, the journal shows at least one `trial` and exactly one `evaluate`, and no holdout was reused.

Run 1 timeline (agent clock): init at +0:36, leakage at +0:58, baseline at +1:12, three trials from +1:39 to +2:15, fit from +2:25 to +11:40, evaluate at +13:10, report at +13:32, final message at +13:41.
Run 2: init at +0:19, leakage and status at +0:26, baseline at +1:01, two trials at +1:15 and +1:42, fit from +1:56 to +10:04, evaluate at +12:09, report at +12:26, final message at +12:32.

The run-2 score is lower most likely mainly because it tried only two modules and never tried the spatial neighbour-target module that was run 1's strongest feature, a consequence of the early stop (deviation 2). This is a hypothesis: there is one run on each side, and the agent's choices vary from run to run.

## Step 2 checklist

| Item | Run 1 | Run 2 |
|---|---|---|
| Phases run in order | Mostly. Frame, leakage, baseline, features, fit, evaluate and report ran in order. `aga leakage` ran before FRAMING.md existed: the first Write failed (deviation 4) and the file was written later. | Mostly. The agent chained `aga init && aga leakage && aga status` and wrote FRAMING.md after the leakage check (minor, see deviation 6). |
| Reasoning logged with `aga note` | **Partial.** Only 3 notes. Nothing was tagged `leakage` or `fit`. The leakage verdict, all three trial lessons and the stop decision went into one `features` note. | **Pass.** 6 notes, one each for frame and leakage, one per trial, plus fit and evaluate. |
| No direct read of `.aga/` files | Pass. No command or Read touched `.aga/`. | Pass. |
| Budget respected (15 min + 5 min grace) | Pass (14.5 min). | Pass (13.2 min). |
| At least one domain feature tried | Pass. Haversine distances to metro centres and to coast points were accepted (−0.022), and a k-nearest-neighbour target mean computed out of fold inside `fit()` was accepted (−0.026). | Pass. Distances to metro centres plus ratios were accepted (−0.020). A coast-distance module was rejected. |
| No questions asked (autonomous) | Pass. | Pass. |
| No models outside `aga`, no stats hard-coded from the raw file | Pass. | Pass. |

## Deviations and fixes

### 1. Reasoning batched into one note (run 1)

Evidence: the run-1 journal has notes only for `frame`, `features` and `evaluate`. The `features` note bundles the leakage verdict, the three trial lessons and the decision to stop. SKILL.md asked for a note "at every decision", but the agent had no concrete list of which notes to write.

Fix (SKILL.md, hard rules): one note per decision, tagged with its phase, and an explicit minimum: `frame`, `leakage` (even "no findings"), `features` after every trial, `fit` (final set and which stop condition was met), and `evaluate`. SKILL.md also now says notes are cheap to chain with the next command. Phase 2 gained step 8 (leakage note), Phase 4 step 4 says "after every trial", and Phase 5 step 1 and Phase 6 step 2 spell out their notes.

Run 2: **fixed.** The journal has exactly the required set of notes, tagged correctly.

### 2. Feature loop stopped before either stop condition (runs 1 and 2)

Evidence: run 1 stopped after 3 trials at 13% of the budget with the last trial accepted, noting "Stopping FE early to give AutoGluon the budget". Run 2 stopped after 2 trials at 12%, noting "stopping loop early to give AutoGluon the budget (benchmark time)". In both runs the agent quoted the SKILL.md sentence "AutoGluon benefits from the remaining time far more than from marginal features" as its reason.

Fix 2a (SKILL.md, before run 2): after the stop rule, SKILL.md now says not to stop earlier "to save time for the fit". The 40% cap already gives AutoGluon the larger share, and a trial costs seconds to a minute. When the FRAMING hypotheses are used up, the agent should move to the next round type instead of stopping. The AutoGluon sentence now reads "Past the cap, ...".

Run 2: **not fixed.** The agent stopped even earlier, citing "benchmark time".

Fix 2b (CLI message plus SKILL.md, after run 2): `aga try` now prints a status line after `Trials:`:

```
Feature loop: 1 rejected in a row, 12% of the budget used. Keep going until 3 rejected in a row or 40% used.
Feature loop: stop condition met (3 rejected in a row). Run 'aga fit' next.
```

The line is guidance only and nothing is enforced. SKILL.md describes the line in Phase 4 step 3 and says not to stop while it says to keep going, "even under a short budget". The aim is to put the rule in front of the agent at the moment it decides, instead of only in SKILL.md. The rejection count matches the SKILL.md stop rule. Accepted trials reset it. Trials that error (the module raised, or broke the contract) or are flagged `target_leak_suspect` are fix-and-retry attempts: they neither count nor reset it. The journal's `trial` entries now record this in a new `errored` field. Tests: `tests/test_features.py::test_try_prints_feature_loop_stop_status`, `test_feature_loop_streak_restarts_after_accept`, `test_feature_loop_error_and_leak_trials_neither_count_nor_reset` and `test_feature_loop_budget_branch`. **The line has not been validated in a live run.** The brief allows only one re-run, and controller ruling R28 makes Task 16's benchmark aga runs its live validation.

### 3. References never read (runs 1 and 2)

Evidence: neither transcript contains a Read of `references/features.md` or `references/leakage.md`. In fact neither run made any Read call. Phase 2 step 5, the manual leakage review against `references/leakage.md`, was skipped in both runs.

Fix (SKILL.md, before run 2): the reading is now an explicit step. Phase 2 step 5 says "Read `references/leakage.md` (the `references/` folder is in this skill's base directory, next to this file), even when the automatic checks found nothing". Phase 4 opens with "Before writing the first module, read `references/features.md`".

Run 2: **not fixed.** The agent still read nothing. On this dataset the impact was nil: all columns are numeric census aggregates known at prediction time, and the run-1 modules followed the fit/transform rules without the playbook. Left open. More SKILL.md wording did not change the behaviour, and the CLI cannot point at the skill's files.

### 4. FRAMING.md write refused because of an 8.3 short path (run 1; benchmark tooling)

Evidence: the workspace was `C:\Users\USERNA~1\AppData\Local\Temp\aga-bench-...`, the short-name form `tempfile.mkdtemp` returns on this machine. Claude Code's Write tool refused it ("suspicious Windows path pattern that requires manual approval"). The agent lost about 20 s, re-resolved the path with `pwd -W` and wrote to the long form. This is also why FRAMING.md appeared after `aga leakage`.

Fix (`benchmark/run.py`): `Path(tempfile.mkdtemp(...)).resolve()` expands the short name to the long form.

Run 2: **fixed.** The agent worked in `C:/Users/username/AppData/Local/Temp/aga-bench-...` and hit no permission refusal. Run 2 created files with Bash heredocs rather than Write, so Write itself was not exercised.

### 5. Feature files numbered twice (run 1, cosmetic)

Evidence: the agent followed SKILL.md's example `features/03_distance.py` and numbered its own files. The harness adds the trial number as well, so the report showed `01_01_geo.py`, `02_02_dwelling.py`, `03_03_knn.py`.

Fix (SKILL.md): the example is now `features/distance.py`, with a note that the harness numbers the stored copy.

Run 2: **fixed** (`01_geo.py`, `02_coast.py`).

### 6. Minor, accepted without a fix

- **FRAMING.md columns grouped (both runs; terser in run 2).** Run 2 has no per-column known/not-known line ("all known (census aggregates, coordinates)") and lists five hypotheses on one line. With 8 columns that are all census aggregates or coordinates, the content is right but thinner than SKILL.md asks for. No fix: SKILL.md already states the format, and more emphasis would compete with the fixes above.
- **`aga status` (`Unresolved: -`) not checked in run 1.** There were no findings to resolve. Run 2 did check it.
- **Leakage check before FRAMING.md in run 2.** The agent chained init and leakage. The automatic check does not depend on FRAMING.md, and the manual review that does was skipped (deviation 3).
- **`aga baseline` run in the background in run 1** (`(aga baseline > baseline.log 2>&1 &)`) while modules were being written. The agent read the log before the first `aga try`, so there was no overlap. Not worth a rule yet.
- **`--time-limit 480` in run 2** where the default would have been about 564 s. This is allowed: it leaves more reserve. The run ended with 1.9 min left.

## Environment observations (not skill deviations)

- AutoGluon in the venv has no CatBoost, XGBoost, torch (no neural nets or FastAI) and no ray (folds fit sequentially). Both runs noted this in NARRATIVE.md. The final ensembles contain only LightGBM, RandomForest and ExtraTrees models. The plain-AutoGluon benchmark system shares the same environment, so the comparison stays fair, but absolute scores are lower than a full install would give.
- The prompt from `run.py` doubles the full stop after the goal ("block group.."), because the registry goal already ends with one. This had no visible effect.
- The first attempt at run 2 crashed at `out.mkdir` with `FileExistsError`: `shutil.rmtree(..., ignore_errors=True)` left an empty `aga_run` directory behind, probably held by OneDrive sync. Removing it by hand and re-launching worked. No fix.
