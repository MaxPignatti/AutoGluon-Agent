# Benchmark data

`python benchmark/prepare.py [--only NAME]` builds the datasets listed in `registry.yaml`, splits each 80/20 (seed 42) and writes:

- `benchmark/workspace/<name>/train.csv` and `task.md` (goal, target, metric): the only files an agent or AutoGluon run may see.
- `benchmark/.hidden/<name>/test.csv` (with target), `test_features_naive.csv` (no target) and `test_features_production.csv` (no target, `leak_columns` blanked to empty values, as they would be at prediction time).

On Windows, `benchmark/run_lanes.ps1 [-Minutes 60] [-Archive NAME]` runs all datasets as two CPU-pinned lanes side by side, then scores (progress in `benchmark/results/raw/_logs/summary.txt`).

`python benchmark/run.py --dataset NAME --system {autogluon,aga}` writes predictions and `meta.json` to `benchmark/results/raw/<name>/<system>/`; `python benchmark/score.py` scores them into `benchmark/results/`.

- `run.py` (aga system) copies the agent's Claude Code transcript to `agent_transcript.jsonl` in the raw output folder (best effort: `meta.json` records `"transcript": null` when it is not found) and records `"refit": true/false` in `meta.json`.
- `score.py` adds an absolute difference (AutoGluon-Agent minus AutoGluon, in metric units) next to the relative error reduction: column `abs_diff` in `results.csv`, "Absolute difference" in `RESULTS.md` and the README table.

Downloads are cached in `benchmark/.cache/`, so reruns are offline. `.cache/`, `.hidden/` and `workspace/` are git-ignored; never commit downloaded or generated data.

## Datasets (real build)

Row and column counts are of the full table (train + hidden test), columns including the target.

| Dataset | Rows | Columns | Train / test rows | Split |
|---|---|---|---|---|
| football-xg | 19,783 | 22 | 15,889 / 3,894 | group `match_id` (800 matches) |
| ames-housing | 1,460 | 80 | 1,168 / 292 | random |
| california-housing | 20,640 | 9 | 16,512 / 4,128 | random |
| bike-sharing | 17,379 | 15 | 13,903 / 3,476 | time, last 20% of `dteday_hour` |
| bank-marketing | 41,188 | 21 | 32,950 / 8,238 | random, stratified |
| adult-planted-leak | 48,842 | 16 | 39,073 / 9,769 | random, stratified |

- **football-xg:** 19,783 shots, 10.2% goals; 99.2% of rows have a non-empty `freeze_frame_json`. The 800-match cap, taken in deterministic (competition, season, match id) order, is filled entirely by two competition-seasons (Premier League, 10,837 shots; Ligue 1, 8,946 shots).
- **bank-marketing:** `duration` is kept in the data and listed as a leak column; the target `y` keeps the original `yes`/`no` labels.
- **adult-planted-leak:** `tax_bracket_following_year` is synthetic and planted (seed 42): `high` for 85% of `>50K` rows and 5% of `<=50K` rows, otherwise `low` or `mid` with equal probability.

## Substitutions

- **NYC taxi trip duration replaced by Ames housing** (spec section 6.2). A coordinate-bearing taxi dataset is not available without a Kaggle login. Ames tests domain feature engineering instead of geo, which California housing already covers.
