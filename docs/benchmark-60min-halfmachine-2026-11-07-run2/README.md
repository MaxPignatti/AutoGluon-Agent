# Benchmark, 60 minutes, half machine per run, all model families, 2026-11-07 (run 2)

All six datasets, one run per system and dataset, 60-minute budget, same setup as [run 1](../benchmark-60min-halfmachine-2026-11-04/README.md): two lanes side by side on a Ryzen 5 9600X (6 cores, 12 threads, 32 GB), each pinned to 6 logical CPUs (3 cores) with a 14 GB AutoGluon memory limit (`benchmark/run.py --cpus/--memory-gb`); both systems were capped alike. The difference from run 1: CatBoost, XGBoost, fastai and torch (CPU) were installed, so AutoGluon `best_quality` (and AutoGluon-Agent's `aga fit`, which also uses AutoGluon) could use every model family and ran close to the full hour. `ray` was not installed in either run, so AutoGluon fell back to its normal (non-memory-safe) fit. `RESULTS.md`, `results.csv` and `chart.png` are the raw output of `benchmark/score.py`.

Code under test: the same as run 1 plus the final-review fixes (commit `de568fb`).

The run was interrupted twice and resumed with `benchmark/run_lanes.ps1 -Resume`, which reruns only runs not marked ok. Every reported run is a complete, uninterrupted 60-minute run: the four aga runs that failed at startup (the headless Claude login had expired, `$0` spent) and the two AutoGluon runs cut by a shutdown and a power cut were re-run from scratch. AutoGluon and aga runs of the same dataset therefore did not always run at the same time of day, but always on the same lane and machine.

## Production test set (leaky columns blank, as at prediction time)

| Dataset | Metric | AutoGluon | AutoGluon-Agent | Absolute difference | Relative error reduction | Run 1 | Claude cost |
|---|---|---|---|---|---|---|---|
| football-xg | ROC AUC | 0.8109 | **0.8285** | +0.018 | +9.3% | +12.6% | $0.95 |
| bank-marketing | ROC AUC | 0.7986 | **0.8166** | +0.018 | +8.9% | +8.1% | $0.76 |
| california-housing | RMSE | 0.4152 | **0.3831** | −0.032 | +7.7% | +8.8% | $0.87 |
| bike-sharing | RMSE | 64.95 | **62.02** | −2.93 | +4.5% | +12.2% | $0.91 |
| adult-planted-leak | ROC AUC | 0.9300 | **0.9328** | +0.003 | +4.1% | +7.4% | $0.77 |
| ames-housing | RMSE | 27,582 | **26,982** | −600 | +2.2% | +5.8% | $0.81 |

Total Claude cost: $5.07.

## What the runs show

- **AutoGluon-Agent still wins on all six datasets, by less.** The extra model families improved AutoGluon on every dataset (for example football-xg 0.8008 → 0.8109, bike-sharing 68.8 → 65.0) and changed AutoGluon-Agent's scores much less, since its gains come mostly from features and leakage handling rather than from the model search. bank-marketing is the only dataset where the margin grew.
- **Leakage track.** AutoGluon-Agent dropped `duration` on bank-marketing and `tax_bracket_following_year` on adult. AutoGluon, trained with them, had validation scores of 0.951 and 0.983 and delivered 0.799 and 0.930 once those columns are blank (gaps −0.152 and −0.053). AutoGluon-Agent's validation-to-test gaps were +0.010 and +0.001.
- **Honest estimates.** AutoGluon-Agent's locked-holdout estimate was within 0.03 AUC or a few percent RMSE of the test score on five datasets. On bike-sharing (temporal split) the holdout RMSE was 84.2 against a test RMSE of 62.0: pessimistic again, as in run 1. AutoGluon's validation promised 29.3 and delivered 65.0.
- **Feature work.** Accepted modules per run: football-xg 4 of 7 trials, california-housing 3 of 9, bike-sharing 3 of 8, bank-marketing 1 of 5, adult 1 of 5, ames-housing 0 of 3. On ames the gain over AutoGluon comes from the leakage/framing work and the refit, not from new features.
- **All six aga runs refit the delivered model** on all labelled rows after their one-shot holdout evaluation.

## Naive test set

With the leaky columns present in the test features, AutoGluon scores 0.958 on bank-marketing and 0.984 on adult (AutoGluon-Agent 0.817 and 0.933). That score cannot be reached in production, where the leaky column is not known at prediction time. See `RESULTS.md`.

## Caveats

Single seed, one run per system and dataset; half a machine per run, with two lanes adding some cache and memory-bandwidth noise. Not comparable with the full-machine numbers in `docs/benchmark-60min-2026-11-04/`.
