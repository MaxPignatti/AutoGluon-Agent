# Benchmark, 60 minutes, half machine per run, 2026-11-04 (run 1)

All six datasets, one run per system and dataset, 60-minute budget. Two lanes ran side by side on a Ryzen 5 9600X (6 cores, 12 threads, 32 GB), each pinned to 6 logical CPUs (3 cores) with a 14 GB AutoGluon memory limit (`benchmark/run.py --cpus/--memory-gb`); both systems were capped alike. The venv had no CatBoost, XGBoost or neural models, so AutoGluon often ran out of models well before its time limit; this applies to both systems. Not comparable with the full-machine numbers in `docs/benchmark-60min-2026-11-04/`. `RESULTS.md`, `results.csv` and `chart.png` are the raw output of `benchmark/score.py`.

Code under test: Task 17 (production refit on all labelled rows, temporal refit, date hint, background-fit guidance) plus the R36 refit gate.

## Production test set (leaky columns blank, as at prediction time)

| Dataset | Metric | AutoGluon | AutoGluon-Agent | Absolute difference | Relative error reduction | Claude cost |
|---|---|---|---|---|---|---|
| football-xg | ROC AUC | 0.8008 | **0.8258** | +0.025 | +12.6% | $1.04 |
| bike-sharing | RMSE | 68.78 | **60.36** | −8.42 | +12.2% | $0.74 |
| california-housing | RMSE | 0.4207 | **0.3836** | −0.037 | +8.8% | $0.93 |
| bank-marketing | ROC AUC | 0.8010 | **0.8171** | +0.016 | +8.1% | $0.95 |
| adult-planted-leak | ROC AUC | 0.9270 | **0.9324** | +0.005 | +7.4% | $0.72 |
| ames-housing | RMSE | 28,336 | **26,680** | −1,656 | +5.8% | $0.76 |

## What the runs show

- **ames-housing** went from a loss (−3.2% in the earlier run) to a win: the delivered model is now refit on all labelled rows, holdout included.
- **bike-sharing:** the agent framed the data as a time series. Its estimate (72.2) was pessimistic against a test RMSE of 60.4; AutoGluon's validation promised 29.7 and delivered 68.8.
- **Honest estimates:** AutoGluon-Agent's own validation score is close to its test score on five of the six datasets; on bike-sharing it was pessimistic (see above). AutoGluon's validation, trained with the leaky columns, promised 0.951 on bank-marketing and 0.983 on adult and delivered 0.801 and 0.927 once those columns are blank.
- **bike-sharing aga took 7 minutes:** in temporal mode, with only LightGBM-family models, the fit finished in 77 s of a 1,568 s limit.

## Run 2

The second run, with CatBoost, XGBoost, fastai and torch (CPU) installed, completed on 2026-11-07: see [run 2](../benchmark-60min-halfmachine-2026-11-07-run2/README.md). AutoGluon-Agent won on all six datasets again, by +2.2% to +9.3%.
