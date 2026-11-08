# Benchmark results

Absolute difference is AutoGluon-Agent's test score minus AutoGluon's, in the metric's units (for `roc_auc` positive is better, for `rmse`, `mae` and `log_loss` negative is better). Relative error reduction is AutoGluon-Agent against AutoGluon `best_quality`: positive means AutoGluon-Agent's error is lower. For `roc_auc` the error is 1 - AUC; for `rmse`, `mae` and `log_loss` it is the metric itself. The CV-vs-test gap is test score minus the system's own validation score, in the metric's units.

## Production

Production variant: leaky columns are blank in the test features, as at prediction time.

| Dataset | Task | Metric | AutoGluon (test) | AutoGluon-Agent (test) | Absolute difference | Relative error reduction | CV-vs-test gap (AutoGluon / AutoGluon-Agent) | Wall-clock min (AutoGluon / AutoGluon-Agent) | Claude cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| football-xg | binary | roc_auc | 0.8008 | 0.8258 | +0.0251 | +12.6% | 0.0144 / 0.0112 | 58.7 / 56.6 | 1.04 |
| ames-housing | regression | rmse | 28335.8690 | 26679.9038 | -1655.9652 | +5.8% | 3672.6400 / 1040.7811 | 11.0 / 32.8 | 0.76 |
| california-housing | regression | rmse | 0.4207 | 0.3836 | -0.0371 | +8.8% | -0.0089 / 0.0000 | 25.7 / 48.6 | 0.93 |
| bike-sharing | regression | rmse | 68.7777 | 60.3582 | -8.4195 | +12.2% | 39.0340 / -11.8322 | 28.9 / 7.0 | 0.74 |
| bank-marketing | binary | roc_auc | 0.8010 | 0.8171 | +0.0161 | +8.1% | -0.1503 / 0.0107 | 21.6 / 31.0 | 0.95 |
| adult-planted-leak | binary | roc_auc | 0.9270 | 0.9324 | +0.0054 | +7.4% | -0.0559 / 0.0023 | 23.2 / 54.4 | 0.72 |

## Naive

Naive variant: test features still contain the leaky columns.

| Dataset | Task | Metric | AutoGluon (test) | AutoGluon-Agent (test) | Absolute difference | Relative error reduction | CV-vs-test gap (AutoGluon / AutoGluon-Agent) | Wall-clock min (AutoGluon / AutoGluon-Agent) | Claude cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| football-xg | binary | roc_auc | 0.8008 | 0.8258 | +0.0251 | +12.6% | 0.0144 / 0.0112 | 58.7 / 56.6 | 1.04 |
| ames-housing | regression | rmse | 28335.8690 | 26679.9038 | -1655.9652 | +5.8% | 3672.6400 / 1040.7811 | 11.0 / 32.8 | 0.76 |
| california-housing | regression | rmse | 0.4207 | 0.3836 | -0.0371 | +8.8% | -0.0089 / 0.0000 | 25.7 / 48.6 | 0.93 |
| bike-sharing | regression | rmse | 68.7777 | 60.3582 | -8.4195 | +12.2% | 39.0340 / -11.8322 | 28.9 / 7.0 | 0.74 |
| bank-marketing | binary | roc_auc | 0.9566 | 0.8171 | -0.1395 | -321.8% | 0.0053 / 0.0107 | 21.6 / 31.0 | 0.95 |
| adult-planted-leak | binary | roc_auc | 0.9837 | 0.9324 | -0.0513 | -315.1% | 0.0008 / 0.0023 | 23.2 / 54.4 | 0.72 |
