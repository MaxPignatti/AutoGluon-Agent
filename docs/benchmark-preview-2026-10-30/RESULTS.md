# Benchmark results

Relative error reduction is AutoGluon-Agent against AutoGluon `best_quality`: positive means AutoGluon-Agent's error is lower. For `roc_auc` the error is 1 - AUC; for `rmse`, `mae` and `log_loss` it is the metric itself. The CV-vs-test gap is test score minus the system's own validation score, in the metric's units.

## Production

Production variant: leaky columns are blank in the test features, as at prediction time.

| Dataset | Task | Metric | AutoGluon (test) | AutoGluon-Agent (test) | Relative error reduction | CV-vs-test gap (AutoGluon / AutoGluon-Agent) | Wall-clock min (AutoGluon / AutoGluon-Agent) | Claude cost (USD) |
|---|---|---|---|---|---|---|---|---|
| football-xg | binary | roc_auc | 0.8002 | 0.8167 | +8.3% | 0.0159 / 0.0125 | 10.9 / 8.6 | 0.19 |
| ames-housing | regression | rmse | 27798.9687 | 27593.0745 | +0.7% | 3040.0419 / 1309.3766 | 8.1 / 7.5 | 0.20 |
| bike-sharing | regression | rmse | 65.4614 | 116.5282 | -78.0% | 35.9373 / 45.3600 | 8.3 / 6.0 | 0.21 |
| adult-planted-leak | binary | roc_auc | 0.9259 | 0.9302 | +5.8% | -0.0568 / -0.0004 | 8.5 / 9.4 | 0.19 |

## Naive

Naive variant: test features still contain the leaky columns.

| Dataset | Task | Metric | AutoGluon (test) | AutoGluon-Agent (test) | Relative error reduction | CV-vs-test gap (AutoGluon / AutoGluon-Agent) | Wall-clock min (AutoGluon / AutoGluon-Agent) | Claude cost (USD) |
|---|---|---|---|---|---|---|---|---|
| football-xg | binary | roc_auc | 0.8002 | 0.8167 | +8.3% | 0.0159 / 0.0125 | 10.9 / 8.6 | 0.19 |
| ames-housing | regression | rmse | 27798.9687 | 27593.0745 | +0.7% | 3040.0419 / 1309.3766 | 8.1 / 7.5 | 0.20 |
| bike-sharing | regression | rmse | 65.4614 | 116.5282 | -78.0% | 35.9373 / 45.3600 | 8.3 / 6.0 | 0.21 |
| adult-planted-leak | binary | roc_auc | 0.9835 | 0.9302 | -321.8% | 0.0007 / -0.0004 | 8.5 / 9.4 | 0.19 |
