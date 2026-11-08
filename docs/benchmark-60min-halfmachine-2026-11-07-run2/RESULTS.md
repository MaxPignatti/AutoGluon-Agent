# Benchmark results

Absolute difference is AutoGluon-Agent's test score minus AutoGluon's, in the metric's units (for `roc_auc` positive is better, for `rmse`, `mae` and `log_loss` negative is better). Relative error reduction is AutoGluon-Agent against AutoGluon `best_quality`: positive means AutoGluon-Agent's error is lower. For `roc_auc` the error is 1 - AUC; for `rmse`, `mae` and `log_loss` it is the metric itself. The CV-vs-test gap is test score minus the system's own validation score, in the metric's units.

## Production

Production variant: leaky columns are blank in the test features, as at prediction time.

| Dataset | Task | Metric | AutoGluon (test) | AutoGluon-Agent (test) | Absolute difference | Relative error reduction | CV-vs-test gap (AutoGluon / AutoGluon-Agent) | Wall-clock min (AutoGluon / AutoGluon-Agent) | Claude cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| football-xg | binary | roc_auc | 0.8109 | 0.8285 | +0.0176 | +9.3% | 0.0128 / 0.0077 | 58.4 / 56.2 | 0.95 |
| ames-housing | regression | rmse | 27581.7750 | 26982.0159 | -599.7591 | +2.2% | 4254.9549 / 2277.4982 | 58.1 / 54.7 | 0.81 |
| california-housing | regression | rmse | 0.4152 | 0.3831 | -0.0321 | +7.7% | -0.0088 / -0.0160 | 58.2 / 56.0 | 0.87 |
| bike-sharing | regression | rmse | 64.9505 | 62.0162 | -2.9343 | +4.5% | 35.6841 / 15.7375 | 58.4 / 37.5 | 0.91 |
| bank-marketing | binary | roc_auc | 0.7986 | 0.8166 | +0.0180 | +8.9% | -0.1522 / 0.0099 | 58.4 / 52.1 | 0.76 |
| adult-planted-leak | binary | roc_auc | 0.9300 | 0.9328 | +0.0029 | +4.1% | -0.0530 / 0.0012 | 58.1 / 51.8 | 0.77 |

## Naive

Naive variant: test features still contain the leaky columns.

| Dataset | Task | Metric | AutoGluon (test) | AutoGluon-Agent (test) | Absolute difference | Relative error reduction | CV-vs-test gap (AutoGluon / AutoGluon-Agent) | Wall-clock min (AutoGluon / AutoGluon-Agent) | Claude cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| football-xg | binary | roc_auc | 0.8109 | 0.8285 | +0.0176 | +9.3% | 0.0128 / 0.0077 | 58.4 / 56.2 | 0.95 |
| ames-housing | regression | rmse | 27581.7750 | 26982.0159 | -599.7591 | +2.2% | 4254.9549 / 2277.4982 | 58.1 / 54.7 | 0.81 |
| california-housing | regression | rmse | 0.4152 | 0.3831 | -0.0321 | +7.7% | -0.0088 / -0.0160 | 58.2 / 56.0 | 0.87 |
| bike-sharing | regression | rmse | 64.9505 | 62.0162 | -2.9343 | +4.5% | 35.6841 / 15.7375 | 58.4 / 37.5 | 0.91 |
| bank-marketing | binary | roc_auc | 0.9575 | 0.8166 | -0.1409 | -331.7% | 0.0067 / 0.0099 | 58.4 / 52.1 | 0.76 |
| adult-planted-leak | binary | roc_auc | 0.9841 | 0.9328 | -0.0513 | -322.1% | 0.0011 / 0.0012 | 58.1 / 51.8 | 0.77 |
