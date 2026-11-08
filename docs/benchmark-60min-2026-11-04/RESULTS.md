# Benchmark results

Relative error reduction is AutoGluon-Agent against AutoGluon `best_quality`: positive means AutoGluon-Agent's error is lower. For `roc_auc` the error is 1 - AUC; for `rmse`, `mae` and `log_loss` it is the metric itself. The CV-vs-test gap is test score minus the system's own validation score, in the metric's units.

## Production

Production variant: leaky columns are blank in the test features, as at prediction time.

| Dataset | Task | Metric | AutoGluon (test) | AutoGluon-Agent (test) | Relative error reduction | CV-vs-test gap (AutoGluon / AutoGluon-Agent) | Wall-clock min (AutoGluon / AutoGluon-Agent) | Claude cost (USD) |
|---|---|---|---|---|---|---|---|---|
| football-xg | binary | roc_auc | 0.8011 | 0.8263 | +12.7% | 0.0124 / 0.0113 | 58.3 / 59.8 | 1.03 |
| ames-housing | regression | rmse | 28335.8690 | 29241.0747 | -3.2% | 3672.6400 / 3601.9519 | 44.3 / 44.1 | 0.79 |
| bike-sharing | regression | rmse | 67.8586 | 65.5089 | +3.5% | 38.1278 / 34.4540 | 54.0 / 42.8 | 0.31 |
| adult-planted-leak | binary | roc_auc | 0.9288 | 0.9317 | +4.0% | -0.0540 / 0.0016 | 29.3 / 36.6 | 0.56 |

## Naive

Naive variant: test features still contain the leaky columns.

| Dataset | Task | Metric | AutoGluon (test) | AutoGluon-Agent (test) | Relative error reduction | CV-vs-test gap (AutoGluon / AutoGluon-Agent) | Wall-clock min (AutoGluon / AutoGluon-Agent) | Claude cost (USD) |
|---|---|---|---|---|---|---|---|---|
| football-xg | binary | roc_auc | 0.8011 | 0.8263 | +12.7% | 0.0124 / 0.0113 | 58.3 / 59.8 | 1.03 |
| ames-housing | regression | rmse | 28335.8690 | 29241.0747 | -3.2% | 3672.6400 / 3601.9519 | 44.3 / 44.1 | 0.79 |
| bike-sharing | regression | rmse | 67.8586 | 65.5089 | +3.5% | 38.1278 / 34.4540 | 54.0 / 42.8 | 0.31 |
| adult-planted-leak | binary | roc_auc | 0.9839 | 0.9317 | -324.9% | 0.0012 / 0.0016 | 29.3 / 36.6 | 0.56 |
