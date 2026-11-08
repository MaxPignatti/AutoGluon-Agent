# Metric guide

The metric decides which feature modules are accepted and what AutoGluon optimises, so choose it in Phase 1 and do not change it afterwards.

## Rule zero

If the task or the user names a metric, use it (mapped to the closest available name below) and note it. Otherwise start from the default and override only for one of the reasons below, with the reason logged via `aga note`.

## Available metrics

| Task | Default | Allowed with `--metric` |
|---|---|---|
| binary | `roc_auc` | `roc_auc`, `pr_auc`, `log_loss`, `accuracy`, `f1_macro` |
| multiclass | `log_loss` | `log_loss`, `accuracy`, `f1_macro` |
| regression | `rmse` | `rmse`, `mae`, `r2` |

`aga evaluate` always reports secondary metrics as well (binary: log loss, accuracy, Brier and calibration; multiclass: accuracy, macro-F1; regression: MAE, R²), so the override only changes what is optimised and compared.

## When to override

- **`pr_auc` for heavy class imbalance** (positives below roughly 5%) when what matters is ranking the rare positives at the top: fraud review queues, defect screening, alerting. ROC-AUC looks flattering there because it rewards ordering the abundant negatives.
- **`log_loss` when probabilities are consumed** downstream: pricing, expected-value decisions, risk scores that are multiplied by costs, or anything compared against a threshold that will change. It rewards calibrated probabilities, not only ranking.
- **`accuracy` or `f1_macro` for hard-label decisions.** Use `accuracy` when classes are balanced and every error costs the same; `f1_macro` when every class matters equally despite imbalance.
- **`mae` for heavy-tailed regression targets** or when cost grows linearly with the error: a few extreme values dominate RMSE, so the model chases outliers. Also use it when the task speaks of typical or median error.
- **`rmse`** (default) when large errors are disproportionately costly, or when the target is roughly symmetric.
- **`r2`** ranks models exactly like `rmse` on a fixed dataset; use it only if the task asks for it.

There is no log-scale metric (such as RMSLE) and no target transform in the harness. If the task is scored on relative error, keep `rmse` or choose `mae`, and say in the narrative that the optimised metric differs from the stated one.

## How to pass it

`--metric` is an option of `aga init`:

```bash
aga init data.csv --target churned --goal "predict churn next month" --metric pr_auc --budget 55
```

If the run already exists with the wrong metric, re-run `aga init` with `--metric` during Phase 1, before `aga leakage` and `aga baseline`. It creates a new run, but while the current run's deadline is still in the future the new run inherits that run's budget clock (`--budget` is ignored; `aga init` prints `Budget inherited ...`). Pass `--fresh-budget` only if you deliberately want a new clock. Scores from different runs or metrics are not comparable; never switch metric mid-loop to rescue a feature.
