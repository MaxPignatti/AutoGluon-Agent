# Benchmark, 60 minutes per run (partial), 2026-11-04

Four of the six datasets, one run per system and dataset, 60-minute budget, runs one at a time on the full machine. Bike sharing was run on 2026-10-30, the others on 2026-11-04. The queue was stopped on purpose before california-housing and bank-marketing so that the fixes below could go in first. `RESULTS.md`, `results.csv` and `chart.png` are the raw output of `benchmark/score.py`. The AutoGluon in the venv had no CatBoost, XGBoost or neural models; this applies to both systems.

## Production test set (leaky columns blank, as at prediction time)

| Dataset | Metric | AutoGluon | AutoGluon-Agent | Absolute difference | Relative error reduction | Claude cost |
|---|---|---|---|---|---|---|
| football-xg | ROC AUC | 0.8011 | **0.8263** | +0.025 | +12.7% | $1.03 |
| adult-planted-leak | ROC AUC | 0.9288 | **0.9317** | +0.003 | +4.0% | $0.56 |
| bike-sharing | RMSE | 67.9 | **65.5** | −2.4 | +3.5% | $0.31 |
| ames-housing | RMSE | **28,336** | 29,241 | +905 | −3.2% | $0.79 |

Relative error reduction uses error = 1 − AUC for ROC AUC and the metric itself for RMSE.

## What the runs show

- **football-xg:** domain features win. The agent kept 5 of 8 feature modules (shot geometry, freeze-frame defenders in the shooting triangle, visible goal, body part × distance, game clock). Its holdout estimate was 0.805 against a test score of 0.826.
- **adult-planted-leak:** the leak was dropped. AutoGluon's validation promised 0.983 and delivered 0.929. AutoGluon-Agent promised 0.930 and delivered 0.932.
- **bike-sharing:** a narrow win, but for the wrong reason. The agent framed the time series as i.i.d., so its estimate was too optimistic (29.0 against 65.5). See `docs/benchmark-preview-2026-10-30/README.md`.
- **ames-housing:** a small loss, as expected. All three feature ideas were rejected, so the final model is plain AutoGluon trained on 80% of the rows: the other 20% is the locked holdout, which is never added back. With 1,168 training rows, losing 20% of them matters.
- **Feature loop at 60 minutes:** the loop worked as designed. It stopped through the `Feature loop:` status line after three rejections in a row, at 5–23% of the budget, and the final fit ran in the background.

## Fixes to make before the remaining runs

1. **Production refit:** after the one-shot holdout evaluation, retrain the final model on all labelled rows (train plus holdout, and the temporal tuning tail). Ames and bike-sharing both show the cost of not doing so.
2. **Temporal detection:** the skill should recognise a time column whose rows form one contiguous block and default to temporal framing.
3. **Long fits run in the background:** the shell tool caps a command at 10 minutes, so any fit longer than about 9 minutes must run in the background. At 60 minutes the agent already did this on its own, after a timeout on football-xg.
