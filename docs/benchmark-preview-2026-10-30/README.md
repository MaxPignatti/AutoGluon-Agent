# Benchmark preview, 2026-10-30 (10-minute budget, 4 datasets)

This is a short exploratory run, not the planned 60-minute benchmark. It covers 4 of the 6 datasets, one run per system and dataset, with a 10-minute budget for each run. The AutoGluon in the venv had no CatBoost, XGBoost or neural models. That applies to both systems. `RESULTS.md`, `results.csv` and `chart.png` are the raw output of `benchmark/score.py`.

## Production test set (leaky columns blank, as at prediction time)

| Dataset | Metric | AutoGluon | AutoGluon-Agent | Relative error reduction |
|---|---|---|---|---|
| football-xg | ROC AUC | 0.8002 | 0.8167 | +8.3% |
| ames-housing | RMSE | 27,799 | 27,593 | +0.7% |
| bike-sharing | RMSE | 65.5 | 116.5 | −78.0% |
| adult-planted-leak | ROC AUC | 0.9259 | 0.9302 | +5.8% |

Claude cost was about $0.19–0.21 per AutoGluon-Agent run.

## Findings

1. **The leak was caught.** On adult-planted-leak, AutoGluon-Agent dropped `tax_bracket_following_year`. AutoGluon kept it and reports a cross-validated AUC of 0.98. On the naive test set, which still contains the leaky column, AutoGluon scores 0.9835. On the production test set, where the column is blank as it would be at prediction time, it falls to 0.9259. AutoGluon-Agent's validation estimate matched its test score to within 0.0004.
2. **Domain features helped football xG.** The agent built a single freeze-frame geometry module (shooting cone, goalkeeper distance). It was accepted and gave +0.016 test AUC.
3. **Temporal mode loses badly on bike sharing.** This is a design gap, not a tuning problem. In temporal mode the final AutoGluon fit never trains on the newest data:
   - the newest 20% is the locked holdout;
   - the newest 15% of the rest is passed only as `tuning_data`.

   Together that is about 32% of the timeline, all of it just before the test period. With a strong growth trend the model under-predicts. AutoGluon alone trains on all rows. The agent itself flagged the trend as the likely cause (holdout RMSE 103.6 against CV about 70). The likely fix is a final refit on all labelled rows after the one-shot holdout evaluation. AutoGluon's `refit_full` covers the tuning tail, and the harness would need a "production refit" step that adds the holdout rows back. This needs a spec decision.
4. **The feature loop still stops early.** In every run the agent stopped after 1–3 trials, at 15–40% of the budget, so that AutoGluon had time for the final fit. On bike sharing it wrote that the stop rule "had not been met". At a 10-minute budget this is arguably sensible, but it should be re-checked at 60 minutes.

## Follow-up: bike sharing at 60 minutes against 60 minutes

This rerun checks whether the 10-minute loss on bike sharing came only from the short budget. Raw rows are in `bike-sharing-60min.csv`.

| Budget | AutoGluon test RMSE | AutoGluon-Agent test RMSE | Relative error reduction | AutoGluon-Agent framing | AutoGluon-Agent validation estimate |
|---|---|---|---|---|---|
| 10 min | 65.5 | 116.5 | −78.0% | temporal | holdout 103.6 |
| 60 min | 67.9 | **65.5** | **+3.5%** | i.i.d. (random) | holdout 29.0 |

At 60 minutes AutoGluon-Agent wins narrowly (+3.5%). The win does not come from fixing the temporal gap, though. It comes from a different framing choice, and that choice was based on a wrong reading of the data:

1. **The agent framed a time series as i.i.d.** It wrote that the training file was "a random sample of hours spread over two years (469 of 731 days)". In fact it is one contiguous block, 2011-01-01 to 2012-08-07 (585 days), and the hidden test set is the period after it (2012-08-07 to 2012-12-31). With i.i.d. framing the temporal-refit gap (finding 3 above) does not apply, which is why the score recovered.
2. **The promise of an honest estimate failed here.** The holdout said RMSE 29.0, but the test came out at 65.5. AutoGluon's own validation was just as optimistic (29.7 against 67.9). Both validate on random rows and are then tested on the future. Under its 10-minute temporal framing, AutoGluon-Agent's estimate was far closer to the truth (103.6 against 116.5).
3. **A tool timeout cut the final fit short.** The agent's first `aga fit` was killed by its shell tool's timeout, and it reran the fit with a 900 s limit. So AutoGluon-Agent's AutoGluon trained for about 15 minutes, against about 58 for AutoGluon alone. SKILL.md says to use a longer timeout or the background. Claude Code's shell tool is capped at 10 minutes, so a fit longer than about 9 minutes must run in the background. The skill should say that plainly.

The fixes this points to, none of them applied yet:
- In the skill: detect a time column whose training rows form one contiguous block, and default to temporal framing.
- In the harness: a production refit on all labelled rows after the one-shot evaluation, so that temporal mode stops discarding the newest data.
- In the skill: run `aga fit` in the background whenever its time limit is above about 9 minutes.
