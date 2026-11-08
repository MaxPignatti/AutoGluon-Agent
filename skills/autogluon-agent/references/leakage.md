# Leakage playbook

Leakage is information in the training data that will not exist when the model is used. It makes CV and holdout scores look great and the deployed model fail. The test for every column is one question: **at the prediction moment written in FRAMING.md, does this value already exist, with this exact meaning?** If not, drop it, however much it helps.

## How to review

1. Run `aga leakage` and read every finding. Critical findings block progress; warnings still need a verdict.
2. Walk the FRAMING.md column list. For each column ask: who records it, when, and could its value depend on the outcome?
3. Read a few sample values of every suspicious column. Names lie; values rarely do.
4. Decide with `aga drop <col> --reason "..."` or `aga keep <col> --reason "..."`. A keep reason names the mechanism and the moment the value is recorded. "Predictive" or "not flagged" is not a reason.
5. When you cannot establish when a column is recorded, drop it and say so in the narrative.

Red flags at any later stage: one feature dominating importance, a score far above what the domain allows, holdout much better than CV, a feature you cannot explain.

## Patterns

### 1. Post-outcome columns
Values recorded after, or because of, the outcome.
*Example:* predicting customer churn with a `cancellation_reason` column. It is filled only for customers who left, so its presence alone reveals the target.
Also watch for arithmetic relatives of the target: parts that sum to it, the target in other units, a balance updated by the event being predicted.

### 2. Columns measured during the event being predicted
The value accumulates while the outcome unfolds, so it encodes the outcome.
*Example:* predicting at ticket creation whether a support ticket will be resolved within a day, using `n_replies` or `last_agent_touch`. Those grow while the ticket is being worked on and are only final after resolution.
Ask: is the value complete at the prediction moment, or only once the event is over?

### 3. Target-encoded IDs
A categorical or ID column already replaced by an outcome statistic computed over all rows, including the row itself.
*Example:* a `merchant_risk` column equal to each merchant's fraud rate over the whole file. Each row's own label is inside its feature, and holdout labels are inside training features.
Drop it, keep the raw ID if it is legitimate, and let a feature module compute the encoding in `fit()` from the training fold only, with smoothing.

### 4. Global aggregates and normalisation over the full dataset
Statistics computed over all rows (including the holdout or the future) baked into a column.
*Example:* a `price_vs_category_mean` column precomputed using the mean over the entire file, or a percentile rank over the full year when predictions are made mid-year.
Inside your own modules, compute every mean, scale, rank or cluster in `fit()` and only apply it in `transform()`.

### 5. Duplicates across splits
The same entity or near-identical rows appear in both training and evaluation, so the model memorises instead of generalising.
*Example:* monthly snapshots of the same customer with an i.i.d. split: the model recognises the customer from stable attributes and copies the label it saw in another month.
Fix by framing: use `--group <entity>` so whole entities fall on one side. `aga leakage` reports duplicate feature rows across train and holdout and group overlap.

### 6. Temporal look-ahead
Features that use information from after the prediction time, or a random split on data where the future predicts the past.
*Example:* a `rolling_mean_sales` computed with a centred window (it averages the next days), or a `next_visit_date` column, when predicting tomorrow's demand.
Use `--time <col>` when predictions concern the future. In modules, lag within each entity and use only history strictly before each row; history that involves the target must be built in `fit()` from the training fold.

### 7. Missingness caused by the outcome
Whether a value is missing depends on the outcome, so the missing indicator alone predicts the target.
*Example:* a `follow_up_score` that is blank for every case that ended before the follow-up visit. Missing means "ended early", which is the outcome.
`aga leakage` flags missingness that predicts the target. Keep the column only if values are missing for reasons that exist before the prediction moment (for example, a sensor not installed).

### 8. Another model's score of the same target
A column that is itself a prediction of the target, made by a model that may have seen these outcomes or that runs after the prediction moment.
*Example:* a vendor `default_probability` or an internal `expected_value` column refreshed nightly from realised outcomes. It can be fitted on these very rows, and it is often unavailable at decision time.
Keep only if you know it is produced before the prediction moment by a model that never saw these rows' outcomes; say so in the keep reason. Otherwise drop it.

### 9. Row identifiers and row order
Sequential IDs, file row numbers or insertion timestamps that correlate with the target because the file was sorted or collected in phases.
*Example:* an auto-increment `record_id` where positive cases were appended at the end of the export.
`aga leakage` flags ID-like and monotonic columns correlated with the target. Drop identifiers that carry no real-world meaning.

## Verdicts in the journal

Write reasons a reviewer can check:

- `aga drop cancellation_reason --reason "filled only after a customer leaves; not known at prediction time"`
- `aga keep sensor_age --reason "installation date is recorded at setup, before any reading; strong because wear drives failures"`
