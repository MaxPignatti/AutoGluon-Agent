---
description: Build a leakage-checked tabular model from a dataset and a goal
argument-hint: <data-path> "<goal>" [budget-minutes]
---

Use the autogluon-agent skill to build a model with these arguments: $ARGUMENTS

The arguments are: the path to a CSV or Parquet file, the prediction goal in quotes, and an optional time budget in minutes. If no budget is given, use 60 minutes. Follow the skill's phases from Setup to Verify and report, and keep every decision within that budget.
