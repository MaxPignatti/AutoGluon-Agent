from types import SimpleNamespace as NS

import numpy as np
import pytest

from autogluon_agent.errors import AgaError
from autogluon_agent.splits import cv_folds, holdout_split


def P(**kw):
    base = dict(task="binary", split="iid", group_col=None, time_col=None, target="target", seed=42)
    return NS(**{**base, **kw})


def test_iid_holdout_is_20pct_stratified(make_binary):
    df = make_binary()
    tr, ho = holdout_split(df, P())
    assert len(ho) == 600 and abs(tr.target.mean() - ho.target.mean()) < 0.02
    assert list(tr.index) == list(range(len(tr)))
    assert list(ho.index) == list(range(len(ho)))


def test_grouped_never_shares_groups(make_binary):
    df = make_binary()
    p = P(split="grouped", group_col="group_id")
    tr, ho = holdout_split(df, p)
    assert not set(tr.group_id) & set(ho.group_id)
    for a, b in cv_folds(tr, p):
        assert not set(tr.group_id.iloc[a]) & set(tr.group_id.iloc[b])


def test_temporal_holdout_is_latest_and_folds_look_forward(make_binary):
    df = make_binary().sample(frac=1, random_state=1)  # shuffled input
    p = P(split="temporal", time_col="ts")
    tr, ho = holdout_split(df, p)
    assert tr.ts.max() <= ho.ts.min()
    folds = cv_folds(tr, p)
    assert len(folds) == 5
    for a, b in folds:
        assert tr.ts.iloc[a].max() <= tr.ts.iloc[b].min()


def test_repeats_default(make_binary):
    assert len(cv_folds(make_binary(), P())) == 15


def test_repeats_explicit_and_regression_iid(make_regression):
    df = make_regression()
    p = P(task="regression")
    assert len(cv_folds(df, p, repeats=1)) == 5
    folds = cv_folds(df, p, repeats=2)
    assert len(folds) == 10
    for a, b in folds:
        assert not set(a) & set(b) and len(a) + len(b) == len(df)


def test_large_data_defaults_to_one_repeat():
    import pandas as pd

    n = 50_000
    df = pd.DataFrame({"x": np.arange(n), "target": np.arange(n) % 2})
    assert len(cv_folds(df, P())) == 5


def test_grouped_repeats_cover_every_row_once_per_repeat(make_binary):
    df = make_binary()
    p = P(split="grouped", group_col="group_id")
    folds = cv_folds(df, p)
    assert len(folds) == 15
    for r in range(3):
        val = np.concatenate([b for _, b in folds[r * 5 : (r + 1) * 5]])
        assert sorted(val) == list(range(len(df)))
    for a, b in folds:
        assert not set(df.group_id.iloc[a]) & set(df.group_id.iloc[b])


def test_temporal_folds_map_back_to_original_positions(make_binary):
    df = make_binary().sample(frac=1, random_state=3).reset_index(drop=True)
    folds = cv_folds(df, P(split="temporal", time_col="ts"))
    for a, b in folds:
        assert df.ts.iloc[a].max() <= df.ts.iloc[b].min()
        assert not set(a) & set(b)


def test_multiclass_holdout_stratified(make_multiclass):
    df = make_multiclass()
    tr, ho = holdout_split(df, P(task="multiclass"))
    assert set(tr.target) == set(ho.target)


def test_deterministic(make_binary):
    df = make_binary()
    a1, _ = holdout_split(df, P())
    a2, _ = holdout_split(df, P())
    assert a1.equals(a2)


def test_unknown_split_raises(make_binary):
    df = make_binary()
    with pytest.raises(AgaError):
        holdout_split(df, P(split="bogus"))
    with pytest.raises(AgaError):
        cv_folds(df, P(split="bogus"))


def test_missing_group_or_time_col_raises(make_binary):
    df = make_binary()
    with pytest.raises(AgaError):
        holdout_split(df, P(split="grouped"))
    with pytest.raises(AgaError):
        cv_folds(df, P(split="temporal"))
