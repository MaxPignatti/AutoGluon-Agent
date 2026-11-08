import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def test_make_binary_shape_and_determinism(make_binary):
    df = make_binary()
    assert len(df) == 3000
    assert set(df["target"].unique()) == {0, 1}
    assert df["cat"].nunique() == 5 and df["cat"].dtype == object
    assert 80 <= df["group_id"].nunique() <= 100
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])
    assert df["ts"].is_monotonic_increasing
    assert 0.005 < (df["leak"] != df["target"]).mean() < 0.04
    pd.testing.assert_frame_equal(df, make_binary())
    assert len(make_binary(n=1500)) == 1500
    assert not df["f_signal1"].equals(make_binary(seed=1)["f_signal1"])


def test_make_binary_signal_structure(make_binary):
    df = make_binary()
    assert roc_auc_score(df["target"], df["f_signal1"]) > 0.7
    assert roc_auc_score(df["target"], df["f_signal2"]) > 0.65
    # group alone (out-of-fold-free proxy: mean target per group) must be weak
    half = len(df) // 2
    rate = df.iloc[:half].groupby("group_id")["target"].mean()
    pred = df.iloc[half:]["group_id"].map(rate).fillna(0.5)
    assert roc_auc_score(df.iloc[half:]["target"], pred) < 0.7
    # target does not depend on ts
    assert abs(np.corrcoef(np.arange(len(df)), df["target"])[0, 1]) < 0.05


def test_make_regression(make_regression):
    df = make_regression()
    assert list(df.columns) == ["x1", "x2", "lat", "lon", "noise1", "target"]
    assert df["lat"].between(37, 39).all() and df["lon"].between(-123, -121).all()
    assert abs(np.corrcoef(df["x1"] * df["x2"], df["target"])[0, 1]) > 0.4
    pd.testing.assert_frame_equal(df, make_regression())


def test_make_multiclass(make_multiclass):
    df = make_multiclass()
    assert set(df["target"].unique()) == {"a", "b", "c"}
    assert df["target"].value_counts(normalize=True).min() > 0.15
    pd.testing.assert_frame_equal(df, make_multiclass())
