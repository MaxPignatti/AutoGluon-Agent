import numpy as np, pandas as pd
from types import SimpleNamespace as NS
from autogluon_agent.proxy import cv_score, dummy_cv_score, encode_target, prepare_pair
from autogluon_agent.splits import cv_folds

def P(task="binary", metric="roc_auc", labels=(0, 1)):
    return NS(task=task, metric=metric, split="iid", group_col=None, time_col=None,
              target="target", seed=42, positive_class=labels[-1] if task == "binary" else None,
              labels=list(labels) if task != "regression" else None)

def test_lgbm_beats_dummy_binary(make_binary):
    df = make_binary().drop(columns=["leak", "group_id"])
    p = P(); folds = cv_folds(df, p)
    X, y = df.drop(columns="target"), df.target
    assert cv_score(X, y, p, folds).mean > dummy_cv_score(y, p, folds).mean + 0.1

def test_regression_scores_are_oriented(make_regression):
    df = make_regression(); p = P("regression", "rmse")
    r = cv_score(df.drop(columns="target"), df.target, p, cv_folds(df, p))
    assert r.mean < 0 and len(r.fold_scores) == 15

def test_unseen_category_in_validation():
    tr = pd.DataFrame({"c": ["a", "b"] * 50}); va = pd.DataFrame({"c": ["z"] * 10})
    _, va2 = prepare_pair(tr, va)
    assert va2["c"].isna().all()

def test_multiclass_runs(make_multiclass):
    df = make_multiclass(); p = P("multiclass", "log_loss", ("a", "b", "c"))
    assert cv_score(df.drop(columns="target"), df.target, p, cv_folds(df, p)).mean < 0

def test_prepare_pair_datetime_text_inf():
    ts = pd.Timestamp("2024-03-05 07:00") + pd.to_timedelta(np.arange(60), unit="h")
    tr = pd.DataFrame({
        "t": ts, "x": np.r_[np.inf, -np.inf, np.arange(58.0)],
        "txt": [f"{'long text ' * 5}{i}" for i in range(60)], "b": [True, False] * 30,
    })
    tr = pd.concat([tr] * 20, ignore_index=True)
    tr["txt"] = [f"{'long text ' * 5}{i}" for i in range(len(tr))]  # > 1000 unique
    out, _ = prepare_pair(tr, tr.head(5))
    assert "t" not in out and "txt" not in out
    assert {"t_year", "t_month", "t_day", "t_dow", "t_hour"} <= set(out.columns)
    assert out["t_year"].iloc[0] == 2024 and out["t_month"].iloc[0] == 3
    assert out["t_day"].iloc[0] == 5 and out["t_dow"].iloc[0] == 1 and out["t_hour"].iloc[0] == 7
    assert isinstance(out["b"].dtype, pd.CategoricalDtype)
    assert not np.isinf(out["x"]).any() and out["x"].isna().sum() == 40

def test_encode_target():
    p = P("multiclass", "log_loss", ("a", "b", "c"))
    assert encode_target(pd.Series(["c", "a", "b"]), p).tolist() == [2, 0, 1]
    pb = P(labels=("no", "yes"))
    assert encode_target(pd.Series(["yes", "no"]), pb).tolist() == [1, 0]
    assert encode_target(pd.Series([1, 2]), P("regression", "rmse")).dtype == float

def test_fold_transform_called_with_original_y_and_flags(make_binary):
    df = make_binary(n=500).drop(columns=["leak", "group_id"])
    p = P(); folds = cv_folds(df, p, repeats=1)
    X, y = df.drop(columns="target"), df.target
    seen = []

    def ft(Xtr, ytr, Xva, yva=None):
        seen.append((len(Xtr), len(ytr), len(yva)))
        return Xtr.drop(columns=["f_noise1"]), Xva.drop(columns=["f_noise1"]), ["flag-a"]

    r = cv_score(X, y, p, folds, fold_transform=ft)
    assert len(seen) == 5 and all(a == b for a, b, _ in seen)
    assert r.flags == ["flag-a"]
