from types import SimpleNamespace as NS

import pandas as pd

from autogluon_agent.cli import main
from autogluon_agent.leakage import run_checks
from autogluon_agent.splits import holdout_split


def P(**kw):
    base = dict(task="binary", metric="roc_auc", split="iid", group_col=None, time_col=None,
                target="target", seed=42, positive_class=1, labels=[0, 1])
    return NS(**{**base, **kw})


def findings(df, p):
    tr, ho = holdout_split(df, p)
    return run_checks(tr, ho, p)


def test_planted_leak_is_critical(make_binary):
    f = findings(make_binary().drop(columns=["ts"]), P())
    assert any(x.column == "leak" and x.severity == "critical" for x in f)
    assert not any(x.column == "f_noise1" and x.severity != "info" for x in f)


def test_suspicious_name_and_missingness(make_binary):
    df = make_binary().drop(columns=["ts", "leak"])
    df["final_status"] = 0
    df["opt"] = df["f_signal1"].where(df.target == 0)  # missing iff positive
    f = findings(df, P())
    assert any(x.column == "final_status" and x.check == "names" for x in f)
    assert any(x.column == "opt" and x.check == "missingness" for x in f)


def test_post_timestamp(make_binary):
    df = make_binary().drop(columns=["leak"])
    df["closed_at"] = df["ts"] + pd.to_timedelta(3, unit="D")
    f = findings(df, P(split="temporal", time_col="ts"))
    assert any(x.column == "closed_at" and x.check == "post_timestamps" for x in f)


def test_cli_blocks_until_drop(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    assert main(["leakage"]) == 0
    assert main(["baseline"]) == 2
    assert main(["drop", "leak", "--reason", "copy of target"]) == 0
    assert main(["baseline"]) == 0


def test_regression_leak_is_critical(make_regression):
    df = make_regression()
    df["copy"] = df["target"] * 2 + 1
    f = findings(df, P(task="regression", metric="rmse", positive_class=None, labels=None))
    assert any(x.column == "copy" and x.severity == "critical" for x in f)
    assert not any(x.column == "noise1" and x.severity != "info" for x in f)


def test_multiclass_runs(make_multiclass):
    df = make_multiclass()
    p = P(task="multiclass", metric="log_loss", positive_class=None, labels=["a", "b", "c"])
    f = findings(df, p)
    assert not any(x.severity == "critical" for x in f)
    assert any(x.check == "adversarial" and x.severity == "info" for x in f)


def test_dropped_columns_are_skipped(make_binary):
    df = make_binary().drop(columns=["ts"])
    p = P()
    tr, ho = holdout_split(df, p)
    f = run_checks(tr, ho, p, dropped=["leak"])
    assert not any(x.column == "leak" for x in f)


def test_duplicates_and_group_overlap(make_binary):
    df = make_binary().drop(columns=["ts"])
    p = P()
    tr, ho = holdout_split(df, p)
    ho = pd.concat([tr.iloc[:20], ho.iloc[20:]], ignore_index=True)  # 20 of ~600 rows copied from train
    f = run_checks(tr, ho, p)
    assert any(x.check == "duplicates" and x.severity == "warning" for x in f)

    pg = P(split="grouped", group_col="group_id")
    f = run_checks(tr, ho, pg)
    assert any(x.check == "duplicates" and x.severity == "critical" and x.column == "group_id" for x in f)


def test_id_like_column(make_binary):
    df = make_binary().drop(columns=["ts", "leak"])
    p = P()
    tr, ho = holdout_split(df, p)
    tr = tr.sort_values("target").reset_index(drop=True)  # row order carries the target
    tr["row_id"] = range(len(tr))
    ho["row_id"] = range(len(tr), len(tr) + len(ho))
    f = run_checks(tr, ho, p)
    assert any(x.column == "row_id" and x.check == "id_like" for x in f)


def test_adversarial_reports_drift(make_binary):
    df = make_binary().drop(columns=["leak"])
    f = findings(df, P(split="temporal", time_col="ts"))
    adv = [x for x in f if x.check == "adversarial"]
    assert len(adv) == 1 and adv[0].severity == "info" and "drift" in adv[0].evidence


def test_cli_keep_resolves_and_writes_json(tmp_path, make_binary, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    assert main(["leakage"]) == 0
    run_dir = next((tmp_path / ".aga").glob("2*"))
    data = json.loads((run_dir / "leakage.json").read_text(encoding="utf-8"))
    assert any(d["column"] == "leak" and d["severity"] == "critical" for d in data)
    assert main(["keep", "leak", "--reason", "legit"]) == 0
    assert main(["baseline"]) == 0
    assert main(["leakage"]) == 0  # kept columns do not block again
    assert main(["baseline"]) == 0


def test_short_target_name_matches_whole_tokens_only(make_binary):
    df = make_binary().drop(columns=["ts", "leak"]).rename(columns={"target": "y"})
    df["day_of_week"] = 1
    df["y_final"] = 1
    df["leak_y"] = 1
    f = findings(df, P(target="y"))
    flagged = {x.column for x in f if x.check == "names"}
    assert "day_of_week" not in flagged
    assert {"y_final", "leak_y"} <= flagged


def test_long_target_name_matches_as_substring(make_binary):
    df = make_binary().drop(columns=["ts", "leak"])
    df["my_target_copy"] = 1
    f = findings(df, P())
    assert any(x.column == "my_target_copy" and x.check == "names" for x in f)


def test_adversarial_skipped_without_usable_columns(make_binary):
    df = make_binary()[["cat", "target"]].copy()
    df["cat"] = [f"long text value number {i} " * 3 for i in range(len(df))]  # text-like: dropped
    f = findings(df, P())
    adv = [x for x in f if x.check == "adversarial"]
    assert len(adv) == 1 and adv[0].severity == "info" and "skipped" in adv[0].evidence
