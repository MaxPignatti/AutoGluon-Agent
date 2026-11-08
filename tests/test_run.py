import json

import pytest

from autogluon_agent.errors import BlockedError, AgaError
from autogluon_agent.journal import Journal
from autogluon_agent.run import Problem, Run


def write(df, d, name="data.csv"):
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    df.to_csv(p, index=False)
    return p


def create(tmp_path, df, metric=None, group=None, time=None, budget=60):
    return Run.create(tmp_path, write(df, tmp_path), "target", "g", metric, group, time, budget)


def test_create_in_dir_with_spaces(tmp_path, make_binary):
    base = tmp_path / "My Data Dir"
    data = write(make_binary(), base)
    run = Run.create(base, data, "target", "predict target", None, None, None, 60)
    assert run.problem.task == "binary" and run.problem.metric == "roc_auc"
    assert len(run.train_df()) == 2400 and len(run.holdout_df()) == 600
    assert (base / ".aga" / "CURRENT").read_text().strip() == run.root.name


def test_missing_target_rows_dropped_and_logged(tmp_path, make_binary):
    df = make_binary()
    df.loc[:9, "target"] = None
    run = Run.create(tmp_path, write(df, tmp_path), "target", "g", None, None, None, 60)
    assert run.journal.entries("init")[0]["dropped_missing_target"] == 10


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda d: d.drop(columns="target"), "target"),
        (lambda d: d.assign(target=1), "single class"),
        (lambda d: d.head(150), "200"),
    ],
)
def test_init_refuses(tmp_path, make_binary, mutate, msg):
    with pytest.raises(AgaError, match=msg):
        Run.create(tmp_path, write(mutate(make_binary()), tmp_path), "target", "g", None, None, None, 60)


def test_reinit_preserves_old_run(tmp_path, make_binary):
    data = write(make_binary(), tmp_path)
    r1 = Run.create(tmp_path, data, "target", "g", None, None, None, 60)
    r2 = Run.create(tmp_path, data, "target", "g", None, "group_id", None, 60)
    assert r1.root.exists() and r1.root != r2.root
    assert Run.load(tmp_path).root == r2.root


def test_blocked_until_resolved(tmp_path, make_binary):
    run = Run.create(tmp_path, write(make_binary(), tmp_path), "target", "g", None, None, None, 60)
    run.state.unresolved = ["leak"]
    run.save()
    with pytest.raises(BlockedError, match="leak"):
        Run.load(tmp_path).require_unblocked()


def test_load_without_run_errors(tmp_path):
    with pytest.raises(AgaError, match="No run found; run 'aga init' first."):
        Run.load(tmp_path)


def test_load_unknown_run_id_errors(tmp_path, make_binary):
    create(tmp_path, make_binary())
    with pytest.raises(AgaError, match="19990101-000000"):
        Run.load(tmp_path, "19990101-000000")


def test_files_written_and_problem_roundtrip(tmp_path, make_binary):
    run = create(tmp_path, make_binary())
    for name in (
        "problem.yaml",
        "profile.json",
        "train.parquet",
        "holdout.parquet",
        "state.json",
        "journal.jsonl",
    ):
        assert (run.root / name).exists(), name
    again = Run.load(tmp_path)
    assert again.problem == run.problem
    assert again.problem.positive_class == 1 and again.problem.labels == [0, 1]
    assert again.problem.split == "iid" and again.problem.seed == 42
    assert again.state.budget_minutes == 60 and again.state.fitted is False
    assert Problem.from_yaml(run.problem.to_yaml()) == run.problem
    assert "n_rows" in json.loads((run.root / "profile.json").read_text())


def test_string_labels_and_regression_problem(tmp_path, make_multiclass, make_regression):
    run = create(tmp_path, make_multiclass())
    assert run.problem.task == "multiclass" and run.problem.metric == "log_loss"
    assert run.problem.labels == ["a", "b", "c"] and run.problem.positive_class is None
    run = create(tmp_path / "reg", make_regression())
    assert run.problem.task == "regression" and run.problem.metric == "rmse"
    assert run.problem.labels is None and run.problem.positive_class is None


def test_split_modes(tmp_path, make_binary):
    assert create(tmp_path / "a", make_binary(), group="group_id").problem.split == "grouped"
    r = create(tmp_path / "b", make_binary(), time="ts")
    assert r.problem.split == "temporal" and r.problem.time_col == "ts"
    assert r.train_df()["ts"].max() <= r.holdout_df()["ts"].min()
    with pytest.raises(AgaError, match="both"):
        create(tmp_path / "c", make_binary(), group="group_id", time="ts")


@pytest.mark.parametrize("kw", [{"group": "nope"}, {"time": "nope"}])
def test_missing_group_or_time_column(tmp_path, make_binary, kw):
    with pytest.raises(AgaError, match="nope"):
        create(tmp_path, make_binary(), **kw)


def test_time_nulls_refused(tmp_path, make_binary):
    df = make_binary()
    df["ts"] = df["ts"].astype(str)
    df.loc[5, "ts"] = "not a date"
    with pytest.raises(AgaError, match="time"):
        create(tmp_path, df, time="ts")


def test_too_few_groups_refused(tmp_path, make_binary):
    df = make_binary()
    df["group_id"] = df["group_id"] % 5
    with pytest.raises(AgaError, match="10"):
        create(tmp_path, df, group="group_id")


def test_rare_class_refused(tmp_path, make_multiclass):
    df = make_multiclass()
    df.loc[:2, "target"] = "rare"
    with pytest.raises(AgaError, match="rare"):
        create(tmp_path, df)


def test_metric_validation(tmp_path, make_binary, make_regression):
    assert create(tmp_path / "a", make_binary(), metric="pr_auc").problem.metric == "pr_auc"
    with pytest.raises(AgaError, match="rmse"):
        create(tmp_path / "b", make_binary(), metric="rmse")
    with pytest.raises(AgaError, match="bogus"):
        create(tmp_path / "c", make_binary(), metric="bogus")
    with pytest.raises(AgaError, match="roc_auc"):
        create(tmp_path / "d", make_regression(), metric="roc_auc")


def test_base_features(tmp_path, make_binary):
    run = create(tmp_path, make_binary(), group="group_id")
    run.state.dropped = {"leak": "post-outcome"}
    X = run.base_features(run.train_df())
    assert "target" not in X and "leak" not in X and "group_id" not in X
    assert "f_signal1" in X


def test_budget(tmp_path, make_binary, monkeypatch):
    run = create(tmp_path, make_binary(), budget=60)
    t0 = run.state.started
    monkeypatch.setattr("autogluon_agent.run.time.time", lambda: t0 + 900)
    assert run.remaining_seconds() == pytest.approx(2700)
    assert run.budget_line() == "Budget: 45.0 min left (25% used)"
    monkeypatch.setattr("autogluon_agent.run.time.time", lambda: t0 + 99999)
    assert run.remaining_seconds() == 0
    assert run.budget_line() == "Budget: 0.0 min left (100% used)"


def test_journal_roundtrip(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    assert j.entries() == []
    e = j.append("note", text="héllo", n=3)
    j.append("drop", column="x")
    assert "ts" in e and e["kind"] == "note"
    assert [x["kind"] for x in j.entries()] == ["note", "drop"]
    assert j.entries("note")[0]["text"] == "héllo"
    assert len(Journal(tmp_path / "j.jsonl").entries("drop")) == 1


def test_profile_computed_on_training_rows_only(tmp_path, make_binary):
    run = create(tmp_path, make_binary())
    prof = json.loads((run.root / "profile.json").read_text())
    assert prof["n_rows"] == len(run.train_df()) == 2400


def test_group_nulls_refused(tmp_path, make_binary):
    df = make_binary()
    df.loc[3, "group_id"] = None
    with pytest.raises(AgaError, match="group_id"):
        create(tmp_path, df, group="group_id")


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_bad_budget_refused(tmp_path, make_binary, budget):
    with pytest.raises(AgaError, match="budget"):
        create(tmp_path, make_binary(), budget=budget)


@pytest.mark.parametrize("kw", [{"group": "target"}, {"time": "target"}])
def test_group_or_time_equal_target_refused(tmp_path, make_binary, kw):
    with pytest.raises(AgaError, match="target"):
        create(tmp_path, make_binary(), **kw)


def test_reinit_inherits_running_budget(tmp_path, make_binary, monkeypatch):
    data = write(make_binary(), tmp_path)
    r1 = Run.create(tmp_path, data, "target", "g", None, None, None, 60)
    t0 = r1.state.started
    monkeypatch.setattr("autogluon_agent.run.time.time", lambda: t0 + 600)
    r2 = Run.create(tmp_path, data, "target", "g", "pr_auc", None, None, 30)
    assert (r2.state.started, r2.state.deadline, r2.state.budget_minutes) == (t0, r1.state.deadline, 60)
    assert r2.journal.entries("init")[0]["budget_inherited_from"] == r1.root.name
    assert r2.budget_line() == "Budget: 50.0 min left (17% used)"


def test_reinit_fresh_budget_or_expired_budget_starts_new_clock(tmp_path, make_binary, monkeypatch):
    data = write(make_binary(), tmp_path)
    r1 = Run.create(tmp_path, data, "target", "g", None, None, None, 60)
    t0 = r1.state.started
    monkeypatch.setattr("autogluon_agent.run.time.time", lambda: t0 + 600)
    r2 = Run.create(tmp_path, data, "target", "g", None, None, None, 30, fresh_budget=True)
    assert r2.state.started == t0 + 600 and r2.state.budget_minutes == 30
    assert r2.journal.entries("init")[0]["budget_inherited_from"] is None
    monkeypatch.setattr("autogluon_agent.run.time.time", lambda: t0 + 600 + 31 * 60)  # r2 has expired
    r3 = Run.create(tmp_path, data, "target", "g", None, None, None, 20)
    assert r3.state.started == t0 + 600 + 31 * 60 and r3.state.budget_minutes == 20


def test_task_override(tmp_path, make_binary, make_multiclass, make_regression):
    df = make_regression()
    df["target"] = (df["target"].rank(pct=True) * 10).round().astype(int)  # 11 integer levels
    assert create(tmp_path / "a", df).problem.task == "multiclass"
    run = Run.create(tmp_path / "b", write(df, tmp_path / "b"), "target", "g", "mae", None, None, 60,
                     task="regression")
    assert run.problem.task == "regression" and run.problem.metric == "mae" and run.problem.labels is None
    assert run.profile_dict()["target"]["task"] == "regression"
    run = Run.create(tmp_path / "c", write(make_binary(), tmp_path / "c"), "target", "g", None, None, None, 60,
                     task="binary")
    assert run.problem.task == "binary"


@pytest.mark.parametrize(
    "fixture,task,msg",
    [
        ("make_multiclass", "binary", "exactly 2"),
        ("make_binary", "multiclass", "at least 3"),
        ("make_multiclass", "regression", "numeric"),
        ("make_binary", "ranking", "unknown task"),
    ],
)
def test_task_override_refused(tmp_path, request, fixture, task, msg):
    df = request.getfixturevalue(fixture)()
    with pytest.raises(AgaError, match=msg):
        Run.create(tmp_path, write(df, tmp_path), "target", "g", None, None, None, 60, task=task)


def test_old_state_without_refit_keys_loads(tmp_path, make_binary):
    run = create(tmp_path, make_binary())
    path = run.root / "state.json"
    raw = json.loads(path.read_text())
    for key in ("refit", "refit_rows", "evaluated"):
        raw.pop(key, None)
    path.write_text(json.dumps(raw))
    run = Run.load(tmp_path)
    assert run.state.refit is False and run.state.refit_rows is None and run.state.evaluated is None
    from autogluon_agent.fit import require_refit_allowed

    run.state.fitted, run.state.eval_count = True, 1
    require_refit_allowed(run)  # no 'evaluated' key: falls back to eval_count >= 1
    run.state.eval_count = 0
    with pytest.raises(AgaError, match="run 'aga evaluate' first"):
        require_refit_allowed(run)
