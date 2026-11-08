import textwrap

import pandas as pd
import pytest

from autogluon_agent.cli import main
from autogluon_agent.run import Run

GOOD = textwrap.dedent("""
    import pandas as pd
    class Features:
        def fit(self, X, y): return self
        def transform(self, X): return pd.DataFrame({"x1x2": X["x1"] * X["x2"]}, index=X.index)
""")


@pytest.mark.slow
def test_fit_and_predict_with_module(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path); df = make_regression(n=1500)
    df.to_csv("d.csv", index=False); (tmp_path / "f.py").write_text(GOOD)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"]); main(["try", "f.py"])
    assert Run.load(tmp_path).state.active_modules  # the module was accepted, so it is fitted and applied
    assert main(["fit", "--time-limit", "60", "--presets", "medium_quality"]) == 0
    new = df.drop(columns="target").head(20)
    new.to_csv("new.csv", index=False)
    assert main(["predict", "new.csv", "--out", "p.csv"]) == 0
    assert len(pd.read_csv("p.csv")) == 20


@pytest.mark.slow
def test_grouped_fit_excludes_group_column(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_binary(n=1500).drop(columns=["leak"]).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g", "--group", "group_id"]); main(["baseline"])
    assert main(["fit", "--time-limit", "60", "--presets", "medium_quality"]) == 0
    from autogluon_agent.fit import FOLD_COL, load_final
    _, pred = load_final(Run.load(tmp_path))
    assert "group_id" not in pred.features() and FOLD_COL not in pred.features()
    assert pred._learner.groups == FOLD_COL  # group-aware: fold assignment comes from our column
    assert pred.info()["num_bag_folds"] == 5  # 5 folds from our assignment, not one per raw group id

    # production refit: group folds are assigned over train + holdout, groups never split
    from autogluon_agent import fit

    seen, original = [], fit.group_folds

    def spy(data, problem, *args, **kwargs):
        folds = original(data, problem, *args, **kwargs)
        seen.append((data, folds))
        return folds

    monkeypatch.setattr(fit, "group_folds", spy)
    run = Run.load(tmp_path)
    run.state.eval_count, run.state.evaluated = 1, True  # skip the (slow) evaluation
    run.save()
    assert main(["refit", "--time-limit", "60"]) == 0
    data, folds = seen[-1]
    assert len(data) == 1500 and sorted(set(folds)) == [0, 1, 2, 3, 4]
    assert (pd.Series(folds).groupby(data["group_id"].to_numpy()).nunique() == 1).all()
    _, pred = load_final(Run.load(tmp_path))
    assert "models_full" in str(pred.path) and pred.info()["num_bag_folds"] == 5


def test_group_folds_keep_groups_whole(make_binary, make_regression):
    import numpy as np

    from autogluon_agent.fit import group_folds
    from autogluon_agent.run import Problem

    for task, df in (("binary", make_binary(n=1000)), ("regression", make_regression(n=1000))):
        df = df.assign(group_id=np.arange(len(df)) % 37)
        p = Problem("target", task, "m", "grouped", "group_id", None, "g", None, None)
        folds = group_folds(df, p)
        assert sorted(set(folds)) == [0, 1, 2, 3, 4]
        assert (pd.Series(folds).groupby(df["group_id"]).nunique() == 1).all()  # no group split across folds


@pytest.mark.slow
def test_predict_handles_unseen_category(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_binary(n=1500).drop(columns=["leak"]).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"])
    main(["fit", "--time-limit", "60", "--presets", "medium_quality"])
    new = make_binary(n=10).drop(columns=["leak", "target"]).assign(cat="never_seen")
    new.to_csv("new.csv", index=False)
    assert main(["predict", "new.csv", "--out", "p.csv"]) == 0


STATEFUL = textwrap.dedent("""
    import pandas as pd
    class Features:
        def fit(self, X, y):
            self.mean = (X["x1"] * X["x2"]).mean(); return self
        def transform(self, X):
            return pd.DataFrame({"x1x2c": X["x1"] * X["x2"] - self.mean}, index=X.index)
""")


def _spy_fit(monkeypatch):
    """Record the number of rows (train + tuning) every TabularPredictor.fit call receives."""
    from autogluon.tabular import TabularPredictor

    seen, original = [], TabularPredictor.fit

    def spy(self, train_data, *args, **kwargs):
        tuning = kwargs.get("tuning_data")
        seen.append(len(train_data) + (0 if tuning is None else len(tuning)))
        return original(self, train_data, *args, **kwargs)

    monkeypatch.setattr(TabularPredictor, "fit", spy)
    return seen


def _spy_fit_time_limit(monkeypatch):
    from autogluon.tabular import TabularPredictor

    limits, original = [], TabularPredictor.fit

    def spy(self, *args, **kwargs):
        limits.append(kwargs.get("time_limit"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TabularPredictor, "fit", spy)
    return limits


def _spy_refit_full(monkeypatch):
    from autogluon.tabular import TabularPredictor

    calls, original = [], TabularPredictor.refit_full

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("time_limit"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TabularPredictor, "refit_full", spy)
    return calls


@pytest.mark.slow
def test_temporal_fit_then_predict_in_fresh_process(tmp_path, make_regression, monkeypatch, capfd):
    import subprocess
    import sys

    from autogluon_agent.fit import load_final

    monkeypatch.chdir(tmp_path)
    df = make_regression(n=1500).assign(ts=lambda d: pd.date_range("2024-01-01", periods=len(d), freq="h"))
    df.to_csv("d.csv", index=False); (tmp_path / "f.py").write_text(STATEFUL)
    main(["init", "d.csv", "--target", "target", "--goal", "g", "--time", "ts"]); main(["baseline"])
    main(["try", "f.py"])
    assert Run.load(tmp_path).state.active_modules  # accepted: the pickled stateful module is exercised
    refits = _spy_refit_full(monkeypatch)
    fit_limits = _spy_fit_time_limit(monkeypatch)
    assert main(["fit", "--time-limit", "30", "--presets", "medium_quality"]) == 0
    assert fit_limits == [pytest.approx(18.0)]  # 60% of the time limit for the fit
    assert len(refits) == 1 and 0 < refits[0] <= 12.0  # the other 40%, shared over the refit models
    out = capfd.readouterr().out
    assert "bagging" in out and "refit_full" in out and "Budget:" in out
    run = Run.load(tmp_path)
    entry = run.journal.entries("fit")[0]
    assert entry["score_val_natural"] == -entry["score_val"]  # rmse: AutoGluon negates error metrics
    assert len(entry["leaderboard_top5"]) >= 1 and any("bagging" in n for n in entry["notes"])
    assert all(r["score_val"] == r["score_val"] for r in entry["leaderboard_top5"])  # pre-refit scores: no NaN
    assert not any(r["model"].endswith("_FULL") for r in entry["leaderboard_top5"])
    assert any("refit_full" in n and "40%" in n for n in entry["notes"]) and entry["n_rows"] == 1200
    assert any("60%" in n for n in entry["notes"])
    # the model that predicts is the refit-full one, trained on the 85% + the newest 15% tuning tail
    _, predictor = load_final(run)
    assert predictor.model_best.endswith("_FULL")

    # production refit on all labelled rows: temporal recipe again, then a fresh-process predict
    run.state.eval_count, run.state.evaluated = 1, True  # skip the (slow) evaluation
    run.save()
    assert main(["refit", "--time-limit", "30"]) == 0
    assert fit_limits[-1] == pytest.approx(18.0) and 0 < refits[-1] <= 12.0
    run = Run.load(tmp_path)
    assert run.state.refit and run.state.refit_rows == 1500 and len(refits) == 2
    assert run.journal.entries("refit")[0]["n_rows"] == 1500
    _, predictor = load_final(run)
    assert predictor.model_best.endswith("_FULL") and "models_full" in str(predictor.path)

    # a fresh process must load the by-value pickled feature module; input lacks a column
    df.drop(columns=["target", "noise1"]).head(12).to_csv("new.csv", index=False)
    code = "import sys; from autogluon_agent.cli import main; sys.exit(main(['predict','new.csv','--out','p.csv']))"
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "noise1" in proc.stdout  # missing-column warning
    assert "Warning: this model was trained on the train portion only" not in proc.stdout
    pred = pd.read_csv("p.csv")
    assert list(pred.columns) == ["prediction"] and len(pred) == 12 and pred["prediction"].notna().all()


TRAIN_ONLY_WARNING = (
    "Warning: this model was trained on the train portion only (80%); run 'aga refit' to train the "
    "delivered model on all labelled rows."
)


@pytest.mark.slow
def test_refit_on_all_labelled_rows(tmp_path, make_binary, monkeypatch, capsys):
    from autogluon.tabular import TabularPredictor

    from autogluon_agent.fit import load_final

    monkeypatch.chdir(tmp_path)
    make_binary(n=1500).drop(columns=["leak"]).to_csv("d.csv", index=False)
    make_binary(n=15, seed=5).drop(columns=["leak", "target"]).to_csv("new.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"])
    seen = _spy_fit(monkeypatch)
    assert main(["fit", "--time-limit", "30", "--presets", "medium_quality"]) == 0
    assert seen == [1200]
    capsys.readouterr()

    assert main(["refit"]) == 1  # before the one-shot evaluation
    assert "run 'aga evaluate' first" in capsys.readouterr().err
    assert main(["evaluate"]) == 0

    assert main(["predict", "new.csv", "--out", "p0.csv"]) == 0
    assert TRAIN_ONLY_WARNING in capsys.readouterr().out
    p0 = pd.read_csv("p0.csv")

    # a refit that fails leaves the evaluated model in charge and no refit claimed
    def boom(self, *args, **kwargs):
        raise RuntimeError("simulated AutoGluon failure")

    with monkeypatch.context() as m:
        m.setattr(TabularPredictor, "fit", boom)
        with pytest.raises(RuntimeError, match="simulated"):
            main(["refit", "--time-limit", "30", "--presets", "medium_quality"])
    run = Run.load(tmp_path)
    assert run.state.refit is False and run.state.refit_rows is None and run.state.fitted
    assert not run.journal.entries("refit")
    assert not (run.root / "models_full.tmp").exists()  # the staging dir is gone
    assert main(["predict", "new.csv", "--out", "p1.csv"]) == 0
    assert TRAIN_ONLY_WARNING in capsys.readouterr().out
    assert pd.read_csv("p1.csv").equals(p0)

    # the real refit trains on train + holdout; the staging -> production rename fails (a locked
    # directory), so the swap falls back to copying
    from pathlib import Path

    rename = Path.rename

    def locked(self, target):
        if Path(target).name == "models_full":
            raise PermissionError("simulated lock")
        return rename(self, target)

    with monkeypatch.context() as m:
        m.setattr(Path, "rename", locked)
        assert main(["refit", "--time-limit", "30"]) == 0  # presets default to the fit's
    out = capsys.readouterr().out
    assert not (Run.load(tmp_path).root / "models_full.tmp").exists()
    assert "1500 labelled rows" in out and out.strip().splitlines()[-1].startswith("Budget:")
    assert seen[-1] == 1500
    run = Run.load(tmp_path)
    assert run.state.refit and run.state.refit_rows == 1500
    entry = run.journal.entries("refit")[-1]
    assert entry["n_rows"] == 1500 and entry["time_limit"] == 30 and entry["presets"] == "medium_quality"
    assert {"time_used", "modules", "notes"} <= set(entry)
    assert not any(p.name.startswith("models_full.") for p in run.root.iterdir() if p.name != "models_full")

    # predict uses the production model, without the warning
    _, predictor = load_final(run)
    assert "models_full" in str(predictor.path)
    assert main(["predict", "new.csv", "--out", "p2.csv"]) == 0
    out = capsys.readouterr().out
    assert TRAIN_ONLY_WARNING not in out
    assert not pd.read_csv("p2.csv").equals(p0)

    # the holdout cannot be evaluated after the refit
    assert main(["evaluate"]) == 1 and main(["evaluate", "--force"]) == 1
    assert "refit on the holdout rows" in capsys.readouterr().err
    assert main(["status"]) == 0 and "Refit: yes (1500 rows)" in capsys.readouterr().out

    # a new fit replaces everything: the refit flag and the production model go
    assert main(["fit", "--time-limit", "30", "--presets", "medium_quality"]) == 0
    run = Run.load(tmp_path)
    assert run.state.refit is False and run.state.refit_rows is None
    assert not (run.root / "models_full").exists()
    capsys.readouterr()
    assert main(["status"]) == 0 and "Refit: no" in capsys.readouterr().out

    # the new model has not been evaluated: no refit until 'evaluate --force'
    assert main(["refit", "--time-limit", "30"]) == 1
    assert "the current model has not been evaluated; run 'aga evaluate --force' first" in capsys.readouterr().err
    assert main(["evaluate", "--force"]) == 0
    assert main(["refit", "--time-limit", "30"]) == 0
    assert Run.load(tmp_path).state.refit


@pytest.mark.slow
def test_multiclass_log_loss_unnegated_and_class_columns(tmp_path, make_multiclass, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_multiclass(n=800).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"])
    assert main(["fit", "--time-limit", "60", "--presets", "medium_quality"]) == 0
    entry = Run.load(tmp_path).journal.entries("fit")[0]
    assert entry["score_val"] <= 0 and entry["score_val_natural"] == -entry["score_val"]  # log_loss
    make_multiclass(n=7, seed=3).drop(columns="target").to_csv("new.csv", index=False)
    assert main(["predict", "new.csv", "--out", "p.csv"]) == 0
    pred = pd.read_csv("p.csv")
    assert list(pred.columns) == ["a", "b", "c"] and len(pred) == 7
    assert (pred.sum(axis=1) - 1).abs().lt(1e-3).all()


@pytest.mark.slow
def test_failed_fit_clears_fitted_flag(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_regression(n=400).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    run = Run.load(tmp_path)
    run.state.fitted = True
    run.state.active_modules = ["99_missing.py"]  # makes the fit fail while fitting modules
    run.save()
    # A missing accepted-module file is an AgaError: main reports it and returns 1.
    assert main(["fit", "--time-limit", "60", "--presets", "medium_quality"]) == 1
    assert Run.load(tmp_path).state.fitted is False
    assert main(["predict", "d.csv", "--out", "p.csv"]) == 1  # "No fitted model"


FAILS_ON_FULL = textwrap.dedent("""
    import pandas as pd
    class Features:
        def fit(self, X, y):
            if len(X) > 1300:  # only the refit on train + holdout (1500 rows) sees this many rows
                raise ValueError("only on the full data")
            return self
        def transform(self, X): return pd.DataFrame({"x1x2": X["x1"] * X["x2"]}, index=X.index)
""")


@pytest.mark.slow
def test_refit_leaves_out_a_module_that_fails_on_the_full_data(tmp_path, make_regression, monkeypatch, capsys):
    from autogluon_agent.report import build_report

    monkeypatch.chdir(tmp_path); df = make_regression(n=1500)
    df.to_csv("d.csv", index=False); (tmp_path / "f.py").write_text(FAILS_ON_FULL)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"]); main(["try", "f.py"])
    run = Run.load(tmp_path)
    (name,) = run.state.active_modules
    assert main(["fit", "--time-limit", "30", "--presets", "medium_quality"]) == 0
    run = Run.load(tmp_path)
    run.state.eval_count, run.state.evaluated = 1, True  # skip the (slow) evaluation
    run.save()
    label, context = run.state.best_label, run.state.best_context
    capsys.readouterr()
    assert main(["refit", "--time-limit", "30"]) == 0
    out = capsys.readouterr().out
    assert f"skipped for the refit: {name} (failed on the full data); the evaluated recipe keeps it" in out
    assert "DEACTIVATED" not in out
    run = Run.load(tmp_path)
    assert not run.journal.entries("deactivate") and run.state.evaluated is True
    assert run.state.refit and run.state.active_modules == [name]  # the evaluated recipe is untouched
    assert run.state.best_label == label and run.state.best_context == context
    entry = run.journal.entries("refit")[-1]
    assert entry["skipped_modules"] == [name] and entry["modules"] == []
    assert entry["presets"] == entry["fit_presets"] == "medium_quality"
    assert any(name in n and "failed on the full data" in n for n in entry["notes"])
    text = build_report(run, tmp_path)
    assert f"module {name}, which failed on the full data, is left out" in text
    assert "same recipe" not in text
    df.drop(columns="target").head(5).to_csv("new.csv", index=False)
    assert main(["predict", "new.csv", "--out", "p.csv"]) == 0
    assert len(pd.read_csv("p.csv")) == 5


class _FakeTrainer:
    def __init__(self, members):
        self.members = members

    def get_minimum_model_set(self, model):
        return self.members


class _FakePredictor:
    def __init__(self, refit_map=None, error=None):
        self.model_best, self._trainer = "WeightedEnsemble_L2", _FakeTrainer(["LightGBM", "CatBoost", "WeightedEnsemble_L2"])
        self.refit_map, self.error, self.calls, self.best_set = refit_map, error, [], None

    def refit_full(self, model="all", **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.refit_map

    def set_model_best(self, model, save_trainer=False):
        self.best_set = (model, save_trainer)


def test_refit_full_splits_the_budget_over_the_ensemble_and_sets_the_refit_model_best():
    from autogluon_agent.fit import _refit_full

    pred = _FakePredictor({"LightGBM": "LightGBM_FULL", "WeightedEnsemble_L2": "WeightedEnsemble_L2_FULL"})
    (note,) = _refit_full(pred, 1000, 12.0)
    assert pred.calls == [{"time_limit": pytest.approx(4.0)}]  # per-model limit: 12 s over 3 models
    assert pred.best_set == ("WeightedEnsemble_L2_FULL", True)
    assert "refit_full" in note and "WeightedEnsemble_L2_FULL" in note and "40%" in note


@pytest.mark.parametrize("refit_map, error", [({}, None), (None, RuntimeError("out of memory"))])
def test_refit_full_failure_keeps_the_tuned_model_and_says_so(refit_map, error):
    from autogluon_agent.fit import _refit_full

    pred = _FakePredictor(refit_map, error)
    (note,) = _refit_full(pred, 1000, 12.0)
    assert pred.best_set is None
    assert note.startswith("the refit_full step failed") and "WeightedEnsemble_L2" in note
    assert "_FULL" not in note
    if error:
        assert "out of memory" in note


def test_refit_full_when_ensemble_members_cannot_be_counted():
    from autogluon_agent.fit import _refit_full

    class Broken:
        def get_minimum_model_set(self, model):
            raise KeyError("private API changed")

    pred = _FakePredictor({"WeightedEnsemble_L2": "WeightedEnsemble_L2_FULL"})
    pred._trainer = Broken()
    notes = _refit_full(pred, 1000, 12.0)
    assert notes[0] == "could not count ensemble members; refit_full time limit applied per model unchanged"
    assert pred.calls == [{"time_limit": pytest.approx(12.0)}]  # n_models falls back to 1
    assert pred.best_set == ("WeightedEnsemble_L2_FULL", True) and "refit_full" in notes[1]


@pytest.mark.slow
def test_temporal_fit_completes_when_ensemble_members_cannot_be_counted(tmp_path, make_regression, monkeypatch):
    import sys

    from autogluon.tabular.trainer.abstract_trainer import AbstractTabularTrainer

    from autogluon_agent.fit import load_final

    original = AbstractTabularTrainer.get_minimum_model_set

    def broken_for_the_harness(self, *args, **kwargs):
        """Raise only for the harness's count in _refit_full; AutoGluon's own calls still work."""
        if sys._getframe(1).f_code.co_name == "_refit_full":
            raise KeyError("private API changed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AbstractTabularTrainer, "get_minimum_model_set", broken_for_the_harness)
    monkeypatch.chdir(tmp_path)
    make_regression(n=800).assign(ts=lambda d: pd.date_range("2024-01-01", periods=len(d), freq="h")).to_csv(
        "d.csv", index=False
    )
    main(["init", "d.csv", "--target", "target", "--goal", "g", "--time", "ts"])
    assert main(["fit", "--time-limit", "20", "--presets", "medium_quality"]) == 0
    run = Run.load(tmp_path)
    notes = run.journal.entries("fit")[-1]["notes"]
    assert "could not count ensemble members; refit_full time limit applied per model unchanged" in notes
    assert any(n.startswith("then the best model was refit") for n in notes)  # refit_full still ran
    _, predictor = load_final(run)  # the fitted model survives
    assert predictor.model_best.endswith("_FULL")
