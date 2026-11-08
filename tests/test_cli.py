import pytest

from autogluon_agent import __version__
from autogluon_agent.cli import main


def test_version(capsys):
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_blocked_error_exits_2(monkeypatch, capsys):
    from autogluon_agent import cli
    from autogluon_agent.errors import BlockedError

    def boom(parser, args):
        raise BlockedError("holdout already used")

    monkeypatch.setattr(cli, "_run", boom)
    assert cli.main([]) == 2
    assert "holdout already used" in capsys.readouterr().err


def test_aga_error_exits_1(monkeypatch, capsys):
    from autogluon_agent import cli
    from autogluon_agent.errors import AgaError

    def boom(parser, args):
        raise AgaError("bad input")

    monkeypatch.setattr(cli, "_run", boom)
    assert cli.main([]) == 1
    assert "bad input" in capsys.readouterr().err


def test_error_hierarchy():
    from autogluon_agent import errors

    assert issubclass(errors.BlockedError, errors.AgaError)
    assert issubclass(errors.HoldoutReusedError, errors.BlockedError)
    assert issubclass(errors.FeatureModuleError, errors.AgaError)
    assert issubclass(errors.ContractError, errors.FeatureModuleError)


def test_usage_error_exits_1(capsys):
    assert main(["--bogus"]) == 1
    assert "unrecognized arguments" in capsys.readouterr().err


def test_help_exits_0(capsys):
    assert main(["--help"]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def _init_data(tmp_path, make_binary):
    data = tmp_path / "data.csv"
    make_binary().to_csv(data, index=False)
    return data


def test_cli_init_drop_status(tmp_path, monkeypatch, capsys, make_binary):
    import json

    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(data), "--target", "target", "--goal", "g"]) == 0
    assert main(["drop", "leak", "--reason", "post-outcome"]) == 0
    capsys.readouterr()
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    run_id = (tmp_path / ".aga" / "CURRENT").read_text().strip()
    state = json.loads((tmp_path / ".aga" / run_id / "state.json").read_text())
    assert "leak" in state["dropped"]
    assert out.strip().splitlines()[-1].startswith("Budget:")


def test_cli_profile_note_keep(tmp_path, monkeypatch, capsys, make_binary):
    import json

    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(data), "--target", "target", "--goal", "g", "--budget", "30"]) == 0
    out = capsys.readouterr().out
    assert "Rows: 2400" in out and "binary" in out and "roc_auc" in out
    assert out.strip().splitlines()[-1].startswith("Budget:")
    assert main(["profile"]) == 0
    out = capsys.readouterr().out
    assert "Rows: 2400" in out and out.strip().splitlines()[-1].startswith("Budget:")
    assert main(["note", "looks fine", "--phase", "eda"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("Budget:")
    run_id = (tmp_path / ".aga" / "CURRENT").read_text().strip()
    root = tmp_path / ".aga" / run_id
    state = json.loads((root / "state.json").read_text())
    state["unresolved"] = ["f_noise1"]
    (root / "state.json").write_text(json.dumps(state))
    assert main(["keep", "f_noise1", "--reason", "checked"]) == 0
    capsys.readouterr()
    state = json.loads((root / "state.json").read_text())
    assert state["unresolved"] == [] and state["kept"] == {"f_noise1": "checked"}
    lines = (root / "journal.jsonl").read_text().splitlines()
    assert [json.loads(x)["kind"] for x in lines] == ["init", "note", "keep"]


def test_cli_run_option_selects_run(tmp_path, monkeypatch, capsys, make_binary):
    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    main(["init", str(data), "--target", "target", "--goal", "first"])
    first = (tmp_path / ".aga" / "CURRENT").read_text().strip()
    main(["init", str(data), "--target", "target", "--goal", "second"])
    capsys.readouterr()
    assert main(["--run", first, "status"]) == 0
    assert "first" in capsys.readouterr().out
    assert main(["status", "--run", first]) == 0
    assert "first" in capsys.readouterr().out


def test_cli_no_run_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["status"]) == 1
    assert "No run found; run 'aga init' first." in capsys.readouterr().err


def test_cli_init_error_exits_1(tmp_path, monkeypatch, capsys, make_binary):
    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(data), "--target", "nope", "--goal", "g"]) == 1
    assert "nope" in capsys.readouterr().err


def test_cli_drop_target_refused_unknown_warns(tmp_path, monkeypatch, capsys, make_binary):
    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    main(["init", str(data), "--target", "target", "--goal", "g"])
    capsys.readouterr()
    assert main(["drop", "target", "--reason", "x"]) == 1
    assert main(["drop", "typo_col", "--reason", "x"]) == 0
    assert "Warning" in capsys.readouterr().out


def test_cli_drop_unresolved_outside_profile_clears_block(tmp_path, monkeypatch, capsys, make_binary):
    import json

    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    main(["init", str(data), "--target", "target", "--goal", "g"])
    root = tmp_path / ".aga" / (tmp_path / ".aga" / "CURRENT").read_text().strip()
    state = json.loads((root / "state.json").read_text())
    state["unresolved"] = ["engineered_feat"]
    (root / "state.json").write_text(json.dumps(state))
    capsys.readouterr()
    assert main(["drop", "engineered_feat", "--reason", "leaky"]) == 0
    assert "Warning" not in capsys.readouterr().out
    state = json.loads((root / "state.json").read_text())
    assert state["unresolved"] == [] and "engineered_feat" in state["dropped"]
    from autogluon_agent.run import Run

    Run.load(tmp_path).require_unblocked()


@pytest.mark.parametrize("budget", ["0", "-5", "nan", "inf"])
def test_cli_init_bad_budget_exits_1(tmp_path, monkeypatch, capsys, make_binary, budget):
    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(data), "--target", "target", "--goal", "g", "--budget", budget]) == 1


def test_cli_baseline(tmp_path, monkeypatch, capsys, make_binary):
    import json

    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(data), "--target", "target", "--goal", "g"]) == 0
    assert main(["drop", "leak", "--reason", "x"]) == 0
    capsys.readouterr()
    assert main(["baseline"]) == 0
    out = capsys.readouterr().out
    assert "dummy" in out and "lightgbm" in out and "±" in out
    assert out.strip().splitlines()[-1].startswith("Budget:")
    root = tmp_path / ".aga" / (tmp_path / ".aga" / "CURRENT").read_text().strip()
    state = json.loads((root / "state.json").read_text())
    assert len(state["best_scores"]) == 15 and state["best_label"] == "baseline"
    assert state["best_scores"] == state["baseline_scores"]["lightgbm"]
    assert len(state["baseline_scores"]["dummy"]) == 15


def test_cli_baseline_blocked_exits_2(tmp_path, monkeypatch, capsys, make_binary):
    import json

    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    main(["init", str(data), "--target", "target", "--goal", "g"])
    root = tmp_path / ".aga" / (tmp_path / ".aga" / "CURRENT").read_text().strip()
    state = json.loads((root / "state.json").read_text())
    state["unresolved"] = ["leak"]
    (root / "state.json").write_text(json.dumps(state))
    capsys.readouterr()
    assert main(["baseline"]) == 2


def test_cli_reinit_inherits_budget_and_task_flag(tmp_path, monkeypatch, capsys, make_binary):
    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(data), "--target", "target", "--goal", "g", "--budget", "40"]) == 0
    assert "inherited" not in capsys.readouterr().out
    assert main(["init", str(data), "--target", "target", "--goal", "g", "--task", "binary"]) == 0
    out = capsys.readouterr().out
    assert "Budget inherited from run" in out and "--fresh-budget" in out
    assert out.strip().splitlines()[-1].startswith("Budget: ")
    assert main(["init", str(data), "--target", "target", "--goal", "g", "--fresh-budget", "--budget", "5"]) == 0
    out = capsys.readouterr().out
    assert "inherited" not in out and "Budget: 5.0 min left" in out
    assert main(["init", str(data), "--target", "target", "--goal", "g", "--task", "multiclass"]) == 1
    assert main(["init", str(data), "--target", "target", "--goal", "g", "--task", "bogus"]) == 1


def _fitted_run(tmp_path, monkeypatch, make_binary, **state):
    from autogluon_agent.run import Run

    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(data), "--target", "target", "--goal", "g"]) == 0
    run = Run.load(tmp_path)
    for key, value in state.items():
        setattr(run.state, key, value)
    run.save()
    return run


class _Calls(list):
    """Recorded time limits; ``.fake.presets`` holds the presets of each call."""


def _capture(monkeypatch, name):
    """Replace autogluon_agent.fit.<name> with a recorder that returns a minimal result."""
    from autogluon_agent import fit

    calls = _Calls()

    def fake(run, time_limit, presets="best_quality"):
        calls.append(time_limit)
        fake.presets.append(presets)
        return {"score_val": 0.8, "leaderboard_top5": [{"model": "M", "score_val": 0.8}],
                "time_used": 1.0, "presets": presets, "notes": [], "n_rows": 3000}

    fake.presets = []
    monkeypatch.setattr(fit, name, fake)
    calls.fake = fake
    return calls


@pytest.mark.parametrize("remaining, expected, warns", [(1380.0, 600.0, False), (400.0, 110.0, False), (250.0, 60.0, True)])
def test_fit_default_time_limit_is_half_the_remaining_budget(
    tmp_path, monkeypatch, capsys, make_binary, remaining, expected, warns
):
    from autogluon_agent.run import Run

    _fitted_run(tmp_path, monkeypatch, make_binary)
    calls = _capture(monkeypatch, "fit_final")
    monkeypatch.setattr(Run, "remaining_seconds", lambda self: remaining)
    capsys.readouterr()
    assert main(["fit"]) == 0
    out = capsys.readouterr().out
    assert calls == [expected]
    assert f"time limit {expected:.0f} s" in out and ("Warning: little budget left" in out) is warns
    assert main(["fit", "--time-limit", "75"]) == 0
    assert calls[-1] == 75.0  # an explicit --time-limit wins


def test_time_limit_help_states_the_defaults():
    from autogluon_agent.cli import build_parser

    choices = build_parser()._subparsers._group_actions[0].choices
    helps = {name: next(a.help for a in choices[name]._actions if a.dest == "time_limit") for name in ("fit", "refit")}
    assert "(remaining budget - 180) / 2" in helps["fit"]
    assert "remaining budget - 180" in helps["refit"] and "/ 2" not in helps["refit"]
    assert all("60%% for the fit, 40%% for refit_full" in h for h in helps.values())


def test_refit_default_time_limit_is_the_remaining_budget(tmp_path, monkeypatch, capsys, make_binary):
    from autogluon_agent.run import Run

    run = _fitted_run(tmp_path, monkeypatch, make_binary, fitted=True, eval_count=1, evaluated=True)
    run.journal.append("fit", presets="medium_quality")
    calls = _capture(monkeypatch, "refit_final")
    monkeypatch.setattr(Run, "remaining_seconds", lambda self: 1380.0)
    capsys.readouterr()
    assert main(["refit"]) == 0
    out = capsys.readouterr().out
    assert calls == [1200.0] and "3000 labelled rows" in out
    assert calls.fake.presets == ["medium_quality"] and "presets=medium_quality" in out  # those of the fit
    assert out.strip().splitlines()[-1].startswith("Budget:")
    assert main(["refit", "--time-limit", "90", "--presets", "good_quality"]) == 0
    assert calls[-1] == 90.0 and calls.fake.presets[-1] == "good_quality"


def test_refit_before_evaluate_is_refused(tmp_path, monkeypatch, capsys, make_binary):
    _fitted_run(tmp_path, monkeypatch, make_binary, fitted=True, eval_count=0)
    capsys.readouterr()
    assert main(["refit"]) == 1
    assert "run 'aga evaluate' first: refitting before the one-shot evaluation would train on the holdout" in capsys.readouterr().err


def test_refit_needs_a_fitted_model(tmp_path, monkeypatch, capsys, make_binary):
    _fitted_run(tmp_path, monkeypatch, make_binary)
    capsys.readouterr()
    assert main(["refit"]) == 1
    assert "No fitted model; run 'aga fit' first." in capsys.readouterr().err


def test_refit_blocked_by_unresolved_leakage(tmp_path, monkeypatch, capsys, make_binary):
    _fitted_run(tmp_path, monkeypatch, make_binary, fitted=True, eval_count=1, unresolved=["leak"])
    assert main(["refit"]) == 2


def test_status_shows_refit(tmp_path, monkeypatch, capsys, make_binary):
    run = _fitted_run(tmp_path, monkeypatch, make_binary)
    capsys.readouterr()
    main(["status"])
    assert "Refit: no" in capsys.readouterr().out
    run.state.refit, run.state.refit_rows = True, 3000
    run.save()
    main(["status"])
    assert "Refit: yes (3000 rows)" in capsys.readouterr().out


def test_init_hints_at_date_columns_unless_time_given(tmp_path, monkeypatch, capsys, make_binary):
    data = _init_data(tmp_path, make_binary)
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    assert main(["init", str(data), "--target", "target", "--goal", "g"]) == 0
    hints = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Hint:")]
    assert len(hints) == 1
    assert hints[0].startswith("Hint: column 'ts' holds dates (2024-01-01 ")
    assert hints[0].endswith("If predictions are about the future, re-run init with --time ts.")
    assert main(["init", str(data), "--target", "target", "--goal", "g", "--time", "ts"]) == 0
    assert "Hint:" not in capsys.readouterr().out



def test_refit_needs_the_current_model_evaluated(tmp_path, monkeypatch, capsys, make_binary):
    _fitted_run(tmp_path, monkeypatch, make_binary, fitted=True, eval_count=1, evaluated=False)
    capsys.readouterr()
    assert main(["refit"]) == 1
    assert (
        "the current model has not been evaluated; run 'aga evaluate --force' first "
        "(the holdout will be marked reused)"
    ) in capsys.readouterr().err


def test_refit_records_recipe_drift_without_changing_the_evaluated_recipe(tmp_path, monkeypatch, capsys, make_binary):
    """AutoGluon is faked: this checks refit_final's bookkeeping (modules, presets, journal, report)."""
    from autogluon_agent import fit
    from autogluon_agent.report import build_report
    from autogluon_agent.run import Run

    run = _fitted_run(tmp_path, monkeypatch, make_binary, fitted=True, eval_count=1, evaluated=True,
                      active_modules=["01_a.py", "02_b.py"], best_label="baseline + 01_a.py, 02_b.py")
    run.journal.append("fit", presets="medium_quality", n_rows=2400, leaderboard_top5=[], notes=[],
                       modules=["01_a.py", "02_b.py"], dropped=[])
    rows = []

    def fake_train(run, data, out, time_limit, presets, during="fit"):
        rows.append(len(data))
        out.mkdir(parents=True, exist_ok=True)
        return {"score_val": -1.0, "score_val_natural": 1.0, "leaderboard_top5": [], "time_used": 2.0,
                "presets": presets, "notes": [], "n_rows": len(data), "modules": ["02_b.py"]}  # 01_a.py failed

    monkeypatch.setattr(fit, "_train", fake_train)
    capsys.readouterr()
    assert main(["refit", "--time-limit", "60", "--presets", "good_quality"]) == 0
    assert rows == [3000]
    run = Run.load(tmp_path)
    assert run.state.refit and run.state.refit_rows == 3000
    assert run.state.active_modules == ["01_a.py", "02_b.py"]  # restored: the evaluated recipe is unchanged
    assert run.state.best_label == "baseline + 01_a.py, 02_b.py"
    entry = run.journal.entries("refit")[-1]
    assert entry["skipped_modules"] == ["01_a.py"] and entry["modules"] == ["02_b.py"]
    assert entry["added_modules"] == entry["removed_modules"] == entry["dropped_after_fit"] == []
    assert entry["presets"] == "good_quality" and entry["fit_presets"] == "medium_quality"
    assert any("01_a.py, which failed on the full data" in n for n in entry["notes"])
    assert any("presets differ" in n and "good_quality" in n and "medium_quality" in n for n in entry["notes"])
    text = build_report(run, tmp_path)
    assert "module 01_a.py, which failed on the full data, is left out" in text
    assert "the refit used presets good_quality instead of medium_quality" in text
    assert "same recipe" not in text

    # a failing refit restores the recipe too and claims no refit
    def broken_train(run, data, out, time_limit, presets, during="fit"):
        out.mkdir(parents=True, exist_ok=True)  # partial staging output
        run.state.active_modules = ["02_b.py"]
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(fit, "_train", broken_train)
    with pytest.raises(RuntimeError):
        main(["refit", "--time-limit", "60"])
    run = Run.load(tmp_path)
    assert run.state.refit is False and run.state.active_modules == ["01_a.py", "02_b.py"]
    assert not (run.root / "models_full.tmp").exists()



def test_refit_refuses_a_recipe_changed_since_the_last_fit(tmp_path, monkeypatch, capsys, make_binary):
    """R36: even with the old model evaluated (e.g. 'evaluate --force' after the change), a recipe that
    differs from the last fit entry must be fitted and evaluated before a refit."""
    from autogluon_agent import fit
    from autogluon_agent.run import Run

    _fitted_run(tmp_path, monkeypatch, make_binary, fitted=True, eval_count=2, evaluated=True,
                active_modules=["01_a.py", "03_c.py"], dropped={"f_noise1": "x"})
    run = Run.load(tmp_path)
    run.journal.append("fit", presets="best_quality", n_rows=2400, leaderboard_top5=[], notes=[],
                       modules=["01_a.py", "02_b.py"], dropped=["leak"])
    monkeypatch.setattr(fit, "_train", lambda *a, **k: pytest.fail("refit must not train"))
    capsys.readouterr()
    assert main(["refit", "--time-limit", "60"]) == 1
    err = capsys.readouterr().err
    assert "the feature recipe changed since the last fit (" in err
    assert "run 'aga fit', then 'aga evaluate --force'" in err
    for phrase in ("module 03_c.py, accepted after the evaluated fit, is added",
                   "module 02_b.py, deactivated after the evaluated fit, is left out",
                   "column f_noise1, dropped after the evaluated fit, is left out",
                   "column leak, re-admitted after the evaluated fit, is used"):
        assert phrase in err
    assert R34 not in err
    run = Run.load(tmp_path)
    assert run.state.refit is False and not run.journal.entries("refit")


GOOD_MODULE = """
import pandas as pd
class Features:
    def fit(self, X, y): return self
    def transform(self, X): return pd.DataFrame({"s12": X["f_signal1"] * 2 + X["f_signal2"]}, index=X.index)
"""
NOISE_MODULE = """
import numpy as np, pandas as pd
class Features:
    def fit(self, X, y): return self
    def transform(self, X):
        return pd.DataFrame({"junk": np.sin(np.arange(len(X)) * 7.3)}, index=X.index)
"""
R34 = "the current model has not been evaluated; run 'aga evaluate --force' first"


def _evaluated_run_with_baseline(tmp_path, monkeypatch, make_binary):
    from autogluon_agent.run import Run

    data = tmp_path / "data.csv"
    make_binary(n=1200).drop(columns=["leak", "ts"]).to_csv(data, index=False)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(data), "--target", "target", "--goal", "g"]) == 0
    assert main(["baseline"]) == 0
    run = Run.load(tmp_path)
    run.state.fitted, run.state.eval_count, run.state.evaluated = True, 1, True  # as after 'aga evaluate'
    run.save()
    run.journal.append("fit", presets="best_quality", n_rows=960, leaderboard_top5=[], notes=[],
                       modules=[], dropped=[])


R36 = "the feature recipe changed since the last fit ("


def test_accepted_try_after_evaluate_requires_reevaluation(tmp_path, monkeypatch, capsys, make_binary):
    from autogluon_agent.run import Run

    _evaluated_run_with_baseline(tmp_path, monkeypatch, make_binary)
    (tmp_path / "noise.py").write_text(NOISE_MODULE)
    capsys.readouterr()
    assert main(["try", "noise.py"]) == 0
    assert "REJECTED" in capsys.readouterr().out
    assert Run.load(tmp_path).state.evaluated is True  # a rejected try changes nothing
    (tmp_path / "good.py").write_text(GOOD_MODULE)
    assert main(["try", "good.py"]) == 0
    assert "ACCEPTED" in capsys.readouterr().out
    assert Run.load(tmp_path).state.evaluated is False
    assert main(["refit"]) == 1
    err = capsys.readouterr().err
    assert R36 in err and "_good.py, accepted after the evaluated fit, is added" in err
    assert R34 not in err  # 'evaluate --force' would only re-score the old model


ZERO_MODULE = """
import pandas as pd
class Features:
    def fit(self, X, y): return self
    def transform(self, X): return pd.DataFrame({"zero": X["f_signal1"] * 0.0}, index=X.index)
"""


def test_rerunning_baseline_keeps_the_accepted_modules_as_reference(tmp_path, monkeypatch, capsys, make_binary):
    from autogluon_agent.run import Run

    _evaluated_run_with_baseline(tmp_path, monkeypatch, make_binary)
    (tmp_path / "good.py").write_text(GOOD_MODULE)
    assert main(["try", "good.py"]) == 0
    before = Run.load(tmp_path).state
    capsys.readouterr()
    assert main(["baseline"]) == 0
    assert "Reference only: the keep/reject reference stays" in capsys.readouterr().out
    after = Run.load(tmp_path).state
    assert after.best_scores == before.best_scores and after.best_label == before.best_label != "baseline"
    assert after.best_scores != after.baseline_scores["lightgbm"]
    (tmp_path / "zero.py").write_text(ZERO_MODULE)
    assert main(["try", "zero.py"]) == 0
    assert "REJECTED" in capsys.readouterr().out  # a no-op module gets no credit for good.py's gain


def test_failed_second_refit_says_the_earlier_refit_is_no_longer_used(tmp_path, monkeypatch, capsys, make_binary):
    import autogluon_agent.fit as fit_mod
    from autogluon_agent.errors import AgaError
    from autogluon_agent.run import Run

    _evaluated_run_with_baseline(tmp_path, monkeypatch, make_binary)
    run = Run.load(tmp_path)
    run.state.refit = True  # an earlier refit succeeded
    run.save()

    def boom(*a, **k):
        raise AgaError("training failed")

    monkeypatch.setattr(fit_mod, "_train", boom)
    capsys.readouterr()
    assert main(["refit", "--time-limit", "60"]) == 1
    assert "the earlier production refit is no longer used" in capsys.readouterr().err
    assert Run.load(tmp_path).state.refit is False


def test_drop_after_evaluate_requires_reevaluation(tmp_path, monkeypatch, capsys, make_binary):
    from autogluon_agent.run import Run

    _evaluated_run_with_baseline(tmp_path, monkeypatch, make_binary)
    capsys.readouterr()
    assert main(["drop", "f_noise1", "--reason", "late decision"]) == 0
    assert main(["refit"]) == 1
    assert "column f_noise1, dropped after the evaluated fit, is left out" in capsys.readouterr().err
    run = Run.load(tmp_path)
    run.state.evaluated = True
    run.save()
    assert main(["drop", "f_noise1", "--reason", "same column, new reason"]) == 0
    assert Run.load(tmp_path).state.evaluated is True  # already dropped: the recipe did not change
    assert main(["keep", "f_noise2", "--reason", "known at prediction time"]) == 0
    assert Run.load(tmp_path).state.evaluated is True  # keeping a column that is in use changes nothing
    assert main(["keep", "f_noise1", "--reason", "re-admitted"]) == 0
    assert Run.load(tmp_path).state.evaluated is False  # re-admitting a dropped column does


def test_deactivation_clears_evaluated(tmp_path, monkeypatch, make_binary):
    from autogluon_agent.run import Run

    run = _fitted_run(tmp_path, monkeypatch, make_binary, evaluated=True, active_modules=["01_a.py"])
    run.deactivate_module("01_a.py", ValueError("x"), "rescore")
    assert Run.load(tmp_path).state.evaluated is False
