import textwrap

import pytest

from autogluon_agent.cli import main
from autogluon_agent.errors import ContractError
from autogluon_agent.features import accept, load_module, make_fold_transform

GOOD = """
import pandas as pd
class Features:
    def fit(self, X, y): return self
    def transform(self, X): return pd.DataFrame({"x1x2": X["x1"] * X["x2"]}, index=X.index)
"""
NOISE = """
import numpy as np, pandas as pd
class Features:
    def fit(self, X, y): return self
    def transform(self, X):
        return pd.DataFrame({"rnd": np.random.default_rng(0).normal(size=len(X))}, index=X.index)
"""
DROPS_ROWS = """
class Features:
    def fit(self, X, y): return self
    def transform(self, X): return X.iloc[:-1][["x1"]].rename(columns={"x1": "y"})
"""
LEAKY_TE = """
import pandas as pd
class Features:
    def fit(self, X, y): self.y = pd.Series(y, index=X.index); return self
    def transform(self, X): return pd.DataFrame({"te": self.y.reindex(X.index)}, index=X.index)
"""


def mod(tmp_path, src, name):
    p = tmp_path / f"{name}.py"
    p.write_text(textwrap.dedent(src))
    return p


def test_accept_rule():
    assert accept([0.82, 0.83, 0.81], [0.80, 0.80, 0.80])[0]
    assert not accept([0.81, 0.79, 0.80], [0.80, 0.80, 0.80])[0]


def test_contract_violation(tmp_path, make_regression):
    df = make_regression()
    X, y = df.drop(columns="target"), df.target
    ft = make_fold_transform([load_module(mod(tmp_path, DROPS_ROWS, "d"))], problem=None)
    with pytest.raises(ContractError):
        ft(X.iloc[:2000], y.iloc[:2000], X.iloc[2000:])


def test_try_accepts_signal_rejects_noise_and_leak(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_regression().to_csv("d.csv", index=False)
    assert main(["init", "d.csv", "--target", "target", "--goal", "g"]) == 0
    assert main(["baseline"]) == 0
    for name, src, expect in [("good", GOOD, True), ("noise", NOISE, False), ("leaky", LEAKY_TE, False)]:
        assert main(["try", str(mod(tmp_path, src, name))]) == 0
    from autogluon_agent.run import Run

    run = Run.load(tmp_path)
    trials = run.journal.entries("trial")
    assert [t["accepted"] for t in trials] == [True, False, False]
    assert any(f.startswith("target_leak_suspect") for f in trials[2]["flags"])
    assert run.state.trial_count == 3
    assert run.state.active_modules == ["01_good.py"]
    assert (run.root / "features" / "02_noise.py").exists()  # rejected files stay for the audit trail


def test_try_logs_module_exception(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_regression().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    main(["baseline"])
    bad = mod(tmp_path, "class Features:\n def fit(self,X,y): raise ValueError('boom')\n def transform(self,X): pass\n", "bad")
    assert main(["try", str(bad)]) == 0
    from autogluon_agent.run import Run

    t = Run.load(tmp_path).journal.entries("trial")[-1]
    assert not t["accepted"] and "boom" in t["error"]


def test_try_requires_baseline(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_regression().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    assert main(["try", str(mod(tmp_path, GOOD, "good"))]) == 1


def test_try_module_without_features_class_or_failing_import(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_regression().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    main(["baseline"])
    assert main(["try", str(mod(tmp_path, "x = 1\n", "nofeat"))]) == 0
    assert main(["try", str(mod(tmp_path, "raise RuntimeError('import-boom')\n", "badimport"))]) == 0
    from autogluon_agent.run import Run

    run = Run.load(tmp_path)
    t1, t2 = run.journal.entries("trial")
    assert not t1["accepted"] and "Features" in t1["error"]
    assert not t2["accepted"] and "import-boom" in t2["error"]
    assert run.state.trial_count == 2 and run.state.active_modules == []


def test_try_blocked_when_unresolved(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_regression().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    main(["baseline"])
    from autogluon_agent.run import Run

    run = Run.load(tmp_path)
    run.state.unresolved = ["lat"]
    run.save()
    assert main(["try", str(mod(tmp_path, GOOD, "good"))]) == 2


def test_merge_adds_and_replaces():
    import pandas as pd

    from autogluon_agent.features import merge

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[5, 7])
    new = pd.DataFrame({"b": [9, 9], "c": [0, 1]}, index=[5, 7])
    out = merge(X, new)
    assert list(out.columns) == ["a", "b", "c"]
    assert out["b"].tolist() == [9, 9] and list(out.index) == [5, 7]
    assert X["b"].tolist() == [3, 4]  # input untouched


def test_try_prints_delta_and_trials(tmp_path, make_regression, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    make_regression().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    main(["baseline"])
    capsys.readouterr()
    assert main(["try", str(mod(tmp_path, GOOD, "good")), "--note", "product"]) == 0
    out = capsys.readouterr().out
    assert "ACCEPTED" in out and "±" in out and "Trials: 1" in out and "Budget:" in out
    # rmse is lower-is-better: an improvement shows as a negative natural-unit delta
    assert "Delta rmse: -" in out


NOOP = """
import pandas as pd
class Features:
    def fit(self, X, y): return self
    def transform(self, X): return pd.DataFrame({"const": 1.0}, index=X.index)
"""


def test_try_rescores_after_drop(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    assert main(["drop", "leak", "--reason", "leaky"]) == 0
    assert main(["baseline"]) == 0
    assert main(["drop", "f_signal1", "--reason", "test"]) == 0
    assert main(["try", str(mod(tmp_path, NOOP, "noop"))]) == 0
    from autogluon_agent.run import Run

    run = Run.load(tmp_path)
    rescores = run.journal.entries("rescore")
    assert len(rescores) == 1 and rescores[0]["new_mean"] < rescores[0]["old_mean"]
    t = run.journal.entries("trial")[-1]
    assert not t["accepted"] and abs(t["mean_delta"]) < 0.005
    assert run.state.best_context == ["f_signal1", "leak"]
    # a second try under the same feature set does not rescore again
    assert main(["try", str(mod(tmp_path, NOOP, "noop2"))]) == 0
    assert len(Run.load(tmp_path).journal.entries("rescore")) == 1


def test_modules_loaded_fresh_per_fold(tmp_path, make_regression):
    src = """
import pandas as pd
CALLS = []
class Features:
    def fit(self, X, y): CALLS.append(1); self.n = len(CALLS); return self
    def transform(self, X): return pd.DataFrame({"n": float(self.n)}, index=X.index)
"""
    df = make_regression()
    X, y = df.drop(columns="target"), df.target
    ft = make_fold_transform([mod(tmp_path, src, "stateful")], problem=None)
    for _ in range(3):
        Xtr, _, _ = ft(X.iloc[:2000], y.iloc[:2000], X.iloc[2000:])
        assert (Xtr["n"] == 1.0).all()


def test_harness_error_is_not_a_rejected_trial(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_regression().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    main(["baseline"])
    from autogluon_agent.run import Run

    (Run.load(tmp_path).root / "train.parquet").unlink()
    with pytest.raises(Exception):
        main(["try", str(mod(tmp_path, GOOD, "good"))])
    run = Run.load(tmp_path)
    assert run.state.trial_count == 0 and run.journal.entries("trial") == []


READS_SIGNAL = """
import pandas as pd
class Features:
    def fit(self, X, y): return self
    def transform(self, X): return pd.DataFrame({"s2": X["f_signal1"] ** 2}, index=X.index)
"""


def _with_accepted_module(tmp_path, make_binary, monkeypatch, *modules):
    """A baselined run whose active set holds ``modules`` ((name, source) pairs) as if accepted."""
    from autogluon_agent.run import Run

    monkeypatch.chdir(tmp_path)
    make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    main(["drop", "leak", "--reason", "leaky"])
    main(["baseline"])
    run = Run.load(tmp_path)
    (run.root / "features").mkdir(exist_ok=True)
    for name, src in modules:
        (run.root / "features" / name).write_text(textwrap.dedent(src))
        run.state.active_modules.append(name)
    run.state.best_label = "baseline + " + ", ".join(run.state.active_modules)
    run.save()
    return run


def test_try_rescore_deactivates_module_broken_by_drop(tmp_path, make_binary, monkeypatch, capsys):
    from autogluon_agent.run import Run

    _with_accepted_module(tmp_path, make_binary, monkeypatch, ("01_reads.py", READS_SIGNAL), ("02_good.py", NOOP))
    assert main(["drop", "f_signal1", "--reason", "late leak verdict"]) == 0
    capsys.readouterr()
    assert main(["try", str(mod(tmp_path, NOOP, "noop"))]) == 0
    out = capsys.readouterr().out
    assert "DEACTIVATED 01_reads.py" in out and "KeyError" in out
    run = Run.load(tmp_path)
    assert run.state.active_modules == ["02_good.py"]
    assert run.state.best_label == "baseline + 02_good.py"
    (d,) = run.journal.entries("deactivate")
    assert d["module"] == "01_reads.py" and d["during"] == "rescore" and "f_signal1" in d["error"]
    assert len(run.journal.entries("rescore")) == 1 and run.state.trial_count == 1
    assert run.journal.entries("trial")[-1]["error"] is None


def test_try_rescore_missing_module_file_is_an_error(tmp_path, make_binary, monkeypatch):
    from autogluon_agent.run import Run

    run = _with_accepted_module(tmp_path, make_binary, monkeypatch, ("01_reads.py", READS_SIGNAL))
    (run.root / "features" / "01_reads.py").unlink()
    main(["drop", "f_noise1", "--reason", "test"])
    assert main(["try", str(mod(tmp_path, NOOP, "noop"))]) == 1
    assert Run.load(tmp_path).state.active_modules == ["01_reads.py"]


def test_fit_modules_deactivates_failing_module(tmp_path, make_binary, monkeypatch, capsys):
    from autogluon_agent.fit import _fit_modules
    from autogluon_agent.run import Run

    _with_accepted_module(tmp_path, make_binary, monkeypatch, ("01_reads.py", READS_SIGNAL), ("02_good.py", NOOP))
    main(["drop", "f_signal1", "--reason", "late leak verdict"])
    run = Run.load(tmp_path)
    train = run.train_df()
    capsys.readouterr()
    instances, X, used = _fit_modules(run, run.base_features(train), train["target"])
    assert len(instances) == 1 and "const" in X and "s2" not in X and used == ["02_good.py"]
    assert "DEACTIVATED 01_reads.py" in capsys.readouterr().out
    run = Run.load(tmp_path)
    assert run.state.active_modules == ["02_good.py"] and run.state.best_context is None
    assert run.journal.entries("deactivate")[0]["during"] == "fit"
    assert run.state.evaluated is False  # a deactivation changes the recipe


def test_fit_modules_during_refit_only_skips_a_failing_module(tmp_path, make_binary, monkeypatch, capsys):
    from autogluon_agent.fit import _fit_modules
    from autogluon_agent.run import Run

    _with_accepted_module(tmp_path, make_binary, monkeypatch, ("01_reads.py", READS_SIGNAL), ("02_good.py", NOOP))
    main(["drop", "f_signal1", "--reason", "late leak verdict"])
    run = Run.load(tmp_path)
    run.state.evaluated = True
    run.save()
    before = (list(run.state.active_modules), run.state.best_context)
    train = run.train_df()
    capsys.readouterr()
    instances, X, used = _fit_modules(run, run.base_features(train), train["target"], during="refit")
    assert len(instances) == 1 and used == ["02_good.py"]
    out = capsys.readouterr().out
    assert "skipped for the refit: 01_reads.py (failed on the full data); the evaluated recipe keeps it" in out
    assert "DEACTIVATED" not in out
    run = Run.load(tmp_path)
    assert (run.state.active_modules, run.state.best_context) == before and run.state.evaluated is True
    assert not run.journal.entries("deactivate")


def _loop_lines(capsys):
    return [line for line in capsys.readouterr().out.splitlines() if line.startswith("Feature loop:")]


BROKEN = """
class Features:
    def fit(self, X, y): raise ValueError("boom")
    def transform(self, X): pass
"""


def _setup_try(tmp_path, make_regression, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    make_regression().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    main(["baseline"])
    capsys.readouterr()


def test_try_prints_feature_loop_stop_status(tmp_path, make_regression, monkeypatch, capsys):
    _setup_try(tmp_path, make_regression, monkeypatch, capsys)
    main(["try", str(mod(tmp_path, GOOD, "good"))])
    (line,) = _loop_lines(capsys)
    assert "Feature loop: 0 rejected in a row" in line and "Keep going until 3 rejected in a row or 40% used" in line
    for i in range(3):
        main(["try", str(mod(tmp_path, NOOP, f"noop{i}"))])
    loop = _loop_lines(capsys)
    assert len(loop) == 3
    assert "2 rejected in a row" in loop[1] and "Keep going" in loop[1]
    assert loop[2].startswith("Feature loop: stop condition met (3 rejected in a row)")
    assert "aga fit" in loop[2]


def test_feature_loop_streak_restarts_after_accept(tmp_path, make_regression, monkeypatch, capsys):
    _setup_try(tmp_path, make_regression, monkeypatch, capsys)
    for name, src in [("noop0", NOOP), ("good", GOOD), ("noop1", NOOP)]:
        main(["try", str(mod(tmp_path, src, name))])
    loop = _loop_lines(capsys)
    assert ["1 rejected in a row" in loop[0], "0 rejected in a row" in loop[1], "1 rejected in a row" in loop[2]] == [True] * 3


def test_feature_loop_error_and_leak_trials_neither_count_nor_reset(tmp_path, make_regression, monkeypatch, capsys):
    _setup_try(tmp_path, make_regression, monkeypatch, capsys)
    for name, src in [("noop0", NOOP), ("broken", BROKEN), ("leaky", LEAKY_TE), ("noop1", NOOP)]:
        main(["try", str(mod(tmp_path, src, name))])
    loop = _loop_lines(capsys)
    assert len(loop) == 4
    assert all("1 rejected in a row" in line for line in loop[:3])
    assert "2 rejected in a row" in loop[3] and "Keep going" in loop[3]
    from autogluon_agent.run import Run

    trials = Run.load(tmp_path).journal.entries("trial")
    assert [t["errored"] for t in trials] == [False, True, True, False]


def test_feature_loop_budget_branch(tmp_path, make_regression, monkeypatch, capsys):
    from autogluon_agent.run import Run

    _setup_try(tmp_path, make_regression, monkeypatch, capsys)
    monkeypatch.setattr(Run, "budget_used_pct", lambda self, remaining=None: 45)
    main(["try", str(mod(tmp_path, GOOD, "good"))])
    (line,) = _loop_lines(capsys)
    assert line.startswith("Feature loop: stop condition met (45% of the budget used)") and "aga fit" in line
