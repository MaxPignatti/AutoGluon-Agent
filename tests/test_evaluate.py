import json, numpy as np, pytest
from autogluon_agent.evaluate import ece
from autogluon_agent.cli import main

def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0); p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert ece(y, p) < 0.02

def test_ece_overconfident_is_large():
    y = np.array([0, 1] * 500); p = np.where(y == 1, 0.99, 0.98)
    assert ece(y, p) > 0.4

def test_ece_multiclass_uses_top_label_confidence():
    y = np.array([0, 1] * 500)
    proba = np.tile([0.98, 0.01, 0.01], (1000, 1))  # always confident in class 0; right half the time
    assert ece(y, proba) == pytest.approx(0.48, abs=1e-6)

def test_evaluate_requires_fitted_model(tmp_path, make_binary, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    make_binary(n=300).drop(columns=["leak"]).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    assert main(["evaluate"]) == 1
    assert "No fitted model; run 'aga fit' first." in capsys.readouterr().err

@pytest.mark.slow
def test_evaluate_once_then_blocked(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_binary(n=1500).drop(columns=["leak"]).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"])
    main(["fit", "--time-limit", "60", "--presets", "medium_quality"])
    assert main(["evaluate"]) == 0
    assert main(["evaluate"]) == 2
    assert main(["evaluate", "--force"]) == 0
    run_dir = next((tmp_path / ".aga").glob("2*"))
    ev = json.loads((run_dir / "evaluation.json").read_text())
    assert ev["holdout_reused"] is True and "ece" in ev and (run_dir / "calibration.png").exists()


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["multiclass", "regression"])
def test_evaluate_other_tasks(tmp_path, make_multiclass, make_regression, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    df = make_multiclass(n=1200) if kind == "multiclass" else make_regression(n=1200)
    df.to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"]); main(["baseline"])
    main(["fit", "--time-limit", "60", "--presets", "medium_quality"])
    assert main(["evaluate"]) == 0
    run_dir = next((tmp_path / ".aga").glob("2*"))
    ev = json.loads((run_dir / "evaluation.json").read_text())
    assert ev["holdout_reused"] is False and ev["gap"] is not None and ev["cv_proxy_best"] is not None
    assert ("ece" in ev) == (kind == "multiclass") and "brier" not in ev
    assert (run_dir / "calibration.png").exists() == (kind == "multiclass")


@pytest.mark.parametrize("extra", [[], ["--force"]])
def test_evaluate_refused_after_refit(tmp_path, make_binary, monkeypatch, capsys, extra):
    from autogluon_agent.run import Run

    monkeypatch.chdir(tmp_path)
    make_binary(n=300).drop(columns=["leak"]).to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    run = Run.load(tmp_path)
    run.state.fitted, run.state.eval_count, run.state.refit, run.state.refit_rows = True, 1, True, 300
    run.save()
    capsys.readouterr()
    assert main(["evaluate", *extra]) == 1
    assert "the production model was refit on the holdout rows; it cannot be evaluated on them" in capsys.readouterr().err
    assert Run.load(tmp_path).state.eval_count == 1
