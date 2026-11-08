import json
import textwrap

import pytest

from autogluon_agent.cli import main
from autogluon_agent.report import build_report
from autogluon_agent.run import Run

HEADINGS = ["## Problem", "## Framing", "## Leakage", "## Baseline", "## Feature experiments",
            "## Final model", "## Holdout results", "## Calibration", "## Feature importance",
            "## Analysis", "## Decision log", "## Run stats"]


def test_report_with_nothing_run(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    assert main(["report"]) == 0
    text = (tmp_path / "REPORT.md").read_text()
    assert all(h in text for h in HEADINGS) and "Not run." in text
    assert text.startswith("# AutoGluon-Agent Report")
    assert [text.index(h) for h in HEADINGS] == sorted(text.index(h) for h in HEADINGS)
    assert text.count("Not written.") == 2
    assert "No notes recorded." in text
    assert "The delivered model was not refit" not in text  # no fit at all: "Not run."
    run = Run.load(tmp_path)
    assert (run.root / "REPORT.md").exists()


def test_report_sections_from_journal(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    (tmp_path / "FRAMING.md").write_text("Predict churn.")
    run = Run.load(tmp_path)
    run.journal.append("baseline", metric="roc_auc", n_folds=5, dummy_mean=0.5, dummy_std=0.0,
                       lightgbm_mean=0.8, lightgbm_std=0.02)
    run.journal.append("trial", module="01_f.py", note="a|b", mean_delta=0.01, se=0.002,
                       delta_natural=0.01, accepted=True, flags=[], new_columns=["c"], reason=None, error=None)
    run.journal.append("rescore", reason="dropped", old_mean=0.8, new_mean=0.79, dropped=["x"], modules=[])
    run.journal.append("trial", module="02_g.py", note=None, mean_delta=0.0, se=0.0, delta_natural=0.0,
                       accepted=False, flags=[], new_columns=[], reason="module raised an error",
                       error="Traceback\nValueError: boom")
    ev = {"metric": "roc_auc", "holdout_score": 0.81, "secondary": {"pr_auc": 0.7}, "cv_proxy_best": 0.8,
          "ag_score_val": 0.79, "gap": 0.02, "n_holdout": 600, "holdout_reused": True,
          "brier": 0.2, "ece": 0.03, "importance": [{"feature": "x1", "importance": 0.1, "stddev": 0.01}]}
    (run.root / "evaluation.json").write_text(json.dumps(ev))
    run.journal.append("evaluate", **{**ev, "holdout_score": 0.83}, eval_count=1)
    run.journal.append("evaluate", **ev, eval_count=2)
    text = build_report(run, tmp_path)
    assert "Predict churn." in text
    assert "0.5000 ± 0.0000" in text and "0.8000 ± 0.0200" in text
    assert r"a\|b" in text and "(rescore)" in text and "2 trials, 1 accepted" in text
    assert "ValueError: boom" in text
    assert "first evaluation" in text and "0.8300" in text and "holdout reused" in text
    assert "![Reliability diagram](.aga/" in text and "/calibration.png)" in text
    assert "![Reliability diagram](calibration.png)" in (run.root / "REPORT.md").read_text()
    assert "| x1 |" in text
    assert "stale holdout result" not in text
    run.state.evaluated = False  # as after evaluate -> fit, or an accepted try after evaluate
    assert "stale holdout result" in build_report(run, tmp_path)


def test_report_decision_log_grouped_by_phase(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    for text, phase in [("untagged first", None), ("feature A taught B", "features"), ("refit done", "refit"),
                        ("framing chosen", "frame"), ("custom phase", "eda"), ("feature C", "features"),
                        ("no leaks", "leakage")]:
        assert main(["note", text] + (["--phase", phase] if phase else [])) == 0
    text = build_report(Run.load(tmp_path), tmp_path)
    log = text[text.index("## Decision log"):text.index("## Run stats")]
    order = ["framing chosen", "no leaks", "feature A taught B", "feature C", "refit done", "custom phase", "untagged first"]
    assert [log.index(n) for n in order] == sorted(log.index(n) for n in order)
    assert "### frame" in log and "### features" in log and "### refit" in log
    assert log.index("### refit") < log.index("custom phase") < log.index("untagged first")
    note_ts = Run.load(tmp_path).journal.entries("note")[0]["ts"]
    assert note_ts.replace("T", " ")[:19] in log  # each entry shows its time
    assert "No notes recorded." not in text


def _decision_log_of(text):
    return text[text.index("## Decision log"):text.index("## Run stats")]


def test_report_decision_log_keeps_each_note_on_one_line(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    assert main(["note", "line one\nline two | with pipe\r\nend", "--phase", "fit"]) == 0
    log = _decision_log_of(build_report(Run.load(tmp_path), tmp_path))
    bullets = [line for line in log.splitlines() if line.startswith("- ")]
    assert len(bullets) == 1
    assert bullets[0].endswith(r"line one line two \| with pipe  end")


def test_report_decision_log_only_untagged_notes(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    run = Run.load(tmp_path)
    run.journal.append("note", text="no phase key")       # older journals may lack the key
    run.journal.append("note", text="empty phase", phase="")  # "" counts as untagged
    run.journal.append("note", text="none phase", phase=None)
    log = _decision_log_of(build_report(run, tmp_path))
    assert log.count("### ") == 1 and "### untagged" in log  # no empty headings for unused phases
    assert log.index("no phase key") < log.index("empty phase") < log.index("none phase")  # journal order


def test_report_decision_log_ignores_non_note_entries(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    run = Run.load(tmp_path)
    run.journal.append("trial", module="01_f.py", note="trial note text", mean_delta=0.0, se=0.0,
                       delta_natural=0.0, accepted=False, flags=[], new_columns=[], reason="r", error=None)
    text = build_report(run, tmp_path)
    assert "No notes recorded." in _decision_log_of(text)
    assert "trial note text" not in _decision_log_of(text)
    run.journal.append("note", text="real note", phase="evaluate")
    log = _decision_log_of(build_report(run, tmp_path))
    assert "### evaluate" in log and "real note" in log and "trial note text" not in log


def _fit_entry(run, n_rows=2400):
    run.journal.append("fit", score_val=0.8, score_val_natural=0.8, leaderboard_top5=[{"model": "M", "score_val": 0.8}],
                       time_used=100.0, time_limit=200.0, presets="best_quality", modules=[], notes=[], n_rows=n_rows)


def test_report_final_model_states_whether_the_model_was_refit(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    run = Run.load(tmp_path)
    _fit_entry(run)
    text = build_report(run, tmp_path)
    assert "The delivered model was not refit; it is trained on the train portion (2400 rows)." in text
    run.journal.append("refit", n_rows=3000, time_used=321.4, time_limit=600.0, presets="best_quality",
                       modules=[], notes=[], leaderboard_top5=[])
    run.state.refit, run.state.refit_rows = True, 3000
    run.save()
    text = build_report(run, tmp_path)
    assert ("The delivered model was refit on all 3000 labelled rows (train + holdout) in 321 s; the holdout "
            "estimate below describes the same recipe trained on 2400 rows.") in text
    assert "was not refit" not in text


def test_report_refit_sentence_follows_state_when_the_journal_entry_is_missing(tmp_path, make_binary, monkeypatch):
    monkeypatch.chdir(tmp_path); make_binary().to_csv("d.csv", index=False)
    main(["init", "d.csv", "--target", "target", "--goal", "g"])
    run = Run.load(tmp_path)
    _fit_entry(run)
    run.state.refit, run.state.refit_rows = True, 3000
    run.save()
    text = build_report(run, tmp_path)
    assert "The delivered model was refit on all 3000 labelled rows (train + holdout)" in text
    assert "journal entry is missing" in text
    assert "not refit" not in text and "same recipe" not in text


@pytest.mark.slow
def test_full_pipeline(tmp_path, make_regression, monkeypatch):
    monkeypatch.chdir(tmp_path); make_regression(n=1500).to_csv("d.csv", index=False)
    (tmp_path / "f.py").write_text(textwrap.dedent("""
        import pandas as pd
        class Features:
            def fit(self, X, y): return self
            def transform(self, X): return pd.DataFrame({"x1x2": X["x1"] * X["x2"]}, index=X.index)
    """))
    (tmp_path / "NARRATIVE.md").write_text("Interaction x1*x2 drives the target.")
    for cmd in (["init", "d.csv", "--target", "target", "--goal", "g", "--budget", "10"],
                ["leakage"], ["baseline"], ["try", "f.py", "--note", "interaction"],
                ["fit", "--time-limit", "60", "--presets", "medium_quality"],
                ["evaluate"], ["refit", "--time-limit", "30", "--presets", "medium_quality"], ["report"]):
        assert main(cmd) == 0, cmd
    text = (tmp_path / "REPORT.md").read_text()
    assert "x1x2" in text and "Interaction x1*x2" in text and "1 accepted" in text
    assert "The delivered model was refit on all 1500 labelled rows (train + holdout)" in text
