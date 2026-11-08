import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from score import relative_error_reduction  # noqa: E402


def test_auc_error_reduction():
    assert relative_error_reduction("roc_auc", 0.80, 0.85) == pytest.approx(0.25)


def test_rmse_error_reduction():
    assert relative_error_reduction("rmse", 10.0, 9.0) == pytest.approx(0.10)


def test_worse_is_negative():
    assert relative_error_reduction("rmse", 10.0, 11.0) < 0


def test_score_predictions_regression_and_binary_use_the_registry_positive_class():
    from score import score_predictions

    reg = {"task": "regression", "metric": "rmse", "target": "y"}
    truth = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    assert score_predictions(reg, truth, pd.DataFrame({"prediction": [1.0, 2.0, 5.0]})) == pytest.approx((4 / 3) ** 0.5)

    binary = {"task": "binary", "metric": "roc_auc", "target": "y", "positive_class": "yes"}
    truth = pd.DataFrame({"y": ["no", "yes", "no", "yes"]})
    good = pd.DataFrame({"prediction": [0.1, 0.9, 0.2, 0.8]})  # P(yes)
    assert score_predictions(binary, truth, good) == pytest.approx(1.0)
    assert score_predictions(binary, truth, 1 - good) == pytest.approx(0.0)


def test_main_writes_failed_row_when_predictions_are_missing(tmp_path):
    import score

    root = tmp_path
    hidden = root / "hidden" / "toy"
    hidden.mkdir(parents=True)
    pd.DataFrame({"y": [1.0, 2.0, 3.0]}).to_csv(hidden / "test.csv", index=False)
    raw = root / "results" / "raw" / "toy"
    (raw / "autogluon").mkdir(parents=True)
    pd.DataFrame({"prediction": [1.0, 2.0, 3.0]}).to_csv(raw / "autogluon" / "preds_production.csv", index=False)
    pd.DataFrame({"prediction": [1.0, 2.0, 4.0]}).to_csv(raw / "autogluon" / "preds_naive.csv", index=False)
    (raw / "autogluon" / "meta.json").write_text('{"wall_seconds": 120, "cv_score": 0.5}', encoding="utf-8")
    (raw / "aga").mkdir()
    (raw / "aga" / "meta.json").write_text('{"status": "failed", "wall_seconds": 60}', encoding="utf-8")
    entries = [{"name": "toy", "task": "regression", "metric": "rmse", "target": "y", "goal": "g"}]

    rows = score.score_all(entries, root / "results" / "raw", root / "hidden")
    by = {(r["system"], r["variant"]): r for r in rows}
    assert by[("autogluon", "production")]["score"] == pytest.approx(0.0)
    assert by[("autogluon", "production")]["cv_test_gap"] == pytest.approx(-0.5)
    assert by[("autogluon", "naive")]["status"] == "ok"
    assert by[("aga", "production")]["status"] == "failed"
    assert by[("aga", "production")]["score"] is None
    assert by[("aga", "production")]["abs_diff"] is None  # no paired score


def _paired_rows(tmp_path):
    import score

    hidden = tmp_path / "hidden" / "toy"
    hidden.mkdir(parents=True)
    pd.DataFrame({"y": [1.0, 2.0, 3.0]}).to_csv(hidden / "test.csv", index=False)
    for system, preds in (("autogluon", [1.0, 2.0, 5.0]), ("aga", [1.0, 2.0, 4.0])):
        folder = tmp_path / "raw" / "toy" / system
        folder.mkdir(parents=True)
        for variant in score.VARIANTS:
            pd.DataFrame({"prediction": preds}).to_csv(folder / f"preds_{variant}.csv", index=False)
        (folder / "meta.json").write_text('{"status": "ok", "wall_seconds": 60}', encoding="utf-8")
    entries = [{"name": "toy", "task": "regression", "metric": "rmse", "target": "y", "goal": "g"}]
    return entries, score.score_all(entries, tmp_path / "raw", tmp_path / "hidden")


def test_abs_diff_is_aga_minus_autogluon_in_metric_units(tmp_path):
    import score

    entries, rows = _paired_rows(tmp_path)
    by = {(r["system"], r["variant"]): r for r in rows}
    expected = (1 / 3) ** 0.5 - (4 / 3) ** 0.5  # rmse(aga) - rmse(autogluon)
    assert by[("aga", "production")]["abs_diff"] == pytest.approx(expected)
    assert by[("autogluon", "production")]["abs_diff"] is None
    assert "abs_diff" in score.COLUMNS
    table = score.markdown_table(entries, rows, "production")
    assert "| Absolute difference |" in table and "Relative error reduction" in table
    assert f"| {expected:+.4f} |" in table
    score.write_results_md(tmp_path / "RESULTS.md", entries, rows)
    assert "Absolute difference" in (tmp_path / "RESULTS.md").read_text(encoding="utf-8")


def test_replace_readme_block_only_touches_text_between_markers():
    from score import replace_block

    text = "before\n<!-- BENCHMARK:START -->\nold\n<!-- BENCHMARK:END -->\nafter\n"
    out = replace_block(text, "| new |")
    assert out == "before\n<!-- BENCHMARK:START -->\n| new |\n<!-- BENCHMARK:END -->\nafter\n"


def test_aga_command_prompt_flags_and_timeout():
    import run

    entry = {"name": "toy", "goal": "Predict toy.", "target": "t", "metric": "rmse"}
    argv = run.build_aga_command("C:/bin/claude.exe", entry, 15)
    assert argv[:2] == ["C:/bin/claude.exe", "-p"]
    assert argv[2] == (
        "Use the autogluon-agent skill. Data: train.csv. Goal: Predict toy.. Target: t. Metric: rmse. "
        "Budget: 15 minutes. Work fully autonomously; do not ask questions."
    )
    assert argv[argv.index("--plugin-dir") + 1] == str(run.REPO_ROOT)
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--allowedTools") + 1] == "Bash,PowerShell,Read,Write,Edit,Glob,Grep"
    assert run.aga_timeout(15) == 15 * 60 + 300


def test_child_env_prepends_interpreter_scripts_dir():
    import run

    env = run.child_env({"PATH": "/usr/bin", "OTHER": "x"})
    scripts = os.path.dirname(sys.executable)
    assert env["PATH"].split(os.pathsep)[0] == scripts
    assert env["PATH"].endswith("/usr/bin")
    assert env["OTHER"] == "x"


def test_parse_cpus_ranges_and_lists():
    import run

    assert run.parse_cpus("0-5") == [0, 1, 2, 3, 4, 5]
    assert run.parse_cpus("6,8-9,6") == [6, 8, 9]
    with pytest.raises(ValueError):
        run.parse_cpus("a-b")


def test_limit_resources_pins_and_sets_env_inherited_by_children():
    import subprocess

    import psutil
    import run

    me = psutil.Process()
    before = me.cpu_affinity()
    env = {}
    try:
        applied = run.limit_resources([before[0]], 2.5, env)
        assert applied == {"cpus": [before[0]], "memory_gb": 2.5}
        assert env == {"AG_CPU_COUNT": "1", "OMP_NUM_THREADS": "1", "AG_MEMORY_LIMIT_IN_GB": "2.5"}
        child = subprocess.run(
            [sys.executable, "-c", "import psutil; print(psutil.Process().cpu_affinity())"],
            capture_output=True, text=True, check=True,
        )
        assert child.stdout.strip() == str([before[0]])  # the agent and aga fit inherit the pin
    finally:
        me.cpu_affinity(before)
    assert run.limit_resources(None, None, env := {}) == {} and env == {}


def test_read_claude_output_records_the_agent_error(tmp_path):
    import json

    import run

    out = tmp_path / "claude_output.json"
    out.write_text(json.dumps({"is_error": True, "total_cost_usd": 0, "usage": {},
                               "result": "Failed to authenticate: OAuth session expired and could not be refreshed"}))
    meta = {}
    run._read_claude_output(out, meta)
    assert meta["claude_error"].startswith("Failed to authenticate")
    out.write_text(json.dumps({"is_error": False, "total_cost_usd": 1.2, "usage": {}, "result": "done"}))
    meta = {}
    run._read_claude_output(out, meta)
    assert "claude_error" not in meta and meta["total_cost_usd"] == 1.2


def test_parse_cpus_edge_cases():
    import run

    assert run.parse_cpus("3") == [3]
    assert run.parse_cpus(" 0 - 1 , 4 ") == [0, 1, 4]
    assert run.parse_cpus("1-") == [1]  # an open upper bound falls back to the lower one
    for bad in ("5-3", "", "-2"):  # a reversed range is empty; a blank lower bound is not a number
        with pytest.raises(ValueError):
            run.parse_cpus(bad)


def test_limit_resources_memory_only_leaves_affinity_alone(monkeypatch):
    import psutil
    import run

    def boom(*_):
        raise AssertionError("cpu_affinity must not be set without --cpus")

    monkeypatch.setattr(psutil.Process, "cpu_affinity", boom)
    env = {}
    assert run.limit_resources(None, 14.0, env) == {"memory_gb": 14.0}
    assert env == {"AG_MEMORY_LIMIT_IN_GB": "14"}  # 14.0 is written as "14"
    assert run.limit_resources([], None, env := {}) == {} and env == {}  # an empty list pins nothing


def test_limit_resources_defaults_to_os_environ(monkeypatch):
    import run

    monkeypatch.delenv("AG_MEMORY_LIMIT_IN_GB", raising=False)
    run.limit_resources(None, 0.5)
    assert os.environ["AG_MEMORY_LIMIT_IN_GB"] == "0.5"  # monkeypatch restores it afterwards


def _fake_main_env(monkeypatch, tmp_path, runner):
    import run

    calls = []
    monkeypatch.setattr(run, "HERE", tmp_path)
    monkeypatch.setattr(run, "load_entry", lambda name: {"name": name})
    monkeypatch.setattr(run, "limit_resources",
                        lambda cpus, mem: calls.append(("limit", cpus, mem))
                        or {k: v for k, v in (("cpus", cpus), ("memory_gb", mem)) if v})

    def fake_runner(entry, minutes, out):
        calls.append(("runner", entry["name"], minutes))
        return runner()

    monkeypatch.setattr(run, "run_aga", fake_runner)
    return run, calls


def test_main_applies_caps_before_the_run_and_records_them(tmp_path, monkeypatch):
    import json

    run, calls = _fake_main_env(monkeypatch, tmp_path, lambda: {"status": "ok"})
    assert run.main(["--dataset", "toy", "--system", "aga", "--minutes", "5",
                     "--cpus", "0-1", "--memory-gb", "14"]) == 0
    assert calls == [("limit", [0, 1], 14.0), ("runner", "toy", 5.0)]  # caps first
    meta = json.loads((tmp_path / "results" / "raw" / "toy" / "aga" / "meta.json").read_text(encoding="utf-8"))
    assert meta["cpus"] == [0, 1] and meta["memory_gb"] == 14.0
    assert meta["dataset"] == "toy" and meta["system"] == "aga" and meta["minutes"] == 5.0


def test_main_without_caps_records_none(tmp_path, monkeypatch):
    import json

    run, calls = _fake_main_env(monkeypatch, tmp_path, lambda: {"status": "failed"})
    assert run.main(["--dataset", "toy", "--system", "aga"]) == 1
    assert calls[0] == ("limit", None, None)
    meta = json.loads((tmp_path / "results" / "raw" / "toy" / "aga" / "meta.json").read_text(encoding="utf-8"))
    assert "cpus" not in meta and "memory_gb" not in meta


def test_main_rejects_a_bad_cpu_spec(tmp_path, monkeypatch, capsys):
    run, calls = _fake_main_env(monkeypatch, tmp_path, lambda: {"status": "ok"})
    with pytest.raises(SystemExit) as exc:
        run.main(["--dataset", "toy", "--system", "aga", "--cpus", "5-3"])
    assert exc.value.code == 2 and calls == []  # nothing pinned, nothing run
    assert "--cpus" in capsys.readouterr().err


def test_popen_group_kwargs_start_a_new_group_or_session():
    import subprocess

    import run

    kwargs = run.popen_group_kwargs()
    if os.name == "nt":
        assert kwargs == {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        assert kwargs == {"start_new_session": True}


def test_timeout_kills_the_whole_process_tree_and_keeps_partial_output(tmp_path):
    import time

    import psutil
    import run

    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(g.pid))\n"
        "print('partial output', flush=True)\n"
        "time.sleep(120)\n"
    )
    started = time.time()
    code, timed_out = run.run_with_timeout(
        [sys.executable, "-c", script], tmp_path, os.environ.copy(), 5, tmp_path / "out.txt", tmp_path / "err.txt"
    )
    assert timed_out and code is None
    assert time.time() - started < 40
    assert "partial output" in (tmp_path / "out.txt").read_text()
    grandchild = int(pid_file.read_text())
    deadline = time.time() + 10
    while psutil.pid_exists(grandchild) and time.time() < deadline:
        time.sleep(0.2)
    assert not psutil.pid_exists(grandchild)


def test_run_with_timeout_returns_exit_code_and_writes_stderr(tmp_path):
    import run

    code, timed_out = run.run_with_timeout(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
        tmp_path, os.environ.copy(), 60, tmp_path / "out.txt", tmp_path / "err.txt",
    )
    assert (code, timed_out) == (3, False)
    assert (tmp_path / "err.txt").read_text() == "boom"


@pytest.mark.parametrize(
    "registry_positive, problem_yaml, mismatch",
    [
        ("yes", "task: binary\npositive_class: 'yes'\n", False),
        (">50K", "task: binary\npositive_class: '>50K'\n", False),
        (1, "task: binary\npositive_class: 1\n", False),
        ("yes", "task: binary\npositive_class: 'no'\n", True),  # target recoded: P(negative)
        (1, "task: binary\npositive_class: 0\n", True),
        ("yes", "task: multiclass\npositive_class: null\n", True),
    ],
)
def test_positive_class_mismatch(tmp_path, registry_positive, problem_yaml, mismatch):
    import run

    (tmp_path / "problem.yaml").write_text(problem_yaml, encoding="utf-8")
    entry = {"task": "binary", "positive_class": registry_positive}
    reason = run.positive_class_mismatch(entry, tmp_path)
    assert (reason is not None) is mismatch
    if mismatch:
        assert "positive class mismatch" in reason


def test_positive_class_check_skips_regression(tmp_path):
    import run

    (tmp_path / "problem.yaml").write_text("task: regression\npositive_class: null\n", encoding="utf-8")
    assert run.positive_class_mismatch({"task": "regression"}, tmp_path) is None

def test_project_dir_name_matches_claude_code_encoding():
    import run

    assert run.project_dir_name(r"C:\Users\username\AppData\Local\Temp\aga-bench-xt49y765-work") == (
        "C--Users-username-AppData-Local-Temp-aga-bench-xt49y765-work"
    )
    assert run.project_dir_name(r"C:\Users\username\OneDrive - Example Ltd\Documenti\AutoGluon-Agent") == (
        "C--Users-username-OneDrive---Example-Ltd-Documenti-AutoGluon-Agent"
    )
    assert run.project_dir_name(r"C:\Temp\aga-bench-s0eo5p_t\work") == "C--Temp-aga-bench-s0eo5p-t-work"
    assert run.project_dir_name("/tmp/aga-bench-x/work") == "-tmp-aga-bench-x-work"


def test_find_transcript_returns_newest_jsonl_or_none(tmp_path, monkeypatch):
    import time as _time

    import run

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    work = tmp_path / "scratch dir" / "work"
    assert run.find_transcript(work) is None  # no projects dir at all
    project = home / ".claude" / "projects" / run.project_dir_name(work)
    project.mkdir(parents=True)
    assert run.find_transcript(work) is None  # dir exists, no transcript
    old, new = project / "old.jsonl", project / "new.jsonl"
    old.write_text("{}\n")
    new.write_text("{}\n")
    now = _time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))
    (project / "memory").mkdir()
    assert run.find_transcript(work) == new


def test_copy_transcript_records_meta(tmp_path, monkeypatch):
    import run

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    work, out = tmp_path / "work", tmp_path / "out"
    out.mkdir()
    meta = {}
    run.copy_transcript(work, out, meta)
    assert meta["transcript"] is None and not (out / "agent_transcript.jsonl").exists()
    project = home / ".claude" / "projects" / run.project_dir_name(work)
    project.mkdir(parents=True)
    (project / "abc.jsonl").write_text('{"type": "x"}\n', encoding="utf-8")
    run.copy_transcript(work, out, meta)
    assert meta["transcript"] == "agent_transcript.jsonl"
    assert (out / "agent_transcript.jsonl").read_text(encoding="utf-8") == '{"type": "x"}\n'


def test_run_refit_flag_read_from_state(tmp_path):
    import json

    import run

    assert run.run_refit(None) is False
    (tmp_path / "state.json").write_text(json.dumps({"fitted": True}), encoding="utf-8")
    assert run.run_refit(tmp_path) is False  # older state without the key
    (tmp_path / "state.json").write_text(json.dumps({"fitted": True, "refit": True}), encoding="utf-8")
    assert run.run_refit(tmp_path) is True



def test_readme_table_matches_score_table_columns():
    import score

    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    block = text.split(score.README_START, 1)[1].split(score.README_END, 1)[0].strip()
    header = score.markdown_table([], [], "production").splitlines()[:2]
    assert block.splitlines()[:2] == header  # filled or placeholder, the columns are score.py's


def test_run_refit_false_when_state_unreadable(tmp_path):
    import run

    assert run.run_refit(tmp_path) is False  # no state.json
    (tmp_path / "state.json").write_text("{not json", encoding="utf-8")
    assert run.run_refit(tmp_path) is False
    (tmp_path / "state.json").write_text("[1, 2]", encoding="utf-8")
    assert run.run_refit(tmp_path) is False


def test_run_aga_records_refit_on_timeout(tmp_path, monkeypatch):
    import json

    import run

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(run.shutil, "which", lambda name, path=None: f"/bin/{name}")
    monkeypatch.setattr(run.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "scratch"))

    def fake_copytree(src, dst, *args, **kwargs):  # the agent's workspace, with a run that was refit
        root = Path(dst) / ".aga" / "20260101-000000"
        root.mkdir(parents=True)
        (Path(dst) / ".aga" / "CURRENT").write_text("20260101-000000", encoding="utf-8")
        (root / "state.json").write_text(json.dumps({"fitted": True, "refit": True}), encoding="utf-8")

    monkeypatch.setattr(run.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(run, "run_with_timeout", lambda *args, **kwargs: (None, True))
    out = tmp_path / "out"
    out.mkdir()
    meta = run.run_aga({"name": "toy", "goal": "g", "target": "t", "metric": "rmse"}, 1, out)
    assert meta["status"] == "failed" and meta["reason"] == "agent timed out"
    assert meta["refit"] is True and meta["transcript"] is None
