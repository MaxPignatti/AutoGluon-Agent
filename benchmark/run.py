"""Run one system on one benchmark dataset and write predictions for both test variants.

    python benchmark/run.py --dataset NAME --system {autogluon,aga} [--minutes 60]
                            [--cpus 0-5] [--memory-gb 14]

``--cpus`` pins the run (and every process it starts: the agent, ``aga fit``, AutoGluon workers)
to those logical CPUs and sets ``AG_CPU_COUNT``/``OMP_NUM_THREADS`` to their number;
``--memory-gb`` sets ``AG_MEMORY_LIMIT_IN_GB``. Use them to run two lanes side by side on disjoint
CPUs, both systems capped alike.

Output: ``benchmark/results/raw/<name>/<system>/`` with ``preds_naive.csv``,
``preds_production.csv`` and ``meta.json``. Both systems write predictions in the format
``aga predict`` writes: a single ``prediction`` column (the value for regression,
P(positive class) for binary), so score.py has one reader.

``meta.json``: ``status`` (ok/failed), ``wall_seconds``, ``cv_score`` (the system's own
validation score in the metric's natural units), and for aga ``usage`` and ``total_cost_usd``
from claude's JSON output, ``refit`` (whether the delivered model was refit on all labelled rows)
and ``transcript`` (``agent_transcript.jsonl``, the agent's Claude Code transcript copied into the
output folder, or null when it was not found).

The AutoGluon system trains on the full ``train.csv``. The aga system runs the headless agent in
a fresh copy of the workspace placed in the system temp directory, outside this repository, so
the test files are not reachable by relative paths from its working directory. This is not a
sandbox: the agent's shell and read tools could still open ``benchmark/.hidden`` by absolute path
(the plugin directory points into this repository). Check the transcript if a result looks suspicious.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
VARIANTS = ("naive", "production")
AG_METRICS = {
    "roc_auc": "roc_auc",
    "rmse": "root_mean_squared_error",
    "mae": "mean_absolute_error",
    "log_loss": "log_loss",
}
HIGHER_IS_BETTER = {"roc_auc"}
AGENT_TOOLS = "Bash,PowerShell,Read,Write,Edit,Glob,Grep"  # Claude Code on Windows may offer PowerShell, not Bash
AG_OVERHEAD_SECONDS = 120  # left for loading, predicting and writing around the fit
MIN_AG_SECONDS = 30
AGA_TIMEOUT_SLACK_SECONDS = 300


def load_entry(name: str) -> dict:
    entries = yaml.safe_load((HERE / "registry.yaml").read_text(encoding="utf-8"))
    for entry in entries:
        if entry["name"] == name:
            return entry
    raise SystemExit(f"no dataset named {name!r} in registry.yaml")


def natural(metric: str, ag_sign_value: float) -> float:
    """AutoGluon reports every metric higher-is-better; error metrics come back negated."""
    return ag_sign_value if metric in HIGHER_IS_BETTER else -ag_sign_value


# ---- autogluon ------------------------------------------------------------------------------
def run_autogluon(entry: dict, minutes: float, out: Path) -> dict:
    from autogluon.tabular import TabularPredictor

    started = time.time()
    train = pd.read_csv(HERE / "workspace" / entry["name"] / "train.csv", low_memory=False)
    positive = entry.get("positive_class") if entry["task"] == "binary" else None
    time_limit = max(MIN_AG_SECONDS, minutes * 60 - AG_OVERHEAD_SECONDS)
    model_dir = tempfile.mkdtemp(prefix="ag-bench-")  # models never go into benchmark/workspace
    try:
        predictor = TabularPredictor(
            label=entry["target"],
            problem_type=entry["task"],
            eval_metric=AG_METRICS[entry["metric"]],
            path=model_dir,
            positive_class=positive,
        ).fit(train, presets="best_quality", time_limit=time_limit)
        score_val = float(predictor.leaderboard(silent=True)["score_val"].iloc[0])
        for variant in VARIANTS:
            features = pd.read_csv(HERE / ".hidden" / entry["name"] / f"test_features_{variant}.csv", low_memory=False)
            if entry["task"] == "regression":
                pred = predictor.predict(features).to_numpy()
            else:
                proba = predictor.predict_proba(features)
                pred = proba[{str(c): c for c in proba.columns}[str(positive)]].to_numpy()
            pd.DataFrame({"prediction": pred}).to_csv(out / f"preds_{variant}.csv", index=False)
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)
    return {
        "status": "ok",
        "wall_seconds": time.time() - started,
        "cv_score": natural(entry["metric"], score_val),
        "cv_score_autogluon_sign": score_val,
        "time_limit": time_limit,
    }


# ---- aga ------------------------------------------------------------------------------------
def build_prompt(entry: dict, minutes: float) -> str:
    return (
        f"Use the autogluon-agent skill. Data: train.csv. Goal: {entry['goal']}. Target: {entry['target']}. "
        f"Metric: {entry['metric']}. Budget: {minutes:g} minutes. "
        "Work fully autonomously; do not ask questions."
    )


def build_aga_command(claude: str, entry: dict, minutes: float) -> list[str]:
    return [
        claude, "-p", build_prompt(entry, minutes),
        "--plugin-dir", str(REPO_ROOT),
        "--output-format", "json",
        "--allowedTools", AGENT_TOOLS,
    ]


def aga_timeout(minutes: float) -> float:
    return minutes * 60 + AGA_TIMEOUT_SLACK_SECONDS


def child_env(base: dict | None = None) -> dict:
    """Environment for child processes: the scripts dir of this interpreter (where ``aga`` is
    installed) comes first on PATH."""
    env = dict(os.environ if base is None else base)
    scripts = os.path.dirname(sys.executable)
    env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
    return env


def _fit_score(run_dir: Path, metric: str) -> float | None:
    """``score_val`` of the last ``fit`` entry in the run's journal, in natural units."""
    path = run_dir / "journal.jsonl"
    if not path.is_file():
        return None
    fits = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fits = [e for e in fits if e.get("kind") == "fit"]
    if not fits:
        return None
    last = fits[-1]
    if last.get("score_val_natural") is not None:
        return float(last["score_val_natural"])
    return natural(metric, float(last["score_val"]))


def popen_group_kwargs() -> dict:
    """Popen flags that put the child in its own process group / session, so the whole tree can
    be killed."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and every process it started (the agent's Bash children, ``aga fit`` and its
    AutoGluon workers). ``Popen.kill`` alone only ends the direct child."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_with_timeout(argv, cwd, env, timeout: float, stdout_path: Path, stderr_path: Path) -> tuple[int | None, bool]:
    """Run ``argv`` with stdout and stderr going to files (not pipes, so a surviving grandchild can
    never block us). Returns ``(returncode, timed_out)``; on timeout the whole process tree is killed
    and whatever was written so far stays in the files."""
    with open(stdout_path, "wb") as out_fh, open(stderr_path, "wb") as err_fh:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out_fh, stderr=err_fh,
            **popen_group_kwargs(),
        )
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            return None, True


def positive_class_mismatch(entry: dict, run_dir: Path) -> str | None:
    """Reason string when the agent's run predicts a different positive class than the registry
    scores against (``aga predict`` writes P(problem.yaml positive_class)); None when they agree."""
    if entry["task"] != "binary":
        return None
    problem = yaml.safe_load((Path(run_dir) / "problem.yaml").read_text(encoding="utf-8"))
    if problem.get("task") != "binary" or str(problem.get("positive_class")) != str(entry["positive_class"]):
        return (
            f"positive class mismatch: aga run has task={problem.get('task')!r} "
            f"positive_class={problem.get('positive_class')!r}, registry expects {entry['positive_class']!r}"
        )
    return None


def _read_claude_output(path: Path, meta: dict) -> None:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    if not text.strip():
        return
    try:
        result = json.loads(text)
    except ValueError:
        meta.setdefault("reason", "claude output was not JSON")
        return
    meta["usage"] = result.get("usage")
    meta["total_cost_usd"] = result.get("total_cost_usd")
    if result.get("is_error"):  # e.g. "Failed to authenticate: OAuth session expired ..."
        meta["claude_error"] = str(result.get("result") or result.get("terminal_reason") or "unknown error")


def project_dir_name(workdir) -> str:
    """Claude Code's folder name under ``~/.claude/projects/`` for a working directory: the absolute
    path with every character other than ASCII letters, digits and ``-`` replaced by ``-``
    (``C:\\Users\\me\\AppData\\Local\\Temp\\aga-bench-x_1\\work`` -> ``C--Users-me-AppData-Local-Temp-aga-bench-x-1-work``)."""
    return re.sub(r"[^A-Za-z0-9-]", "-", str(workdir))


def find_transcript(workdir) -> Path | None:
    """The newest Claude Code transcript (``*.jsonl``) recorded for ``workdir``, or None."""
    folder = Path.home() / ".claude" / "projects" / project_dir_name(workdir)
    if not folder.is_dir():
        return None
    transcripts = [p for p in folder.glob("*.jsonl") if p.is_file()]
    return max(transcripts, key=lambda p: p.stat().st_mtime) if transcripts else None


def copy_transcript(workdir, out: Path, meta: dict) -> None:
    """Best effort: copy the agent's transcript to ``out/agent_transcript.jsonl``; record it in ``meta``."""
    meta["transcript"] = None
    try:
        src = find_transcript(workdir)
        if src is not None:
            shutil.copyfile(src, out / "agent_transcript.jsonl")
            meta["transcript"] = "agent_transcript.jsonl"
    except OSError as exc:
        meta["transcript_error"] = f"{type(exc).__name__}: {exc}"


def run_refit(run_dir: Path | None) -> bool:
    """Whether the run's delivered model was refit on all labelled rows (``state.json`` ``refit``);
    False when there is no run or its state cannot be read."""
    if run_dir is None:
        return False
    try:
        return bool(json.loads((Path(run_dir) / "state.json").read_text(encoding="utf-8")).get("refit"))
    except (AttributeError, OSError, ValueError):  # missing, unreadable or not a JSON object
        return False


def run_aga(entry: dict, minutes: float, out: Path) -> dict:
    claude = shutil.which("claude")
    if claude is None:
        raise SystemExit("`claude` not found on PATH")
    env = child_env()
    aga = shutil.which("aga", path=env["PATH"]) or "aga"
    # resolve(): on Windows the temp dir can come back as an 8.3 short path (USERNA~1), which
    # Claude Code's Write tool refuses as suspicious; the long form is accepted.
    scratch = Path(tempfile.mkdtemp(prefix="aga-bench-")).resolve()
    work = scratch / "work"
    shutil.copytree(HERE / "workspace" / entry["name"], work)
    meta: dict = {"status": "failed", "refit": False}
    started = time.time()
    try:
        code, timed_out = run_with_timeout(
            build_aga_command(claude, entry, minutes), work, env, aga_timeout(minutes),
            out / "claude_output.json", out / "claude_stderr.txt",
        )
        meta["claude_exit_code"] = code
        _read_claude_output(out / "claude_output.json", meta)  # also on timeout: keep any partial usage
        copy_transcript(work, out, meta)
        current = work / ".aga" / "CURRENT"
        run_id = current.read_text(encoding="utf-8").strip() if current.is_file() else None
        run_dir = work / ".aga" / run_id if run_id else None
        meta["refit"] = run_refit(run_dir)  # also on timeout: the state may already record a refit
        if timed_out:
            meta["reason"] = "agent timed out"  # failed run: no predictions, even if a model exists
            return meta

        fitted = False
        if run_dir is not None and (run_dir / "state.json").is_file():
            fitted = bool(json.loads((run_dir / "state.json").read_text(encoding="utf-8")).get("fitted"))
        if run_dir is not None and run_dir.is_dir():
            keep = out / "aga_run"
            keep.mkdir(exist_ok=True)
            for name in ("journal.jsonl", "problem.yaml", "REPORT.md"):
                if (run_dir / name).is_file():
                    shutil.copyfile(run_dir / name, keep / name)
        if not fitted:
            meta["reason"] = f"claude error: {meta['claude_error']}" if meta.get("claude_error") else "no fitted model"
            return meta
        mismatch = positive_class_mismatch(entry, run_dir)
        if mismatch:
            meta["reason"] = mismatch
            return meta
        meta["cv_score"] = _fit_score(run_dir, entry["metric"])
        meta["run_id"] = run_id
        ok = True
        for variant in VARIANTS:
            features = HERE / ".hidden" / entry["name"] / f"test_features_{variant}.csv"
            cmd = [aga, "predict", str(features), "--out", str(out / f"preds_{variant}.csv"), "--run", run_id]
            p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if p.returncode != 0:
                ok = False
                meta["reason"] = f"aga predict ({variant}) failed: {p.stderr.strip()[-500:]}"
                break
        meta["status"] = "ok" if ok else "failed"
    finally:
        meta["wall_seconds"] = time.time() - started
        shutil.rmtree(scratch, ignore_errors=True)
    return meta


# ---- resource caps --------------------------------------------------------------------------
def parse_cpus(spec: str) -> list[int]:
    """``"0-5"`` or ``"0,2,4-5"`` -> sorted, distinct logical CPU ids."""
    cpus: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    if not cpus:
        raise ValueError(f"no CPUs in {spec!r}")
    return sorted(cpus)


def limit_resources(cpus: list[int] | None, memory_gb: float | None, env: dict | None = None) -> dict:
    """Pin this process to ``cpus`` (children inherit the affinity) and set the variables AutoGluon
    and OpenMP read in ``env`` (default ``os.environ``, which ``child_env`` passes on to the agent).
    Returns what was applied, for ``meta.json``."""
    env = os.environ if env is None else env
    applied: dict = {}
    if cpus:
        import psutil

        psutil.Process().cpu_affinity(cpus)
        env["AG_CPU_COUNT"] = env["OMP_NUM_THREADS"] = str(len(cpus))
        applied["cpus"] = cpus
    if memory_gb:
        env["AG_MEMORY_LIMIT_IN_GB"] = f"{memory_gb:g}"
        applied["memory_gb"] = memory_gb
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--system", required=True, choices=("autogluon", "aga"))
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--cpus", type=parse_cpus, default=None, help="logical CPUs to pin to, e.g. 0-5")
    parser.add_argument("--memory-gb", type=float, default=None, help="AutoGluon memory limit")
    args = parser.parse_args(argv)
    resources = limit_resources(args.cpus, args.memory_gb)  # before AutoGluon is imported
    entry = load_entry(args.dataset)
    out = HERE / "results" / "raw" / entry["name"] / args.system
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)  # rmtree can leave the dir behind under OneDrive
    runner = run_autogluon if args.system == "autogluon" else run_aga
    try:
        meta = runner(entry, args.minutes, out)
    except Exception as exc:  # a crashed run is recorded, not lost
        meta = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        if args.system == "aga":
            meta["refit"] = False
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        raise
    meta.update(dataset=entry["name"], system=args.system, minutes=args.minutes, **resources)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "usage"}, default=str))
    return 0 if meta["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
