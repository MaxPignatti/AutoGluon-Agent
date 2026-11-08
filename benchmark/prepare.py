"""Build each benchmark dataset and split it into an agent-visible workspace and a hidden test set.

Layout written under ``root`` (default: this directory):

    workspace/<name>/train.csv, task.md        what the agent / AutoGluon may see
    .hidden/<name>/test.csv                    test rows with the target (for scoring)
    .hidden/<name>/test_features_naive.csv     test rows without the target
    .hidden/<name>/test_features_production.csv
                                               same, with leak_columns blanked, as
                                               they would be at prediction time

Usage: python benchmark/prepare.py [--only NAME]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split

HERE = Path(__file__).resolve().parent
SEED = 42
TEST_SIZE = 0.2


def _split(df: pd.DataFrame, entry: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    kind = entry["test_split"]["kind"]
    if kind == "random":
        stratify = df[entry["target"]] if entry["task"] == "binary" else None
        train, test = train_test_split(
            df, test_size=TEST_SIZE, random_state=SEED, stratify=stratify
        )
        return train.sort_index(), test.sort_index()
    column = entry["test_split"]["column"]
    if kind == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        train_idx, test_idx = next(splitter.split(df, groups=df[column]))
        return df.iloc[train_idx], df.iloc[test_idx]
    if kind == "time":
        order = df.sort_values(column, kind="stable").index
        n_test = int(round(len(df) * TEST_SIZE))
        return df.loc[order[:-n_test]], df.loc[order[-n_test:]]
    raise ValueError(f"unknown test_split kind: {kind!r}")


def prepare(entry: dict, root: Path, df: pd.DataFrame | None = None) -> None:
    """Split ``df`` (or build it from the registry entry) and write the files."""
    root = Path(root)
    if df is None:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import datasets  # lazy: benchmark/datasets.py, only needed when building for real

        df = getattr(datasets, entry["builder"])(HERE / ".cache")
    df = df.reset_index(drop=True)
    name, target = entry["name"], entry["target"]
    train, test = _split(df, entry)

    workspace, hidden = root / "workspace" / name, root / ".hidden" / name
    for folder in (workspace, hidden):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)  # rmtree can leave the dir behind under OneDrive

    train.to_csv(workspace / "train.csv", index=False)
    (workspace / "task.md").write_text(
        f"Goal: {entry['goal']}\nTarget: {target}\nMetric: {entry['metric']}\n",
        encoding="utf-8",
    )

    test.to_csv(hidden / "test.csv", index=False)
    features = test.drop(columns=[target])
    features.to_csv(hidden / "test_features_naive.csv", index=False)
    production = features.copy()
    for column in entry.get("leak_columns") or []:
        production[column] = np.nan
    production.to_csv(hidden / "test_features_production.csv", index=False)


def load_registry() -> list[dict]:
    return yaml.safe_load((HERE / "registry.yaml").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", metavar="NAME", help="prepare a single dataset")
    args = parser.parse_args(argv)
    entries = load_registry()
    if args.only:
        entries = [e for e in entries if e["name"] == args.only]
        if not entries:
            parser.error(f"no dataset named {args.only!r} in registry.yaml")
    for entry in entries:
        prepare(entry, HERE)
        train = pd.read_csv(HERE / "workspace" / entry["name"] / "train.csv", low_memory=False)
        test_rows = len(pd.read_csv(HERE / ".hidden" / entry["name"] / "test.csv", low_memory=False))
        print(
            f"{entry['name']}: train {train.shape[0]} rows x {train.shape[1]} cols, "
            f"test {test_rows} rows",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
