"""Builders for the benchmark datasets. Each ``build_<name>(cache) -> DataFrame`` downloads its
sources into ``cache`` once and is offline on reruns."""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

STATSBOMB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
MAX_MATCHES = 800
SEED = 42


def _download(url: str, dest: Path, retries: int = 5) -> Path:
    """Fetch ``url`` to ``dest`` unless already cached (atomic write, retried)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "autogluon-agent-benchmark"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as err:
            if err.code == 404 or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _json(url: str, dest: Path):
    return json.loads(_download(url, dest).read_text(encoding="utf-8"))


# --- football-xg ---------------------------------------------------------------------------


def _shot_rows(match: dict, competition: str, cache: Path) -> list[dict]:
    """Shot rows of one match; cached as a small per-match file (raw events are large)."""
    dest = cache / "statsbomb" / "shots" / f"{match['match_id']}.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    events_path = cache / "statsbomb" / "_events_tmp" / f"{match['match_id']}.json"
    events = _json(f"{STATSBOMB}events/{match['match_id']}.json", events_path)
    events_path.unlink()
    rows = []
    for event in events:
        if event["type"]["name"] != "Shot":
            continue
        shot = event["shot"]
        freeze = shot.get("freeze_frame")
        location = event.get("location") or [None, None]
        rows.append(
            {
                "match_id": match["match_id"],
                "competition": competition,
                "period": event["period"],
                "minute": event["minute"],
                "second": event["second"],
                "team": event["team"]["name"],
                "player": event.get("player", {}).get("name"),
                "position": event.get("position", {}).get("name"),
                "play_pattern": event["play_pattern"]["name"],
                "x": location[0],
                "y": location[1],
                "body_part": shot["body_part"]["name"],
                "technique": shot["technique"]["name"],
                "shot_type": shot["type"]["name"],
                "first_time": bool(shot.get("first_time", False)),
                "under_pressure": bool(event.get("under_pressure", False)),
                "one_on_one": bool(shot.get("one_on_one", False)),
                "open_goal": bool(shot.get("open_goal", False)),
                "aerial_won": bool(shot.get("aerial_won", False)),
                "follows_dribble": bool(shot.get("follows_dribble", False)),
                "freeze_frame_json": json.dumps(freeze, separators=(",", ":")) if freeze else "",
                "is_goal": int(shot["outcome"]["name"] == "Goal"),
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def build_football_xg(cache: Path) -> pd.DataFrame:
    """One row per StatsBomb open-data shot; statsbomb_xg, end_location and outcome are excluded."""
    cache = Path(cache)
    competitions = _json(f"{STATSBOMB}competitions.json", cache / "statsbomb" / "competitions.json")
    todo: list[tuple[dict, str]] = []
    for comp in sorted(competitions, key=lambda c: (c["competition_id"], c["season_id"])):
        try:
            matches = _json(
                f"{STATSBOMB}matches/{comp['competition_id']}/{comp['season_id']}.json",
                cache / "statsbomb" / "matches" / f"{comp['competition_id']}_{comp['season_id']}.json",
            )
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
            continue
        todo += [(m, comp["competition_name"]) for m in sorted(matches, key=lambda m: m["match_id"])]
        if len(todo) >= MAX_MATCHES:
            break
    todo = todo[:MAX_MATCHES]
    with ThreadPoolExecutor(max_workers=8) as pool:  # map keeps the deterministic order
        per_match = list(pool.map(lambda item: _shot_rows(item[0], item[1], cache), todo))
    return pd.DataFrame([row for rows in per_match for row in rows])


# --- regression datasets ---------------------------------------------------------------------


def build_ames(cache: Path) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(
        data_id=42165, as_frame=True, data_home=str(Path(cache) / "openml"), parser="auto"
    ).frame
    return frame.drop(columns=["Id"]).reset_index(drop=True)


def build_california(cache: Path) -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing

    return fetch_california_housing(
        as_frame=True, data_home=str(Path(cache) / "sklearn")
    ).frame.reset_index(drop=True)


def _uci_zip(url: str, dest: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(_download(url, dest))


def build_bike(cache: Path) -> pd.DataFrame:
    zf = _uci_zip(
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        Path(cache) / "uci" / "bike.zip",
    )
    df = pd.read_csv(io.BytesIO(zf.read("hour.csv")))
    df = df.drop(columns=["casual", "registered", "instant"])
    df["dteday_hour"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    return df


# --- classification datasets -----------------------------------------------------------------


def build_bank(cache: Path) -> pd.DataFrame:
    outer = _uci_zip(
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        Path(cache) / "uci" / "bank.zip",
    )
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
    member = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
    return pd.read_csv(io.BytesIO(inner.read(member)), sep=";")


ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]  # fmt: skip


def build_adult_leak(cache: Path) -> pd.DataFrame:
    """UCI Adult plus a planted post-outcome column, ``tax_bracket_following_year``."""
    zf = _uci_zip(
        "https://archive.ics.uci.edu/static/public/2/adult.zip", Path(cache) / "uci" / "adult.zip"
    )
    parts = []
    for member, skip in (("adult.data", 0), ("adult.test", 1)):  # adult.test has a junk first line
        parts.append(
            pd.read_csv(
                io.BytesIO(zf.read(member)),
                names=ADULT_COLUMNS,
                skiprows=skip,
                skipinitialspace=True,
                na_values="?",
            )
        )
    df = pd.concat(parts, ignore_index=True)
    df["income"] = df["income"].str.rstrip(".")
    rich = (df["income"] == ">50K").to_numpy()
    rng = np.random.default_rng(SEED)
    is_high = rng.random(len(df)) < np.where(rich, 0.85, 0.05)
    other = np.where(rng.random(len(df)) < 0.5, "low", "mid")
    df["tax_bracket_following_year"] = np.where(is_high, "high", other)
    return df
