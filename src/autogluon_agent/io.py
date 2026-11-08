"""Table reading."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .errors import AgaError

_COMMA_DECIMAL = r"^\s*-?\d+,\d+\s*$"
_NUMBER_MAYBE_COMMA = r"^\s*-?\d+(,\d+)?\s*$"


def _has_comma_decimal(df: pd.DataFrame) -> bool:
    """True if some object column holds comma-decimal numbers (possibly mixed with integers)."""
    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue
        s = s.dropna().astype(str)
        if len(s) and s.str.match(_COMMA_DECIMAL).any() and s.str.match(_NUMBER_MAYBE_COMMA).all():
            return True
    return False


def _sniff_delimiter(path: Path, df: pd.DataFrame) -> str:
    """Delimiter pandas' python engine would sniff (from the header line)."""
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        return ";"
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
        return csv.Sniffer().sniff(header).delimiter
    except csv.Error:
        return ";"


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in (".parquet", ".csv", ".txt"):
        raise AgaError(f"Unsupported file type {ext!r} for {path.name}; use .parquet, .csv or .txt")
    try:
        if ext == ".parquet":
            return pd.read_parquet(path)
        df = pd.read_csv(path, sep=None, engine="python")
        single_semicolon = df.shape[1] == 1 and ";" in str(df.columns[0])
        if single_semicolon or _has_comma_decimal(df):
            sep = _sniff_delimiter(path, df)
            # A comma-delimited file cannot use comma as decimal mark: keep the default read.
            if sep != ",":
                df = pd.read_csv(path, sep=sep, decimal=",")
        return df
    except AgaError:
        raise
    except Exception as e:  # unreadable / corrupt / missing file
        raise AgaError(f"Could not read {path}: {e}") from e
