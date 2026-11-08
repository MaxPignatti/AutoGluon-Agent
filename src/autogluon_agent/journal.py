"""Append-only JSON-lines journal of everything that happens in a run."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, kind: str, **payload) -> dict:
        """Append one entry (with an ISO ``ts``) as a JSON line and return it."""
        entry = {"ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, **payload}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
            fh.flush()
        return entry

    def entries(self, kind: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if kind is None or entry.get("kind") == kind:
                    out.append(entry)
        return out
