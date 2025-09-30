"""Lightweight JSONL logger for run metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class JsonlLogger:
    """Append-only JSONL writer for scalar metrics."""

    path: Path
    flush: bool = True
    _file: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, step: int, **scalars: float) -> None:
        record: Dict[str, float] = {"step": int(step)}
        record.update(scalars)
        json.dump(record, self._file)
        self._file.write("\n")
        if self.flush:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
