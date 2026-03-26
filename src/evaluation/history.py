"""
Evaluation History
==================
Lightweight JSON-file store for evaluation results.
Each entry is appended to ``logs/eval_history.json``.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_DEFAULT_PATH = Path(__file__).parent.parent.parent / "logs" / "eval_history.json"


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")


def append_result(record: dict[str, Any], path: Path | None = None) -> None:
    """Append *record* to the history file (thread-safe)."""
    store = Path(path) if path else _DEFAULT_PATH
    with _LOCK:
        _ensure_file(store)
        try:
            data: list = json.loads(store.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            data = []
        data.append(record)
        store.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_history(path: Path | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Return up to *limit* most-recent evaluation records."""
    store = Path(path) if path else _DEFAULT_PATH
    _ensure_file(store)
    try:
        data: list = json.loads(store.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []
    return data[-limit:]


def clear_history(path: Path | None = None) -> None:
    """Wipe all stored evaluation records."""
    store = Path(path) if path else _DEFAULT_PATH
    _ensure_file(store)
    store.write_text("[]", encoding="utf-8")
