"""
Central config loader — every module imports from here.
The YAML file is the single source of truth; nothing is hard-coded.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Load .env BEFORE anything else so every module sees the variables
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=True)  # override=True to refresh
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Loaded .env from %s", _env_path)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning("python-dotenv not found — using shell environment only")

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class Config:
    """Thin wrapper around the raw YAML dict with helper accessors."""

    def __init__(self, raw: dict):
        self._raw = raw
        self._configure_logging()

    def _configure_logging(self):
        level = self._raw.get("logging", {}).get("level", "INFO")
        logging.basicConfig(
            format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
            level=getattr(logging, level, logging.INFO),
        )

    # ── top-level section accessors ───────────────────────────────────────
    @property
    def embedding(self) -> dict:
        return self._raw.get("embedding", {})

    @property
    def llm(self) -> dict:
        return self._raw.get("llm", {})

    @property
    def vision_llm(self) -> dict:
        return self._raw.get("vision_llm", {})

    @property
    def vectorstore(self) -> dict:
        return self._raw.get("vectorstore", {})

    @property
    def indexing(self) -> dict:
        return self._raw.get("indexing", {})

    @property
    def retrieval(self) -> dict:
        return self._raw.get("retrieval", {})

    @property
    def sql(self) -> dict:
        return self._raw.get("sql", {})

    @property
    def reflection(self) -> dict:
        return self._raw.get("reflection", {})

    @property
    def cache(self) -> dict:
        return self._raw.get("cache", {})

    @property
    def web_search(self) -> dict:
        return self._raw.get("web_search", {})

    @property
    def deduplication(self) -> dict:
        return self._raw.get("deduplication", {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Config:
    """Load and cache config (call with explicit path to override)."""
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)

