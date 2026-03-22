"""
Top-level Indexer facade — delegates to the full indexing pipeline
(src/indexing/pipeline.py) which implements the enhanced_rag_cache approach:
  PDF → pymupdf4llm → image enrichment → chunking → embeddings → Chroma.

Usage:
    from src.indexer import Indexer
    idx = Indexer()
    idx.index_file("report.pdf")         # single file
    idx.index_folder("./docs")           # whole folder
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import load_config
from .indexing.pipeline import IndexingPipeline

logger = logging.getLogger(__name__)


class Indexer:
    """
    Facade around IndexingPipeline.
    All config is read from config.yaml — swap embedding or chunking
    strategy there without touching this file.
    """

    def __init__(self, cfg_path: str | None = None):
        self.config = load_config(cfg_path)
        self._pipeline = IndexingPipeline(self.config)

    def index_file(self, file_path: str) -> dict:
        """Index a single PDF / TXT / MD file."""
        logger.info("Indexing file: %s", file_path)
        return self._pipeline.run(file_path)

    def index_folder(
        self,
        folder: str,
        patterns: tuple[str, ...] = ("**/*.pdf", "**/*.txt", "**/*.md"),
    ) -> list[dict]:
        """Recursively index all matching files under `folder`."""
        base = Path(folder)
        results = []
        for pat in patterns:
            for f in base.glob(pat):
                if f.is_file():
                    try:
                        results.append(self.index_file(str(f)))
                    except Exception as exc:
                        logger.warning("Skipping %s — %s", f.name, exc)
        return results

