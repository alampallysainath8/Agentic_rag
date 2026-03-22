from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ..config import Config
from ..embeddings import get_embedding_provider
from ..vectorstore import get_vectorstore
from .chunker import chunk_text_docs, extract_tables_and_text
from .image_enricher import (
    enrich_images,
    replace_images_with_placeholders,
    restore_image_placeholders,
)
from .pdf_parser import clean_markdown, load_pdf_with_docling

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _load_hash_store(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as fh:
            return json.load(fh)
    return {}


def _save_hash_store(path: str, store: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(store, fh, indent=2)


# ── pipeline ──────────────────────────────────────────────────────────────────

class IndexingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self._embedder = get_embedding_provider(config.embedding)
        self._vs = get_vectorstore(config.vectorstore)
        dedup_cfg = config.deduplication if hasattr(config, "deduplication") else {}
        self._hash_store_path = dedup_cfg.get(
            "hash_store_path", "./indexing/file_hashes.json"
        )
        logger.info(
            "IndexingPipeline ready | embedding=%s | vectorstore=%s",
            config.embedding.get("provider"),
            config.vectorstore.get("provider"),
        )

    # ── public entry point ────────────────────────────────────────────────

    def run(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        doc_id = path.name
        suffix = path.suffix.lower()

        # ── deduplication check ───────────────────────────────────────────
        hash_store = _load_hash_store(self._hash_store_path)
        current_hash = _compute_hash(file_path)
        if hash_store.get(doc_id) == current_hash:
            logger.info("Skipping '%s' — already indexed (hash unchanged).", doc_id)
            return {
                "doc_id": doc_id,
                "source": str(path),
                "num_chunks": 0,
                "skipped": True,
                "reason": "duplicate",
            }

        if suffix != ".pdf":
            raise ValueError(f"Only PDF files are supported. Got: '{path.name}'")

        logger.info("=== Indexing: %s ===", doc_id)

        # ── config values ─────────────────────────────────────────────────
        idx_cfg = self.config.indexing
        md_rec = idx_cfg.get("markdown_recursive", {})
        chunk_size = md_rec.get("recursive_chunk_size", 500)
        chunk_overlap = md_rec.get("recursive_overlap", 50)
        do_enrich = idx_cfg.get("embed_images", False)
        scratch_dir = idx_cfg.get("scratch_dir", "./indexing/scratch_images")
        vision_model = self.config.vision_llm.get("groq_model", "")
        api_key = os.getenv("GROQ_API_KEY", "")

        all_chunks = self._index_pdf(
            file_path=str(path),
            doc_id=doc_id,
            scratch_dir=scratch_dir,
            do_enrich=do_enrich,
            vision_model=vision_model,
            api_key=api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not all_chunks:
            logger.warning("No chunks produced from '%s'", doc_id)
            return {"doc_id": doc_id, "source": str(path), "num_chunks": 0}

        # ── embed + upsert ────────────────────────────────────────────────
        self._upsert(all_chunks)

        hash_store[doc_id] = current_hash
        _save_hash_store(self._hash_store_path, hash_store)

        logger.info("Indexed %d chunk(s) for '%s'", len(all_chunks), doc_id)
        return {
            "doc_id":     doc_id,
            "source":     str(path),
            "num_chunks": len(all_chunks),
            "skipped":    False,
        }

    # ── private helpers ───────────────────────────────────────────────────

    def _index_pdf(
        self,
        file_path: str,
        doc_id: str,
        scratch_dir: str,
        do_enrich: bool,
        vision_model: str,
        api_key: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]:
        """6-step PDF indexing flow."""

        # 1. Parse PDF → raw markdown + extracted image file paths
        raw_md, img_paths = load_pdf_with_docling(file_path, scratch_dir)
        logger.info("Step 1: parsed '%s' (%d chars, %d images)", doc_id, len(raw_md), len(img_paths))

        # 2. Replace ![alt](ref) refs with {{IMG_N}} tokens BEFORE any processing
        ph_md, ph_map = replace_images_with_placeholders(raw_md, img_paths)
        logger.info("Step 2: inserted %d image placeholder(s)", len(ph_map))

        # 3. Clean markdown (strip page numbers, orphaned headers, excess blank lines)
        clean_md = clean_markdown(ph_md)
        logger.info("Step 3: cleaned markdown (%d chars)", len(clean_md))

        # 4. Enrich placeholders — context is read from the already-tokenised markdown
        ph_map = enrich_images(ph_map, clean_md, do_enrich, vision_model, api_key)
        logger.info("Step 4: image descriptions filled")

        # 5. Split tables out as atomic docs; collect remaining text
        text_docs, table_docs = extract_tables_and_text(clean_md, file_path)
        logger.info("Step 5: %d text doc(s), %d table doc(s)", len(text_docs), len(table_docs))

        # 6. Two-pass chunk the text docs
        text_chunks = chunk_text_docs(text_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info("Step 6: produced %d text chunk(s)", len(text_chunks))

        # 7. Restore {{IMG_N}} tokens in chunks with real descriptions
        restored = restore_image_placeholders(text_chunks, ph_map)

        return restored + table_docs

    def _upsert(self, chunks: list[Document]) -> None:
        """Embed and upsert Documents into the vector store."""
        texts = [d.page_content for d in chunks]
        embeddings = self._embedder.embed(texts)
        ids = [
            d.metadata.get("chunk_id", str(i))
            for i, d in enumerate(chunks)
        ]
        metadatas = [d.metadata for d in chunks]
        self._vs.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
