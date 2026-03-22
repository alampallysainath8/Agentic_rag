"""
BM25 sparse keyword retriever.

Wraps langchain_community.retrievers.BM25Retriever so it fits
the same interface used by VectorRetriever and HybridRetriever.

Public API
----------
    from src.retrieval.bm25_retriever import BM25Retriever
    bm25 = BM25Retriever.from_documents(docs, k=10)
    results: List[Document] = bm25.retrieve("supplier quality")
"""
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.documents import Document

from src.utils.logger import setup_logger
from src.utils.custom_exception import RetrievalException

logger = setup_logger(__name__)


class BM25Retriever:
    """LangChain-backed BM25 keyword retriever.

    Parameters
    ----------
    documents: Corpus as LangChain Documents.
    k:         Number of results per query.
    """

    def __init__(self, documents: List[Document], k: int = 10):
        try:
            from langchain_community.retrievers import BM25Retriever as _BM25
        except ImportError as exc:
            raise RetrievalException(
                "langchain_community not installed — "
                "run: pip install langchain-community rank_bm25"
            ) from exc

        self._retriever: Any = _BM25.from_documents(documents)
        self._retriever.k = k
        self._documents = documents
        self._k = k
        logger.info("BM25Retriever: indexed %d docs, k=%d", len(documents), k)

    # ── factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_documents(cls, documents: List[Document], k: int = 10) -> "BM25Retriever":
        """Build from LangChain Documents."""
        return cls(documents, k=k)

    @classmethod
    def from_chunks(cls, chunks: list, k: int = 10) -> "BM25Retriever":
        """Build from src.indexing.chunker.Chunk objects."""
        docs = [c.to_langchain_document() for c in chunks]
        return cls(docs, k=k)

    @classmethod
    def from_chroma(cls, vectorstore: Any, k: int = 10) -> "BM25Retriever":
        """Populate BM25 corpus from an existing Chroma vectorstore.

        Supports:
        - ``ChromaVectorStore`` (src.vectorstore) via ``.get_all_documents()``
        - Raw chromadb Collection via ``.get()``
        - Any object that exposes ``._collection.get()``
        """
        try:
            # 1. Our own ChromaVectorStore wrapper
            if hasattr(vectorstore, "get_all_documents"):
                raw = vectorstore.get_all_documents()
            # 2. Raw chromadb Collection (has a .get() method)
            elif hasattr(vectorstore, "get") and callable(vectorstore.get):
                raw = vectorstore.get(include=["documents", "metadatas"])
            # 3. Wrapped object exposing the underlying _collection
            elif hasattr(vectorstore, "_collection"):
                raw = vectorstore._collection.get(include=["documents", "metadatas"])
            else:
                raise AttributeError(
                    f"{type(vectorstore).__name__!r} has no recognised document-access method. "
                    "Expected get_all_documents(), get(), or _collection."
                )

            docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(
                    raw.get("documents", []),
                    raw.get("metadatas", []),
                )
                if text
            ]
            logger.info("BM25Retriever.from_chroma: %d docs", len(docs))
            return cls(docs, k=k)
        except Exception as exc:
            raise RetrievalException("BM25Retriever.from_chroma failed", exc) from exc

    # ── retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Return top-k BM25 documents for *query*."""
        try:
            if k is not None:
                self._retriever.k = k
            results: List[Document] = self._retriever.invoke(query)
            logger.info("BM25 retrieve: '%s' → %d hits", query[:60], len(results))
            return results
        except Exception as exc:
            raise RetrievalException(f"BM25 retrieve failed for '{query}'", exc) from exc

    def as_langchain_retriever(self, k: Optional[int] = None) -> Any:
        """Return the inner LangChain retriever (for EnsembleRetriever)."""
        if k is not None:
            self._retriever.k = k
        return self._retriever

    @property
    def documents(self) -> List[Document]:
        return self._documents

    def __len__(self) -> int:
        return len(self._documents)
