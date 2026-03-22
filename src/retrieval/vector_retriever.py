"""
Vector retriever — wraps ChromaVectorStore with the configured embedding model.

Returns LangChain Documents so the same objects flow through BM25,
HybridRetriever, and the EnsembleRetriever without conversion.

Public API
----------
    from src.retrieval.vector_retriever import VectorRetriever
    vr = VectorRetriever()                          # uses config.yaml
    docs = vr.retrieve("What is supplier turnover?", k=5)
    retriever = vr.as_langchain_retriever(k=10)     # for EnsembleRetriever
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.config import load_config
from src.embeddings import get_embedding_provider
from src.utils.logger import setup_logger
from src.utils.custom_exception import RetrievalException
from src.vectorstore import get_vectorstore

logger = setup_logger(__name__)


class VectorRetriever:
    """Semantic dense retriever backed by Chroma + configurable embedding model.

    Parameters
    ----------
    config: src.config.Config object.  Reads ``embedding`` and ``vectorstore``
            sections.  Defaults to ``load_config()`` when None.
    """

    def __init__(self, config=None):
        try:
            cfg = config or load_config()
            self._embedder = get_embedding_provider(cfg.embedding)
            self._vs       = get_vectorstore(cfg.vectorstore)
            self._default_k = cfg.retrieval.get("top_k", 5)
            logger.info("VectorRetriever ready (k=%d)", self._default_k)
        except Exception as exc:
            raise RetrievalException("VectorRetriever init failed", exc) from exc

    # ── retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        """Embed *query* and return the top-k most similar Documents.

        Parameters
        ----------
        query:  Natural language query string.
        k:      Number of results. Defaults to ``retrieval.top_k`` from config.
        filter: Optional Chroma metadata filter dict.

        Returns
        -------
        List[Document]
        """
        try:
            k = k or self._default_k
            query_vec = self._embedder.embed_query(query)

            # Chroma .query() returns dicts; convert to LangChain Documents
            raw_results = self._vs.query(query_vec, k=k, filter=filter)
            docs = [
                Document(
                    page_content=r.get("document", r.get("page_content", "")),
                    metadata=r.get("metadata", {}),
                )
                for r in raw_results
            ]
            logger.info("VectorRetriever: '%s' → %d docs", query[:60], len(docs))
            return docs
        except Exception as exc:
            raise RetrievalException(f"VectorRetriever.retrieve failed for '{query}'", exc) from exc

    def as_langchain_retriever(self, k: Optional[int] = None) -> Any:
        """Return a LangChain-compatible retriever for use in chains / EnsembleRetriever."""
        effective_k = k or self._default_k
        return self._vs.as_retriever(search_kwargs={"k": effective_k})
