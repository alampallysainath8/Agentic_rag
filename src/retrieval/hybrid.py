from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.config import load_config
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.utils.custom_exception import RetrievalException
from src.utils.logger import setup_logger
from src.vectorstore import get_vectorstore

logger = setup_logger(__name__)


def cohere_rerank(
    query: str,
    documents: List[Document],
    top_n: int = 5,
    api_key: str = "",
    model: str = "rerank-english-v3.0",
) -> List[Document]:
    """Cross-encoder reranking via Cohere Rerank API.

    Falls back to original order if Cohere is unavailable or key is missing.
    """
    key = api_key or os.getenv("COHERE_API_KEY", "")
    if not key or not documents:
        return documents[:top_n]
    try:
        import cohere
        co = cohere.Client(key)
        texts = [doc.page_content for doc in documents]
        response = co.rerank(
            model=model,
            query=query,
            documents=texts,
            top_n=min(top_n, len(documents)),
            return_documents=True,
        )
        reranked = [
            Document(
                page_content=result.document.text,
                metadata={
                    **documents[result.index].metadata,
                    "cohere_score": result.relevance_score,
                    "cohere_rank": i,
                },
            )
            for i, result in enumerate(response.results)
        ]
        logger.info(
            "Cohere rerank: %d -> %d docs for query '%s'",
            len(documents), len(reranked), query[:60],
        )
        return reranked
    except ImportError:
        logger.warning("cohere package not installed -- skipping rerank. Run: pip install cohere")
        return documents[:top_n]
    except Exception as exc:
        logger.warning("Cohere rerank failed (%s) -- returning original order", exc)
        return documents[:top_n]


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
    top_k: int = 10,
) -> List[str]:
    """Merge multiple ranked document-ID lists with RRF."""
    rrf_scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            rrf_scores[doc_id] += 1.0 / (rank + 1 + k)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in fused[:top_k]]


class HybridRetriever:
    """BM25 + dense vector retriever fused with Reciprocal Rank Fusion.

    Parameters
    ----------
    config:  src.config.Config - reads retrieval section for defaults.
    bm25:    Optional pre-built BM25Retriever.  Build via load_bm25_from_store().
    vector:  Optional pre-built VectorRetriever.  Built automatically if None.
    """

    def __init__(self, config=None, bm25: Any = None, vector: Any = None):
        cfg = config or load_config()
        ret_cfg = cfg.retrieval
        self._default_k   = ret_cfg.get("top_k", 5)
        self._candidate_k = ret_cfg.get("candidate_k", 20)
        self._rrf_k       = ret_cfg.get("hybrid_rrf_k", 60)
        self._use_rerank  = ret_cfg.get("use_cohere_rerank", False)
        self._bm25   = bm25
        self._vector = vector or VectorRetriever(cfg)
        logger.info(
            "HybridRetriever ready (candidate_k=%d, rrf_k=%d, rerank=%s)",
            self._candidate_k, self._rrf_k, self._use_rerank,
        )

    def load_bm25_from_store(self, vectorstore: Any = None) -> None:
        """Populate BM25 from documents already indexed in Chroma."""
        if vectorstore is None:
            vectorstore = get_vectorstore(load_config().vectorstore)
        self._bm25 = BM25Retriever.from_chroma(vectorstore, k=self._candidate_k)
        logger.info("HybridRetriever: BM25 corpus loaded from vectorstore")

    def load_bm25_from_documents(self, documents: List[Document]) -> None:
        """Build BM25 index from a list of Documents."""
        self._bm25 = BM25Retriever.from_documents(documents, k=self._candidate_k)
        logger.info(
            "HybridRetriever: BM25 corpus loaded from %d documents", len(documents)
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        """Retrieve top-k documents using BM25 + dense RRF fusion.

        Falls back to dense-only if BM25 is not loaded.
        """
        k = k or self._default_k
        ck = self._candidate_k

        dense_docs = self._vector.retrieve(query, k=ck, filter=filter)

        if self._bm25 is None:
            logger.warning("HybridRetriever: BM25 not loaded -- dense-only retrieval.")
            return dense_docs[:k]

        sparse_docs = self._bm25.retrieve(query, k=ck)

        dense_ids  = [d.metadata.get("chunk_id", d.page_content[:40]) for d in dense_docs]
        sparse_ids = [d.metadata.get("chunk_id", d.page_content[:40]) for d in sparse_docs]

        fused_ids = reciprocal_rank_fusion(
            [dense_ids, sparse_ids], k=self._rrf_k, top_k=k
        )

        doc_map: Dict[str, Document] = {}
        for doc in dense_docs + sparse_docs:
            did = doc.metadata.get("chunk_id", doc.page_content[:40])
            doc_map.setdefault(did, doc)

        result = [doc_map[did] for did in fused_ids if did in doc_map]
        logger.info(
            "HybridRetriever: %d docs via RRF (dense=%d, sparse=%d)",
            len(result), len(dense_docs), len(sparse_docs),
        )
        if self._use_rerank:
            return cohere_rerank(query, result, top_n=k)
        return result
