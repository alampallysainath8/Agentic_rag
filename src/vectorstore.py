"""
Plug-and-play vector store adapters.
Phase-1: Chroma only.  Future phases: add FAISS/Pinecone here + factory switch.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Base interface ────────────────────────────────────────────────────────────

class VectorStore:
    """Minimal interface all adapters must implement."""

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
    ) -> None:
        raise NotImplementedError

    def query(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Return list of {id, document, metadata, score} dicts."""
        raise NotImplementedError

    def delete(self, ids: List[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


# ── Chroma ────────────────────────────────────────────────────────────────────

class ChromaVectorStore(VectorStore):
    """
    Persistent Chroma vector store.
    Collection name + persist path configured via config.yaml under
    vectorstore.chroma section.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        collection_name: str = "agentic_rag",
        distance_function: str = "cosine",
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError("chromadb required: pip install chromadb") from e

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_function},
        )
        logger.info(
            "Chroma collection '%s' at '%s' — %d docs",
            collection_name, persist_directory, self._collection.count(),
        )

    # ── write ─────────────────────────────────────────────────────────────
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete(self, ids: List[str]) -> None:
        self._collection.delete(ids=ids)

    # ── read ──────────────────────────────────────────────────────────────
    def query(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = dict(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        if filter:
            kwargs["where"] = filter
        res = self._collection.query(**kwargs)
        results = []
        for i, doc_id in enumerate(res["ids"][0]):
            results.append({
                "id":       doc_id,
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score":    1.0 - res["distances"][0][i],  # cosine: higher = better
            })
        return results

    def count(self) -> int:
        return self._collection.count()

    def get_all_documents(self) -> Dict[str, Any]:
        """Return all stored documents and metadatas as a raw dict.

        Returns dict with keys 'documents' (List[str]) and 'metadatas' (List[dict]).
        """
        return self._collection.get(include=["documents", "metadatas"])

    # LangChain interop ────────────────────────────────────────────────────
    def as_langchain_retriever(self, embedding_fn, k: int = 5):
        """Return a LangChain Chroma retriever (requires langchain-chroma)."""
        from langchain_chroma import Chroma as LCChroma
        lc_chroma = LCChroma(
            client=self._client,
            collection_name=self._collection.name,
            embedding_function=embedding_fn,
        )
        return lc_chroma.as_retriever(search_kwargs={"k": k})


# ── Factory ───────────────────────────────────────────────────────────────────

def get_vectorstore(cfg: dict) -> VectorStore:
    """
    Factory: read `vectorstore` section from config dict.
    """
    prov = cfg.get("provider", "chroma")
    if prov == "chroma":
        c = cfg.get("chroma", {})
        return ChromaVectorStore(
            persist_directory=c.get("persist_directory", "./chroma_data"),
            collection_name=c.get("collection_name", "agentic_rag"),
            distance_function=c.get("distance_function", "cosine"),
        )
    raise ValueError(f"Unknown vectorstore provider: {prov!r}. Only 'chroma' is supported in phase-1.")

