import logging
import os
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_cache_store = None


def _get_store(persist_dir: str, embeddings):
    global _cache_store
    if _cache_store is None:
        from langchain_community.vectorstores import FAISS
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        index_path = os.path.join(persist_dir, "cache_index")
        if os.path.exists(index_path):
            _cache_store = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            logger.info("Semantic cache loaded from disk: %s", index_path)
        else:
            _cache_store = None
    return _cache_store


class SemanticCache:
    def __init__(self, config):
        cache_cfg = config.cache if hasattr(config, "cache") else {}
        self.enabled = cache_cfg.get("enabled", True)
        self.threshold = float(cache_cfg.get("distance_threshold", 0.25))
        self.persist_dir = cache_cfg.get("persist_directory", "./cache_data")
        self._embeddings = None
        self._store = None

    def _lazy_init(self):
        if self._embeddings is None:
            from sentence_transformers import SentenceTransformer
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        if self._store is None:
            global _cache_store
            _cache_store = None
            self._store = _get_store(self.persist_dir, self._embeddings)

    def lookup(self, query: str):
        if not self.enabled:
            return None
        self._lazy_init()
        if self._store is None:
            return None
        try:
            hits = self._store.similarity_search_with_score(query, k=1)
        except Exception:
            return None
        if not hits:
            return None
        doc, distance = hits[0]
        logger.info("Cache distance: %.4f (threshold: %.4f)", distance, self.threshold)
        if distance <= self.threshold:
            answer = doc.metadata.get("answer", "")
            if answer:
                logger.info("Cache HIT for query: %s", query[:60])
                return answer
        logger.info("Cache MISS")
        return None

    def store(self, query: str, answer: str):
        if not self.enabled:
            return
        self._lazy_init()
        from langchain_community.vectorstores import FAISS
        doc = Document(page_content=query, metadata={"answer": answer})
        if self._store is None:
            self._store = FAISS.from_documents([doc], self._embeddings)
        else:
            self._store.add_documents([doc])
        index_path = os.path.join(self.persist_dir, "cache_index")
        self._store.save_local(index_path)
        # update module-level singleton
        global _cache_store
        _cache_store = self._store
        logger.info("Stored in cache for query: %s", query[:60])
