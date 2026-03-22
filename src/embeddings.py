"""
Plug-and-play embedding adapters with SINGLETON caching.
Switch provider in config.yaml — no code changes needed.

IMPORTANT: Models are cached globally so they are never reloaded.
This saves 1-3 seconds per query.
"""
from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# ── Global model caches (singleton pattern) ──────────────────────────────────
_HF_MODELS_CACHE: dict[str, "SentenceTransformer"] = {}  # model_name → loaded model
_OPENAI_CLIENT_CACHE: dict[str, "OpenAI"] = {}          # api_key_env → client


# ── Base interface ─────────────────────────────────────────────────────────────

class EmbeddingProvider:
    """Common interface that all embedding adapters must implement."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        """Convenience: embed a single query."""
        return self.embed([text])[0]

    # LangChain compatibility shim
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)


# ── HuggingFace ───────────────────────────────────────────────────────────────

class HuggingFaceEmbedding(EmbeddingProvider):
    """
    Sentence-Transformers embedding via HuggingFace.
    Configured by embedding.huggingface_model in config.yaml.
    """

    def __init__(self, model_name: str, batch_size: int = 64):
        global _HF_MODELS_CACHE
        if model_name in _HF_MODELS_CACHE:
            logger.debug("Using cached HuggingFace model: %s", model_name)
            self.model = _HF_MODELS_CACHE[model_name]
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("sentence-transformers required: pip install sentence-transformers") from e
            logger.info("Loading HuggingFace embedding model: %s (first use—cached for reuse)", model_name)
            self.model = SentenceTransformer(model_name)
            _HF_MODELS_CACHE[model_name] = self.model  # cache globally
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIEmbedding(EmbeddingProvider):
    """
    OpenAI embedding via the openai client.
    API key read from env-var named by embedding.openai_api_key_env in config.yaml.
    """

    def __init__(self, model_name: str, api_key_env: str = "OPENAI_API_KEY", batch_size: int = 64):
        global _OPENAI_CLIENT_CACHE
        if api_key_env in _OPENAI_CLIENT_CACHE:
            logger.debug("Using cached OpenAI client")
            self.client = _OPENAI_CLIENT_CACHE[api_key_env]
        else:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("openai package required: pip install openai") from e
            api_key = os.getenv(api_key_env)
            if not api_key:
                available = [k for k in os.environ.keys() if "OPENAI" in k.upper() or "API" in k.upper()]
                raise EnvironmentError(
                    f"{api_key_env} env-var not set or empty. "
                    f"Available keys: {available}. "
                    f"Set {api_key_env} in .env or shell environment."
                )
            logger.info("Initializing OpenAI client (cached for reuse)")
            self.client = OpenAI(api_key=api_key)
            _OPENAI_CLIENT_CACHE[api_key_env] = self.client  # cache globally
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info("OpenAI embedding model: %s", model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Batch requests to stay within API limits
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            results.extend([d.embedding for d in resp.data])
        return results


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedding_provider(cfg: dict) -> EmbeddingProvider:
    """
    Factory: read `embedding` section from config dict and return the right adapter.
    """
    prov = cfg.get("provider", "huggingface")
    batch = cfg.get("batch_size", 64)
    if prov == "huggingface":
        return HuggingFaceEmbedding(cfg["huggingface_model"], batch_size=batch)
    if prov == "openai":
        return OpenAIEmbedding(
            cfg["openai_model"],
            api_key_env=cfg.get("openai_api_key_env", "OPENAI_API_KEY"),
            batch_size=batch,
        )
    raise ValueError(f"Unknown embedding provider: {prov!r}. Choose 'huggingface' or 'openai'.")

