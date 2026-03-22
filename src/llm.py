"""
Plug-and-play LLM adapters (Groq / Vertex AI).
Switch provider in config.yaml — no code changes needed.
Returns LangChain ChatModel objects so they drop straight into
LangChain chains and LangGraph nodes.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ── Base interface ────────────────────────────────────────────────────────────

class LLMProvider:
    """Common interface; all adapters return plain strings from .generate()."""

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def get_langchain_llm(self):
        """Return a LangChain BaseChatModel compatible object."""
        raise NotImplementedError


# ── Groq ──────────────────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    """
    Groq-hosted LLM adapter.
    API key read from env-var named by llm.groq_api_key_env in config.yaml.
    Default model: llama-3.3-70b-versatile
    """

    def __init__(self, model_name: str, api_key_env: str = "GROQ_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 2048):
        from groq import Groq
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            available = [k for k in os.environ.keys() if "GROQ" in k.upper() or "API" in k.upper()]
            raise EnvironmentError(
                f"{api_key_env} env-var not set or empty. "
                f"Available keys: {available}. "
                f"Set {api_key_env} in .env or shell environment."
            )
        self._client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("Groq LLM: %s", model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content

    def get_langchain_llm(self):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


# ── Vertex AI ────────────────────────────────────────────────────────────────

class VertexAIProvider(LLMProvider):
    """
    Google Vertex AI adapter.
    Auth: application-default credentials or GOOGLE_APPLICATION_CREDENTIALS.
    Project read from env-var named by llm.vertex_project_env.
    """

    def __init__(self, model_name: str, project_env: str = "GOOGLE_CLOUD_PROJECT",
                 location: str = "us-central1",
                 temperature: float = 0.0, max_tokens: int = 2048):
        import vertexai
        from vertexai.generative_models import GenerativeModel
        project = os.getenv(project_env, "").strip()
        if not project:
            available = [k for k in os.environ.keys() if "GOOGLE" in k.upper() or "GCP" in k.upper()]
            raise EnvironmentError(
                f"{project_env} env-var not set or empty. "
                f"Available keys: {available}. "
                f"Set {project_env} in .env or shell environment."
            )
        vertexai.init(project=project, location=location)
        self._model = GenerativeModel(model_name)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("VertexAI LLM: %s (project=%s)", model_name, project)

    def generate(self, prompt: str, **kwargs) -> str:
        from vertexai.generative_models import GenerationConfig
        cfg = GenerationConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        resp = self._model.generate_content(prompt, generation_config=cfg)
        return resp.text

    def get_langchain_llm(self):
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_provider(cfg: dict) -> LLMProvider:
    """
    Factory: read `llm` section from config dict and return the right adapter.
    """
    prov = cfg.get("provider", "groq")
    temp = cfg.get("temperature", 0.0)
    max_tok = cfg.get("max_tokens", 2048)
    if prov == "groq":
        return GroqProvider(
            cfg["groq_model"],
            api_key_env=cfg.get("groq_api_key_env", "GROQ_API_KEY"),
            temperature=temp,
            max_tokens=max_tok,
        )
    if prov == "vertexai":
        return VertexAIProvider(
            cfg["vertex_model"],
            project_env=cfg.get("vertex_project_env", "GOOGLE_CLOUD_PROJECT"),
            location=cfg.get("vertex_location", "us-central1"),
            temperature=temp,
            max_tokens=max_tok,
        )
    raise ValueError(f"Unknown LLM provider: {prov!r}. Choose 'groq' or 'vertexai'.")

