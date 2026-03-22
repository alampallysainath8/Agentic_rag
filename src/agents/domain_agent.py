"""
Domain Classifier Agent — identifies which knowledge domain a query belongs to.

Runs immediately after query expansion, before the Decision Agent.
The detected domain is threaded through the rest of the pipeline so that the
Decision Agent and Router Agent can apply domain-aware constraints.

Domains
-------
hr_domain         Employees, salaries, leaves, attendance, performance,
                  training — all backed by the SQLite HR database.
research_domain   AI, RAG, LLMs, embeddings, research papers — backed by
                  the vector document store.
general           Greetings, chit-chat, generic world knowledge.

Usage
-----
    agent = DomainAgent.from_config()
    result = agent.classify("What is the average salary in Engineering?")
    print(result.domain)   # "hr_domain"
    print(result.reason)
"""
from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import load_config
from src.utils.custom_exception import LLMException
from src.utils.logger import setup_logger
from src.agents.prompt_manager import DOMAIN_CLASSIFIER_PROMPT

logger = setup_logger(__name__)

# ── Pydantic output schema ────────────────────────────────────────────────────

class DomainDecision(BaseModel):
    """Structured output from the domain classifier agent."""

    domain: Literal["hr_domain", "research_domain", "general"] = Field(
        description=(
            "hr_domain        = employees, salaries, HR records; "
            "research_domain  = AI, RAG, LLMs, research papers; "
            "general          = greetings, general world knowledge"
        )
    )
    reason: str = Field(
        description="One sentence explaining the domain classification.",
        max_length=200,
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the domain classification (0–1).",
    )


# Fast-path: recognise trivial greetings without an LLM call.
_GREETING_SIGNALS = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "good morning",
    "good afternoon", "good evening", "what is ai", "help", "who are you",
}


class DomainAgent:
    """Classifies queries into knowledge domains using structured LLM output.

    Parameters
    ----------
    llm:  LangChain chat model.  A small, fast model is sufficient.
    """

    def __init__(self, llm: Any):
        self._llm = llm
        try:
            self._structured_llm = llm.with_structured_output(DomainDecision)
        except Exception as exc:
            logger.warning(
                "DomainAgent: structured output not supported (%s) — "
                "falling back to heuristics.", exc
            )
            self._structured_llm = None

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config=None) -> "DomainAgent":
        """Build from config.yaml using the small model."""
        cfg = config or load_config()
        try:
            small_model = cfg.llm.get(
                "small_model", cfg.llm.get("model_name", "llama-3.1-8b-instant")
            )
            api_key_env = cfg.llm.get("groq_api_key_env", "GROQ_API_KEY")
            api_key = os.getenv(api_key_env, "")
            llm = ChatGroq(model=small_model, temperature=0.0, api_key=api_key)
            logger.info("DomainAgent: using model %s", small_model)
            return cls(llm)
        except Exception as exc:
            raise LLMException("DomainAgent.from_config failed", exc) from exc

    # ── public API ────────────────────────────────────────────────────────────

    def classify(self, query: str) -> DomainDecision:
        """Classify *query* into a domain.

        Returns
        -------
        DomainDecision with ``domain``, ``reason``, ``confidence``.
        """
        # Fast path: obvious greetings
        if query.strip().lower().rstrip("!?.") in _GREETING_SIGNALS:
            return DomainDecision(
                domain="general",
                reason="Greeting or general conversational query.",
                confidence=1.0,
            )

        # Structured LLM path
        if self._structured_llm is not None:
            try:
                result: DomainDecision = self._structured_llm.invoke([
                    SystemMessage(content=DOMAIN_CLASSIFIER_PROMPT.format(query=query)),
                ])
                logger.info(
                    "DomainAgent: '%s' → %s (%.0f%% confident)",
                    query[:60], result.domain, result.confidence * 100,
                )
                return result
            except Exception as exc:
                logger.warning(
                    "DomainAgent LLM call failed (%s) — heuristic fallback", exc
                )

        # Heuristic fallback
        return self._heuristic(query)

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _heuristic(query: str) -> DomainDecision:
        """Minimal fallback used only when the LLM call fails."""
        return DomainDecision(
            domain="general",
            reason="LLM unavailable — defaulting to general domain.",
            confidence=0.3,
        )


