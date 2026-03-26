
from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import load_config
from src.utils.custom_exception import LLMException
from src.utils.logger import setup_logger
from src.agents.prompt_manager import DECISION_SYSTEM_PROMPT

DEFAULT_DOMAIN = "hr_domain"  # safe fallback when domain is unknown

logger = setup_logger(__name__)

# ── Pydantic output schema ────────────────────────────────────────────────────

class RoutingDecision(BaseModel):
    """Structured output from the decision agent."""

    strategy: Literal["no_retrieval", "single_retrieval", "multi_hop", "web_search"] = Field(
        description=(
            "no_retrieval     = greetings / general knowledge; "
            "single_retrieval = one focused lookup; "
            "multi_hop        = MULTIPLE REASONING STEPS (chaining, analysis, comparison); "
            "web_search       = requires current/external web information"
        )
    )
    reason: str = Field(
        description="One sentence explaining the routing decision.",
        max_length=200,
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the decision (0-1).",
    )


# Fast-path: recognise trivial greetings without an LLM call.
_GREETINGS = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "good morning",
    "good afternoon", "good evening", "help", "who are you", "what is ai",
}


class DecisionAgent:
    """Classifies queries into retrieval strategies using structured LLM output.

    Parameters
    ----------
    llm:      LangChain chat model.  Should be a *small, fast* model
              (llama-3.1-8b or gemma-2-9b-it recommended).
    """

    def __init__(self, llm: Any):
        self._llm = llm
        try:
            self._structured_llm = llm.with_structured_output(RoutingDecision)
        except Exception as exc:
            logger.warning(
                "Structured output not supported (%s) — falling back to heuristics.", exc
            )
            self._structured_llm = None

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config=None) -> "DecisionAgent":
        """Build from config.yaml using the small model."""
        cfg = config or load_config()
        try:
            small_model = cfg.llm.get("small_model", cfg.llm.get("model_name", "llama-3.1-8b-instant"))
            api_key_env = cfg.llm.get("groq_api_key_env", "GROQ_API_KEY")
            api_key = os.getenv(api_key_env, "")
            llm = ChatGroq(model=small_model, temperature=0.0, api_key=api_key)
            logger.info("DecisionAgent: using model %s", small_model)
            return cls(llm)
        except Exception as exc:
            raise LLMException("DecisionAgent.from_config failed", exc) from exc

    # ── public API ────────────────────────────────────────────────────────────

    def decide(self, query: str, domain: str = DEFAULT_DOMAIN) -> RoutingDecision:
        """Classify *query* into a retrieval strategy.

        Parameters
        ----------
        query:  The (expanded) user query.
        domain: Detected domain — hr_domain / research_domain / general.

        Returns
        -------
        RoutingDecision with ``strategy``, ``reason``, ``confidence``.
        """
        # Fast path: greetings and general domain
        q_norm = query.strip().lower().rstrip("!?.")
        if q_norm in _GREETINGS or domain == "general":
            return RoutingDecision(
                strategy="no_retrieval",
                reason="Greeting or general-domain query — no retrieval needed.",
                confidence=1.0,
            )

        # Structured LLM path
        if self._structured_llm is not None:
            try:
                result: RoutingDecision = self._structured_llm.invoke([
                    SystemMessage(content=DECISION_SYSTEM_PROMPT.format(domain=domain)),
                    HumanMessage(content=f"QUERY: {query}"),
                ])
                logger.info(
                    "DecisionAgent: '%s' [domain=%s] → %s (%.0f%% confident)",
                    query[:60], domain, result.strategy, result.confidence * 100,
                )
                return result
            except Exception as exc:
                logger.warning("DecisionAgent LLM call failed (%s) — heuristic fallback", exc)

        return self._heuristic(query, domain)

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _heuristic(query: str, domain: str) -> RoutingDecision:
        """Minimal fallback used only when the structured LLM call fails."""
        return RoutingDecision(
            strategy="single_retrieval",
            reason="LLM unavailable — defaulting to single_retrieval.",
            confidence=0.3,
        )
