
from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import load_config
from src.agents.prompt_manager import ROUTER_SYSTEM_PROMPT
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Pydantic output schema ────────────────────────────────────────────────────

class RetrievalSourceDecision(BaseModel):
    """Structured output for retrieval source selection."""

    source: Literal["sql", "hybrid"] = Field(
        description=(
            "sql    = structured query against the HR SQLite database (hr_domain only); "
            "hybrid = BM25 keyword + dense vector search over the document corpus (any domain)"
        )
    )
    reason: str = Field(description="One sentence rationale.", max_length=200)


class RouterAgent:
    """Routes single_retrieval queries to the best retrieval source.

    Parameters
    ----------
    llm: LangChain ChatModel.
    """

    def __init__(self, llm: Any):
        self._llm = llm
        try:
            self._structured_llm = llm.with_structured_output(RetrievalSourceDecision)
        except Exception as exc:
            logger.warning("Structured output unsupported (%s) — defaulting to vector.", exc)
            self._structured_llm = None

    @classmethod
    def from_config(cls, config=None) -> "RouterAgent":
        cfg = config or load_config()
        api_key_env = cfg.llm.get("groq_api_key_env", "GROQ_API_KEY")
        model = cfg.llm.get("small_model", cfg.llm.get("model_name", "llama-3.1-8b-instant"))
        llm = ChatGroq(model=model, temperature=0.0, api_key=os.getenv(api_key_env, ""))
        return cls(llm)

    def route(self, query: str, domain: str = "hr_domain", strategy: str = "single_retrieval") -> str:
        """Return the best retrieval source for *query*.

        Parameters
        ----------
        query:    The (expanded/rewritten) user query.
        domain:   Detected domain — constrains available sources.
        strategy: Decision agent strategy — informs source selection.

        Returns one of: "sql", "hybrid".
        web_search is handled upstream and never returned here.
        """
        _fallback = "sql" if (domain == "hr_domain" and strategy != "multi_hop") else "hybrid"

        if self._structured_llm is not None:
            try:
                result: RetrievalSourceDecision = self._structured_llm.invoke([
                    SystemMessage(
                        content=ROUTER_SYSTEM_PROMPT.format(
                            query=query,
                            domain=domain,
                            strategy=strategy,
                        )
                    ),
                ])
                source = result.source
                # Hard guard: research_domain must never use sql
                if domain == "research_domain" and source == "sql":
                    logger.warning(
                        "RouterAgent: research_domain cannot use 'sql' — overriding to hybrid",
                    )
                    source = "hybrid"
                logger.info(
                    "RouterAgent: '%s' [domain=%s, strategy=%s] → %s (%s)",
                    query[:60], domain, strategy, source, result.reason[:80],
                )
                return source
            except Exception as exc:
                logger.warning(
                    "RouterAgent LLM call failed (%s) — defaulting to %s", exc, _fallback
                )

        return _fallback
