
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger
from src.agents.prompt_manager import (
    GENERATOR_SYSTEM_WITH_CONTEXT,
    GENERATOR_SYSTEM_NO_RETRIEVAL,
    GENERATOR_REWRITE_PROMPT,
)

logger = setup_logger(__name__)

def _format_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history as a readable string for prompt injection."""
    if not chat_history:
        return ""
    lines = ["PREVIOUS CONVERSATION:"]
    for turn in chat_history[-6:]:    # max 6 prior turns to stay within tokens
        role = turn.get("role", "user").capitalize()
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


class GeneratorAgent:
    """Generates answers grounded in context or directly via LLM.

    Parameters
    ----------
    llm_provider: LLMProvider instance (from src.llm).
    use_rewriting: If True, rewrite the query before generation.
    """

    def __init__(self, llm_provider: Any, use_rewriting: bool = False):
        self._llm = llm_provider
        self._use_rewriting = use_rewriting

    # ── query rewriting ───────────────────────────────────────────────────────

    def rewrite_query(self, query: str) -> str:
        """Rewrite *query* for improved retrieval (reduces lexical mismatch)."""
        try:
            rewritten = self._llm.generate(
                GENERATOR_REWRITE_PROMPT.format(query=query)
            ).strip().strip('"')
            logger.debug("Query rewritten: '%s' → '%s'", query[:60], rewritten[:60])
            return rewritten
        except Exception as exc:
            logger.warning("Query rewrite failed: %s — using original", exc)
            return query

    # ── context-grounded generation (single_retrieval / multi_hop) ────────────

    def generate(
        self,
        context: str,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate an answer grounded in *context*.

        Parameters
        ----------
        context:      Retrieved text (documents / SQL results).
        question:     Current user question.
        chat_history: Optional prior conversation turns.

        Returns
        -------
        Answer string.
        """
        if not context.strip():
            return self.generate_no_retrieval(question, chat_history)

        history_str = _format_history(chat_history or [])
        system = GENERATOR_SYSTEM_WITH_CONTEXT.format(context=context)
        prompt = (
            f"{system}\n\n{history_str}\n\nQuestion: {question}\n\nAnswer:"
            if history_str
            else f"{system}\n\nQuestion: {question}\n\nAnswer:"
        )
        return self._llm.generate(prompt).strip()

    # ── direct LLM generation (no_retrieval) ──────────────────────────────────

    def generate_no_retrieval(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate an answer without any retrieved context.

        Includes chat history so follow-up questions are coherent.

        Parameters
        ----------
        question:     Current user question.
        chat_history: Optional prior conversation turns.

        Returns
        -------
        Answer string.
        """
        history_str = _format_history(chat_history or [])
        prompt = (
            f"{GENERATOR_SYSTEM_NO_RETRIEVAL}\n\n{history_str}\n\nQuestion: {question}\n\nAnswer:"
            if history_str
            else f"{GENERATOR_SYSTEM_NO_RETRIEVAL}\n\nQuestion: {question}\n\nAnswer:"
        )
        return self._llm.generate(prompt).strip()
