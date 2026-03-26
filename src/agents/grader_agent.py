
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.utils.logger import setup_logger
from src.agents.prompt_manager import GRADER_SYSTEM_PROMPT, GRADER_REWRITE_SYSTEM_PROMPT

logger = setup_logger(__name__)


# ── Pydantic schemas ────────────────────────────────────────────────

class GradeResult(BaseModel):
    """Structured grading output."""

    score: int = Field(
        ge=1, le=5,
        description="Quality score: 1 (terrible) to 5 (excellent).",
    )
    rationale: str = Field(description="One-sentence explanation of the score.")
    verdict: Literal["GOOD", "BAD"] = Field(
        description="GOOD if score >= 3, BAD if score < 3.",
    )


class GraderAgent:
    """Grades answer quality and produces a rewritten query when grade is BAD.

    Parameters
    ----------
    llm: LangChain ChatModel (must support .invoke() and ideally .with_structured_output()).
    """

    def __init__(self, llm: Any):
        self._llm = llm
        try:
            self._structured_llm = llm.with_structured_output(GradeResult)
        except Exception as exc:
            logger.warning("Structured output unavailable (%s) — JSON fallback.", exc)
            self._structured_llm = None

    # ── grade ─────────────────────────────────────────────────────────

    def grade(self, question: str, answer: str) -> GradeResult:
        """Grade the *answer* given *question*.

        Returns
        -------
        GradeResult(score, rationale, verdict)
        """
        user_content = f"QUESTION: {question}\n\nANSWER: {answer}"

        if self._structured_llm is not None:
            try:
                result: GradeResult = self._structured_llm.invoke([
                    SystemMessage(content=GRADER_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ])
                logger.info(
                    "GraderAgent: score=%d verdict=%s for '%s'",
                    result.score, result.verdict, question[:60],
                )
                return result
            except Exception as exc:
                logger.warning("GraderAgent structured call failed (%s) — fallback", exc)

        return self._grade_fallback(question, answer)

    def _grade_fallback(self, question: str, answer: str) -> GradeResult:
        prompt = (
            f"{GRADER_SYSTEM_PROMPT}\n\nQUESTION: {question}\n\nANSWER: {answer}\n\n"
            'Respond with JSON only: {"score": <int>, "rationale": "<one sentence>", "verdict": "GOOD" or "BAD"}'
        )
        try:
            raw = self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            data = json.loads(raw)
            score = int(data.get("score", 3))
            return GradeResult(
                score=score,
                rationale=data.get("rationale", ""),
                verdict=data.get("verdict", "GOOD" if score >= 3 else "BAD"),
            )
        except Exception:
            return GradeResult(score=3, rationale="Parse error", verdict="GOOD")

    # ── rewrite ──────────────────────────────────────────────────────

    def rewrite(self, query: str, issues: str = "") -> str:
        """Rewrite *query* to retrieve better evidence.

        Parameters
        ----------
        query:  Original query that produced a BAD answer.
        issues: Rationale from the grade (used as rewrite context).

        Returns
        -------
        Improved query string.
        """
        prompt = (
            f"ORIGINAL QUERY: {query}\n"
            f"GRADING ISSUES: {issues}\n"
            "IMPROVED QUERY:"
        )
        try:
            result = self._llm.invoke([
                SystemMessage(content=GRADER_REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            new_query = result.content.strip().strip('"')
        except Exception as exc:
            logger.warning("GraderAgent.rewrite failed: %s", exc)
            return query

        logger.info("GraderAgent: rewrite '%s' → '%s'", query[:50], new_query[:50])
        return new_query
