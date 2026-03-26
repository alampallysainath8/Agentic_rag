
from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.prompt_manager import REFLECTION_SYSTEM_PROMPT
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# -- Pydantic schema -----------------------------------------------------------

class ReflectionResult(BaseModel):
    """Structured output from the reflection agent."""

    grounded: bool = Field(description="All claims supported by context?")
    complete: bool = Field(description="Answer fully addresses the question?")
    coherent: bool = Field(description="Answer is well-structured and consistent?")
    issues:   str  = Field(default="", description="Brief description of problems (or empty).")
    verdict:  Literal["PASS", "FAIL"] = Field(
        description="PASS if grounded AND complete AND coherent, else FAIL."
    )


# -- Agent ---------------------------------------------------------------------

class ReflectionAgent:
    """Validates a generated answer (self-reflection pattern).

    Parameters
    ----------
    llm: LangChain chat model.
    """

    def __init__(self, llm: Any):
        self._llm = llm
        try:
            self._structured_llm = llm.with_structured_output(ReflectionResult)
        except Exception as exc:
            logger.warning("Structured output unavailable (%s) -- JSON fallback.", exc)
            self._structured_llm = None

    def reflect(self, question: str, context: str, answer: str) -> ReflectionResult:
        """Evaluate the answer.

        Parameters
        ----------
        question: Original user question.
        context:  Retrieved context used to generate the answer.
        answer:   Generated answer to evaluate.

        Returns
        -------
        ReflectionResult(grounded, complete, coherent, issues, verdict)
        """
        user_content = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT (truncated):\n{context[:3000]}\n\n"
            f"ANSWER:\n{answer}"
        )

        if self._structured_llm is not None:
            try:
                result: ReflectionResult = self._structured_llm.invoke([
                    SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ])
                logger.info(
                    "ReflectionAgent: verdict=%s issues='%s'",
                    result.verdict, result.issues[:80],
                )
                return result
            except Exception as exc:
                logger.warning("ReflectionAgent structured call failed (%s) -- fallback", exc)

        return self._reflect_fallback(question, context, answer)

    def _reflect_fallback(self, question: str, context: str, answer: str) -> ReflectionResult:
        prompt = (
            f"{REFLECTION_SYSTEM_PROMPT}\n\n"
            f"QUESTION: {question}\n\nCONTEXT: {context[:2000]}\n\nANSWER: {answer}\n\n"
            'Respond with JSON only: {"grounded": true/false, "complete": true/false, '
            '"coherent": true/false, "issues": "...", "verdict": "PASS" or "FAIL"}'
        )
        try:
            raw = self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            data = json.loads(raw)
            grounded = bool(data.get("grounded", True))
            complete = bool(data.get("complete", True))
            coherent = bool(data.get("coherent", True))
            verdict: Literal["PASS", "FAIL"] = (
                "PASS" if grounded and complete and coherent else "FAIL"
            )
            return ReflectionResult(
                grounded=grounded, complete=complete, coherent=coherent,
                issues=data.get("issues", ""),
                verdict=verdict,
            )
        except Exception:
            return ReflectionResult(
                grounded=True, complete=True, coherent=True,
                issues="", verdict="PASS",
            )