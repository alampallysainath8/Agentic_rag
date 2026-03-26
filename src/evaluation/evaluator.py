"""
RAG Evaluator Service — simple, single-metric mode
====================================================
Evaluates a query/answer/context triple using Faithfulness (DeepEval).
One metric, sequential calls, no batch logic — easy to verify works first.

To add more metrics later, enable them in rag_metrics.py.
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ── Set DeepEval env vars BEFORE deepeval is imported anywhere ─────────────────
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_SECONDS_OVERRIDE", "300")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    eval_id: str
    timestamp: str
    query: str
    answer: str
    context_chunks: list[str]
    expected_output: str | None
    scores: dict[str, float]
    reasons: dict[str, str | None]
    passed: dict[str, bool]
    threshold: float
    model: str
    overall_pass: bool = field(init=False)

    def __post_init__(self) -> None:
        self.overall_pass = all(self.passed.values()) if self.passed else False

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_id":         self.eval_id,
            "timestamp":       self.timestamp,
            "query":           self.query,
            "answer":          self.answer,
            "context_chunks":  self.context_chunks,
            "expected_output": self.expected_output,
            "scores":          self.scores,
            "reasons":         self.reasons,
            "passed":          self.passed,
            "threshold":       self.threshold,
            "model":           self.model,
            "overall_pass":    self.overall_pass,
        }


class RAGEvaluatorService:
    """
    Simple evaluator — faithfulness only.

    Parameters
    ----------
    model     : judge model name (default: env EVAL_MODEL or gpt-4o-mini).
    threshold : pass/fail threshold (default: 0.5).
    """

    _instance: "RAGEvaluatorService | None" = None

    def __init__(self, model: str | None = None, threshold: float = 0.5) -> None:
        self.model     = model or os.getenv("EVAL_MODEL", "gpt-4o-mini")
        self.threshold = threshold
        self._metrics  = None   # lazy-loaded on first call

    @classmethod
    def get_instance(
        cls, model: str | None = None, threshold: float = 0.5
    ) -> "RAGEvaluatorService":
        resolved = model or os.getenv("EVAL_MODEL", "gpt-4o-mini")
        if (
            cls._instance is None
            or cls._instance.model != resolved
            or cls._instance.threshold != threshold
        ):
            cls._instance = cls(model=resolved, threshold=threshold)
        return cls._instance

    def _get_metrics(self):
        if self._metrics is None:
            from .rag_metrics import RAGMetrics
            self._metrics = RAGMetrics(
                model=self.model,
                threshold=self.threshold,
                include_reason=True,
            )
        return self._metrics

    def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: list[str],
        expected_output: str | None = None,
    ) -> EvaluationResult:
        """Run faithfulness evaluation and persist result."""
        if not context_chunks:
            raise ValueError("context_chunks must be non-empty — faithfulness requires retrieved passages.")

        metrics = self._get_metrics()
        raw = metrics.measure_all(
            input=query,
            actual_output=answer,
            retrieval_context=context_chunks,
            expected_output=expected_output,
        )

        scores:  dict[str, float]      = {n: r.score  for n, r in raw.items()}
        reasons: dict[str, str | None] = {n: r.reason for n, r in raw.items()}
        passed:  dict[str, bool]       = {n: r.passed for n, r in raw.items()}

        result = EvaluationResult(
            eval_id        = str(uuid.uuid4()),
            timestamp      = datetime.now(timezone.utc).isoformat(),
            query          = query,
            answer         = answer,
            context_chunks = context_chunks,
            expected_output= expected_output,
            scores         = scores,
            reasons        = reasons,
            passed         = passed,
            threshold      = self.threshold,
            model          = str(self.model),
        )

        try:
            from .history import append_result
            append_result(result.to_dict())
        except Exception as exc:
            logger.warning("Could not persist evaluation result: %s", exc)

        return result
