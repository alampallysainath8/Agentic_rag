"""
RAG Metrics — 4 metrics
========================
Metrics run on every call:
  - faithfulness       : Is the answer grounded in the retrieved context?
  - answer_relevancy   : Is the answer on-topic to the query?
  - context_relevancy  : Are the retrieved chunks relevant to the query?

Metric run only when expected_output is supplied:
  - context_recall     : Did retrieval cover all information needed to answer?

Timeout notes
-------------
- async_mode=False  → sequential internal LLM calls, no burst.
- Context truncated to _MAX_CHUNKS × _MAX_CHUNK_CHARS before sending to judge.
- inter_metric_delay seconds between each metric call prevents rate-limit bursting.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    name: str
    score: float
    reason: str | None
    passed: bool
    threshold: float = 0.5


# ── context limits ─────────────────────────────────────────────────────────────
_MAX_CHUNK_CHARS: int = 1_500   # ~375 tokens per chunk
_MAX_CHUNKS: int = 6            # 6 chunks max forwarded to the judge


def _truncate_context(chunks: list[str]) -> list[str]:
    """Trim chunks to avoid LLM timeouts."""
    trimmed = chunks[:_MAX_CHUNKS]
    return [c[:_MAX_CHUNK_CHARS] if len(c) > _MAX_CHUNK_CHARS else c for c in trimmed]


class RAGMetrics:
    """
    Full RAG metric suite — faithfulness, answer relevancy, context relevancy,
    and (optionally) context recall.

    Parameters
    ----------
    model         : model name string (e.g. ``"gpt-4o-mini"``) or a DeepEval
                    model object.
    threshold     : pass/fail threshold in [0, 1].  Default 0.5.
    include_reason: ask the judge LLM to explain each score.
    """

    def __init__(
        self,
        model: str | object = "gpt-4o-mini",
        threshold: float = 0.5,
        include_reason: bool = True,
    ) -> None:
        try:
            from deepeval.metrics import (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualRelevancyMetric,
                ContextualRecallMetric,
            )
        except ImportError as exc:
            raise ImportError("deepeval is not installed. Run: pip install deepeval") from exc

        self.threshold = threshold

        # async_mode=False → sequential LLM calls, prevents rate-limit bursting
        _kw = dict(model=model, threshold=threshold,
                   include_reason=include_reason, async_mode=False)

        self.faithfulness          = FaithfulnessMetric(**_kw)
        self.answer_relevancy      = AnswerRelevancyMetric(**_kw)
        self.contextual_relevancy  = ContextualRelevancyMetric(**_kw)
        self.contextual_recall     = ContextualRecallMetric(**_kw)

    # ── individual measures ────────────────────────────────────────────────────

    def measure_faithfulness(self, input: str, actual_output: str,
                             retrieval_context: list[str]) -> MetricResult:
        """Is the answer grounded in the retrieved context (no hallucination)?"""
        from deepeval.test_case import LLMTestCase
        tc = LLMTestCase(input=input, actual_output=actual_output,
                         retrieval_context=_truncate_context(retrieval_context))
        self.faithfulness.measure(tc)
        return MetricResult("faithfulness",
                            round(self.faithfulness.score or 0.0, 4),
                            self.faithfulness.reason,
                            (self.faithfulness.score or 0.0) >= self.threshold,
                            self.threshold)

    def measure_answer_relevancy(self, input: str,
                                 actual_output: str) -> MetricResult:
        """Is the answer relevant to the query?"""
        from deepeval.test_case import LLMTestCase
        tc = LLMTestCase(input=input, actual_output=actual_output)
        self.answer_relevancy.measure(tc)
        return MetricResult("answer_relevancy",
                            round(self.answer_relevancy.score or 0.0, 4),
                            self.answer_relevancy.reason,
                            (self.answer_relevancy.score or 0.0) >= self.threshold,
                            self.threshold)

    def measure_context_relevancy(self, input: str, actual_output: str,
                                  retrieval_context: list[str]) -> MetricResult:
        """Are the retrieved chunks relevant to the query?"""
        from deepeval.test_case import LLMTestCase
        tc = LLMTestCase(input=input, actual_output=actual_output,
                         retrieval_context=_truncate_context(retrieval_context))
        self.contextual_relevancy.measure(tc)
        return MetricResult("context_relevancy",
                            round(self.contextual_relevancy.score or 0.0, 4),
                            self.contextual_relevancy.reason,
                            (self.contextual_relevancy.score or 0.0) >= self.threshold,
                            self.threshold)

    def measure_context_recall(self, input: str, actual_output: str,
                               expected_output: str,
                               retrieval_context: list[str]) -> MetricResult:
        """Did retrieval cover all information needed to answer? (needs expected_output)"""
        from deepeval.test_case import LLMTestCase
        tc = LLMTestCase(input=input, actual_output=actual_output,
                         expected_output=expected_output,
                         retrieval_context=_truncate_context(retrieval_context))
        self.contextual_recall.measure(tc)
        return MetricResult("context_recall",
                            round(self.contextual_recall.score or 0.0, 4),
                            self.contextual_recall.reason,
                            (self.contextual_recall.score or 0.0) >= self.threshold,
                            self.threshold)

    # ── combined runner ────────────────────────────────────────────────────────

    def measure_all(
        self,
        input: str,
        actual_output: str,
        retrieval_context: list[str],
        expected_output: str | None = None,
        inter_metric_delay: float = 0.5,
    ) -> dict[str, MetricResult]:
        """
        Run all metrics sequentially.

        - faithfulness, answer_relevancy, context_relevancy always run.
        - context_recall runs only when expected_output is provided.
        - inter_metric_delay (seconds) pauses between calls to avoid rate-limit burst.
        """
        results: dict[str, MetricResult] = {}

        def _run(name: str, fn, *args):
            try:
                results[name] = fn(*args)
            except Exception as exc:
                logger.warning("%s failed: %s", name, exc)

        _run("faithfulness", self.measure_faithfulness,
             input, actual_output, retrieval_context)

        if inter_metric_delay > 0:
            time.sleep(inter_metric_delay)

        _run("answer_relevancy", self.measure_answer_relevancy,
             input, actual_output)

        if inter_metric_delay > 0:
            time.sleep(inter_metric_delay)

        _run("context_relevancy", self.measure_context_relevancy,
             input, actual_output, retrieval_context)

        if expected_output:
            if inter_metric_delay > 0:
                time.sleep(inter_metric_delay)
            _run("context_recall", self.measure_context_recall,
                 input, actual_output, expected_output, retrieval_context)

        return results
