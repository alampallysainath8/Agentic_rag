"""Custom exceptions for Agentic RAG.

Provides a structured exception type that captures the originating file,
line number, and full traceback in a logger-friendly format.
"""
from __future__ import annotations

import sys
import traceback
from typing import Optional, cast


class AgenticRAGException(Exception):
    """Exception wrapper that captures context and pretty traceback.

    Parameters
    ----------
    error_message:
        Human-friendly message or an ``Exception`` instance.
    error_details:
        Optional object with ``exc_info`` (e.g. ``sys``) or an
        ``Exception`` to extract traceback information from.
        When *None*, ``sys.exc_info()`` is used automatically.
    """

    def __init__(
        self,
        error_message: object,
        error_details: Optional[object] = None,
    ):
        norm_msg = str(error_message)

        # ── resolve exc_info ───────────────────────────────────────────────
        exc_type = exc_value = exc_tb = None
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        elif hasattr(error_details, "exc_info"):          # sys-like module
            exc_type, exc_value, exc_tb = cast(sys, error_details).exc_info()  # type: ignore[attr-defined]
        elif isinstance(error_details, BaseException):
            exc_type, exc_value, exc_tb = (
                type(error_details),
                error_details,
                error_details.__traceback__,
            )
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk to the last frame for the most relevant location
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name: str = (
            last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        )
        self.lineno: int = last_tb.tb_lineno if last_tb else -1
        self.error_message: str = norm_msg

        if exc_type and exc_tb:
            self.traceback_str: str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self) -> str:
        base = (
            f"Error in [{self.file_name}] at line [{self.lineno}] "
            f"| Message: {self.error_message}"
        )
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self) -> str:
        return (
            f"AgenticRAGException(file={self.file_name!r}, "
            f"line={self.lineno}, message={self.error_message!r})"
        )


# ── Specialised sub-classes ───────────────────────────────────────────────────

class IndexingException(AgenticRAGException):
    """Raised during document indexing / chunking."""


class RetrievalException(AgenticRAGException):
    """Raised during retrieval (vector, BM25, SQL, hybrid)."""


class SQLAgentException(AgenticRAGException):
    """Raised inside the Plan & Execute SQL agent."""


class LLMException(AgenticRAGException):
    """Raised on LLM provider failures."""


class WorkflowException(AgenticRAGException):
    """Raised inside the LangGraph workflow."""
