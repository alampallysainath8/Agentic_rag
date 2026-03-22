"""
Convenience re-export: import the compiled LangGraph workflow from here.

Usage:
    from src.langgraph_adapter import build_workflow
    app = build_workflow()
    result = app.invoke({"query": "What is the supplier turnover?"})
"""
from .graph.workflow import build_workflow  # noqa: F401

__all__ = ["build_workflow"]

