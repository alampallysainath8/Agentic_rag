"""LangGraph state schema for the Agentic RAG workflow."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages:      Annotated[List[BaseMessage], operator.add]
    user_id:       str
    chat_history:  List[Dict[str, str]]

    query:             str
    expanded_query:    str            # after query expansion / typo fix
    rewritten_query:   str           # corrective rewrite by grading node
    domain:            str           # hr_domain | research_domain | general
    strategy:          str           # no_retrieval | single_retrieval | multi_hop | web_search
    retrieval_source:  str           # sql | vector | bm25 | hybrid

    context:       str
    sources:       List[str]         # document/URL sources for the answer
    citations:     List[Dict[str, Any]]  # structured citation list [{index, source, page, chunk_id}]
    answer:        str
    final_answer:  Optional[str]

    cache_hit:     bool              # True when answer served from semantic cache
    hop_count:     int               # multi-hop iteration counter

    reflection:    Dict[str, Any]
    grade:         Dict[str, Any]
    retry_count:   int
