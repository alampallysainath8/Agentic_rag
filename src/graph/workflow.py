from __future__ import annotations

import json
import os
from typing import Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from .state import AgentState
from src.agents.decision_agent import DecisionAgent
from src.agents.domain_agent import DomainAgent
from src.agents.generator_agent import GeneratorAgent
from src.agents.grader_agent import GraderAgent
from src.agents.prompt_manager import MULTIHOP_FINAL_PROMPT, MULTIHOP_PLAN_PROMPT, QUERY_EXPANSION_PROMPT
from src.agents.reflection_agent import ReflectionAgent
from src.agents.router_agent import RouterAgent
from src.cache.semantic_cache import SemanticCache
from src.config import load_config
from src.llm import get_llm_provider
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sql_agent import SQLAgent
from src.utils.custom_exception import WorkflowException
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MAX_RETRIES = 2

# Module-level checkpointer — single instance shared across all workflow builds
_checkpointer = InMemorySaver()


# ── citation helpers ─────────────────────────────────────────────────────────

def _format_docs_with_citations(docs):
    """Format a list of Documents into a numbered context string + citations list.

    Returns
    -------
    context_str : str   — numbered passages for the generator prompt
    citations   : list  — [{index, source, page, chunk_id}]
    """
    parts = []
    citations = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("source", meta.get("filename", meta.get("file_name", "unknown")))
        page   = meta.get("page", meta.get("page_number", ""))
        chunk  = meta.get("chunk_id", "")
        meta_str = f"source: {source}" + (f", page {page}" if page else "")
        parts.append(f"[{i}] ({meta_str})\n{doc.page_content}")
        citations.append({"index": i, "source": source, "page": str(page), "chunk_id": chunk})
    return "\n\n".join(parts), citations


def make_cache_check_node(cache: SemanticCache):
    def cache_check_node(state: AgentState) -> AgentState:
        msgs = state.get("messages", [])
        query = ""
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                query = m.content
                break
        query = query or state.get("query", "")
        cached = cache.lookup(query)
        if cached:
            ai_msg = AIMessage(content=cached)
            logger.info("cache_check_node: HIT for '%s'", query[:60])
            return {**state, "query": query, "final_answer": cached, "cache_hit": True, "messages": [ai_msg]}
        return {**state, "query": query, "cache_hit": False}
    return cache_check_node


def make_expand_query_node(lc_llm):
    def expand_query_node(state: AgentState) -> AgentState:
        query = state.get("query", "")
        try:
            resp = lc_llm.invoke(QUERY_EXPANSION_PROMPT.format(query=query))
            expanded = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            if expanded and expanded != query:
                logger.info("expand_query_node: '%s' expanded", query[:50])
            else:
                expanded = query
        except Exception as exc:
            logger.warning("expand_query_node failed (%s) — using original query", exc)
            expanded = query
        return {**state, "expanded_query": expanded}
    return expand_query_node


def make_domain_classify_node(domain_agent: DomainAgent):
    """Runs domain classification on the expanded query.

    Populates state['domain'] with one of:
      hr_domain | research_domain | general
    """
    def domain_classify_node(state: AgentState) -> AgentState:
        query = state.get("expanded_query") or state.get("query", "")
        result = domain_agent.classify(query)
        logger.info(
            "domain_classify_node: '%s' \u2192 domain=%s (%.0f%% confident)",
            query[:60], result.domain, result.confidence * 100,
        )
        return {**state, "domain": result.domain}
    return domain_classify_node


def make_decision_node(decision_agent):
    def decision_node(state: AgentState) -> AgentState:
        query = state.get("expanded_query") or state.get("query", "")
        domain = state.get("domain", "hr_domain")
        decision = decision_agent.decide(query, domain=domain)
        logger.info(
            "decision_node: '%s' [domain=%s] -> strategy=%s",
            query[:60], domain, decision.strategy,
        )
        return {**state, "strategy": decision.strategy}
    return decision_node


def make_no_retrieval_node(generator):
    def no_retrieval_node(state: AgentState) -> AgentState:
        query = state.get("expanded_query") or state.get("query", "")
        chat_history = state.get("chat_history", [])
        answer = generator.generate_no_retrieval(query, chat_history)
        ai_msg = AIMessage(content=answer)
        new_chat = (chat_history or []) + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        return {**state, "context": "", "answer": answer, "final_answer": answer,
                "messages": [ai_msg], "chat_history": new_chat}
    return no_retrieval_node


def make_route_node(router_agent):
    def route_node(state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state.get("expanded_query") or state.get("query", "")
        domain = state.get("domain", "hr_domain")
        strategy = state.get("strategy", "single_retrieval")
        source = router_agent.route(query, domain=domain, strategy=strategy)
        logger.info("route_node: '%s' [domain=%s] -> %s", query[:60], domain, source)
        return {**state, "retrieval_source": source}
    return route_node


def make_sql_node(sql_agent):
    def sql_node(state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state.get("expanded_query") or state.get("query", "")
        result = sql_agent.run(query)
        ctx = result["final_answer"]
        logger.info("sql_node: %d rows, success=%s", len(result.get("rows", [])), result["success"])
        return {**state, "context": ctx, "sources": ["sqlite"], "retrieval_source": "sql"}
    return sql_node


def make_hybrid_node(hybrid_ret):
    def hybrid_node(state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state.get("expanded_query") or state.get("query", "")
        docs = hybrid_ret.retrieve(query, k=5)
        ctx, citations = _format_docs_with_citations(docs)
        sources = list(dict.fromkeys(c["source"] for c in citations))
        logger.info("hybrid_node: %d docs (reranked=%s)", len(docs), hybrid_ret._use_rerank)
        return {**state, "context": ctx, "sources": sources, "citations": citations, "retrieval_source": "hybrid"}
    return hybrid_node


def make_web_search_node(tavily_api_key: str, max_results: int = 5):
    def web_search_node(state: AgentState) -> AgentState:
        query = state.get("expanded_query") or state.get("query", "")
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_api_key)
            results = client.search(query, max_results=max_results)
            snippets = [r["content"] for r in results.get("results", [])]
            sources = [r["url"] for r in results.get("results", [])]
            ctx = "\n\n".join(snippets)
            logger.info("web_search_node: %d results from Tavily", len(snippets))
        except Exception as exc:
            logger.error("web_search_node: Tavily failed (%s)", exc)
            ctx, sources = "", []
        return {**state, "context": ctx, "sources": sources, "retrieval_source": "web_search"}
    return web_search_node


def make_multihop_node(llm, sql_agent, hybrid_ret):
    """Execute a multi-step reasoning plan using only sql and hybrid tools.

    Each step in the LLM-generated plan uses:
      - 'hybrid' — BM25 + vector search over documents (any domain)
      - 'sql'    — structured query against HR tables (hr_domain only)
    web_search is NOT available inside multihop.
    """
    def multihop_node(state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state.get("expanded_query") or state.get("query", "")
        domain = state.get("domain", "hr_domain")

        plan_raw = llm.invoke(MULTIHOP_PLAN_PROMPT.format(query=query, domain=domain))
        plan_text = plan_raw.content if hasattr(plan_raw, "content") else str(plan_raw)
        try:
            plan: List[Dict] = json.loads(plan_text)
            if not isinstance(plan, list) or not plan:
                raise ValueError("empty plan")
        except Exception as exc:
            logger.warning("multihop_node: plan parse failed (%s) — fallback", exc)
            plan = [{"step": 1, "tool": "hybrid", "task": query}]

        results: Dict[str, str] = {}
        contexts: List[str] = []
        all_sources: List[str] = []
        all_citations: List[Dict] = []
        citation_offset = 0

        for step in plan:
            tool = step.get("tool", "hybrid")
            task = step.get("task", query)
            label = step.get("step", "?")
            prev = "\nPrevious results: " + str(results) if results else ""
            enriched = task + prev

            # Domain guard: research_domain must never use sql
            if tool == "sql" and domain == "research_domain":
                logger.warning(
                    "multihop_node: research_domain step %s tried 'sql' -- overriding to hybrid",
                    label,
                )
                tool = "hybrid"

            if tool == "sql":
                res = sql_agent.run(enriched)
                output = res.get("final_answer", "")
                results["sql"] = output
                contexts.append("[SQL - step {}]\n{}".format(label, output))
                all_sources.append("sqlite")
            else:  # hybrid (default)
                docs = hybrid_ret.retrieve(enriched, k=5)
                step_ctx, step_cites = _format_docs_with_citations(docs)
                # Re-index citations to be globally unique
                for c in step_cites:
                    c["index"] = c["index"] + citation_offset
                citation_offset += len(step_cites)
                results["hybrid"] = step_ctx
                contexts.append("[Hybrid - step {}]\n{}".format(label, step_ctx))
                all_sources.extend(c["source"] for c in step_cites)
                all_citations.extend(step_cites)

        logger.info("multihop_node: %d/%d steps completed", len(contexts), len(plan))

        combined = "\n\n".join(contexts)
        final_raw = llm.invoke(MULTIHOP_FINAL_PROMPT.format(context=combined, query=query))
        answer = final_raw.content.strip() if hasattr(final_raw, "content") else str(final_raw).strip()

        return {
            **state,
            "context": combined,
            "answer": answer,
            "sources": list(dict.fromkeys(all_sources)),
            "citations": all_citations,
            "retrieval_source": "multihop",
        }
    return multihop_node


def make_generator_node(generator):
    def generator_node(state: AgentState) -> AgentState:
        query = state.get("expanded_query") or state.get("query", "")
        context = state.get("context", "")
        chat_history = state.get("chat_history", [])
        answer = generator.generate(context, query, chat_history)
        ai_msg = AIMessage(content=answer)
        new_chat = (chat_history or []) + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        return {**state, "answer": answer, "messages": [ai_msg], "chat_history": new_chat}
    return generator_node


def make_reflection_node(reflector):
    def reflection_node(state: AgentState) -> AgentState:
        result = reflector.reflect(
            question=state.get("query", ""),
            context=state.get("context", ""),
            answer=state.get("answer", ""),
        )
        logger.info("reflection_node: verdict=%s", result.verdict)
        return {**state, "reflection": result.model_dump()}
    return reflection_node


def make_grading_node(grader, max_retries: int = MAX_RETRIES):
    def grading_node(state: AgentState) -> AgentState:
        reflection = state.get("reflection", {})
        retry_count = state.get("retry_count", 0)
        query = state.get("query", "")
        answer = state.get("answer", "")

        if reflection.get("verdict") == "PASS" or retry_count >= max_retries:
            if retry_count >= max_retries:
                logger.warning("grading_node: max retries reached — accepting answer")
            grade_result = grader.grade(query, answer)
            return {**state, "grade": grade_result.model_dump(), "final_answer": answer}

        grade_result = grader.grade(query, answer)
        logger.info("grading_node: score=%d verdict=%s retry=%d",
                    grade_result.score, grade_result.verdict, retry_count)

        if grade_result.verdict == "GOOD":
            return {**state, "grade": grade_result.model_dump(), "final_answer": answer}

        issues = grade_result.rationale + " " + reflection.get("issues", "")
        new_query = grader.rewrite(query, issues)
        logger.info("grading_node: rewriting for retry %d: '%s'", retry_count + 1, new_query[:60])

        return {**state, "grade": grade_result.model_dump(), "rewritten_query": new_query,
                "retry_count": retry_count + 1, "final_answer": None}
    return grading_node


def make_cache_store_node(cache: SemanticCache):
    def cache_store_node(state: AgentState) -> AgentState:
        query = state.get("query", "")
        answer = state.get("final_answer") or state.get("answer", "")
        if query and answer and not state.get("cache_hit"):
            cache.store(query, answer)
        return state
    return cache_store_node


def route_after_cache(state: AgentState) -> str:
    return "end" if state.get("cache_hit") else "expand_query"


def route_after_decision(state: AgentState) -> str:
    strategy = state.get("strategy", "single_retrieval")
    if strategy == "no_retrieval":
        return "no_retrieval"
    if strategy == "multi_hop":
        return "multi_hop"
    if strategy == "web_search":
        return "web_search"
    return "single_retrieval"


def route_after_route_node(state: AgentState) -> str:
    """Route to sql_node or hybrid_node based on the router's decision."""
    source = state.get("retrieval_source", "hybrid")
    # Normalise: any legacy vector/bm25 values fall back to hybrid
    return source if source in ("sql", "hybrid") else "hybrid"


def route_after_grading(state: AgentState) -> Literal["accept", "retry"]:
    return "accept" if state.get("final_answer") is not None else "retry"


def build_workflow(cfg_path: str | None = None):
    try:
        config = load_config(cfg_path)
        llm = get_llm_provider(config.llm)
        lc_llm = llm.get_langchain_llm()
    except Exception as exc:
        raise WorkflowException("build_workflow: LLM init failed", exc) from exc

    cache = SemanticCache(config)
    domain_agent = DomainAgent(lc_llm)
    decision_agent = DecisionAgent(lc_llm)
    router_agent = RouterAgent(lc_llm)
    generator = GeneratorAgent(llm)
    reflector = ReflectionAgent(lc_llm)
    grader = GraderAgent(lc_llm)
    sql_agent = SQLAgent.from_config(config, lc_llm)
    hybrid_ret = HybridRetriever(config)
    try:
        hybrid_ret.load_bm25_from_store()
    except Exception:
        logger.info("BM25 corpus empty — hybrid will fall back to dense-only.")

    tavily_key = os.getenv(config.web_search.get("tavily_api_key_env", "TAVILY_API_KEY"), "")
    max_web_results = config.web_search.get("max_results", 5)
    max_retries = config.reflection.get("max_retries", MAX_RETRIES)

    cache_check_fn = make_cache_check_node(cache)
    expand_fn = make_expand_query_node(lc_llm)
    domain_classify_fn = make_domain_classify_node(domain_agent)
    decision_fn = make_decision_node(decision_agent)
    no_ret_fn = make_no_retrieval_node(generator)
    route_fn = make_route_node(router_agent)
    sql_fn = make_sql_node(sql_agent)
    hybrid_fn = make_hybrid_node(hybrid_ret)
    web_search_fn = make_web_search_node(tavily_key, max_web_results)
    multihop_fn = make_multihop_node(lc_llm, sql_agent, hybrid_ret)
    generator_fn = make_generator_node(generator)
    reflection_fn = make_reflection_node(reflector)
    grading_fn = make_grading_node(grader, max_retries)
    cache_store_fn = make_cache_store_node(cache)

    g = StateGraph(AgentState)

    g.add_node("cache_check_node", cache_check_fn)
    g.add_node("expand_query_node", expand_fn)
    g.add_node("domain_classify_node", domain_classify_fn)
    g.add_node("decision_node", decision_fn)
    g.add_node("no_retrieval_node", no_ret_fn)
    g.add_node("route_node", route_fn)
    g.add_node("sql_node", sql_fn)
    g.add_node("hybrid_node", hybrid_fn)
    g.add_node("web_search_node", web_search_fn)
    g.add_node("multihop_node", multihop_fn)
    g.add_node("generator_node", generator_fn)
    g.add_node("reflection_node", reflection_fn)
    g.add_node("grading_node", grading_fn)
    g.add_node("cache_store_node", cache_store_fn)

    g.set_entry_point("cache_check_node")

    g.add_conditional_edges(
        "cache_check_node",
        route_after_cache,
        {"end": END, "expand_query": "expand_query_node"},
    )
    g.add_edge("expand_query_node", "domain_classify_node")
    g.add_edge("domain_classify_node", "decision_node")
    g.add_conditional_edges(
        "decision_node",
        route_after_decision,
        {
            "no_retrieval": "no_retrieval_node",
            "single_retrieval": "route_node",
            "multi_hop": "multihop_node",
            "web_search": "web_search_node",
        },
    )

    g.add_edge("no_retrieval_node", "cache_store_node")

    g.add_conditional_edges(
        "route_node",
        route_after_route_node,
        {"sql": "sql_node", "hybrid": "hybrid_node"},
    )

    for node in ("sql_node", "hybrid_node", "web_search_node"):
        g.add_edge(node, "generator_node")

    g.add_edge("multihop_node", "reflection_node")
    g.add_edge("generator_node", "reflection_node")
    g.add_edge("reflection_node", "grading_node")

    g.add_conditional_edges(
        "grading_node",
        route_after_grading,
        {"accept": "cache_store_node", "retry": "route_node"},
    )

    g.add_edge("cache_store_node", END)

    return g.compile(checkpointer=_checkpointer)
