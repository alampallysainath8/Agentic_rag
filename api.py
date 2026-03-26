import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage as _HumanMessage, AIMessage as _AIMessage

from src.config import load_config
from src.graph.workflow import build_workflow
from src.indexing.pipeline import IndexingPipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_workflow = None
_pipeline = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = IndexingPipeline(config)
    return _pipeline


class QueryRequest(BaseModel):
    query: str
    session_id: str = ""
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    expanded_query: str = ""
    answer: str
    context: str = ""          # raw retrieval context text from the workflow state
    retrieval_source: str = ""
    domain: str = ""
    sources: List[str] = []
    citations: List[dict] = []
    session_id: str
    cache_hit: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.session_id:
        request = request.model_copy(update={"session_id": str(uuid.uuid4())})

    session_id = request.session_id
    workflow = get_workflow()
    # thread_id ties this invocation to its LangGraph checkpoint
    thread_config = {"configurable": {"thread_id": session_id}}

    # Only send the new user message — InMemorySaver restores prior state
    # (messages accumulate via operator.add reducer; chat_history persists in checkpoint)
    initial_state = {
        "messages":       [_HumanMessage(content=request.query)],
        "query":          request.query,
        "expanded_query": "",
        "retry_count":    0,
        "cache_hit":      False,
        "sources":        [],
    }

    try:
        result = workflow.invoke(initial_state, config=thread_config)
    except Exception as exc:
        logger.exception("Workflow error")
        raise HTTPException(status_code=500, detail=str(exc))

    answer = result.get("final_answer") or result.get("answer", "")

    return QueryResponse(
        query=request.query,
        expanded_query=result.get("expanded_query", ""),
        answer=answer,
        context=result.get("context", ""),
        retrieval_source=result.get("retrieval_source", ""),
        domain=result.get("domain", ""),
        sources=list(dict.fromkeys(result.get("sources", []))),
        citations=result.get("citations", []),
        session_id=session_id,
        cache_hit=result.get("cache_hit", False),
    )


@app.post("/index")
async def index_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    upload_dir = Path("uploaded_docs")
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())

    def _run_index(path: str):
        try:
            result = get_pipeline().run(path)
            if result.get("skipped"):
                logger.info("File '%s' already indexed — skipped.", file.filename)
            else:
                logger.info("Indexed '%s': %d chunks", file.filename, result["num_chunks"])
        except Exception as exc:
            logger.error("Indexing failed for '%s': %s", file.filename, exc)

    background_tasks.add_task(_run_index, str(dest))
    return {"message": f"Indexing started for '{file.filename}'"}


# ── Evaluation endpoints ──────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    query: str
    answer: str
    context_chunks: List[str]
    expected_output: Optional[str] = None
    model: Optional[str] = None          # defaults to EVAL_MODEL env var or gpt-4o-mini
    threshold: float = 0.5


class MetricDetail(BaseModel):
    score: float
    reason: Optional[str] = None
    passed: bool
    threshold: float


class EvaluateResponse(BaseModel):
    eval_id: str
    timestamp: str
    query: str
    answer: str
    scores: Dict[str, float]
    reasons: Dict[str, Optional[str]]
    passed: Dict[str, bool]
    overall_pass: bool
    threshold: float
    model: str


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate(request: EvaluateRequest):
    """Evaluate a RAG query/answer pair — Faithfulness metric."""
    from src.evaluation.evaluator import RAGEvaluatorService  # lazy import

    if not request.context_chunks:
        raise HTTPException(
            status_code=422,
            detail="context_chunks must contain at least one retrieved passage.",
        )

    svc = RAGEvaluatorService(
        model=request.model or None,
        threshold=request.threshold,
    )
    try:
        result = svc.evaluate(
            query=request.query,
            answer=request.answer,
            context_chunks=request.context_chunks,
            expected_output=request.expected_output,
        )
    except Exception as exc:
        logger.exception("Evaluation error")
        raise HTTPException(status_code=500, detail=str(exc))

    return EvaluateResponse(
        eval_id=result.eval_id,
        timestamp=result.timestamp,
        query=result.query,
        answer=result.answer,
        scores=result.scores,
        reasons=result.reasons,
        passed=result.passed,
        overall_pass=result.overall_pass,
        threshold=result.threshold,
        model=result.model,
    )


@app.get("/evaluations", tags=["Evaluation"])
def get_evaluations(limit: int = 100):
    """Return the most-recent stored evaluation results."""
    from src.evaluation.history import load_history  # lazy import
    return {"evaluations": load_history(limit=limit)}


@app.delete("/evaluations", tags=["Evaluation"])
def clear_evaluations():
    """Delete all stored evaluation records."""
    from src.evaluation.history import clear_history  # lazy import
    clear_history()
    return {"cleared": True}


# ── Batch evaluation (Excel-friendly) ────────────────────────────────────────

class BatchEvalItem(BaseModel):
    row_id: Optional[str] = None
    query: str
    answer: str
    context_chunks: List[str]           # list of context strings
    expected_output: Optional[str] = None  # enables context_recall when provided


class BatchEvalRequest(BaseModel):
    items: List[BatchEvalItem]
    model: Optional[str] = None
    threshold: float = 0.5
    delay_between_rows: float = 1.0     # seconds to wait between rows


@app.post("/evaluate/batch", tags=["Evaluation"])
def evaluate_batch(request: BatchEvalRequest):
    """
    Evaluate multiple Q/A pairs in one call.
    Each item follows the same schema as ``/evaluate``.
    A short delay between rows avoids rate-limit bursting.
    Returns a list of per-row results plus aggregate averages.
    """
    import time
    from src.evaluation.evaluator import RAGEvaluatorService

    if len(request.items) > 30:
        raise HTTPException(status_code=422, detail="Max 30 items per batch request.")

    svc = RAGEvaluatorService(model=request.model or None, threshold=request.threshold)
    results = []
    metric_sums: dict = {}
    metric_cnts: dict = {}

    for idx, item in enumerate(request.items):
        if idx > 0 and request.delay_between_rows > 0:
            time.sleep(request.delay_between_rows)
        try:
            res = svc.evaluate(
                query=item.query,
                answer=item.answer,
                context_chunks=item.context_chunks,
                expected_output=item.expected_output,
            )
            row = {
                "row_id":       item.row_id or str(idx + 1),
                "query":        item.query,
                "answer":       item.answer,
                "scores":       res.scores,
                "reasons":      res.reasons,
                "passed":       res.passed,
                "overall_pass": res.overall_pass,
                "error":        None,
            }
            for mk, sc in res.scores.items():
                metric_sums[mk] = metric_sums.get(mk, 0.0) + sc
                metric_cnts[mk] = metric_cnts.get(mk, 0) + 1
        except Exception as exc:
            logger.warning("Batch row %s failed: %s", idx, exc)
            row = {
                "row_id":       item.row_id or str(idx + 1),
                "query":        item.query,
                "answer":       item.answer,
                "scores":       {},
                "reasons":      {},
                "passed":       {},
                "overall_pass": False,
                "error":        str(exc),
            }
        results.append(row)

    n_pass = sum(1 for r in results if r["overall_pass"] and not r["error"])
    n_err  = sum(1 for r in results if r["error"])
    avgs   = {mk: round(metric_sums[mk] / metric_cnts[mk], 4)
              for mk in metric_sums if metric_cnts[mk]}

    return {
        "total":           len(results),
        "passed":          n_pass,
        "failed":          len(results) - n_pass - n_err,
        "errored":         n_err,
        "metric_averages": avgs,
        "results":         results,
    }


# ── Pipeline batch evaluation ─────────────────────────────────────────────────
# Each row only needs a query (+ optional expected_output).
# The endpoint calls the RAG workflow to produce answer + context,
# then runs the evaluation metrics on those real outputs.

class PipelineBatchItem(BaseModel):
    row_id: Optional[str] = None
    query: str
    expected_output: Optional[str] = None  # ground-truth — enables context_recall


class PipelineBatchRequest(BaseModel):
    items: List[PipelineBatchItem]
    top_k: int = 5
    model: Optional[str] = None
    threshold: float = 0.5
    delay_between_rows: float = 2.0


@app.post("/evaluate/pipeline-batch", tags=["Evaluation"])
def evaluate_pipeline_batch(request: PipelineBatchRequest):
    """
    End-to-end pipeline evaluation.

    For each item the endpoint:
      1. Runs the RAG workflow on the *query* to get a real *answer* + *context*.
      2. Splits the context string into chunks.
      3. Evaluates the answer against the context with all configured metrics.

    Excel input format (pipeline mode):
      • ``query``           — the question to ask the RAG system  (required)
      • ``expected_output`` — ground-truth answer, enables Context Recall  (optional)

    No ``answer`` or ``context`` columns needed — they come from the pipeline.
    """
    import time as _time

    if len(request.items) > 30:
        raise HTTPException(status_code=422, detail="Max 30 items per pipeline-batch request.")

    from src.evaluation.evaluator import RAGEvaluatorService
    workflow = get_workflow()
    svc      = RAGEvaluatorService(model=request.model or None, threshold=request.threshold)

    results      = []
    metric_sums: dict = {}
    metric_cnts: dict = {}

    def _context_to_chunks(ctx: str) -> list[str]:
        """Split raw context string into a list of non-empty chunks."""
        if not ctx:
            return []
        # Try double-newline split first, then single newline
        chunks = [c.strip() for c in ctx.split("\n\n") if c.strip()]
        if not chunks:
            chunks = [c.strip() for c in ctx.split("\n") if c.strip()]
        if not chunks:
            chunks = [ctx.strip()]
        return chunks

    for idx, item in enumerate(request.items):
        if idx > 0 and request.delay_between_rows > 0:
            _time.sleep(request.delay_between_rows)

        # Step 1 — run the RAG pipeline
        try:
            thread_id = str(uuid.uuid4())
            thread_cfg = {"configurable": {"thread_id": thread_id}}
            initial_state = {
                "messages":       [_HumanMessage(content=item.query)],
                "query":          item.query,
                "expanded_query": "",
                "retry_count":    0,
                "cache_hit":      False,
                "sources":        [],
            }
            wf_result = workflow.invoke(initial_state, config=thread_cfg)
            answer    = wf_result.get("final_answer") or wf_result.get("answer", "")
            ctx_str   = wf_result.get("context", "")
            chunks    = _context_to_chunks(ctx_str)

            # Fallback: use citation sources as minimal context placeholder
            if not chunks:
                chunks = [c.get("source", "") for c in wf_result.get("citations", [])
                          if c.get("source")]

            if not answer:
                raise ValueError("Pipeline returned an empty answer.")
            if not chunks:
                raise ValueError("Pipeline returned no retrieval context — cannot evaluate faithfulness.")

        except Exception as exc:
            logger.warning("Pipeline call failed for row %s: %s", idx, exc)
            results.append({
                "row_id":          item.row_id or str(idx + 1),
                "query":           item.query,
                "pipeline_answer": "",
                "pipeline_context":  "",
                "scores":          {},
                "reasons":         {},
                "passed":          {},
                "overall_pass":    False,
                "error":           f"Pipeline error: {exc}",
            })
            continue

        # Step 2 — evaluate the real answer + context
        try:
            res = svc.evaluate(
                query=item.query,
                answer=answer,
                context_chunks=chunks,
                expected_output=item.expected_output,
            )
            row = {
                "row_id":          item.row_id or str(idx + 1),
                "query":           item.query,
                "pipeline_answer": answer,
                "pipeline_context":  ctx_str[:500],   # preview only
                "retrieval_source":wf_result.get("retrieval_source", ""),
                "scores":          res.scores,
                "reasons":         res.reasons,
                "passed":          res.passed,
                "overall_pass":    res.overall_pass,
                "error":           None,
            }
            for mk, sc in res.scores.items():
                metric_sums[mk] = metric_sums.get(mk, 0.0) + sc
                metric_cnts[mk] = metric_cnts.get(mk, 0) + 1
        except Exception as exc:
            logger.warning("Evaluation failed for row %s: %s", idx, exc)
            row = {
                "row_id":          item.row_id or str(idx + 1),
                "query":           item.query,
                "pipeline_answer": answer,
                "pipeline_context":  ctx_str[:500],
                "scores":          {},
                "reasons":         {},
                "passed":          {},
                "overall_pass":    False,
                "error":           f"Eval error: {exc}",
            }
        results.append(row)

    n_pass = sum(1 for r in results if r["overall_pass"] and not r["error"])
    n_err  = sum(1 for r in results if r["error"])
    avgs   = {mk: round(metric_sums[mk] / metric_cnts[mk], 4)
              for mk in metric_sums if metric_cnts[mk]}

    return {
        "total":           len(results),
        "passed":          n_pass,
        "failed":          len(results) - n_pass - n_err,
        "errored":         n_err,
        "metric_averages": avgs,
        "results":         results,
    }


@app.get("/graph/viz")
def graph_viz():
    try:
        g = get_workflow()
        return {"mermaid": g.get_graph().draw_mermaid()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Session memory endpoints ───────────────────────────────────────────────────

@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Return conversation turns stored in the LangGraph checkpoint for this session."""
    workflow = get_workflow()
    config = {"configurable": {"thread_id": session_id}}
    snapshot = workflow.get_state(config)
    if not snapshot or not snapshot.values:
        return {"session_id": session_id, "turns": [], "count": 0}
    msgs = snapshot.values.get("messages", [])
    turns = [
        {"role": "user" if isinstance(m, _HumanMessage) else "assistant", "content": m.content}
        for m in msgs
    ]
    return {"session_id": session_id, "turns": turns, "count": len(turns)}


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """InMemorySaver does not support deletion — return a new session_id to start fresh."""
    new_sid = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "cleared": False,
        "new_session_id": new_sid,
        "note": "Use new_session_id to start a fresh conversation.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
