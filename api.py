import uuid
import logging
from pathlib import Path
from typing import List, Optional

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
