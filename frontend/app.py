"""
Agentic RAG — Frontend
======================
Clean ChatGPT-style Streamlit UI.
Communicates with the FastAPI backend at RAG_API_URL (default http://localhost:8000).

Run:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

RETRIEVAL_LABELS = {
    "sql":          "SQL",
    "hybrid":       "Hybrid (BM25 + Vector)",
    "multihop":     "Multi-hop",
    "web_search":   "Web Search",
    "no_retrieval": "Direct LLM",
    "":             "—",
}

DOMAIN_LABELS = {
    "hr_domain":       "HR",
    "research_domain": "Research",
    "general":         "General",
    "":                "—",
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal CSS — ChatGPT-like look ───────────────────────────────────────────
st.markdown("""
<style>
/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* full-height chat area */
.main .block-container {
    max-width: 860px;
    padding: 2rem 1.5rem 6rem;
    margin: 0 auto;
}

/* user bubble */
.msg-user {
    background: #f4f4f5;
    border-radius: 12px;
    padding: 10px 16px;
    margin: 8px 0;
    max-width: 78%;
    margin-left: auto;
    font-size: 0.95rem;
    color: #111;
    word-break: break-word;
}

/* assistant bubble */
.msg-assistant {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    padding: 10px 16px;
    margin: 8px 0;
    max-width: 90%;
    font-size: 0.95rem;
    color: #111;
    word-break: break-word;
    line-height: 1.6;
}

/* meta info under assistant bubble */
.msg-meta {
    font-size: 0.72rem;
    color: #a1a1aa;
    margin: 2px 0 10px 4px;
}

/* citation expander styling */
.citation-item {
    font-size: 0.82rem;
    color: #52525b;
    margin: 2px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "meta"}]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Agentic RAG")
    st.caption(f"Session `{st.session_state.session_id[:8]}…`")

    # Backend health
    try:
        h = requests.get(f"{API_URL}/health", timeout=2)
        if h.ok:
            st.success("Backend online")
        else:
            st.warning("Backend responded with error")
    except Exception:
        st.error("Backend offline — run: `uvicorn api:app --reload`")

    st.divider()

    top_k = st.slider("Docs retrieved (top-k)", 1, 15, 5)

    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Document upload
    st.markdown("**Upload document**")
    uploaded = st.file_uploader(
        "PDF, TXT or MD",
        type=["pdf", "txt", "md"],
        label_visibility="collapsed",
    )
    if uploaded:
        if st.button("Index document", use_container_width=True):
            with st.spinner("Indexing…"):
                r = requests.post(
                    f"{API_URL}/index",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=120,
                )
            if r.ok:
                st.success(r.json().get("message", "Indexing started"))
            else:
                st.error(f"Failed ({r.status_code}): {r.text[:200]}")


# ── Chat title ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        "<h2 style='text-align:center; margin-top:3rem; color:#18181b;'>What can I help you with?</h2>",
        unsafe_allow_html=True,
    )


# ── Message history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("meta", {})

    if role == "user":
        st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-assistant">{content}</div>', unsafe_allow_html=True)

        if meta:
            retrieval = RETRIEVAL_LABELS.get(meta.get("retrieval_source", ""), meta.get("retrieval_source", ""))
            domain    = DOMAIN_LABELS.get(meta.get("domain", ""), meta.get("domain", ""))
            parts = []
            if retrieval and retrieval != "—":
                parts.append(f"retrieval: **{retrieval}**")
            if domain and domain != "—":
                parts.append(f"domain: **{domain}**")
            if meta.get("cache_hit"):
                parts.append("cache hit")
            if meta.get("expanded_query") and meta["expanded_query"] != meta.get("query", ""):
                parts.append(f'expanded: *{meta["expanded_query"][:80]}*')

            if parts:
                st.markdown(
                    f'<div class="msg-meta">{" &nbsp;·&nbsp; ".join(parts)}</div>',
                    unsafe_allow_html=True,
                )

            # Citations expander
            citations = meta.get("citations", [])
            if citations:
                with st.expander(f"References ({len(citations)})"):
                    for c in citations:
                        page_str = f" — p. {c['page']}" if c.get("page") else ""
                        st.markdown(
                            f'<div class="citation-item">[{c["index"]}] {c["source"]}{page_str}</div>',
                            unsafe_allow_html=True,
                        )


# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything…")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input, "meta": {}})

    # Call backend — server manages session memory via session_id
    with st.spinner(""):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={
                    "query": user_input,
                    "session_id": st.session_state.session_id,
                    "top_k": top_k,
                },
                timeout=120,
            )
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Run: `uvicorn api:app --reload`")
            st.stop()
        except Exception as exc:
            st.error(f"Request error: {exc}")
            st.stop()

    if resp.ok:
        data = resp.json()
        answer = data.get("answer", "No answer returned.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": {**data, "query": user_input},
        })
    else:
        error_msg = f"API error {resp.status_code}: {resp.text[:300]}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "meta": {},
        })

    st.rerun()
