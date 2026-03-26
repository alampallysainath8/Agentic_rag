"""
📊 RAG Evaluation Page
======================
Three tabs:
  1. Manual     – paste a Q/A pair, run all 4 metrics, see scores + reasoning
  2. Excel Batch – upload an .xlsx file and evaluate every row
  3. History    – browse & search past evaluation records
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

# ── colours per metric ────────────────────────────────────────────────────────
METRIC_META = {
    "faithfulness":      {"label": "Faithfulness",       "color": "#6366f1"},
    "answer_relevancy":  {"label": "Answer Relevancy",   "color": "#22c55e"},
    "context_relevancy": {"label": "Context Relevancy",  "color": "#f59e0b"},
    "context_recall":    {"label": "Context Recall",     "color": "#3b82f6"},
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Evaluation", page_icon="📊", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.main .block-container { max-width: 1050px; padding: 1.5rem 2rem 4rem; }
.metric-card {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 14px 18px 10px; text-align: center;
}
.metric-card .mlabel { font-size: .74rem; color: #6b7280; font-weight: 600;
    text-transform: uppercase; letter-spacing: .04em; margin-bottom: 4px; }
.metric-card .mscore { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.metric-card .mbadge { display: inline-block; font-size: .72rem; border-radius: 999px;
    padding: 2px 9px; margin-top: 5px; font-weight: 600; }
.badge-pass { background: #dcfce7; color: #166534; }
.badge-fail { background: #fee2e2; color: #991b1b; }
.reason-box { background: #f1f5f9; border-left: 3px solid #94a3b8;
    border-radius: 0 8px 8px 0; padding: 8px 14px;
    font-size: .83rem; color: #374151; margin-top: 4px; }
.overall-pass { color: #16a34a; font-weight: 700; font-size: 1rem; }
.overall-fail { color: #dc2626; font-weight: 700; font-size: 1rem; }
.tip-box { background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 14px 18px; font-size: .87rem; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _metric_cards(scores: dict, passed: dict) -> None:
    if not scores:
        return
    cols = st.columns(len(scores))
    for i, (mk, sc) in enumerate(scores.items()):
        m    = METRIC_META.get(mk, {"label": mk.replace("_", " ").title(), "color": "#6b7280"})
        ok   = passed.get(mk, False)
        bdg  = "badge-pass" if ok else "badge-fail"
        with cols[i]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="mlabel">{m["label"]}</div>'
                f'<div class="mscore" style="color:{m["color"]};">{sc:.2f}</div>'
                f'<span class="mbadge {bdg}">{"PASS" if ok else "FAIL"}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _reason_blocks(reasons: dict, scores: dict) -> None:
    for mk, reason in reasons.items():
        if reason:
            m   = METRIC_META.get(mk, {"label": mk, "color": "#6b7280"})
            sc  = scores.get(mk, 0)
            st.markdown(
                f'<div style="font-weight:600;font-size:.84rem;color:#374151;margin-top:10px;">'
                f'{m["label"]} &nbsp;<span style="background:#e0f2fe;color:#0369a1;'
                f'border-radius:999px;padding:1px 9px;font-size:.75rem;">{sc:.2f}</span></div>'
                f'<div class="reason-box">{reason}</div>',
                unsafe_allow_html=True,
            )


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Evaluation")
    try:
        h = requests.get(f"{API_URL}/health", timeout=2)
        st.success("Backend online") if h.ok else st.warning("Backend error")
    except Exception:
        st.error("Backend offline")
    st.divider()
    threshold = st.slider("Pass threshold", 0.0, 1.0, 0.5, 0.05)
    delay     = st.slider("Delay between metrics (s)", 0.0, 3.0, 0.5, 0.25,
                           help="Pause between each LLM metric call — prevents rate-limit errors.")
    st.divider()
    if st.button("🗑 Clear history", use_container_width=True):
        try:
            r = requests.delete(f"{API_URL}/evaluations", timeout=10)
            st.success("History cleared") if r.ok else st.error(r.text[:200])
        except Exception as exc:
            st.error(str(exc))
    st.divider()
    st.page_link("app.py", label="← Back to chat", icon="💬")


# ── title ─────────────────────────────────────────────────────────────────────
st.title("📊 RAG Evaluation")
st.caption(
    "Metrics: **Faithfulness** · **Answer Relevancy** · **Context Relevancy** "
    "· **Context Recall** *(requires expected answer)*"
)

tab_manual, tab_excel, tab_history = st.tabs([
    "✏️ Manual",
    "📂 Excel Batch",
    "📋 History",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual
# ══════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Evaluate a single Q/A pair")

    c1, c2 = st.columns(2)
    with c1:
        q_in = st.text_area("Query", height=100, key="m_q",
                             placeholder="e.g. How many active employees are in Engineering?")
        ctx_in = st.text_area("Retrieved context  *(one chunk per line)*", height=130,
                              key="m_ctx",
                              placeholder="Paste retrieved passages — one chunk per line.")
    with c2:
        a_in = st.text_area("Answer  (RAG output)", height=100, key="m_a",
                             placeholder="e.g. There are 7 active employees in Engineering.")
        exp_in = st.text_area(
            "Expected / ground-truth answer  *(optional — unlocks Context Recall)*",
            height=130, key="m_exp",
            placeholder="Leave blank to skip Context Recall.",
        )

    if st.button("▶ Run evaluation", type="primary", use_container_width=True, key="m_run"):
        if not q_in.strip() or not a_in.strip() or not ctx_in.strip():
            st.warning("Query, answer, and at least one context chunk are required.")
            st.stop()

        chunks  = [c.strip() for c in ctx_in.splitlines() if c.strip()]
        payload = {
            "query":          q_in.strip(),
            "answer":         a_in.strip(),
            "context_chunks": chunks,
            "threshold":      threshold,
        }
        if exp_in.strip():
            payload["expected_output"] = exp_in.strip()

        with st.spinner("Evaluating… this takes 30-120 s depending on how many metrics run."):
            try:
                resp = requests.post(f"{API_URL}/evaluate", json=payload, timeout=300)
            except Exception as exc:
                st.error(str(exc)); st.stop()

        if not resp.ok:
            st.error(f"API error {resp.status_code}: {resp.text[:400]}"); st.stop()

        data = resp.json()
        ok   = data.get("overall_pass")
        st.markdown(
            f'<div class="{"overall-pass" if ok else "overall-fail"}">'
            f'{"✅ Overall PASS" if ok else "❌ Overall FAIL"}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        _metric_cards(data.get("scores", {}), data.get("passed", {}))
        st.markdown("#### Reasoning")
        _reason_blocks(data.get("reasons", {}), data.get("scores", {}))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Excel Batch
# ══════════════════════════════════════════════════════════════════════════════
with tab_excel:
    st.subheader("Batch evaluation from Excel")

    # ── mode selector ──────────────────────────────────────────────────────
    mode = st.radio(
        "Evaluation mode",
        ["🔁 Pipeline mode  (recommended)",
         "📋 Manual mode"],
        horizontal=True,
        help=(
            "**Pipeline mode** — Excel has only *query* (+ optional *expected_output*). "
            "The RAG pipeline runs first and produces the real answer + context automatically.\n\n"
            "**Manual mode** — Excel has pre-filled *answer* and *context* columns."
        ),
    )
    is_pipeline = mode.startswith("🔁")

    # ── format instructions ────────────────────────────────────────────────
    if is_pipeline:
        st.markdown("""
<div class="tip-box">
<strong>📋 Excel columns — Pipeline mode</strong><br><br>

| Column | Required | Description |
|---|---|---|
| <code>query</code> | ✅ | The question to send to the RAG pipeline |
| <code>expected_output</code> | ☑️ optional | Ground-truth answer — enables <strong>Context Recall</strong> |

<br>
The pipeline fetches the <strong>answer</strong> and <strong>retrieval context</strong> automatically.<br>
No <code>answer</code> or <code>context</code> columns needed.
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="tip-box">
<strong>📋 Excel columns — Manual mode</strong><br><br>

| Column | Required | Description |
|---|---|---|
| <code>query</code> | ✅ | The question asked to the RAG system |
| <code>answer</code> | ✅ | The answer produced by the RAG pipeline |
| <code>context</code> | ✅ | Retrieved passage(s). Separate multiple chunks with&nbsp;<code>|||</code> |
| <code>expected_output</code> | ☑️ optional | Ground-truth answer — enables <strong>Context Recall</strong> |

<br>
<strong>Multi-chunk context separator:</strong>
<code>Employee count is 7 ||| Manager is Alice Morgan ||| Budget $2M</code>
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    # ── template download ──────────────────────────────────────────────────
    if is_pipeline:
        _tpl = pd.DataFrame({
            "query": [
                "How many active employees are in the Engineering department?",
                "Who has the highest base salary?",
                "List each department with its manager and their salary.",
            ],
            "expected_output": [
                "There are 7 active employees in the Engineering department.",
                "Alice Morgan (Engineering Manager) has the highest base salary at $150,000.",
                "",   # leave blank → context_recall skipped for this row
            ],
        })
    else:
        _tpl = pd.DataFrame({
            "query": [
                "How many active employees are in the Engineering department?",
                "Who has the highest base salary?",
            ],
            "answer": [
                "There are 7 active employees in Engineering.",
                "Alice Morgan has the highest salary at $150,000.",
            ],
            "context": [
                "active_count: 7 ||| department: Engineering",
                "employee_name: Alice Morgan ||| BaseSalary: 150000.0 ||| JobTitle: Engineering Manager",
            ],
            "expected_output": [
                "There are 7 active employees in the Engineering department.",
                "Alice Morgan (Engineering Manager) has the highest base salary at $150,000.",
            ],
        })

    _buf = io.BytesIO()
    _tpl.to_excel(_buf, index=False, sheet_name="eval_data")
    st.download_button(
        "⬇️ Download template.xlsx",
        data=_buf.getvalue(),
        file_name=f"eval_template_{'pipeline' if is_pipeline else 'manual'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()

    # ── file upload ────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload your .xlsx file", type=["xlsx", "xls"],
                                key=f"xl_upload_{mode}")

    if uploaded:
        try:
            df = pd.read_excel(uploaded)
        except Exception as exc:
            st.error(f"Could not read Excel file: {exc}"); st.stop()

        df.columns = [c.strip().lower() for c in df.columns]

        # validate columns
        if is_pipeline:
            required = ["query"]
        else:
            required = ["query", "answer", "context"]

        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required column(s): `{'`, `'.join(missing)}`")
            st.stop()

        preview_cols = [c for c in ["query", "answer", "context", "expected_output"]
                        if c in df.columns]
        st.success(f"Loaded **{len(df)} rows** — preview:")
        st.dataframe(
            df[preview_cols].head(5).map(lambda x: str(x)[:90] if pd.notna(x) else ""),
            use_container_width=True, hide_index=True,
        )

        row_delay = st.slider(
            "Delay between rows (s)", 0.5, 5.0,
            2.0 if is_pipeline else 1.5, 0.5,
            key="xl_delay",
            help="Pipeline mode needs a bit more time per row (pipeline + eval both run).",
        )
        max_rows = st.slider(
            "Max rows to evaluate", 1, min(30, len(df)), min(5, len(df)),
            key="xl_max",
        )

        if st.button("🚀 Run batch evaluation", type="primary",
                     use_container_width=True, key="xl_run"):

            if is_pipeline:
                # ── pipeline mode ──────────────────────────────────────────
                items = []
                for i, row in df.head(max_rows).iterrows():
                    item: dict = {"row_id": str(i + 1), "query": str(row["query"])}
                    exp = row.get("expected_output", "")
                    if pd.notna(exp) and str(exp).strip():
                        item["expected_output"] = str(exp).strip()
                    items.append(item)

                payload = {
                    "items":              items,
                    "threshold":          threshold,
                    "delay_between_rows": row_delay,
                }
                endpoint  = f"{API_URL}/evaluate/pipeline-batch"
                est_base  = max_rows * (row_delay + 45)  # pipeline + eval time
            else:
                # ── manual mode ────────────────────────────────────────────
                items = []
                for i, row in df.head(max_rows).iterrows():
                    raw_ctx = str(row.get("context", ""))
                    chunks  = [c.strip() for c in raw_ctx.split("|||") if c.strip()] or [raw_ctx]
                    item = {
                        "row_id":         str(i + 1),
                        "query":          str(row["query"]),
                        "answer":         str(row["answer"]),
                        "context_chunks": chunks,
                    }
                    exp = row.get("expected_output", "")
                    if pd.notna(exp) and str(exp).strip():
                        item["expected_output"] = str(exp).strip()
                    items.append(item)

                payload  = {"items": items, "threshold": threshold,
                            "delay_between_rows": row_delay}
                endpoint = f"{API_URL}/evaluate/batch"
                est_base = max_rows * (row_delay + 30)

            with st.spinner(f"Evaluating {max_rows} rows… (~{est_base:.0f}–{est_base*2:.0f} s)"):
                try:
                    resp = requests.post(endpoint, json=payload, timeout=900)
                except Exception as exc:
                    st.error(str(exc)); st.stop()

            if not resp.ok:
                st.error(f"API error {resp.status_code}: {resp.text[:400]}"); st.stop()

            rpt = resp.json()
            total, n_pass, n_fail, n_err = (
                rpt["total"], rpt["passed"], rpt["failed"], rpt["errored"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total",  total)
            m2.metric("Passed", n_pass, delta=f"{n_pass/total*100:.0f}%" if total else "")
            m3.metric("Failed", n_fail)
            m4.metric("Errors", n_err)

            avgs = rpt.get("metric_averages", {})
            if avgs:
                st.markdown("**Average scores across all rows**")
                _metric_cards(avgs, {k: v >= threshold for k, v in avgs.items()})

            st.markdown("---")
            st.markdown("#### Per-row results")

            for rec in rpt.get("results", []):
                row_id = rec.get("row_id", "?")
                ok     = rec.get("overall_pass")
                err    = rec.get("error")
                icon   = "✅" if ok else ("⚠️" if err else "❌")
                q_txt  = rec.get("query", "")[:80]
                with st.expander(f"{icon} Row {row_id} — {q_txt}"):
                    if err:
                        st.error(err)
                    else:
                        if is_pipeline:
                            st.markdown(f"**Pipeline answer:** {rec.get('pipeline_answer','')[:300]}")
                            ctx_prev = rec.get("pipeline_context", "")
                            if ctx_prev:
                                st.caption(f"Context preview: {ctx_prev[:200]}…")
                        _metric_cards(rec.get("scores", {}), rec.get("passed", {}))
                        _reason_blocks(rec.get("reasons", {}), rec.get("scores", {}))

            # ── export results ─────────────────────────────────────────────
            result_rows = []
            for rec in rpt.get("results", []):
                row_out: dict = {
                    "row_id":       rec.get("row_id"),
                    "query":        rec.get("query", ""),
                    "overall_pass": rec.get("overall_pass"),
                    "error":        rec.get("error", ""),
                }
                if is_pipeline:
                    row_out["pipeline_answer"]  = rec.get("pipeline_answer", "")
                    row_out["pipeline_context"] = rec.get("pipeline_context", "")
                    row_out["retrieval_source"] = rec.get("retrieval_source", "")
                else:
                    row_out["answer"] = rec.get("answer", "")
                for mk in METRIC_META:
                    row_out[f"{mk}_score"]  = rec.get("scores", {}).get(mk, "")
                    row_out[f"{mk}_passed"] = rec.get("passed", {}).get(mk, "")
                result_rows.append(row_out)

            res_df  = pd.DataFrame(result_rows)
            res_buf = io.BytesIO()
            res_df.to_excel(res_buf, index=False, sheet_name="results")
            st.download_button(
                "⬇️ Download results.xlsx",
                data=res_buf.getvalue(),
                file_name="eval_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Evaluation History")
    if st.button("🔄 Refresh", key="h_refresh"):
        st.rerun()

    try:
        ev_resp = requests.get(f"{API_URL}/evaluations?limit=100", timeout=5)
        records = ev_resp.json().get("evaluations", []) if ev_resp.ok else []
    except Exception:
        records = []

    if not records:
        st.info("No evaluations yet. Run a manual or batch evaluation first.")
    else:
        n_total = len(records)
        n_pass  = sum(1 for r in records if r.get("overall_pass"))

        c1, c2 = st.columns(2)
        c1.metric("Total evaluations", n_total)
        c2.metric("Pass rate", f"{n_pass / n_total * 100:.1f}%" if n_total else "—")

        # Aggregate average scores
        m_sums: dict = {}
        m_cnts: dict = {}
        for rec in records:
            for mk, sc in rec.get("scores", {}).items():
                m_sums[mk] = m_sums.get(mk, 0.0) + sc
                m_cnts[mk] = m_cnts.get(mk, 0) + 1
        avgs = {mk: round(m_sums[mk] / m_cnts[mk], 3) for mk in m_sums if m_cnts[mk]}
        thr  = records[-1].get("threshold", 0.5)
        st.markdown("**Average scores (all time)**")
        _metric_cards(avgs, {k: v >= thr for k, v in avgs.items()})

        st.markdown("---")

        # Summary table
        rows = []
        for rec in reversed(records):
            sc  = rec.get("scores", {})
            row: dict = {
                "Time":  rec.get("timestamp", "")[:19].replace("T", " "),
                "Pass?": "✅" if rec.get("overall_pass") else "❌",
                "Query": rec.get("query", "")[:60],
            }
            for mk, m in METRIC_META.items():
                row[m["label"]] = f"{sc[mk]:.2f}" if mk in sc else "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Detailed expandable records
        st.markdown("---")
        st.markdown("#### Detailed records  (last 20)")
        for rec in list(reversed(records))[:20]:
            ts   = rec.get("timestamp", "")[:19].replace("T", " ")
            ok   = rec.get("overall_pass")
            icon = "✅" if ok else "❌"
            with st.expander(f"{icon} [{ts}]  {rec.get('query','')[:75]}"):
                st.write("**Query:**",  rec.get("query", ""))
                st.write("**Answer:**", rec.get("answer", "")[:400])
                _metric_cards(rec.get("scores", {}), rec.get("passed", {}))
                _reason_blocks(rec.get("reasons", {}), rec.get("scores", {}))
