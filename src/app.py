"""Streamlit chat interface, feedback collector, and monitoring dashboard."""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.ingestion import ingest
from src.retrieval import AdvancedRetriever, answer_question

FEEDBACK_DB = Path("data/feedback.db")
KNOWLEDGE_DB = Path("data/knowledge.duckdb")
SAMPLE_QUERIES = [
    "What was Vietnam's GDP growth rate in 2024?",
    "Chính sách tiền tệ năm 2024 hỗ trợ tăng trưởng như thế nào?",
    "Which tax measures supported households and businesses?",
]


def get_connection() -> sqlite3.Connection:
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(FEEDBACK_DB)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            response_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            query TEXT NOT NULL,
            rewritten_query TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            token_count INTEGER NOT NULL,
            rerank_score REAL NOT NULL,
            retrieval_method TEXT NOT NULL DEFAULT 'hybrid',
            score_label TEXT NOT NULL DEFAULT 'CrossEncoder score',
            feedback INTEGER
        )
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(interactions)").fetchall()}
    if "retrieval_method" not in columns:
        connection.execute(
            "ALTER TABLE interactions ADD COLUMN retrieval_method TEXT NOT NULL DEFAULT 'hybrid'"
        )
    if "score_label" not in columns:
        connection.execute(
            "ALTER TABLE interactions ADD COLUMN score_label "
            "TEXT NOT NULL DEFAULT 'CrossEncoder score'"
        )
    connection.commit()
    return connection


def record_interaction(
    response_id: str,
    query: str,
    rewritten_query: str,
    answer: str,
    sources: str,
    latency_ms: float,
    token_count: int,
    rerank_score: float,
    retrieval_method: str,
    score_label: str,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO interactions (
                response_id, created_at, query, rewritten_query, answer, sources,
                latency_ms, token_count, rerank_score, retrieval_method, score_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_id,
                datetime.now(UTC).isoformat(),
                query,
                rewritten_query,
                answer,
                sources,
                latency_ms,
                token_count,
                rerank_score,
                retrieval_method,
                score_label,
            ),
        )


def save_feedback(response_id: str, value: int) -> None:
    with get_connection() as connection:
        connection.execute(
            "UPDATE interactions SET feedback = ? WHERE response_id = ?", (value, response_id)
        )


@st.cache_resource(show_spinner="Preparing the retrieval pipeline...")
def get_retriever() -> AdvancedRetriever:
    if not KNOWLEDGE_DB.exists():
        ingest(db_path=KNOWLEDGE_DB)
    return AdvancedRetriever(KNOWLEDGE_DB)


def run_query(query: str, method: str = "bm25") -> dict[str, object]:
    started = time.perf_counter()
    rewritten, contexts = get_retriever().search(query, method=method)  # type: ignore[arg-type]
    answer, token_count = answer_question(query, contexts)
    latency_ms = (time.perf_counter() - started) * 1000
    response_id = str(uuid.uuid4())
    sources = " | ".join(f"{item.filename}#{item.chunk_id}" for item in contexts)
    best_score = max((item.score for item in contexts), default=0.0)
    score_label = "BM25 score" if method == "bm25" else "CrossEncoder score"
    record_interaction(
        response_id,
        query,
        rewritten,
        answer,
        sources,
        latency_ms,
        token_count,
        best_score,
        method,
        score_label,
    )
    return {
        "response_id": response_id,
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "method": method,
        "score_label": score_label,
    }


def render_feedback(response_id: str) -> None:
    columns = st.columns([1, 1, 8])
    if columns[0].button("👍", key=f"up-{response_id}", help="Helpful"):
        save_feedback(response_id, 1)
        st.toast("Feedback saved. Thank you.")
    if columns[1].button("👎", key=f"down-{response_id}", help="Not helpful"):
        save_feedback(response_id, 0)
        st.toast("Feedback saved. Thank you.")


def render_chat() -> None:
    st.subheader("Ask about Vietnam's economy and financial policies")
    st.caption(
        "Questions may be in Vietnamese or English; answers are returned in English with citations."
    )
    mode_label = st.selectbox(
        "Retrieval mode",
        ("Fast — BM25", "Advanced — Hybrid + reranker"),
        help=(
            "Fast mode uses local keyword search immediately. Advanced mode downloads Hugging Face "
            "models on first use and can take several minutes."
        ),
    )
    method = "bm25" if mode_label == "Fast — BM25" else "hybrid"
    if method == "bm25":
        st.caption("Fast mode uses BM25 only; no Hugging Face model download is required.")
    else:
        st.info(
            "Advanced mode loads MiniLM and a CrossEncoder. The first run can take several minutes."
        )
    if not os.getenv("DEEPSEEK_API_KEY"):
        st.caption(
            "No DeepSeek key is configured, so answers will use cited extracts from retrieved text."
        )
    button_columns = st.columns(3)
    selected_query = None
    for index, sample in enumerate(SAMPLE_QUERIES):
        if button_columns[index].button(sample, use_container_width=True):
            selected_query = sample

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                with st.expander("Retrieved evidence"):
                    for context in message["contexts"]:
                        st.markdown(
                            f"**{context.filename} — {context.chunk_id}** "
                            f"({message['score_label']} {context.score:.3f})"
                        )
                        st.write(context.content)
                render_feedback(message["response_id"])

    query = selected_query or st.chat_input("Ask a macroeconomic or policy question")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.status("Processing question...", expanded=True) as status:
            st.write("Loading the retrieval index...")
            if method == "hybrid":
                st.write("Loading models, retrieving documents, and reranking passages...")
            else:
                st.write("Retrieving documents with BM25...")
            result = run_query(query, method)
            st.write("Generating an answer or preparing cited extracts...")
            status.update(label="Answer ready", state="complete", expanded=False)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "contexts": result["contexts"],
                "response_id": result["response_id"],
                "score_label": result["score_label"],
            }
        )
        st.rerun()


def load_analytics() -> pd.DataFrame:
    with get_connection() as connection:
        return pd.read_sql_query("SELECT * FROM interactions ORDER BY created_at", connection)


def render_dashboard() -> None:
    st.subheader("Real-time RAG monitoring")
    frame = load_analytics()
    if frame.empty:
        st.info("No queries recorded yet. Use the Chat Assistant tab to generate monitoring data.")
        return
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    frame["date"] = frame["created_at"].dt.date.astype(str)
    frame["request_number"] = range(1, len(frame) + 1)

    st.markdown("#### 1. Total Queries Over Time")
    daily = frame.groupby("date", as_index=False).size().rename(columns={"size": "queries"})
    st.bar_chart(daily, x="date", y="queries", use_container_width=True)

    st.markdown("#### 2. User Satisfaction: Thumbs Up vs Down")
    feedback = frame.dropna(subset=["feedback"]).copy()
    if feedback.empty:
        st.caption("No feedback has been submitted yet.")
    else:
        feedback["rating"] = feedback["feedback"].map({1.0: "Thumbs up", 0.0: "Thumbs down"})
        counts = feedback.groupby("rating", as_index=False).size().rename(columns={"size": "count"})
        pie = (
            alt.Chart(counts)
            .mark_arc(innerRadius=45)
            .encode(
                theta=alt.Theta("count:Q"), color=alt.Color("rating:N"), tooltip=["rating", "count"]
            )
        )
        st.altair_chart(pie, use_container_width=True)

    st.markdown("#### 3. End-to-End Pipeline Latency Distribution")
    latency = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("request_number:Q", title="Request"),
            y=alt.Y("latency_ms:Q", title="Latency (ms)"),
            tooltip=["created_at:T", "latency_ms:Q", "query:N"],
        )
    )
    st.altair_chart(latency, use_container_width=True)

    st.markdown("#### 4. Token Consumption per Request")
    tokens = (
        alt.Chart(frame)
        .mark_area(opacity=0.55, line=True)
        .encode(
            x=alt.X("request_number:Q", title="Request"),
            y=alt.Y("token_count:Q", title="Tokens"),
            tooltip=["created_at:T", "token_count:Q", "query:N"],
        )
    )
    st.altair_chart(tokens, use_container_width=True)

    st.markdown("#### 5. Retrieval Score Distribution")
    histogram = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("rerank_score:Q", bin=alt.Bin(maxbins=20), title="Retrieval score"),
            y=alt.Y("count():Q", title="Requests"),
            color=alt.Color("score_label:N", title="Score type"),
            tooltip=[alt.Tooltip("count():Q", title="Requests"), "score_label:N"],
        )
    )
    st.altair_chart(histogram, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Macro Policy RAG", page_icon="🏦", layout="wide")
    st.title("🏦 Macroeconomic & Policy RAG Assistant")
    chat_tab, dashboard_tab = st.tabs(["Chat Assistant", "Monitoring Dashboard"])
    with chat_tab:
        render_chat()
    with dashboard_tab:
        render_dashboard()


if __name__ == "__main__":
    main()
