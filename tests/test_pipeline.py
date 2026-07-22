from pathlib import Path

import pytest

from src.app import get_connection, record_interaction, save_feedback
from src.eval import hit_rate, mean_reciprocal_rank
from src.ingestion import build_chunks
from src.retrieval import (
    AdvancedRetriever,
    RetrievedChunk,
    answer_question,
    chat_completion,
    reciprocal_rank_fusion,
    tokenize,
)


def test_recursive_chunking_has_overlap_and_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "policy.md").write_text("Vietnam policy. " * 120, encoding="utf-8")

    chunks = build_chunks(raw_dir, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 2
    assert chunks[0]["filename"] == "policy.md"
    assert chunks[0]["chunk_id"] == "policy-0000"
    assert chunks[0]["created_at"]
    assert len(str(chunks[0]["id"])) == 20
    assert len(str(chunks[0]["content"])) <= 200


def test_rrf_combines_keyword_and_vector_ranks() -> None:
    result = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [item[0] for item in result[:2]] == ["a", "b"]
    assert result[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_unicode_tokenizer_supports_vietnamese() -> None:
    assert tokenize("Tăng trưởng GDP Việt Nam") == ["tăng", "trưởng", "gdp", "việt", "nam"]


def test_retrieval_metrics() -> None:
    expected = ["a.md", "b.md", "c.md"]
    predictions = [["a.md"], ["x.md", "b.md"], ["x.md"]]
    assert hit_rate(expected, predictions) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(expected, predictions) == pytest.approx(0.5)


def test_feedback_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "feedback.db"
    monkeypatch.setattr("src.app.FEEDBACK_DB", database)
    record_interaction("r1", "q", "rq", "a", "source", 12.5, 42, 0.7, "bm25", "BM25 score")
    save_feedback("r1", 1)
    with get_connection() as connection:
        row = connection.execute(
            "SELECT query, latency_ms, token_count, retrieval_method, score_label, feedback "
            "FROM interactions WHERE response_id = 'r1'"
        ).fetchone()
    assert row == ("q", 12.5, 42, "bm25", "BM25 score", 1)


def test_invalid_chunk_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        build_chunks(tmp_path, chunk_size=100, chunk_overlap=100)


def _contexts() -> list[RetrievedChunk]:
    return [
        RetrievedChunk("one", "report.md", "report-0000", "GDP grew by 7.09 percent.", 1.0),
        RetrievedChunk("two", "policy.md", "policy-0000", "The central bank lowered rates.", 0.5),
    ]


def test_no_api_key_returns_cited_extractive_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    answer, tokens = answer_question("How did GDP grow?", _contexts())

    assert "DeepSeek generation is disabled" in answer
    assert "GDP grew by 7.09 percent." in answer
    assert "[Source 1: report.md#report-0000]" in answer
    assert tokens > 0


def test_deepseek_failure_returns_extractive_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.retrieval.chat_completion", lambda *_: (_ for _ in ()).throw(OSError())
    )

    answer, _ = answer_question("How did GDP grow?", _contexts())

    assert "DeepSeek generation failed" in answer
    assert "[Source 2: policy.md#policy-0000]" in answer


def test_chat_completion_uses_wrapper_calling_convention() -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> dict[str, object]:
            self.kwargs = kwargs
            return {"ok": True}

    class FakeClient:
        class chat:  # noqa: N801 - mimics the provider client API.
            completions = FakeCompletions()

    client = FakeClient()
    result = chat_completion(client, "model", [{"role": "user", "content": "hello"}])  # type: ignore[arg-type]

    assert result == {"ok": True}
    assert client.chat.completions.kwargs == {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }


def test_bm25_fast_mode_does_not_load_ml_models() -> None:
    retriever = object.__new__(AdvancedRetriever)
    retriever.documents = [
        {"id": "one", "filename": "report.md", "chunk_id": "report-0000", "content": "GDP growth"}
    ]
    retriever._by_id = {"one": retriever.documents[0]}
    retriever.rewrite_query = lambda query: query  # type: ignore[method-assign]
    retriever.bm25_search = lambda query, limit: [("one", 3.0)]  # type: ignore[method-assign]
    retriever.vector_search = lambda *_: pytest.fail("Fast mode must not load embeddings")  # type: ignore[method-assign]

    _, chunks = retriever.search("GDP", method="bm25")

    assert chunks[0].filename == "report.md"
    assert chunks[0].score == 3.0


def test_hybrid_mode_selects_vector_and_reranker_with_mocks() -> None:
    retriever = object.__new__(AdvancedRetriever)
    calls: list[str] = []
    retriever.rewrite_query = lambda query: query  # type: ignore[method-assign]
    retriever.bm25_search = lambda *_: calls.append("bm25") or [("one", 1.0)]  # type: ignore[method-assign]
    retriever.vector_search = lambda *_: calls.append("vector") or [("two", 1.0)]  # type: ignore[method-assign]
    expected = _contexts()
    retriever.rerank = lambda query, candidates, limit: calls.append("rerank") or expected  # type: ignore[method-assign]

    _, chunks = retriever.search("GDP", method="hybrid")

    assert calls == ["bm25", "vector", "rerank"]
    assert chunks == expected
