from pathlib import Path

import pytest

from src.app import get_connection, record_interaction, save_feedback
from src.eval import hit_rate, mean_reciprocal_rank
from src.ingestion import build_chunks
from src.retrieval import reciprocal_rank_fusion, tokenize


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
    record_interaction("r1", "q", "rq", "a", "source", 12.5, 42, 0.7)
    save_feedback("r1", 1)
    with get_connection() as connection:
        row = connection.execute(
            "SELECT query, latency_ms, token_count, feedback "
            "FROM interactions WHERE response_id = 'r1'"
        ).fetchone()
    assert row == ("q", 12.5, 42, 1)


def test_invalid_chunk_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        build_chunks(tmp_path, chunk_size=100, chunk_overlap=100)
