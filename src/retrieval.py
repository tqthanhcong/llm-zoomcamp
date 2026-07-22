"""Query rewriting, hybrid retrieval with RRF, and CrossEncoder re-ranking."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

DEFAULT_DB_PATH = Path("data/knowledge.duckdb")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_TIMEOUT_SECONDS = 20.0


def get_deepseek_client() -> OpenAI:
    """Create an OpenAI-compatible client configured for DeepSeek."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
        timeout=DEEPSEEK_TIMEOUT_SECONDS,
        max_retries=1,
    )


def chat_completion(client: OpenAI, model: str, messages: list[dict[str, str]]):
    """Request a deterministic chat completion through DeepSeek's compatible API."""
    return client.chat.completions.create(model=model, messages=messages, temperature=0)


def tokenize(text: str) -> list[str]:
    """Unicode-friendly tokenizer used by BM25 and deterministic fallbacks."""
    import re

    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked IDs using score = sum(1 / (k + rank)), with rank starting at 1."""
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, document_id in enumerate(ranked, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    filename: str
    chunk_id: str
    content: str
    score: float


class AdvancedRetriever:
    """Three-stage retrieval: rewrite, BM25/vector RRF, then CrossEncoder."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
    ) -> None:
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.documents = self._load_documents()
        if not self.documents:
            raise RuntimeError("The knowledge base is empty. Run `make setup` first.")
        self._by_id = {str(row["id"]): row for row in self.documents}
        self._bm25 = BM25Okapi([tokenize(str(row["content"])) for row in self.documents])
        self._embedding_model = None
        self._reranker = None
        self._document_embeddings: np.ndarray | None = None

    def _load_documents(self) -> list[dict[str, str]]:
        if not self.db_path.exists():
            return []
        connection = duckdb.connect(str(self.db_path), read_only=True)
        try:
            rows = connection.execute(
                "SELECT id, filename, chunk_id, content FROM macro_policy.chunks "
                "ORDER BY filename, chunk_id"
            ).fetchall()
        finally:
            connection.close()
        return [
            {"id": row[0], "filename": row[1], "chunk_id": row[2], "content": row[3]}
            for row in rows
        ]

    def rewrite_query(self, query: str) -> str:
        """Use an LLM to make a search query explicit; fall back safely without an API key."""
        clean_query = " ".join(query.split())
        if not clean_query:
            raise ValueError("Query must not be empty")
        if not os.getenv("DEEPSEEK_API_KEY"):
            return clean_query

        try:
            response = chat_completion(
                get_deepseek_client(),
                os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
                [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user's Vietnamese or English question as one concise, "
                            "explicit retrieval query. "
                            "Preserve names, dates, rates, and policy terminology. "
                            "Return only the rewritten query and do not answer it."
                        ),
                    },
                    {"role": "user", "content": clean_query},
                ],
            )
            return (response.choices[0].message.content or "").strip() or clean_query
        except Exception:
            return clean_query

    def bm25_search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        indices = np.argsort(scores)[::-1][:limit]
        return [(str(self.documents[index]["id"]), float(scores[index])) for index in indices]

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def vector_search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        model = self._get_embedding_model()
        if self._document_embeddings is None:
            self._document_embeddings = np.asarray(
                model.encode(
                    [str(row["content"]) for row in self.documents],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
        query_vector = np.asarray(
            model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        )
        scores = self._document_embeddings @ query_vector
        indices = np.argsort(scores)[::-1][:limit]
        return [(str(self.documents[index]["id"]), float(scores[index])) for index in indices]

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def _as_chunks(
        self, candidates: Sequence[tuple[str, float]], limit: int
    ) -> list[RetrievedChunk]:
        chunks = []
        for document_id, score in candidates[:limit]:
            document = self._by_id[document_id]
            chunks.append(
                RetrievedChunk(
                    id=str(document["id"]),
                    filename=str(document["filename"]),
                    chunk_id=str(document["chunk_id"]),
                    content=str(document["content"]),
                    score=float(score),
                )
            )
        return chunks

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, float]], limit: int = 3
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        documents = [self._by_id[document_id] for document_id, _ in candidates]
        pairs = [(query, str(document["content"])) for document in documents]
        scores = np.asarray(self._get_reranker().predict(pairs, show_progress_bar=False))
        ordered = np.argsort(scores)[::-1][:limit]
        return [
            RetrievedChunk(
                id=str(documents[index]["id"]),
                filename=str(documents[index]["filename"]),
                chunk_id=str(documents[index]["chunk_id"]),
                content=str(documents[index]["content"]),
                score=float(scores[index]),
            )
            for index in ordered
        ]

    def search(
        self,
        query: str,
        *,
        rewrite: bool = True,
        candidate_limit: int = 10,
        result_limit: int = 3,
        method: Literal["bm25", "vector", "hybrid"] = "bm25",
    ) -> tuple[str, list[RetrievedChunk]]:
        rewritten = self.rewrite_query(query) if rewrite else " ".join(query.split())
        if method == "bm25":
            candidates = self.bm25_search(rewritten, candidate_limit)
            return rewritten, self._as_chunks(candidates, result_limit)
        elif method == "vector":
            candidates = self.vector_search(rewritten, candidate_limit)
            return rewritten, self._as_chunks(candidates, result_limit)
        elif method == "hybrid":
            keyword = self.bm25_search(rewritten, candidate_limit)
            semantic = self.vector_search(rewritten, candidate_limit)
            candidates = reciprocal_rank_fusion(
                [[item[0] for item in keyword], [item[0] for item in semantic]], k=60
            )[:candidate_limit]
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        return rewritten, self.rerank(rewritten, candidates, result_limit)


def answer_question(
    query: str,
    contexts: Sequence[RetrievedChunk],
    *,
    prompt_style: Literal["basic", "cot"] = "basic",
) -> tuple[str, int]:
    """Generate an answer, falling back to cited extracts when generation is unavailable."""
    if not contexts:
        return "I could not find relevant evidence in the indexed documents.", 0

    context_text = "\n\n".join(
        f"[Source {index}: {chunk.filename}#{chunk.chunk_id}]\n{chunk.content}"
        for index, chunk in enumerate(contexts, start=1)
    )
    system = (
        "You are a macroeconomic and financial-policy research assistant. Answer in clear English, "
        "even when the question is Vietnamese. Use only the supplied evidence. Cite every material "
        "claim inline as [Source N]. If evidence is insufficient, say so explicitly."
    )
    if prompt_style == "cot":
        system += (
            " Reason through consistency, dates, and possible conflicts internally before "
            "answering. "
            "Do not reveal private reasoning; provide a concise evidence-backed conclusion."
        )

    def fallback(reason: str | None = None) -> tuple[str, int]:
        notice = f"⚠️ {reason}\n\n" if reason else ""
        extracts = "\n\n".join(
            f"[Source {index}: {chunk.filename}#{chunk.chunk_id}] {chunk.content}"
            for index, chunk in enumerate(contexts, start=1)
        )
        return notice + "Most relevant retrieved evidence:\n\n" + extracts, max(
            1, len((query + context_text).split())
        )

    if not os.getenv("DEEPSEEK_API_KEY"):
        return fallback("DeepSeek generation is disabled; showing cited extracts.")

    try:
        response = chat_completion(
            get_deepseek_client(),
            os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {query}\n\nEvidence:\n{context_text}"},
            ],
        )
        usage = response.usage
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return (response.choices[0].message.content or "").strip(), total_tokens
    except Exception:
        return fallback("DeepSeek generation failed; showing cited extracts instead.")
