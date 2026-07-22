"""Retrieval and LLM-as-a-Judge evaluation for the macro-policy assistant."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

from src.retrieval import (
    AdvancedRetriever,
    RetrievedChunk,
    answer_question,
    chat_completion,
    get_deepseek_client,
)

DEFAULT_GROUND_TRUTH = Path("data/ground_truth.csv")


def hit_rate(expected: Sequence[str], predictions: Sequence[Sequence[str]]) -> float:
    """Share of queries whose relevant document appears in the result list."""
    if len(expected) != len(predictions):
        raise ValueError("Expected labels and predictions must have equal lengths")
    if not expected:
        return 0.0
    return mean(label in result for label, result in zip(expected, predictions, strict=True))


def mean_reciprocal_rank(expected: Sequence[str], predictions: Sequence[Sequence[str]]) -> float:
    """Mean inverse rank of the first relevant document, or zero when absent."""
    if len(expected) != len(predictions):
        raise ValueError("Expected labels and predictions must have equal lengths")
    if not expected:
        return 0.0
    reciprocal_ranks = []
    for label, result in zip(expected, predictions, strict=True):
        try:
            reciprocal_ranks.append(1.0 / (list(result).index(label) + 1))
        except ValueError:
            reciprocal_ranks.append(0.0)
    return mean(reciprocal_ranks)


def load_ground_truth(path: Path = DEFAULT_GROUND_TRUTH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"question", "answer", "source_file"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Ground truth must contain columns: {sorted(required)}")
    return rows


def _filenames(chunks: Sequence[RetrievedChunk]) -> list[str]:
    return [chunk.filename for chunk in chunks]


def evaluate_retrieval(
    retriever: AdvancedRetriever, ground_truth: Sequence[dict[str, str]], limit: int = 10
) -> dict[str, dict[str, float]]:
    """Compare BM25, vector search, and the selected hybrid re-ranked pipeline."""
    expected = [row["source_file"] for row in ground_truth]
    results: dict[str, dict[str, float]] = {}
    methods = {
        "Text Search (BM25)": "bm25",
        "Vector Search": "vector",
        "Hybrid + Re-ranker": "hybrid",
    }
    for label, method in methods.items():
        predictions = []
        for row in ground_truth:
            _, chunks = retriever.search(
                row["question"],
                rewrite=False,
                candidate_limit=max(10, limit),
                result_limit=limit,
                method=method,  # type: ignore[arg-type]
            )
            predictions.append(_filenames(chunks))
        results[label] = {
            "hit_rate": hit_rate(expected, predictions),
            "mrr": mean_reciprocal_rank(expected, predictions),
        }
    return results


def judge_answer(question: str, reference: str, answer: str) -> dict[str, int | str]:
    """Score relevance and faithfulness from 1-5 using a structured LLM judge."""
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required for LLM-as-a-Judge evaluation")
    response = chat_completion(
        get_deepseek_client(),
        os.getenv("DEEPSEEK_JUDGE_MODEL", os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")),
        [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator. Score the candidate answer for relevance to the "
                    "question and faithfulness to the reference, each from 1 (poor) to 5 "
                    "(excellent). Return only JSON with integer keys relevance and faithfulness "
                    "plus a short rationale."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nReference answer: {reference}\n"
                    f"Candidate answer: {answer}"
                ),
            },
        ],
    )
    payload = (response.choices[0].message.content or "").strip()
    payload = payload.removeprefix("```json").removesuffix("```").strip()
    result = json.loads(payload)
    for key in ("relevance", "faithfulness"):
        result[key] = max(1, min(5, int(result[key])))
    return result


def evaluate_llm(
    retriever: AdvancedRetriever, ground_truth: Sequence[dict[str, str]], sample_size: int = 10
) -> dict[str, dict[str, float]]:
    """Compare basic and reasoning-guided prompts with an LLM judge."""
    scores: dict[str, list[dict[str, int | str]]] = {"Basic Prompt": [], "CoT Prompt": []}
    for row in list(ground_truth)[:sample_size]:
        _, contexts = retriever.search(row["question"], result_limit=3)
        for label, style in (("Basic Prompt", "basic"), ("CoT Prompt", "cot")):
            answer, _ = answer_question(row["question"], contexts, prompt_style=style)  # type: ignore[arg-type]
            scores[label].append(judge_answer(row["question"], row["answer"], answer))
    return {
        label: {
            "relevance": mean(int(score["relevance"]) for score in values),
            "faithfulness": mean(int(score["faithfulness"]) for score in values),
        }
        for label, values in scores.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=Path("data/knowledge.duckdb"))
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--judge-sample", type=int, default=10)
    args = parser.parse_args()

    retriever = AdvancedRetriever(args.db_path)
    ground_truth = load_ground_truth(args.ground_truth)
    retrieval_results = evaluate_retrieval(retriever, ground_truth)
    print("\nRetrieval evaluation")
    print("Method                         Hit Rate    MRR")
    for method, metrics in retrieval_results.items():
        print(f"{method:<30} {metrics['hit_rate']:.3f}      {metrics['mrr']:.3f}")

    if args.llm_judge:
        print("\nLLM-as-a-Judge evaluation (1-5)")
        print(json.dumps(evaluate_llm(retriever, ground_truth, args.judge_sample), indent=2))


if __name__ == "__main__":
    main()
