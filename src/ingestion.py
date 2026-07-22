"""Automated document ingestion into a local DuckDB knowledge base via dlt."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("RUNTIME__DLTHUB_TELEMETRY", "false")
os.environ.setdefault("DLT_DATA_DIR", str(Path(".dlt/runtime").resolve()))

import dlt  # noqa: E402
from dlt.common.runtime import run_context  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_DB_PATH = Path("data/knowledge.duckdb")
DEFAULT_PIPELINES_DIR = Path(".dlt/pipelines")
SUPPORTED_SUFFIXES = {".md", ".markdown", ".pdf", ".txt"}


def read_document(path: Path) -> str:
    """Return text from a supported UTF-8 text or PDF document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported document type: {path.suffix}")


def build_chunks(
    raw_dir: Path = DEFAULT_RAW_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, object]]:
    """Split documents recursively and attach stable source metadata."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    created_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, object]] = []

    paths = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in paths:
        text = read_document(path).strip()
        if not text:
            continue
        relative_name = path.relative_to(raw_dir).as_posix()
        for index, chunk in enumerate(splitter.split_text(text)):
            chunk_id = f"{Path(relative_name).stem}-{index:04d}"
            rows.append(
                {
                    "id": hashlib.sha256(f"{relative_name}:{index}:{chunk}".encode()).hexdigest()[
                        :20
                    ],
                    "filename": relative_name,
                    "chunk_id": chunk_id,
                    "content": chunk,
                    "created_at": created_at,
                }
            )
    return rows


@dlt.resource(name="chunks", write_disposition="replace", primary_key="id")
def chunk_resource(rows: list[dict[str, object]]) -> Iterator[dict[str, object]]:
    """Expose document chunks as a replaceable dlt resource."""
    yield from rows


def ingest(
    raw_dir: Path = DEFAULT_RAW_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """Load all source chunks into DuckDB and return the loaded row count."""
    rows = build_chunks(raw_dir, chunk_size, chunk_overlap)
    if not rows:
        raise RuntimeError(f"No supported, non-empty documents found in {raw_dir}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    # dlt's root-user fallback is /var/dlt; keep all runtime state project-local.
    run_context.active()._global_dir = str(Path(".dlt/runtime").resolve())
    pipeline = dlt.pipeline(
        pipeline_name="macro_policy_ingestion",
        pipelines_dir=str(DEFAULT_PIPELINES_DIR),
        destination=dlt.destinations.duckdb(credentials=str(db_path)),
        dataset_name="macro_policy",
    )
    pipeline.run(chunk_resource(rows), table_name="chunks")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()
    count = ingest(args.raw_dir, args.db_path, args.chunk_size, args.chunk_overlap)
    print(f"Loaded {count} chunks into {args.db_path}")


if __name__ == "__main__":
    main()
