.PHONY: setup run eval test lint clean

setup:
	uv sync --extra dev
	uv run python -m src.ingestion

run:
	uv run streamlit run src/app.py

eval:
	uv run python -m src.eval

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

clean:
	rm -f data/knowledge.duckdb data/knowledge.duckdb.wal data/feedback.db
