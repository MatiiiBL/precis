.PHONY: install check lint format typecheck test clean

install:
	uv sync

check: lint typecheck test

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:
	uv run mypy

test:
	uv run pytest

clean:
	rm -rf build dist .mypy_cache .pytest_cache .ruff_cache
