.PHONY: install check lint format typecheck test clean

install:
	uv sync

check: lint typecheck test

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy

test:
	uv run pytest

clean:
	rm -rf build dist .mypy_cache .pytest_cache .ruff_cache
