"""``precis eval`` and ``precis bench``: quality evaluation and throughput benchmarks."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    """Add the evaluation and benchmark commands to ``app``."""
