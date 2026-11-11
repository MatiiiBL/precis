"""``precis run`` and ``precis status``: the per-paper agent pipeline."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    """Add the pipeline commands to ``app``."""
