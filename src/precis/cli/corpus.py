"""``precis topics``, ``precis report`` and ``precis site``: corpus-level outputs."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    """Add the corpus commands to ``app``."""
