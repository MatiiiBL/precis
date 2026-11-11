"""``precis fetch``: find papers and download their full text."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    """Add the ingestion commands to ``app``."""
