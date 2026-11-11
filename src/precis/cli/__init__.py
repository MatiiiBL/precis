"""The ``precis`` command line.

Each area registers its own subcommands from its own module, so the packages
can grow without touching each other's CLI code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from precis import __version__
from precis.cli import corpus, evaluate, fetch, run
from precis.config import Config, load_config

app = typer.Typer(
    name="precis",
    help="Summarize research papers at scale with LLM agents.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@dataclass
class AppState:
    config: Config


def get_config(ctx: typer.Context) -> Config:
    state = ctx.ensure_object(AppState)
    return state.config


def _version(value: bool) -> None:
    if value:
        typer.echo(f"precis {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c", help="TOML config file (default: ./precis.toml if present)."
        ),
    ] = None,
    store: Annotated[
        Path | None, typer.Option("--store", help="SQLite store path (overrides the config).")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Print the version.")
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(config)
    if store is not None:
        cfg = cfg.model_copy(update={"store_path": store})
    ctx.obj = AppState(config=cfg)


for _module in (fetch, run, corpus, evaluate):
    _module.register(app)


def main() -> None:
    app()
