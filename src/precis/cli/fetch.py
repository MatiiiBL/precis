"""``precis fetch``: find papers and download their full text.

Each command runs in two phases. First the metadata: an arXiv search, an id
lookup or a directory listing, stored as ``PaperMeta`` as it arrives. Then
the full text of every paper whose ``fetch`` stage is not already ``done``,
recorded per paper as it finishes (see ``precis.sources.runner``). Re-running
a command after an interruption or a failure picks up where it stopped.
"""

from __future__ import annotations

import asyncio
import secrets
import statistics
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from precis.config import Config, FetchConfig
from precis.models import PaperMeta
from precis.sources.arxiv import fetch_metadata, search
from precis.sources.atom import ArxivApiError
from precis.sources.fulltext import FullTextFetcher
from precis.sources.http import ArxivHttp, FetchError
from precis.sources.local import find_pdfs
from precis.sources.raw import RawStore
from precis.sources.runner import (
    FetchRunner,
    FetchStats,
    Outcome,
    OutcomeStatus,
    fetch_config_hash,
)
from precis.store import Store

fetch_app = typer.Typer(
    help="Find papers and fetch their full text into the store.", no_args_is_help=True
)
console = Console()

# Papers found, then a callback per paper; the CLI draws both on one progress view.
type Finder = Callable[[ArxivHttp, Callable[[PaperMeta], None]], Awaitable[list[PaperMeta]]]

DATE_FORMATS = ["%Y-%m-%d"]
_SHOWN: tuple[OutcomeStatus, ...] = ("html", "pdf", "failed", "skipped")
_LABELS: dict[OutcomeStatus, str] = {
    "html": "parsed from HTML",
    "pdf": "parsed from PDF",
    "failed": "failed",
    "skipped": "already done",
}


def register(app: typer.Typer) -> None:
    """Add the ingestion commands to ``app``."""
    app.add_typer(fetch_app, name="fetch")


def open_http(config: FetchConfig) -> ArxivHttp:
    """The client every command uses (tests replace it with one on a mock transport)."""
    return ArxivHttp(config)


@fetch_app.command("arxiv")
def fetch_arxiv(
    ctx: typer.Context,
    category: Annotated[
        list[str], typer.Option("--category", help="arXiv category, e.g. cs.AR; repeatable.")
    ],
    since: Annotated[
        datetime, typer.Option(formats=DATE_FORMATS, help="First submission date (inclusive).")
    ],
    until: Annotated[
        datetime | None,
        typer.Option(formats=DATE_FORMATS, help="Last submission date (inclusive; default today)."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Stop after this many papers (oldest first).")
    ] = None,
    metadata_only: Annotated[
        bool, typer.Option("--metadata-only", help="Store metadata; skip full text.")
    ] = False,
) -> None:
    """Search arXiv by category and submission date, then fetch full text."""
    config = _config(ctx)
    last = (until or datetime.now(UTC)).date()

    async def find(http: ArxivHttp, found: Callable[[PaperMeta], None]) -> list[PaperMeta]:
        papers: list[PaperMeta] = []
        results = search(
            http,
            category,
            since.date(),
            last,
            page_size=config.fetch.page_size,
            max_attempts=config.fetch.max_attempts,
        )
        async for meta in results:
            papers.append(meta)
            found(meta)
            if limit is not None and len(papers) >= limit:
                break
        return papers

    params = {
        "categories": category,
        "since": since.date().isoformat(),
        "until": last.isoformat(),
        "limit": limit,
    }
    _run_arxiv(config, "fetch arxiv", params, find, metadata_only=metadata_only)


@fetch_app.command("ids")
def fetch_ids(
    ctx: typer.Context,
    ids: Annotated[list[str], typer.Argument(help="arXiv ids, e.g. 2406.01234 or 2406.01234v2.")],
    metadata_only: Annotated[
        bool, typer.Option("--metadata-only", help="Store metadata; skip full text.")
    ] = False,
) -> None:
    """Fetch specific arXiv papers by id."""
    config = _config(ctx)

    async def find(http: ArxivHttp, found: Callable[[PaperMeta], None]) -> list[PaperMeta]:
        papers = await fetch_metadata(http, ids, max_attempts=config.fetch.max_attempts)
        for meta in papers:
            found(meta)
        return papers

    _run_arxiv(config, "fetch ids", {"ids": ids}, find, metadata_only=metadata_only)


@fetch_app.command("local")
def fetch_local(
    ctx: typer.Context,
    directory: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Directory of PDF files.")
    ],
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N files.")] = None,
) -> None:
    """Parse a directory of PDFs (papers that are not on arXiv)."""
    config = _config(ctx)
    paths = find_pdfs(directory)[:limit]
    run_id = _run_id()
    with Store(config.store_path) as store, _progress() as progress:
        store.start_run(run_id, {"command": "fetch local", "directory": str(directory)})
        task = progress.add_task("Parsing PDFs", total=len(paths), status="")
        runner = _runner(store, config, run_id, progress, task)
        runner.stats.found = len(paths)
        ending = _execute(store, run_id, runner, lambda: runner.fetch_local(paths))
    _report(runner.stats, ending)


def _run_arxiv(
    config: Config,
    command: str,
    params: dict[str, Any],
    find: Finder,
    *,
    metadata_only: bool,
) -> None:
    run_id = _run_id()
    with Store(config.store_path) as store, _progress() as progress:
        store.start_run(run_id, {"command": command, **params, "metadata_only": metadata_only})
        searching = progress.add_task("Finding papers", total=None, status="")
        fetching = progress.add_task("Fetching text", total=0, status="", visible=False)
        runner = _runner(store, config, run_id, progress, fetching)

        def found(meta: PaperMeta) -> None:
            runner.save_metadata(meta)
            if metadata_only:
                runner.mark_pending(meta.id)
            runner.stats.found += 1
            n = runner.stats.found
            progress.update(searching, completed=n, status=f"{n} papers")

        async def work() -> None:
            async with open_http(config.fetch) as http:
                try:
                    papers = await find(http, found)
                    progress.update(searching, total=len(papers), completed=len(papers))
                    if metadata_only:
                        return
                    progress.update(fetching, total=len(papers), visible=True)
                    raw = RawStore(config.fetch.raw_dir)
                    fetcher = FullTextFetcher(http, raw, config.fetch.prefer)
                    await runner.fetch_arxiv(fetcher, papers)
                except (ArxivApiError, FetchError, ValueError) as exc:
                    raise _Aborted(f"looking up papers failed: {exc}") from exc
                finally:
                    runner.stats.requests = http.requests

        ending = _execute(store, run_id, runner, work)
    _report(runner.stats, ending)


class _Aborted(Exception):
    """The command cannot go on (the search itself failed)."""


@dataclass(frozen=True)
class _Ending:
    exit_code: int = 0
    message: str | None = None


def _execute(
    store: Store, run_id: str, runner: FetchRunner, work: Callable[[], Coroutine[Any, Any, None]]
) -> _Ending:
    """Run ``work`` and record the run's outcome; Ctrl-C keeps what is done."""
    try:
        asyncio.run(work())
    except KeyboardInterrupt:
        store.finish_run(run_id, {**runner.stats.as_dict(), "interrupted": True})
        return _Ending(130, "[yellow]Interrupted.[/] Run the same command again to resume.")
    except _Aborted as exc:
        store.finish_run(run_id, {**runner.stats.as_dict(), "error": str(exc)})
        return _Ending(1, f"[red]Error:[/] {exc}")
    store.finish_run(run_id, runner.stats.as_dict())
    return _Ending()


def _report(stats: FetchStats, ending: _Ending) -> None:
    """Print the summary once the progress view is gone, then exit with the run's code."""
    if ending.exit_code != 1:
        _summary(stats)
    if ending.message:
        console.print(ending.message)
    if ending.exit_code:
        raise typer.Exit(ending.exit_code)


def _runner(
    store: Store, config: Config, run_id: str, progress: Progress, task: TaskID
) -> FetchRunner:
    def on_outcome(outcome: Outcome) -> None:
        progress.advance(task)
        counts = runner.stats.counts
        progress.update(task, status="  ".join(f"{name} {counts[name]}" for name in _SHOWN))
        if outcome.status == "failed":
            progress.console.print(f"[red]failed[/] {outcome.paper_id}: {outcome.detail}")

    # on_outcome reads the runner's running totals, so it closes over the name.
    runner = FetchRunner(
        store, config_hash=fetch_config_hash(config.fetch), run_id=run_id, on_outcome=on_outcome
    )
    if reset := runner.reset_interrupted():
        progress.console.print(f"reset {reset} fetch stages left running by an earlier crash")
    return runner


def _summary(stats: FetchStats) -> None:
    counts = stats.counts
    table = Table(title="Fetch summary", show_header=False, title_justify="left")
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("papers found", str(stats.found))
    for name in _SHOWN:
        table.add_row(_LABELS[name], str(counts[name]))
    if stats.requests:
        table.add_row("requests to arXiv", str(stats.requests))
    if stats.tokens:
        tokens = sorted(stats.tokens)
        table.add_row("tokens per document (median)", f"{statistics.median(tokens):,.0f}")
        table.add_row("tokens per document (min-max)", f"{tokens[0]:,}-{tokens[-1]:,}")
    console.print(table)
    for paper_id, reason in list(stats.failures.items())[:20]:
        console.print(f"  [red]{paper_id}[/]: {reason}")
    if len(stats.failures) > 20:
        console.print(f"  ... and {len(stats.failures) - 20} more (see `precis status`)")


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        console=console,
    )


def _config(ctx: typer.Context) -> Config:
    # Imported here: precis.cli imports this module while it is being set up.
    from precis.cli import get_config

    return get_config(ctx)


def _run_id() -> str:
    return f"fetch-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"
