"""``precis topics``, ``precis report`` and ``precis site``: corpus-level outputs.

Each command reads the finished summaries from the store. ``topics`` and
``report`` make requests (recorded as their own run, and served from the
response cache when nothing changed); ``site`` only renders.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from precis.config import Config
from precis.corpus.load import FinishedPaper, load_papers
from precis.corpus.report import render_report
from precis.corpus.site import SiteContent, build_site
from precis.corpus.stats import faithfulness, spend
from precis.corpus.synthesis import (
    REPORT_ARTIFACT,
    REPORT_STATS_ARTIFACT,
    ReportStats,
    write_report,
)
from precis.corpus.topics import TOPICS_ARTIFACT, TopicModel, build_topics
from precis.llm import AsyncAnthropicLike, Gateway, make_client
from precis.models import CorpusReport
from precis.offline import offline_client
from precis.store import Store
from precis.util import sha256_hex

console = Console()

Offline = Annotated[
    bool, typer.Option("--offline", help="Answer requests with deterministic offline stand-ins.")
]


# --------------------------------------------------------------------------
# Shared with ``precis eval``
# --------------------------------------------------------------------------


def config_of(ctx: typer.Context) -> Config:
    from precis.cli import get_config  # the CLI root imports this module

    return get_config(ctx)


def model_client(config: Config, *, offline: bool) -> AsyncAnthropicLike:
    """The real SDK client, or the offline client that plays every role."""
    return offline_client() if offline else make_client(config)


async def with_gateway[T](
    store: Store,
    config: Config,
    command: str,
    work: Callable[[Gateway], Awaitable[T]],
    *,
    offline: bool,
) -> tuple[T, Gateway]:
    """Run ``work`` with a gateway that is recorded as its own run in the store.

    The run's stats are stored even when ``work`` fails, so the spend of a
    failed command is still accounted for.
    """
    run_id = f"{command}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    store.start_run(run_id, {"command": command, "config": config.model_dump(mode="json")})
    gateway = Gateway(model_client(config, offline=offline), store, run_id=run_id, config=config)
    try:
        result = await work(gateway)
    finally:
        await gateway.aclose()
        store.finish_run(run_id, asdict(gateway.stats()))
    return result, gateway


def require_papers(store: Store) -> list[FinishedPaper]:
    papers = load_papers(store)
    if not papers:
        console.print("[red]No summarized papers in the store.[/] Run [bold]precis run[/] first.")
        raise typer.Exit(1)
    return papers


def fail(message: str) -> typer.Exit:
    console.print(f"[red]{message}[/]")
    return typer.Exit(1)


# --------------------------------------------------------------------------
# Stored corpus artifacts
# --------------------------------------------------------------------------


def load_topics(store: Store) -> TopicModel | None:
    raw = store.get_artifact(TOPICS_ARTIFACT)
    return None if raw is None else TopicModel.model_validate(raw)


def topics_digest(model: TopicModel) -> str:
    """Identifies a topic model, so a report written from another one is detected."""
    return sha256_hex(model.model_dump_json())[:16]


def load_report(store: Store, model: TopicModel) -> tuple[CorpusReport, ReportStats] | None:
    """The stored report, if it was written from ``model``."""
    raw_report = store.get_artifact(REPORT_ARTIFACT)
    raw_stats = store.get_artifact(REPORT_STATS_ARTIFACT)
    if raw_report is None or raw_stats is None:
        return None
    stats = ReportStats.model_validate(raw_stats)
    if stats.topics_digest != topics_digest(model):
        return None
    return CorpusReport.model_validate(raw_report), stats


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _topics_table(model: TopicModel) -> Table:
    table = Table(title=f"{len(model.topics)} topics")
    table.add_column("#", justify="right")
    table.add_column("Topic")
    table.add_column("Papers", justify="right")
    table.add_column("Keywords")
    for topic in model.topics:
        table.add_row(
            str(topic.id), topic.label, str(len(topic.paper_ids)), ", ".join(topic.keywords[:5])
        )
    return table


def topics_command(
    ctx: typer.Context,
    k: Annotated[
        int | None, typer.Option("--k", min=1, help="Number of topics (default: silhouette).")
    ] = None,
    offline: Offline = False,
) -> None:
    """Cluster the summaries into topics and name each one."""
    config = config_of(ctx)
    with Store(config.store_path) as store:
        papers = require_papers(store)
        agent = config.agents.synthesizer

        async def work(gw: Gateway) -> TopicModel:
            return await build_topics(gw, papers, config.corpus, agent, k=k)

        model, gateway = asyncio.run(with_gateway(store, config, "topics", work, offline=offline))
        store.put_artifact(TOPICS_ARTIFACT, model)
    console.print(_topics_table(model))
    if model.silhouette:
        best = model.silhouette[model.k]
        console.print(f"k = {model.k} by silhouette ({best:.3f}; tried {min(model.silhouette)}"
                      f"-{max(model.silhouette)}).")  # fmt: skip
    console.print(f"Spent ${gateway.spent_usd:.2f} in {gateway.stats().calls} requests.")


def report_command(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option("--out", help="Markdown file to write.")] = Path(
        "reports/report.md"
    ),
    offline: Offline = False,
) -> None:
    """Write a synthesis of every topic and render the report as Markdown."""
    config = config_of(ctx)
    with Store(config.store_path) as store:
        papers = require_papers(store)
        model = load_topics(store)
        if model is None:
            raise fail("No topics yet. Run precis topics first.")

        async def work(gw: Gateway) -> tuple[CorpusReport, ReportStats]:
            return await write_report(
                gw,
                model.topics,
                papers,
                config.agents.synthesizer,
                budget=config.corpus.synthesis_input_tokens,
            )

        (report, stats), gateway = asyncio.run(
            with_gateway(store, config, "report", work, offline=offline)
        )
        stats = stats.model_copy(update={"topics_digest": topics_digest(model)})
        store.put_artifact(REPORT_ARTIFACT, report)
        store.put_artifact(REPORT_STATS_ARTIFACT, stats)
        text = render_report(
            report,
            model.topics,
            papers,
            stats=stats,
            faith=faithfulness(papers),
            cost=spend(store.usage()),
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    console.print(f"Wrote [bold]{out}[/]: {report.title}")
    console.print(
        f"{len(report.topics)} topic syntheses, {stats.citations} citations kept, "
        f"{stats.invalid_citations} removed; spent ${gateway.spent_usd:.2f}."
    )


def site_command(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option("--out", help="Directory to write the site into.")] = Path(
        "build/site"
    ),
) -> None:
    """Render the static website (run precis topics, and ideally precis report, first)."""
    config = config_of(ctx)
    with Store(config.store_path) as store:
        papers = require_papers(store)
        model = load_topics(store)
        if model is None:
            raise fail("No topics yet. Run precis topics first.")
        stored = load_report(store, model)
        if stored is None:
            console.print(
                "[yellow]No report for the current topics; topic pages will have no "
                "synthesis.[/] Run precis report to add it."
            )
        content = SiteContent(
            papers=papers,
            topics=model.topics,
            report=stored[0] if stored else None,
            cost=spend(store.usage()),
        )
        build = build_site(content, out)
    console.print(
        f"Wrote [bold]{out}[/]: {build.pages:,} pages, {build.bytes / 1e6:.1f} MB "
        f"in {build.seconds:.2f}s. Open {out / 'index.html'}"
    )


def register(app: typer.Typer) -> None:
    """Add the corpus commands to ``app``."""
    app.command("topics")(topics_command)
    app.command("report")(report_command)
    app.command("site")(site_command)
