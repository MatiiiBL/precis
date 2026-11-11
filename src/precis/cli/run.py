"""``precis run``, ``precis status`` and ``precis export``: the per-paper agent pipeline."""

from __future__ import annotations

import asyncio
import json
import secrets
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from precis.config import Config
from precis.corpus.load import load_papers
from precis.llm import Gateway, ScaledClock, UnpricedModelError, make_client
from precis.models import Stage, StageStatus
from precis.offline import offline_client, offline_conflict, spend_phrase
from precis.pipeline import PIPELINE_STAGES, LiveProgress, Pipeline, batch_settings
from precis.store import Store

console = Console()
INTERRUPTED_EXIT_CODE = 130
OFFLINE_TIME_SCALE = 600.0
"""Simulated seconds per wall-clock second in ``--offline`` runs."""


class Strategy(StrEnum):
    AGENTIC = "agentic"
    FULLTEXT = "fulltext"
    MAPREDUCE = "mapreduce"


class Mode(StrEnum):
    ONLINE = "online"
    BATCH = "batch"


def _config(ctx: typer.Context) -> Config:
    # Imported here: precis.cli imports this module while it is being set up.
    from precis.cli import get_config

    return get_config(ctx)


def parse_stages(value: str) -> list[Stage]:
    """``"read,verify"`` -> stages; ``fetch`` belongs to ``precis fetch``."""
    names = [v.strip() for v in value.split(",") if v.strip()]
    allowed = {s.value: s for s in PIPELINE_STAGES}
    unknown = [n for n in names if n not in allowed]
    if unknown or not names:
        raise typer.BadParameter(
            f"choose from {', '.join(allowed)} (got {', '.join(unknown) or 'nothing'})"
        )
    return [allowed[n] for n in names]


def select_papers(store: Store, ids: list[str] | None, limit: int | None) -> list[str]:
    """The given ids (comma-separated or repeated), or the newest papers with their text.

    Without ``ids``, only papers whose fetch stage is done are chosen, so that
    ``--limit`` is not spent on papers a running fetch has not reached yet.
    """
    if ids:
        chosen = list(dict.fromkeys(i.strip() for v in ids for i in v.split(",") if i.strip()))
    else:
        fetched = set(store.papers_with_status(Stage.FETCH, StageStatus.DONE))
        chosen = [meta.id for meta in store.list_papers() if meta.id in fetched]
    return chosen[:limit] if limit is not None else chosen


def new_run_id() -> str:
    return f"run-{datetime.now(UTC):%Y%m%dT%H%M%S}-{secrets.token_hex(3)}"


def run_command(
    ctx: typer.Context,
    stages: Annotated[
        str, typer.Option(help="Comma-separated stages to run, in pipeline order.")
    ] = "read,verify,revise",
    strategy: Annotated[
        Strategy | None, typer.Option(help="Summarization strategy (overrides the config).")
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="At most this many papers.")] = None,
    ids: Annotated[
        list[str] | None, typer.Option(help="Paper ids, comma-separated or repeated.")
    ] = None,
    mode: Annotated[
        Mode, typer.Option(help="online, or batch for the Message Batches API at half price.")
    ] = Mode.ONLINE,
    max_cost: Annotated[
        float | None,
        typer.Option(
            min=0,
            help="Stop admitting papers once the store's total recorded spend, over all runs, "
            "reaches this many USD.",
        ),
    ] = None,
    offline: Annotated[
        bool, typer.Option(help="Answer every request with the deterministic offline client.")
    ] = False,
) -> None:
    """Read, verify and revise papers with LLM agents."""
    config = _config(ctx)
    updates: dict[str, Any] = {}
    if strategy is not None:
        updates["agents"] = config.agents.model_copy(update={"strategy": strategy.value})
    if max_cost is not None:
        updates["budget"] = config.budget.model_copy(update={"max_cost_usd": max_cost})
    config = config.model_copy(update=updates)
    wanted = parse_stages(stages)

    store = Store(config.store_path)
    try:
        paper_ids = select_papers(store, ids, limit)
        if not paper_ids:
            console.print("No fetched papers in the store; run [bold]precis fetch[/bold] first.")
            raise typer.Exit(1)
        if offline and (conflict := offline_conflict(store)):
            console.print(f"[red]Cannot run offline:[/red] {conflict}")
            raise typer.Exit(1)
        if mode is Mode.BATCH:
            config = batch_settings(config, len(paper_ids))
        # Offline, simulated time runs fast so that batch waits (minutes of
        # queueing and polling) take a fraction of a second.
        clock = ScaledClock(OFFLINE_TIME_SCALE) if offline else None
        client = offline_client(clock=clock) if offline else make_client(config)
        try:
            gateway = Gateway(
                client,
                store,
                run_id=new_run_id(),
                config=config,
                clock=clock,
                mode=mode.value,
                offline=offline,
            )
        except UnpricedModelError as e:
            console.print(f"[red]Cannot start the run:[/red] {e}")
            raise typer.Exit(2) from e
        pipeline = Pipeline(config, store, gateway)
        stats = asyncio.run(_run(pipeline, paper_ids, wanted))
    finally:
        store.close()
    _print_run_summary(gateway.run_id, stats, offline=offline)
    if stats["interrupted"]:
        raise typer.Exit(INTERRUPTED_EXIT_CODE)


async def _run(pipeline: Pipeline, paper_ids: list[str], stages: list[Stage]) -> dict[str, Any]:
    try:
        with LiveProgress(pipeline, console=console):
            return await pipeline.run(paper_ids, stages)
    finally:
        await pipeline.gateway.aclose()


def _print_run_summary(run_id: str, stats: dict[str, Any], *, offline: bool) -> None:
    gateway = stats["gateway"]
    console.print(
        f"\n[bold]{run_id}[/bold]: {stats['papers_finished']} of {stats['papers']} papers "
        f"finished, {stats['papers_blocked']} blocked, in {stats['elapsed_s']:.1f}s; "
        f"{gateway['calls']:,} requests ({gateway['cached_calls']:,} from cache), "
        f"{spend_phrase(gateway['cost_usd'], offline=offline, digits=4)}."
    )
    if stats["blocked"]:
        reasons = "; ".join(f"{n:,} {reason}" for reason, n in stats["blocked"].items())
        console.print(f"Blocked: {reasons}.")
    faithfulness = stats["faithfulness"]
    final, first = faithfulness["final"], faithfulness["first_drafts"]
    if final["claims"]:
        console.print(
            f"Final summaries of {faithfulness['papers']:,} papers: "
            f"{final['supported_rate']:.1%} of {final['claims']:,} checked claims supported "
            f"(first drafts: {first['supported_rate']:.1%})."
        )
    revised = faithfulness["revised"]
    if revised["papers"]:
        console.print(
            f"Revised {revised['papers']:,} summaries: "
            f"{revised['before']['supported_rate']:.1%} of their claims supported before, "
            f"{revised['after']['supported_rate']:.1%} after."
        )
    if stats["budget_reached"]:
        console.print(
            "[yellow]The budget was reached: it caps the store's total spend, so raise "
            "--max-cost (or budget.max_cost_usd) to finish the rest.[/yellow]"
        )
    if stats["interrupted"]:
        console.print(
            "[yellow]Interrupted; unfinished stages are pending for the next run.[/yellow]"
        )


def status_command(ctx: typer.Context) -> None:
    """Show stage counts, recent runs and model usage."""
    config = _config(ctx)
    store = Store(config.store_path)
    try:
        console.print(f"[bold]{store.count_papers():,} papers[/bold] in {config.store_path}")
        console.print(_stage_table(store))
        runs = store.runs()[-5:]
        if runs:
            console.print(_runs_table(runs))
        usage = store.usage()
        if usage:
            console.print(_usage_table(store))
    finally:
        store.close()


def _stage_table(store: Store) -> Table:
    counts = store.stage_counts()
    table = Table(title="Stages", title_justify="left")
    table.add_column("stage")
    for status in StageStatus:
        table.add_column(status.value, justify="right")
    for stage in Stage:
        table.add_row(stage.value, *(f"{counts[stage][s]:,}" for s in StageStatus))
    return table


def _runs_table(runs: list[dict[str, Any]]) -> Table:
    table = Table(title="Recent runs", title_justify="left")
    for name in ("run", "started", "papers", "finished", "requests", "cost", "note"):
        table.add_column(name, justify="left" if name in ("run", "started", "note") else "right")
    for run in runs:
        stats = run["stats"] or {}
        # precis run nests the gateway's counters; corpus and eval commands
        # store them as the run's stats; fetch runs make no model requests.
        gateway = stats.get("gateway", stats if "calls" in stats else None)
        note = (
            "running or crashed"
            if run["finished_at"] is None
            else "interrupted"
            if stats.get("interrupted")
            else "budget reached"
            if stats.get("budget_reached")
            else ""
        )
        if (run["config"] or {}).get("offline"):
            note = f"offline, {note}" if note else "offline"
        table.add_row(
            run["run_id"],
            f"{run['started_at']:%Y-%m-%d %H:%M}",
            str(stats.get("papers", "")),
            str(stats.get("papers_finished", "")),
            f"{gateway['calls']:,}" if gateway else "",
            f"${gateway['cost_usd']:.4f}" if gateway else "",
            note,
        )
    return table


def _usage_table(store: Store) -> Table:
    table = Table(title="Usage (all runs)", title_justify="left")
    for name in ("stage", "model", "calls", "cached", "input", "cache read", "output", "cost"):
        table.add_column(name, justify="left" if name in ("stage", "model") else "right")
    for row in store.usage():
        table.add_row(
            row.stage,
            f"{row.model} (offline)" if row.offline else row.model,
            f"{row.calls:,}",
            f"{row.cached_calls:,}",
            f"{row.input_tokens + row.cache_creation_input_tokens:,}",
            f"{row.cache_read_input_tokens:,}",
            f"{row.output_tokens:,}",
            f"${row.cost_usd:.4f}",
        )
    return table


def export_command(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option(help="JSON Lines file to write.")] = Path("summaries.jsonl"),
) -> None:
    """Write every paper's final summary and verification as JSON Lines, by paper id."""
    config = _config(ctx)
    store = Store(config.store_path)
    try:
        papers = load_papers(store)
    finally:
        store.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for paper in papers:
            verification = paper.verification
            record = {
                "paper": paper.meta.model_dump(
                    mode="json",
                    include={"id", "title", "authors", "published", "primary_category", "url"},
                ),
                "summary": paper.summary.model_dump(mode="json"),
                "verification": verification.model_dump(mode="json") if verification else None,
                "supported_fraction": verification.supported_fraction() if verification else None,
                "revised": paper.revised,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"Wrote {len(papers):,} summaries to {out}.")


def register(app: typer.Typer) -> None:
    """Add the pipeline commands to ``app``."""
    app.command("run")(run_command)
    app.command("status")(status_command)
    app.command("export")(export_command)
