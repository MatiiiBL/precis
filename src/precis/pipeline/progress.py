"""The live progress view of ``precis run``.

It re-renders a few times per second from two snapshots that cost nothing
to take: the scheduler's in-memory ``RunStats`` and ``Gateway.stats()``.
It never queries the store, so it stays cheap on a run of thousands of
papers.
"""

from __future__ import annotations

from types import TracebackType

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from precis.llm import GatewayStats
from precis.pipeline.scheduler import Pipeline
from precis.pipeline.stats import RunStats


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = round(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def render_progress(
    run_id: str, stats: RunStats, gateway: GatewayStats, *, budget_usd: float | None
) -> RenderableType:
    """Stage counts, requests in flight, concurrency, tokens, spend and ETA."""
    finished = stats.papers_finished + stats.papers_blocked
    header = Text.assemble(
        ("Run ", "bold"),
        (run_id, "bold cyan"),
        f"  {finished}/{stats.papers} papers  ·  {format_duration(stats.elapsed_s)} elapsed"
        f"  ·  ETA {format_duration(stats.eta_s())}",
    )
    if stats.interrupted:
        header.append("  ·  interrupted, finishing up", style="bold yellow")
    elif stats.budget_reached:
        header.append("  ·  budget reached, no new papers", style="bold yellow")

    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("stage")
    for name in ("running", "done", "cached", "skipped", "failed", "refused"):
        table.add_column(name, justify="right")
    for stage in stats.stages:
        t = stats.tally[stage]
        table.add_row(
            stage.value,
            str(t.running),
            str(t.done),
            str(t.cached),
            str(t.skipped),
            Text(str(t.failed), style="red" if t.failed else ""),
            Text(str(t.refused), style="yellow" if t.refused else ""),
        )

    g = gateway
    requests = Text(
        f"Requests  {g.in_flight} in flight, {g.queued} queued, concurrency limit "
        f"{g.concurrency_limit}  ·  {g.calls:,} calls ({g.cached_calls:,} cached, "
        f"{g.retries:,} retries, {g.rate_limited:,} rate-limited, {g.overloaded:,} overloaded)"
    )
    spend = f"${g.cost_usd:,.2f}" + (f" of ${budget_usd:,.2f}" if budget_usd is not None else "")
    tokens = Text(
        f"Tokens    {g.input_tokens:,} in, {g.cache_read_input_tokens:,} cache read, "
        f"{g.cache_creation_input_tokens:,} cache write, {g.output_tokens:,} out  ·  {spend}"
    )
    if g.batches_pending:
        tokens.append(f"  ·  {g.batches_pending} batches pending")
    return Group(header, table, requests, tokens)


class LiveProgress:
    """Context manager that shows ``render_progress`` while a run is going."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        console: Console | None = None,
        refresh_per_second: float = 4,
    ) -> None:
        self._pipeline = pipeline
        self._live = Live(
            get_renderable=self.render,
            console=console,
            refresh_per_second=refresh_per_second,
            transient=False,
        )

    def render(self) -> RenderableType:
        pipeline = self._pipeline
        return render_progress(
            pipeline.run_id,
            pipeline.stats,
            pipeline.gateway.stats(),
            budget_usd=pipeline.config.budget.max_cost_usd,
        )

    def __enter__(self) -> LiveProgress:
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._live.__exit__(exc_type, exc, tb)
