"""``precis eval`` and ``precis bench``: quality evaluation and throughput benchmarks.

Every command writes ``<name>.json`` and ``<name>.md`` to ``eval.results_dir``
(``results/`` by default), so the numbers in the docs can be regenerated; the
benchmarks also draw a figure into ``eval.figures_dir``.

Offline, the evaluations run on virtual time against a fake API with a
latency model (the one the benchmarks use), so their latency columns are
simulated rather than zero.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown

from precis.cli.corpus import config_of, fail, require_papers, with_gateway
from precis.config import Config
from precis.eval.ablation import ablation_markdown, run_ablation
from precis.eval.abstract import abstract_markdown, evaluate_abstract
from precis.eval.bench import (
    LATENCY,
    ThroughputPoint,
    faults_markdown,
    run_faults,
    run_throughput,
    throughput_figure,
    throughput_markdown,
)
from precis.eval.faithfulness import evaluate_faithfulness, faithfulness_markdown
from precis.eval.results import write_results
from precis.eval.runs import Paper, sample_papers
from precis.eval.simclock import VirtualClock
from precis.llm import AsyncAnthropicLike, Clock, Gateway, RealClock, make_client
from precis.llm.fake import FakeAnthropic
from precis.models import Document
from precis.offline import RESPONDERS
from precis.store import Store

console = Console()

eval_app = typer.Typer(help="Evaluate the quality of the summaries.", no_args_is_help=True)
bench_app = typer.Typer(help="Benchmark the pipeline offline.", no_args_is_help=True)

N = Annotated[int | None, typer.Option("--n", min=1, help="Papers to sample (default: eval.n).")]
Offline = Annotated[
    bool, typer.Option("--offline", help="Answer requests with deterministic offline stand-ins.")
]


def _show(markdown: str, paths: Sequence[Path]) -> None:
    console.print(Markdown(markdown))
    console.print("Wrote " + ", ".join(str(p) for p in paths))


def _client(config: Config, *, offline: bool) -> tuple[AsyncAnthropicLike, Clock]:
    if offline:
        clock = VirtualClock()
        return FakeAnthropic(RESPONDERS, clock=clock, latency=LATENCY), clock
    return make_client(config), RealClock()


def _run[T](clock: Clock, coro: Coroutine[Any, Any, T]) -> T:
    if isinstance(clock, VirtualClock):
        return asyncio.run(clock.run(coro))
    return asyncio.run(coro)


def _evaluate[R: BaseModel](
    ctx: typer.Context,
    n: int | None,
    offline: bool,
    command: str,
    evaluate: Callable[[Gateway, Store, Clock, list[Paper], Config], Awaitable[R]],
) -> tuple[Config, R]:
    """Sample papers and run ``evaluate`` over them with a recorded gateway."""
    config = config_of(ctx)
    with Store(config.store_path) as store:
        papers = sample_papers(store, n or config.eval.n, seed=config.eval.seed)
        if not papers:
            raise fail("No papers with full text and an abstract. Run precis fetch first.")
        client, clock = _client(config, offline=offline)

        async def work(gw: Gateway) -> R:
            return await evaluate(gw, store, clock, papers, config)

        result, gateway = _run(
            clock, with_gateway(store, config, command, work, client=client, clock=clock)
        )
    console.print(f"Spent ${gateway.spent_usd:.2f} in {gateway.stats().calls} requests.")
    return config, result


@eval_app.command("abstract")
def abstract_command(ctx: typer.Context, n: N = None, offline: Offline = False) -> None:
    """Summarize sampled papers without their abstract and compare with it."""
    config, result = _evaluate(
        ctx,
        n,
        offline,
        "eval-abstract",
        lambda gw, store, clock, papers, config: evaluate_abstract(
            gw, store, clock, papers, config.agents, seed=config.eval.seed
        ),
    )
    markdown = abstract_markdown(result, model=config.agents.judge.model)
    _show(markdown, write_results(config.eval.results_dir, "abstract", result, markdown))


@eval_app.command("ablation")
def ablation_command(ctx: typer.Context, n: N = None, offline: Offline = False) -> None:
    """Compare the agentic, fulltext and mapreduce strategies on the same papers."""
    config, result = _evaluate(
        ctx,
        n,
        offline,
        "eval-ablation",
        lambda gw, store, clock, papers, config: run_ablation(
            gw, store, clock, papers, config.agents, seed=config.eval.seed
        ),
    )
    markdown = ablation_markdown(result)
    _show(markdown, write_results(config.eval.results_dir, "ablation", result, markdown))


@eval_app.command("faithfulness")
def faithfulness_command(ctx: typer.Context) -> None:
    """Verdict rates before and after revision, and an audit of the evidence quotes."""
    config = config_of(ctx)
    with Store(config.store_path) as store:
        papers = require_papers(store)
        documents: dict[str, Document] = {}
        for paper in papers:
            doc = store.get_document(paper.id)
            if doc is not None:
                documents[paper.id] = doc
    result = evaluate_faithfulness(papers, documents)
    markdown = faithfulness_markdown(result)
    _show(markdown, write_results(config.eval.results_dir, "faithfulness", result, markdown))


@bench_app.command("throughput")
def throughput_command(
    ctx: typer.Context,
    papers: Annotated[
        int | None, typer.Option("--papers", min=1, help="Papers (default: eval.bench.papers).")
    ] = None,
    concurrency: Annotated[
        list[int] | None,
        typer.Option("--concurrency", min=1, help="Ceilings to sweep; repeat the option."),
    ] = None,
) -> None:
    """Run the full pipeline offline over synthetic papers at several concurrency ceilings."""
    config = config_of(ctx)
    bench = config.eval.bench
    if concurrency:
        bench = bench.model_copy(update={"concurrency": sorted(set(concurrency))})

    def progress(point: ThroughputPoint) -> None:
        console.print(
            f"ceiling {point.concurrency:>4}: {point.papers_per_minute:6.1f} papers/min, "
            f"{point.requests_per_minute:6.0f} requests/min, "
            f"{100 * point.rate_limited_share:.2f}% 429s ({point.wall_seconds:.0f} s)"
        )

    result = asyncio.run(run_throughput(config, bench, papers=papers, on_point=progress))
    markdown = throughput_markdown(result)
    paths = write_results(config.eval.results_dir, "bench-throughput", result, markdown)
    figure = throughput_figure(result, config.eval.figures_dir / "bench-throughput.png")
    _show(markdown, [*paths, figure])


@bench_app.command("faults")
def faults_command(
    ctx: typer.Context,
    papers: Annotated[
        int | None,
        typer.Option("--papers", min=1, help="Papers (default: eval.bench.faults_papers)."),
    ] = None,
) -> None:
    """Inject API errors, kill the run with SIGKILL, resume it, and check nothing is lost."""
    config = config_of(ctx)
    bench = config.eval.bench
    result = run_faults(
        config,
        bench,
        papers=papers or bench.faults_papers,
        concurrency=bench.faults_concurrency,
    )
    markdown = faults_markdown(result)
    _show(markdown, write_results(config.eval.results_dir, "bench-faults", result, markdown))
    if not result.passed:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    """Add the evaluation and benchmark commands to ``app``."""
    app.add_typer(eval_app, name="eval")
    app.add_typer(bench_app, name="bench")
