import io

from rich.console import Console

from precis.llm import GatewayStats
from precis.models import Stage
from precis.pipeline import LiveProgress, RunStats, render_progress
from precis.pipeline.progress import format_duration

from .helpers import add_papers, setup


def gateway_stats(**overrides: int | float) -> GatewayStats:
    values: dict[str, int | float] = dict.fromkeys(
        (
            "in_flight",
            "queued",
            "concurrency_limit",
            "calls",
            "cached_calls",
            "retries",
            "rate_limited",
            "overloaded",
            "errors",
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "batches_pending",
        ),
        0,
    )
    values["cost_usd"] = 0.0
    values.update(overrides)
    return GatewayStats(**values)


def text(renderable: object) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=140, color_system=None).print(renderable)
    return buffer.getvalue()


def test_durations() -> None:
    assert format_duration(None) == "-"
    assert format_duration(42.4) == "42s"
    assert format_duration(125) == "2m05s"
    assert format_duration(3725) == "1h02m"


def test_view_shows_stages_requests_tokens_spend_and_eta() -> None:
    stats = RunStats(papers=10, stages=(Stage.READ, Stage.VERIFY, Stage.REVISE))
    stats.papers_finished = 4
    stats.tally[Stage.READ].done = 4
    stats.tally[Stage.READ].running = 3
    stats.tally[Stage.VERIFY].failed = 1
    g = gateway_stats(
        in_flight=5, concurrency_limit=16, calls=1234, input_tokens=98765, cost_usd=1.5
    )
    out = text(render_progress("run-x", stats, g, budget_usd=10.0))
    assert "Run run-x  4/10 papers" in out
    assert "ETA " in out
    assert "ETA -" not in out
    assert "read" in out
    assert "verify" in out
    assert "revise" in out
    assert "5 in flight, 0 queued, concurrency limit 16" in out
    assert "1,234 calls" in out
    assert "98,765 in" in out
    assert "$1.50 of $10.00" in out

    stats.interrupted = True
    assert "interrupted, finishing up" in text(render_progress("r", stats, g, budget_usd=None))


async def test_live_view_renders_a_real_pipeline() -> None:
    s = setup()
    ids = add_papers(s.store, 1)
    pipeline = s.pipeline("run-live")
    console = Console(file=io.StringIO(), width=140, color_system=None, force_terminal=False)
    with LiveProgress(pipeline, console=console) as live:
        await pipeline.run(ids)
        out = text(live.render())
    assert "1/1 papers" in out
    assert "Run run-live" in out
