from datetime import UTC, datetime
from pathlib import Path

from precis.config import BenchConfig, Config
from precis.eval.bench import (
    faults_markdown,
    paid_twice,
    request_hash,
    run_faults,
    run_throughput,
    throughput_figure,
    throughput_markdown,
)
from precis.models import LLMCall
from precis.util import canonical_json, sha256_hex


async def test_throughput_sweep_runs_every_paper_at_every_ceiling(tmp_path: Path) -> None:
    bench = BenchConfig(papers=30, concurrency=[2, 16])
    seen = []
    result = await run_throughput(Config(), bench, on_point=seen.append)
    assert [p.concurrency for p in result.points] == [2, 16]
    assert seen == result.points
    low, high = result.points
    assert low.finished == high.finished == 30
    assert high.papers_per_minute > 2 * low.papers_per_minute
    assert high.max_in_flight > low.max_in_flight
    assert low.paper_latency_p50_s <= low.paper_latency_p95_s
    assert "| 16 |" in throughput_markdown(result)
    figure = throughput_figure(result, tmp_path / "bench.png")
    assert figure.stat().st_size > 10_000


async def test_throughput_is_reproducible() -> None:
    bench = BenchConfig(papers=12, concurrency=[8])
    first = await run_throughput(Config(), bench)
    second = await run_throughput(Config(), bench)
    keys = {"wall_seconds"}
    assert first.points[0].model_dump(exclude=keys) == second.points[0].model_dump(exclude=keys)


async def test_the_gateway_paces_itself_under_a_tight_limit() -> None:
    # 60 requests/min with 64 slots: the gateway learns the limit from the
    # response headers and paces requests instead of collecting 429s.
    bench = BenchConfig(papers=40, concurrency=[64], rpm=60)
    point = (await run_throughput(Config(), bench)).points[0]
    assert point.finished == 40
    assert point.requests_per_minute <= 60 * 1.1
    assert point.rate_limited_share < 0.05


def test_request_hash_matches_the_gateway() -> None:
    params = {"model": "m", "max_tokens": 10, "messages": []}
    sent = {**params, "betas": ["x"], "fallbacks": "default"}
    assert request_hash(sent) == sha256_hex(canonical_json(params))


def test_paid_twice_counts_repeated_paid_hashes() -> None:
    def call(i: int, request_hash: str, status: str) -> LLMCall:
        return LLMCall(call_id=str(i), run_id="r", paper_id="p", stage="read", model="m",
                       request_hash=request_hash, status=status,  # type: ignore[arg-type]
                       started_at=datetime(2026, 1, 1, tzinfo=UTC))  # fmt: skip

    calls = [call(0, "a", "ok"), call(1, "a", "cached"), call(2, "b", "ok"), call(3, "b", "ok")]
    assert paid_twice(calls) == 1


def test_a_crash_and_resume_loses_nothing_and_pays_nothing_twice(tmp_path: Path) -> None:
    result = run_faults(Config(), BenchConfig(), papers=40, concurrency=8, workdir=tmp_path)
    assert result.passed
    assert (result.papers_lost, result.paid_twice) == (0, 0)
    assert result.papers_done + result.papers_failed == 40
    assert 16 <= result.finished_at_kill < 40
    assert result.resent_requests <= result.in_flight_at_kill
    assert result.kept_usd > 0
    assert result.retries > 0
    assert "passed" in faults_markdown(result)
