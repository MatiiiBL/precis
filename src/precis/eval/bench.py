"""Offline benchmarks of the full per-paper pipeline.

Both benchmarks run the real scheduler, agents, gateway and store against
``FakeAnthropic`` playing every role with the offline responders, on the
synthetic corpus (``precis.eval.synthetic``) and on virtual time
(``precis.eval.simclock``), so they measure the pipeline's behaviour under
rate limits and faults rather than the speed of the machine.

- ``throughput``: the same papers at each concurrency ceiling, against
  simulated server limits and a latency model.
- ``faults``: injected 429, 529, 500 and timeouts, plus a real crash: the
  first run is a child process killed with SIGKILL mid-run, and a second run
  resumes from the store it left behind.

Each run uses a store file in a scratch directory rather than ``:memory:``,
and turns off the fake's log of request bodies: both would otherwise keep
every document and conversation in memory (about 1 MB per paper).
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from precis.config import BenchConfig, Config
from precis.eval.figures import Panel, Series, small_multiples
from precis.eval.metrics import percentile
from precis.eval.results import markdown_table
from precis.eval.simclock import VirtualClock
from precis.eval.synthetic import synthetic_paper
from precis.llm import Clock, Gateway, Reply
from precis.llm.fake import FakeAnthropic, FakeReply, Faults, LatencyModel, Responder, ServerLimits
from precis.models import LLMCall, StageStatus
from precis.offline import RESPONDERS
from precis.pipeline import PIPELINE_STAGES, Pipeline
from precis.store import Store
from precis.util import canonical_json, sha256_hex

log = logging.getLogger(__name__)

# Time to first token plus about 50 output tokens per second (thinking
# included), with log-normal jitter: roughly what a large model with
# adaptive thinking does under load. A 500-token reply takes ~12 s.
LATENCY = LatencyModel(first_token_s=2.0, seconds_per_output_token=0.02, jitter=0.35)

# Per-request fault probabilities for the fault benchmark; each is far above
# what the API shows on a normal day.
FAULTS = Faults(
    rate_limit=0.04, overloaded=0.02, server_error=0.02, timeout=0.01, retry_after_s=5.0
)

TERMINAL = {StageStatus.DONE, StageStatus.SKIPPED, StageStatus.REFUSED, StageStatus.FAILED}


class _Discard(list[dict[str, Any]]):
    """A request log that keeps nothing."""

    def append(self, item: dict[str, Any]) -> None:
        pass


class TimedGateway(Gateway):
    """A ``Gateway`` that records, per paper, when its first request was made and
    its last one answered (on the gateway's clock).

    A paper's latency is the span between the two: the time it was being
    worked on, including waits for admission and retries.
    """

    def __init__(self, *args: Any, clock: Clock, **kwargs: Any) -> None:
        super().__init__(*args, clock=clock, **kwargs)
        self._timer = clock
        self.first: dict[str, float] = {}
        self.last: dict[str, float] = {}

    async def create(
        self, params: dict[str, Any], *, stage: str, paper_id: str | None = None
    ) -> Reply:
        started = self._timer.now()
        try:
            return await super().create(params, stage=stage, paper_id=paper_id)
        finally:
            if paper_id is not None:
                self.first.setdefault(paper_id, started)
                self.last[paper_id] = self._timer.now()

    def paper_latencies(self) -> list[float]:
        return [self.last[p] - self.first[p] for p in self.first]


def _client(
    clock: Clock,
    *,
    seed: int,
    limits: ServerLimits,
    faults: Faults | None = None,
    responders: Sequence[Responder] = RESPONDERS,
) -> FakeAnthropic:
    fake = FakeAnthropic(
        responders, clock=clock, seed=seed, latency=LATENCY, faults=faults, limits=limits
    )
    fake.requests = _Discard()
    return fake


def populate(store: Store, papers: int, seed: int) -> list[str]:
    """Write ``papers`` synthetic papers (metadata and text) into ``store``."""
    ids = []
    for i in range(papers):
        meta, doc = synthetic_paper(i, seed=seed)
        store.upsert_paper(meta)
        store.put_document(doc)
        ids.append(meta.id)
    return ids


def bench_config(base: Config, concurrency: int) -> Config:
    """``base`` with a concurrency ceiling and enough papers in flight to fill it.

    The response cache is on (resuming after a crash depends on it) and the
    budget is off: the simulated spend is reported, not enforced.
    """
    limits = base.limits.model_copy(
        update={
            "max_concurrency": concurrency,
            "initial_concurrency": min(base.limits.initial_concurrency, concurrency),
        }
    )
    pipeline = base.pipeline.model_copy(
        update={"max_papers_in_flight": 2 * concurrency, "use_cache": True}
    )
    budget = base.budget.model_copy(update={"max_cost_usd": None})
    return base.model_copy(update={"limits": limits, "pipeline": pipeline, "budget": budget})


def _remove_store(path: Path) -> None:
    """Delete a scratch store (about 100 MB for 2,000 papers) as soon as it is measured."""
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _limits(bench: BenchConfig) -> ServerLimits:
    return ServerLimits(rpm=bench.rpm, itpm=bench.itpm, otpm=bench.otpm)


# --------------------------------------------------------------------------
# Throughput
# --------------------------------------------------------------------------


class ThroughputPoint(BaseModel):
    concurrency: int
    papers: int
    finished: int
    simulated_minutes: float
    papers_per_minute: float
    requests: int
    requests_per_minute: float
    input_tokens_per_minute: float = Field(description="Uncached input plus cache writes.")
    output_tokens_per_minute: float
    rate_limited: int
    rate_limited_share: float = Field(description="429 responses / all responses.")
    final_concurrency_limit: int = Field(description="The AIMD window at the end of the run.")
    max_in_flight: int
    paper_latency_p50_s: float
    paper_latency_p95_s: float
    cost_usd: float
    wall_seconds: float


class ThroughputResult(BaseModel):
    papers: int
    rpm_limit: int
    itpm_limit: int
    otpm_limit: int
    latency: dict[str, float]
    points: list[ThroughputPoint]


async def throughput_point(
    base: Config, bench: BenchConfig, concurrency: int, *, papers: int, workdir: Path
) -> ThroughputPoint:
    """Run ``papers`` synthetic papers through every stage at one concurrency ceiling."""
    started = time.perf_counter()
    clock = VirtualClock()
    fake = _client(clock, seed=bench.seed, limits=_limits(bench))
    config = bench_config(base, concurrency)
    path = workdir / f"throughput-{concurrency}.db"
    with Store(path) as store:
        ids = populate(store, papers, bench.seed)
        gateway = TimedGateway(
            fake, store, run_id=f"bench-{concurrency}", config=config, clock=clock
        )
        pipeline = Pipeline(config, store, gateway)
        await clock.run(pipeline.run(ids, PIPELINE_STAGES))
        stats = gateway.stats()
        finished = pipeline.stats.papers_finished
    _remove_store(path)
    minutes = clock.now() / 60 or 1.0
    ok = [e for e in fake.events if e.outcome == "ok"]
    latencies = gateway.paper_latencies()
    return ThroughputPoint(
        concurrency=concurrency,
        papers=papers,
        finished=finished,
        simulated_minutes=clock.now() / 60,
        papers_per_minute=finished / minutes,
        requests=len(ok),
        requests_per_minute=len(ok) / minutes,
        input_tokens_per_minute=sum(e.input_tokens for e in ok) / minutes,
        output_tokens_per_minute=sum(e.output_tokens for e in ok) / minutes,
        rate_limited=fake.count("429"),
        rate_limited_share=fake.count("429") / len(fake.events) if fake.events else 0.0,
        final_concurrency_limit=stats.concurrency_limit,
        max_in_flight=fake.max_in_flight,
        paper_latency_p50_s=percentile(latencies, 50),
        paper_latency_p95_s=percentile(latencies, 95),
        cost_usd=stats.cost_usd,
        wall_seconds=time.perf_counter() - started,
    )


async def run_throughput(
    base: Config,
    bench: BenchConfig,
    *,
    papers: int | None = None,
    on_point: Callable[[ThroughputPoint], None] | None = None,
) -> ThroughputResult:
    """Sweep ``bench.concurrency``; ``on_point`` is called as each ceiling finishes."""
    n = papers or bench.papers
    points = []
    with tempfile.TemporaryDirectory(prefix="precis-bench-") as scratch:
        for concurrency in bench.concurrency:
            point = await throughput_point(
                base, bench, concurrency, papers=n, workdir=Path(scratch)
            )
            log.info("concurrency %d: %.1f papers/min", concurrency, point.papers_per_minute)
            if on_point is not None:
                on_point(point)
            points.append(point)
    return ThroughputResult(
        papers=n,
        rpm_limit=bench.rpm,
        itpm_limit=bench.itpm,
        otpm_limit=bench.otpm,
        latency={
            "first_token_s": LATENCY.first_token_s,
            "seconds_per_output_token": LATENCY.seconds_per_output_token,
            "jitter": LATENCY.jitter,
        },
        points=points,
    )


def throughput_markdown(result: ThroughputResult) -> str:
    rows = [
        [
            p.concurrency,
            f"{p.papers_per_minute:.1f}",
            round(p.requests_per_minute),
            f"{100 * p.requests_per_minute / result.rpm_limit:.0f}%",
            f"{100 * p.input_tokens_per_minute / result.itpm_limit:.0f}%",
            f"{100 * p.output_tokens_per_minute / result.otpm_limit:.0f}%",
            f"{100 * p.rate_limited_share:.2f}%",
            p.final_concurrency_limit,
            f"{p.paper_latency_p50_s / 60:.1f}",
            f"{p.paper_latency_p95_s / 60:.1f}",
            f"{p.simulated_minutes:.1f}",
            f"{p.wall_seconds:.0f}",
        ]
        for p in result.points
    ]
    headers = [
        "Ceiling",
        "Papers/min",
        "Requests/min",
        "RPM used",
        "ITPM used",
        "OTPM used",
        "429s",
        "Final window",
        "p50 paper (min)",
        "p95 paper (min)",
        "Run (sim. min)",
        "Wall (s)",
    ]
    return (
        f"# Throughput: {result.papers:,} papers, offline\n\n"
        f"Simulated limits: {result.rpm_limit:,} requests, {result.itpm_limit:,} input tokens "
        f"and {result.otpm_limit:,} output tokens per minute. Latency: "
        f"{result.latency['first_token_s']} s to first token plus "
        f"{result.latency['seconds_per_output_token']} s per output token, log-normal jitter "
        f"{result.latency['jitter']}. 'Final window' is the adaptive concurrency limit at the "
        "end of the run.\n\n" + markdown_table(headers, rows)
    )


def throughput_figure(result: ThroughputResult, path: Path) -> Path:
    x = [float(p.concurrency) for p in result.points]
    share = [
        Series("requests", [100 * p.requests_per_minute / result.rpm_limit for p in result.points]),
        Series(
            "input tokens",
            [100 * p.input_tokens_per_minute / result.itpm_limit for p in result.points],
        ),
        Series(
            "output tokens",
            [100 * p.output_tokens_per_minute / result.otpm_limit for p in result.points],
        ),
    ]
    papers = Series("papers", [p.papers_per_minute for p in result.points])
    latency = [
        Series("p50", [p.paper_latency_p50_s / 60 for p in result.points]),
        Series("p95", [p.paper_latency_p95_s / 60 for p in result.points]),
    ]
    window = [
        Series("ceiling", [float(p.concurrency) for p in result.points]),
        Series("adaptive window", [float(p.final_concurrency_limit) for p in result.points]),
    ]
    panels = [
        Panel("Papers per minute", "papers/min", [papers]),
        Panel("Achieved rate, share of limit", "% of limit", share, 100, "limit"),
        Panel("Paper latency", "minutes", latency),
        Panel("Concurrency at the end of the run", "requests in flight", window),
    ]
    limited = sum(p.rate_limited for p in result.points)
    errors = "no 429s" if limited == 0 else f"{limited:,} 429s"
    title = f"Throughput: {result.papers:,} papers, read, verify and revise (offline; {errors})"
    return small_multiples(x, panels, path, title=title)


# --------------------------------------------------------------------------
# Faults and crash recovery
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _CrashSettings:
    store_path: str
    witness_path: str
    seed: int
    concurrency: int
    rpm: int
    itpm: int
    otpm: int
    config: str
    """The base ``Config`` as JSON."""


def request_hash(params: dict[str, Any]) -> str:
    """The gateway's hash of a request, from the arguments the client received.

    The gateway adds the server-side fallback arguments after hashing, so
    they are removed again here.
    """
    original = {k: v for k, v in params.items() if k not in ("betas", "fallbacks")}
    return sha256_hex(canonical_json(original))


def _witness(path: str) -> Responder:
    """A responder that answers nothing but logs every request that reached the
    model, so the requests in flight at a crash can be counted afterwards."""
    log_file = open(path, "a", buffering=1)  # noqa: SIM115 - lives as long as the child

    def witness(params: dict[str, Any]) -> FakeReply | None:
        log_file.write(request_hash(params) + "\n")
        return None

    return witness


async def _faulty_run(
    settings: _CrashSettings, store: Store, run_id: str, seed: int, *, witness: bool
) -> Gateway:
    clock = VirtualClock()
    responders = [_witness(settings.witness_path), *RESPONDERS] if witness else RESPONDERS
    limits = ServerLimits(rpm=settings.rpm, itpm=settings.itpm, otpm=settings.otpm)
    fake = _client(clock, seed=seed, limits=limits, faults=FAULTS, responders=responders)
    config = bench_config(Config.model_validate_json(settings.config), settings.concurrency)
    gateway = Gateway(fake, store, run_id=run_id, config=config, clock=clock)
    ids = [meta.id for meta in store.list_papers()]
    await clock.run(Pipeline(config, store, gateway).run(ids, PIPELINE_STAGES))
    return gateway


def _child(settings: _CrashSettings) -> None:
    """The run that gets killed: it works until the parent sends SIGKILL."""
    with Store(settings.store_path) as store:
        asyncio.run(_faulty_run(settings, store, "faults-crashed", settings.seed, witness=True))


class FaultsResult(BaseModel):
    papers: int
    fault_rates: dict[str, float]
    killed_after_s: float = Field(description="Wall time of the first run before SIGKILL.")
    finished_at_kill: int = Field(description="Papers whose stages were all done at the kill.")
    running_at_kill: int = Field(description="Stages in progress at the kill.")
    in_flight_at_kill: int = Field(description="Requests sent but not answered at the kill.")
    first_run_calls: int
    resumed_calls: int
    resumed_cached_calls: int = Field(description="Requests the resume served from the cache.")
    resent_requests: int = Field(
        description="Requests in flight at the kill that the resume sent again."
    )
    kept_usd: float = Field(
        description="Cost of the stages finished before the kill, which the resume skipped."
    )
    cache_savings_usd: float = Field(
        description="What the requests the resume served from the cache cost the first time."
    )
    total_cost_usd: float
    retries: int
    resume_faults: dict[str, int]
    papers_done: int
    papers_failed: int
    papers_lost: int = Field(description="Papers with a stage not in a terminal state.")
    paid_twice: int = Field(description="Request hashes with more than one paid response.")

    @property
    def passed(self) -> bool:
        return self.papers_lost == 0 and self.paid_twice == 0


def paid_twice(calls: Sequence[LLMCall]) -> int:
    """Request hashes with more than one paid ("ok") response."""
    paid = Counter(c.request_hash for c in calls if c.status == "ok")
    return sum(n > 1 for n in paid.values())


def _paper_states(store: Store) -> tuple[int, int, int]:
    """(done, failed, lost) over every paper in the store."""
    done = failed = lost = 0
    for meta in store.list_papers():
        records = [store.get_stage(meta.id, s) for s in PIPELINE_STAGES]
        if any(r is None or r.status not in TERMINAL for r in records):
            lost += 1
        elif any(r is not None and r.status is StageStatus.FAILED for r in records):
            failed += 1
        else:
            done += 1
    return done, failed, lost


def run_faults(
    base: Config,
    bench: BenchConfig,
    *,
    papers: int,
    concurrency: int = 32,
    kill_at: float = 0.4,
    workdir: Path | None = None,
) -> FaultsResult:
    """Kill a faulty run with SIGKILL once ``kill_at`` of the papers are finished,
    resume it, and check that no paper is lost and no request is paid for twice."""
    with tempfile.TemporaryDirectory(prefix="precis-faults-") as scratch:
        directory = workdir or Path(scratch)
        return _run_faults(base, bench, directory, papers, concurrency, kill_at)


def _run_faults(
    base: Config, bench: BenchConfig, directory: Path, papers: int, concurrency: int, kill_at: float
) -> FaultsResult:
    store_path, witness_path = directory / "faults.db", directory / "sent.log"
    with Store(store_path) as store:
        populate(store, papers, bench.seed)
    settings = _CrashSettings(
        store_path=str(store_path),
        witness_path=str(witness_path),
        seed=bench.seed,
        concurrency=concurrency,
        rpm=bench.rpm,
        itpm=bench.itpm,
        otpm=bench.otpm,
        config=base.model_dump_json(),
    )
    child = multiprocessing.get_context("spawn").Process(target=_child, args=(settings,))
    started = time.perf_counter()
    child.start()
    with Store(store_path) as store:
        while child.is_alive() and _paper_states(store)[0] < kill_at * papers:
            time.sleep(0.05)
        child.kill()
        child.join()
        killed_after = time.perf_counter() - started
        finished_at_kill = _paper_states(store)[0]
        running = sum(c.get(StageStatus.RUNNING, 0) for c in store.stage_counts().values())
        first_calls = store.calls(run_id="faults-crashed")
        answered = {c.request_hash for c in first_calls}
        sent = set(witness_path.read_text().split()) if witness_path.exists() else set()
        in_flight = sent - answered
        finished_stages = {
            (meta.id, stage.value)
            for meta in store.list_papers()
            for stage in PIPELINE_STAGES
            if (record := store.get_stage(meta.id, stage)) is not None and record.status in TERMINAL
        }
        kept = sum(c.cost_usd for c in first_calls if (c.paper_id, c.stage) in finished_stages)

        gateway = asyncio.run(
            _faulty_run(settings, store, "faults-resumed", bench.seed + 1, witness=False)
        )
        resumed = store.calls(run_id="faults-resumed")
        first_paid = {c.request_hash: c for c in first_calls if c.status == "ok"}
        cached = [c for c in resumed if c.status == "cached"]
        all_calls = first_calls + resumed
        stats = gateway.stats()
        done, failed, lost = _paper_states(store)
    return FaultsResult(
        papers=papers,
        fault_rates={
            "429": FAULTS.rate_limit,
            "529": FAULTS.overloaded,
            "500": FAULTS.server_error,
            "timeout": FAULTS.timeout,
        },
        killed_after_s=killed_after,
        finished_at_kill=finished_at_kill,
        running_at_kill=running,
        in_flight_at_kill=len(in_flight),
        first_run_calls=len(first_calls),
        resumed_calls=len(resumed),
        resumed_cached_calls=len(cached),
        resent_requests=sum(c.request_hash in in_flight for c in resumed if c.status == "ok"),
        kept_usd=kept,
        cache_savings_usd=sum(
            first_paid[c.request_hash].cost_usd for c in cached if c.request_hash in first_paid
        ),
        total_cost_usd=sum(c.cost_usd for c in all_calls),
        retries=sum(max(0, c.attempts - 1) for c in all_calls),
        resume_faults={
            "429": stats.rate_limited,
            "529": stats.overloaded,
            "500": stats.server_errors,
            "timeout": stats.timeouts,
        },
        papers_done=done,
        papers_failed=failed,
        papers_lost=lost,
        paid_twice=paid_twice(all_calls),
    )


def faults_markdown(result: FaultsResult) -> str:
    verdict = "passed" if result.passed else "FAILED"
    rates = ", ".join(f"{k} {100 * v:.0f}%" for k, v in result.fault_rates.items())
    faults = ", ".join(f"{n} x {k}" for k, n in result.resume_faults.items())
    done = result.papers_done
    rows: list[list[object]] = [
        ["Papers", result.papers],
        ["Papers finished when the first run was killed", result.finished_at_kill],
        ["Stages running at the kill", result.running_at_kill],
        ["Requests in flight at the kill", result.in_flight_at_kill],
        ["Requests answered in the killed run", result.first_run_calls],
        ["Requests in the resumed run", result.resumed_calls],
        ["of which served from the cache", result.resumed_cached_calls],
        ["of which in flight at the kill, sent again", result.resent_requests],
        ["Work kept: stages finished before the kill", f"${result.kept_usd:,.2f}"],
        ["Work kept: replayed from the cache on resume", f"${result.cache_savings_usd:,.2f}"],
        ["Total cost of both runs", f"${result.total_cost_usd:,.2f}"],
        ["Retries in both runs", result.retries],
        ["Faults injected in the resumed run", faults],
        ["Papers done / failed / lost", f"{done} / {result.papers_failed} / {result.papers_lost}"],
        ["Requests paid for twice", result.paid_twice],
    ]
    return (
        f"# Faults and crash recovery: {verdict}\n\n"
        f"Injected per request: {rates}. The first run was killed with SIGKILL; a second run "
        "resumed from the store it left behind.\n\n" + markdown_table(["", "Value"], rows)
    )
