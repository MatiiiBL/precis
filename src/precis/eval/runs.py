"""Running a summarization strategy over sampled papers, for the evaluations.

Both ``eval abstract`` and ``eval ablation`` summarize a fixed sample of
papers without the abstract and need, per paper, the summary, what it cost
and how long it took. Cost and tokens come from the gateway's ``LLMCall``
records: a request served from the response cache is counted with the usage
and cost of the response it replays, so a re-run reports what the summaries
cost when they were made. Latency is wall time on the gateway's clock and is
only reported for papers whose requests were all sent, since a cached reply
takes no time.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from precis.agents import run_summarize
from precis.config import AgentsConfig
from precis.llm import Clock, Gateway
from precis.models import Document, LLMCall, PaperMeta, PaperSummary
from precis.store import Store

log = logging.getLogger(__name__)

type Paper = tuple[PaperMeta, Document]


def sample_papers(store: Store, n: int, *, seed: int) -> list[Paper]:
    """Up to ``n`` papers with full text and an abstract, the same ones for the same seed."""
    ids = sorted(meta.id for meta in store.list_papers())
    random.Random(seed).shuffle(ids)
    chosen: list[Paper] = []
    for paper_id in ids:
        meta = store.get_paper(paper_id)
        doc = store.get_document(paper_id)
        if meta is not None and doc is not None and meta.abstract.strip():
            chosen.append((meta, doc))
            if len(chosen) == n:
                break
    return sorted(chosen, key=lambda p: p[0].id)


@dataclass(frozen=True)
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    """At least one request was served from the response cache."""


@dataclass(frozen=True)
class Summarized:
    meta: PaperMeta
    doc: Document
    summary: PaperSummary
    usage: Usage
    latency_s: float | None
    """Wall time to summarize, or None when part of it came from the cache."""


@dataclass(frozen=True)
class StrategyRun:
    strategy: str
    papers: list[Summarized]
    failed: list[str]


class _Originals:
    """Paid responses by request hash, so cached replays can be priced."""

    def __init__(self, store: Store) -> None:
        self.calls = {c.request_hash: c for c in store.calls() if c.status == "ok"}

    def usage(self, calls: Sequence[LLMCall]) -> Usage:
        resolved = [c if c.status != "cached" else self.calls.get(c.request_hash, c) for c in calls]
        return Usage(
            requests=len(calls),
            input_tokens=sum(c.input_tokens for c in resolved),
            output_tokens=sum(c.output_tokens for c in resolved),
            cache_read_input_tokens=sum(c.cache_read_input_tokens for c in resolved),
            cache_creation_input_tokens=sum(c.cache_creation_input_tokens for c in resolved),
            cost_usd=sum(c.cost_usd for c in resolved),
            cached=any(c.status == "cached" for c in calls),
        )


async def summarize_papers(
    gw: Gateway,
    store: Store,
    clock: Clock,
    papers: Sequence[Paper],
    agents: AgentsConfig,
    strategy: str,
) -> StrategyRun:
    """Summarize ``papers`` with ``strategy`` and without the abstract, concurrently.

    Callers run one strategy at a time, so a strategy's latencies are not
    inflated by another's traffic.
    """
    config = agents.model_copy(update={"include_abstract": False, "strategy": strategy})
    started_at = datetime.now(UTC)

    async def one(meta: PaperMeta, doc: Document) -> tuple[PaperSummary, float]:
        start = clock.now()
        result = await run_summarize(gw, meta, doc, config)
        return result.value, clock.now() - start

    outcomes = await asyncio.gather(*(one(m, d) for m, d in papers), return_exceptions=True)
    originals = _Originals(store)
    done: list[Summarized] = []
    failed: list[str] = []
    for (meta, doc), outcome in zip(papers, outcomes, strict=True):
        if isinstance(outcome, Exception):
            log.warning("%s: %s summary failed: %s", meta.id, strategy, outcome)
            failed.append(meta.id)
            continue
        if isinstance(outcome, BaseException):
            raise outcome
        summary, latency = outcome
        calls = [
            c
            for c in store.calls(run_id=gw.run_id, paper_id=meta.id)
            if c.stage == "read" and c.started_at >= started_at
        ]
        usage = originals.usage(calls)
        done.append(Summarized(meta, doc, summary, usage, None if usage.cached else latency))
    return StrategyRun(strategy, done, failed)
