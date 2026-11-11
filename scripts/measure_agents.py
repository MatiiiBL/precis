"""Measure the agents offline: the numbers in docs/agents.md, section 7.

    uv run python scripts/measure_agents.py [--papers 200] [--json out.json]

Runs the whole per-paper pipeline (read, verify, revise) over the synthetic
corpus of ``precis.eval.synthetic`` with the offline client, the real
gateway and an in-memory store, once per summarization strategy, and three
more times for the agentic strategy to compare online and batch mode. Every
run is on a ``VirtualClock``, so the result does not depend on the machine.
It prints Markdown tables; ``--json`` also writes the raw figures. The
offline responders follow a fixed reading pattern, so the figures describe
requests, tokens, caching and cost for that pattern, not how much a real
model would read.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from anthropic.types.messages import MessageBatch

from precis.config import Config
from precis.eval.simclock import VirtualClock
from precis.eval.synthetic import synthetic_corpus
from precis.llm import Gateway
from precis.llm.fake import FakeAnthropic
from precis.models import Document, PaperMeta
from precis.offline import offline_client
from precis.pipeline import PIPELINE_STAGES, Pipeline, batch_settings
from precis.store import Store

STRATEGIES = ("agentic", "fulltext", "mapreduce")
SHORT, LONG = 7_000, 12_000
"""Body sizes, in tokens, below and above which papers count as short and long."""


def corpus_store(corpus: list[tuple[PaperMeta, Document]]) -> Store:
    store = Store(":memory:")
    for meta, doc in corpus:
        store.upsert_paper(meta)
        store.put_document(doc)
    return store


class BatchSizes:
    """Records the size of every batch the fake client receives."""

    def __init__(self, client: FakeAnthropic) -> None:
        self.sizes: list[int] = []
        self._create = client.messages.batches.create
        client.messages.batches.create = self.create  # type: ignore[method-assign]

    async def create(self, *, requests: Iterable[Any]) -> MessageBatch:
        items = list(requests)
        self.sizes.append(len(items))
        return await self._create(requests=items)


@dataclass(frozen=True)
class Run:
    stats: dict[str, Any]
    store: Store
    batch_sizes: list[int]
    simulated_s: float


async def run(
    corpus: list[tuple[PaperMeta, Document]],
    config: Config,
    *,
    mode: Literal["online", "batch"] = "online",
) -> Run:
    """Run every stage over ``corpus`` in virtual time."""
    clock = VirtualClock()
    store = corpus_store(corpus)
    client = offline_client(clock=clock)
    batches = BatchSizes(client)
    gateway = Gateway(client, store, run_id="measure", config=config, clock=clock, mode=mode)

    async def pipeline() -> dict[str, Any]:
        ids = [meta.id for meta, _ in corpus]
        stats = await Pipeline(config, store, gateway).run(ids, PIPELINE_STAGES)
        await gateway.aclose()
        return stats

    stats = await clock.run(pipeline())
    return Run(stats, store, batches.sizes, clock.now())


def per_paper(store: Store) -> dict[str, dict[str, dict[str, float]]]:
    """Requests, tokens and cost of every (stage, paper)."""
    rows: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for call in store.calls():
        row = rows[call.stage][call.paper_id or ""]
        row["requests"] += 1
        row["tokens"] += (
            call.input_tokens
            + call.output_tokens
            + call.cache_creation_input_tokens
            + call.cache_read_input_tokens
        )
        row["cache_read"] += call.cache_read_input_tokens
        row["cost"] += call.cost_usd
    return rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def stage_figures(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    papers = list(rows.values())
    tokens = [p["tokens"] for p in papers]
    return {
        "papers": len(papers),
        "requests_mean": statistics.fmean(p["requests"] for p in papers),
        "tokens_median": statistics.median(tokens),
        "tokens_p90": percentile(tokens, 0.9),
        "cache_read_share": sum(p["cache_read"] for p in papers) / sum(tokens),
        "cost": sum(p["cost"] for p in papers),
    }


def length_scaling(
    corpus: list[tuple[PaperMeta, Document]], read: dict[str, dict[str, float]]
) -> dict[str, float]:
    """How read-stage tokens grow with body length, over every paper."""
    pairs = [(float(doc.n_tokens), read[meta.id]["tokens"]) for meta, doc in corpus]
    slope = statistics.linear_regression([x for x, _ in pairs], [y for _, y in pairs]).slope
    return {
        "slope": slope,
        "median_short": statistics.median(y for x, y in pairs if x < SHORT),
        "median_long": statistics.median(y for x, y in pairs if x > LONG),
    }


async def measure(n_papers: int) -> dict[str, Any]:
    corpus = synthetic_corpus(n_papers)
    base = Config.model_validate({"pipeline": {"fallbacks": False}})
    results: dict[str, Any] = {
        "papers": n_papers,
        "body_tokens_median": statistics.median(d.n_tokens for _, d in corpus),
        "strategies": {},
    }
    for strategy in STRATEGIES:
        config = base.model_copy(
            update={"agents": base.agents.model_copy(update={"strategy": strategy})}
        )
        outcome = await run(corpus, config)
        stats, rows = outcome.stats, per_paper(outcome.store)
        gateway = stats["gateway"]
        total_tokens = sum(
            gateway[k]
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
        results["strategies"][strategy] = {
            "stages": {stage: stage_figures(rows[stage]) for stage in rows},
            "length": length_scaling(corpus, rows["read"]),
            "agents": {name: s.get("agents", {}) for name, s in stats["stages"].items()},
            "faithfulness": stats["faithfulness"],
            "tokens_per_paper": total_tokens / n_papers,
            "cost_per_paper": gateway["cost_usd"] / n_papers,
            "cost": gateway["cost_usd"],
        }
    agentic = results["strategies"]["agentic"]
    results["modes"] = {
        "online": {"cost": agentic["cost"], "batches": 0, "requests_per_batch": None, "hours": None}
    }
    # The fake answers a batch 300 simulated seconds after it is sent.
    for name, config in {
        "batch, online settings": base,
        "batch, batch_settings": batch_settings(base, n_papers),
    }.items():
        outcome = await run(corpus, config, mode="batch")
        sizes = outcome.batch_sizes
        results["modes"][name] = {
            "cost": outcome.stats["gateway"]["cost_usd"],
            "batches": len(sizes),
            "requests_per_batch": statistics.fmean(sizes),
            "hours": outcome.simulated_s / 3600,
        }
    return results


def markdown(results: dict[str, Any]) -> str:
    lines = [
        f"{results['papers']} synthetic papers, median body "
        f"{results['body_tokens_median']:,.0f} tokens.",
        "",
        "| Read stage | requests/paper | tokens/paper (median) | p90 | body read | read cost |",
        "|---|---|---|---|---|---|",
    ]
    strategies = results["strategies"]
    for name, r in strategies.items():
        read = r["stages"]["read"]
        fraction = r["agents"]["read"].get("fraction_read_mean", 0) or 0
        lines.append(
            f"| {name} | {read['requests_mean']:.2f} | {read['tokens_median']:,.0f} | "
            f"{read['tokens_p90']:,.0f} | {fraction:.0%} | ${read['cost']:.2f} |"
        )
    lines += [
        "",
        "| Strategy | extra read tokens per body token | median, bodies < 7,000 | "
        "median, bodies > 12,000 |",
        "|---|---|---|---|",
    ]
    for name, r in strategies.items():
        length = r["length"]
        lines.append(
            f"| {name} | {length['slope']:.2f} | {length['median_short']:,.0f} | "
            f"{length['median_long']:,.0f} |"
        )
    agentic = strategies["agentic"]
    lines += [
        "",
        "| Agentic stage | requests/paper | tokens/paper (median) | cache reads | cost |",
        "|---|---|---|---|---|",
    ]
    for stage, figures in agentic["stages"].items():
        lines.append(
            f"| {stage} ({figures['papers']} papers) | {figures['requests_mean']:.2f} | "
            f"{figures['tokens_median']:,.0f} | {figures['cache_read_share']:.0%} | "
            f"${figures['cost']:.2f} |"
        )
    faith = agentic["faithfulness"]
    revised = faith["revised"]
    read_agents = agentic["agents"]["read"]
    lines += [
        "",
        f"Agentic, whole pipeline: {agentic['tokens_per_paper']:,.0f} tokens and "
        f"${agentic['cost_per_paper']:.3f} per paper; "
        f"{read_agents.get('rejected_submissions', 0)} reader submissions rejected and fixed.",
        f"Final summaries: {faith['final']['supported_rate']:.1%} of "
        f"{faith['final']['claims']:,} checked claims supported (first drafts "
        f"{faith['first_drafts']['supported_rate']:.1%}); {revised['papers']} revised papers: "
        f"{revised['before']['supported_rate']:.1%} -> {revised['after']['supported_rate']:.1%}.",
        "",
        "| Mode (agentic) | cost | batches | requests per batch | simulated hours |",
        "|---|---|---|---|---|",
    ]
    for mode, m in results["modes"].items():
        per_batch = f"{m['requests_per_batch']:.0f}" if m["requests_per_batch"] else "-"
        hours = f"{m['hours']:.1f}" if m["hours"] is not None else "-"
        lines.append(
            f"| {mode} | ${m['cost']:.2f} | {m['batches'] or '-'} | {per_batch} | {hours} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--papers", type=int, default=200)
    parser.add_argument("--json", type=Path, help="Also write the raw figures here.")
    args = parser.parse_args()
    results = asyncio.run(measure(args.papers))
    print(markdown(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
