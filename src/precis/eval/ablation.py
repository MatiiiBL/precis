"""The three summarization strategies on the same papers (``precis eval ablation``).

``agentic`` (the reader navigates the paper with tools), ``fulltext`` (one
request with the whole body) and ``mapreduce`` (notes per chunk, then a
combining request) each summarize the same sampled papers, all without the
abstract. Every summary is then checked by the same critic and scored by the
same judge against the abstract, so the strategies differ only in how they
read. Strategies run one after another, so each has the gateway to itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from pydantic import BaseModel, Field

from precis.agents import run_critic
from precis.config import AgentsConfig
from precis.corpus.stats import VerdictCounts
from precis.eval.abstract import score_paper
from precis.eval.metrics import Estimate, bootstrap, percentile
from precis.eval.results import markdown_table
from precis.eval.runs import Paper, StrategyRun, summarize_papers
from precis.llm import Clock, Gateway
from precis.models import Verdict
from precis.store import Store

STRATEGIES = ("agentic", "fulltext", "mapreduce")


class StrategyResult(BaseModel):
    strategy: str
    papers: int
    failed: int
    requests_per_paper: float
    input_tokens_per_paper: float = Field(description="Uncached input tokens.")
    cache_write_tokens_per_paper: float
    cache_read_tokens_per_paper: float
    output_tokens_per_paper: float
    cost_per_paper_usd: float
    total_cost_usd: float
    latency_p50_s: float | None
    latency_p95_s: float | None
    latency_measured: int = Field(description="Papers whose requests were all sent.")
    recall: Estimate
    rouge_l: Estimate
    supported: float = Field(description="Share of checked claims the critic found supported.")
    fully_supported: float = Field(description="Share of papers with every claim supported.")


class AblationEval(BaseModel):
    papers: list[str]
    reader_model: str
    judge_model: str
    strategies: list[StrategyResult]


async def _score(gw: Gateway, run: StrategyRun, agents: AgentsConfig, seed: int) -> StrategyResult:
    async def judge_and_check(index: int) -> tuple[float, float, list[Verdict]]:
        paper = run.papers[index]
        score = await score_paper(gw, paper.meta, paper.summary, agents.judge)
        critic = await run_critic(gw, paper.meta, paper.doc, paper.summary, agents)
        return score.recall, score.rouge.rougeL, [c.verdict for c in critic.value.checks]

    scored = await asyncio.gather(*(judge_and_check(i) for i in range(len(run.papers))))
    n = len(run.papers) or 1
    latencies = [p.latency_s for p in run.papers if p.latency_s is not None]
    verdicts = [v for _, _, vs in scored for v in vs]
    counts = VerdictCounts(**{v.value: verdicts.count(v) for v in Verdict})
    return StrategyResult(
        strategy=run.strategy,
        papers=len(run.papers),
        failed=len(run.failed),
        requests_per_paper=sum(p.usage.requests for p in run.papers) / n,
        input_tokens_per_paper=sum(p.usage.input_tokens for p in run.papers) / n,
        cache_write_tokens_per_paper=sum(p.usage.cache_creation_input_tokens for p in run.papers)
        / n,
        cache_read_tokens_per_paper=sum(p.usage.cache_read_input_tokens for p in run.papers) / n,
        output_tokens_per_paper=sum(p.usage.output_tokens for p in run.papers) / n,
        cost_per_paper_usd=sum(p.usage.cost_usd for p in run.papers) / n,
        total_cost_usd=sum(p.usage.cost_usd for p in run.papers),
        latency_p50_s=percentile(latencies, 50) if latencies else None,
        latency_p95_s=percentile(latencies, 95) if latencies else None,
        latency_measured=len(latencies),
        recall=bootstrap([r for r, _, _ in scored], seed=seed),
        rouge_l=bootstrap([r for _, r, _ in scored], seed=seed),
        supported=counts.rate(Verdict.SUPPORTED),
        fully_supported=sum(all(v is Verdict.SUPPORTED for v in vs) for _, _, vs in scored) / n,
    )


async def run_ablation(
    gw: Gateway,
    store: Store,
    clock: Clock,
    papers: Sequence[Paper],
    agents: AgentsConfig,
    *,
    seed: int = 0,
    strategies: Sequence[str] = STRATEGIES,
) -> AblationEval:
    results = []
    for strategy in strategies:
        run = await summarize_papers(gw, store, clock, papers, agents, strategy)
        results.append(await _score(gw, run, agents, seed))
    return AblationEval(
        papers=[meta.id for meta, _ in papers],
        reader_model=agents.reader.model,
        judge_model=agents.judge.model,
        strategies=results,
    )


def _seconds(value: float | None) -> object:
    return "n/a" if value is None else round(value, 1)


def ablation_markdown(result: AblationEval) -> str:
    rows: list[list[object]] = [
        [
            r.strategy,
            r.papers,
            round(r.requests_per_paper, 1),
            round(r.input_tokens_per_paper),
            round(r.cache_write_tokens_per_paper),
            round(r.cache_read_tokens_per_paper),
            round(r.output_tokens_per_paper),
            f"${r.cost_per_paper_usd:.3f}",
            _seconds(r.latency_p50_s),
            _seconds(r.latency_p95_s),
            f"{r.recall.mean:.2f} [{r.recall.low:.2f}, {r.recall.high:.2f}]",
            f"{r.rouge_l.mean:.3f}",
            f"{100 * r.supported:.1f}%",
        ]
        for r in result.strategies
    ]
    headers = [
        "Strategy",
        "Papers",
        "Requests",
        "Input tok.",
        "Cache-write tok.",
        "Cache-read tok.",
        "Output tok.",
        "Cost",
        "p50 s",
        "p95 s",
        "Key-point recall",
        "ROUGE-L",
        "Supported claims",
    ]
    return (
        f"# Summarization strategies on the same {len(result.papers)} papers\n\n"
        f"Reader `{result.reader_model}`, judge `{result.judge_model}`; no abstract shown. "
        "Requests, tokens and cost are per paper and cover summarizing only; latency is "
        "wall time per paper. Brackets are 95% bootstrap intervals.\n\n"
        + markdown_table(headers, rows)
    )
