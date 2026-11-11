"""How much of the authors' abstract does a summary recover without seeing it?

The reader normally gets the abstract; for this evaluation it does not, so
the abstract becomes an independent reference written by the authors.
Two measures are reported:

- ROUGE-1/2/L F1 between the summary and the abstract. ROUGE counts shared
  words, so it is low for any abstractive summary; it is reported next to a
  mismatched baseline (each summary against another paper's abstract from
  the same corpus), which shows how much overlap shared field vocabulary
  alone produces.
- Key-point recall, judged by a model (``precis.eval.judge``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from pydantic import BaseModel

from precis.config import AgentConfig
from precis.eval.judge import coverage, key_points
from precis.eval.metrics import Estimate, Rouge, bootstrap, mean_rouge, rouge, summary_text
from precis.eval.results import markdown_table
from precis.llm.base import ModelGateway
from precis.models import PaperMeta, PaperSummary


class AbstractScore(BaseModel):
    paper_id: str
    rouge: Rouge
    key_points: list[str]
    covered: list[bool]

    @property
    def recall(self) -> float:
        return sum(self.covered) / len(self.covered) if self.covered else 0.0


class AbstractEval(BaseModel):
    papers: int
    rouge: Rouge
    rouge_mismatched: Rouge
    recall: Estimate
    rouge_l: Estimate
    scores: list[AbstractScore]


async def score_paper(
    gw: ModelGateway, meta: PaperMeta, summary: PaperSummary, judge: AgentConfig
) -> AbstractScore:
    points = await key_points(gw, meta, judge)
    covered = await coverage(gw, meta, points, summary, judge) if points else []
    return AbstractScore(
        paper_id=meta.id,
        rouge=rouge(summary_text(summary), meta.abstract),
        key_points=points,
        covered=covered,
    )


async def score_against_abstracts(
    gw: ModelGateway,
    items: Sequence[tuple[PaperMeta, PaperSummary]],
    judge: AgentConfig,
    *,
    seed: int = 0,
) -> AbstractEval:
    """Score each summary against its paper's abstract (papers judged concurrently)."""
    scores = list(await asyncio.gather(*(score_paper(gw, m, s, judge) for m, s in items)))
    # A derangement by rotation: every summary meets an abstract that is not its own.
    mismatched = [
        rouge(summary_text(summary), items[(i + 1) % len(items)][0].abstract)
        for i, (_, summary) in enumerate(items)
    ]
    with_points = [s for s in scores if s.key_points]
    return AbstractEval(
        papers=len(scores),
        rouge=mean_rouge([s.rouge for s in scores]),
        rouge_mismatched=mean_rouge(mismatched if len(items) > 1 else []),
        recall=bootstrap([s.recall for s in with_points], seed=seed),
        rouge_l=bootstrap([s.rouge.rougeL for s in scores], seed=seed),
        scores=scores,
    )


def _interval(e: Estimate) -> str:
    return f"{e.mean:.3f} [{e.low:.3f}, {e.high:.3f}]"


def abstract_markdown(result: AbstractEval, *, model: str) -> str:
    rows = [
        ["ROUGE-1 F1", f"{result.rouge.rouge1:.3f}", f"{result.rouge_mismatched.rouge1:.3f}"],
        ["ROUGE-2 F1", f"{result.rouge.rouge2:.3f}", f"{result.rouge_mismatched.rouge2:.3f}"],
        ["ROUGE-L F1", _interval(result.rouge_l), f"{result.rouge_mismatched.rougeL:.3f}"],
        ["Key-point recall", _interval(result.recall), ""],
    ]
    return (
        f"# Summaries written without the abstract, compared with it\n\n"
        f"{result.papers} papers; key points extracted and judged by `{model}`. "
        "Brackets are 95% bootstrap intervals; the mismatched column pairs each summary "
        "with another paper's abstract.\n\n"
        + markdown_table(["Metric", "Own abstract", "Mismatched abstract"], rows)
    )
