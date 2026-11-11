"""Key-point recall: does a summary say what the authors' abstract says?

Two judge requests per paper. The first extracts 3-6 key points from the
abstract; it depends only on the abstract, so the gateway's response cache
serves it once per paper however many summaries are compared against it.
The second shows the judge the key points and a summary, never the
abstract itself, and asks which points the summary conveys. Recall is the
share of key points covered.

ROUGE rewards copying the abstract's words; this asks whether the facts
are there, which is the property that matters for a summary written
without the abstract.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from precis.config import AgentConfig
from precis.corpus.structured import request_structured, structured_params
from precis.llm.base import ModelGateway
from precis.models import PaperMeta, PaperSummary

logger = logging.getLogger(__name__)


class KeyPoints(BaseModel):
    key_points: list[str] = Field(
        description="3-6 key points, each one self-contained sentence, in the abstract's order."
    )


class PointJudgment(BaseModel):
    point: int = Field(description="The key point's number, starting at 1.")
    covered: bool
    reason: str = Field(description="One sentence.")


class Coverage(BaseModel):
    judgments: list[PointJudgment] = Field(description="One judgment per key point, in order.")


EXTRACT_SYSTEM = """\
You extract the key points of a research paper from its abstract, so that a \
summary of the paper can be checked against them.

Write 3-6 key points: the problem, the approach, and each main result. Each is one \
self-contained sentence that can be checked without the abstract, keeps the \
abstract's numbers, and adds nothing that the abstract does not say. Skip \
motivation and boilerplate."""

COVERAGE_SYSTEM = """\
You check whether a summary of a research paper conveys each of the paper's key \
points.

A key point is covered when the summary states the same fact, in any words and \
possibly with less detail; when the key point has a number, the summary must \
not contradict it. A key point is not covered when the summary omits it, only \
hints at it, or says something different. Judge every key point, in order, by \
its number."""


def summary_for_judge(summary: PaperSummary) -> str:
    lines = [
        f"TL;DR: {summary.tldr}",
        f"Problem: {summary.problem}",
        f"Approach: {summary.approach}",
        "Findings:",
        *(f"- {f.claim}" for f in summary.findings),
        "Contributions:",
        *(f"- {c}" for c in summary.contributions),
        "Limitations:",
        *(f"- {item}" for item in summary.limitations),
    ]
    return "\n".join(lines)


async def key_points(gw: ModelGateway, meta: PaperMeta, judge: AgentConfig) -> list[str]:
    user = f"Title: {meta.title}\n\nAbstract:\n{meta.abstract}"
    params = structured_params(judge, system=EXTRACT_SYSTEM, user=user, output=KeyPoints)
    result = await request_structured(gw, params, KeyPoints, stage="judge", paper_id=meta.id)
    return [p.strip() for p in result.key_points if p.strip()]


async def coverage(
    gw: ModelGateway,
    meta: PaperMeta,
    points: list[str],
    summary: PaperSummary,
    judge: AgentConfig,
) -> list[bool]:
    """Which of ``points`` the summary covers.

    A key point the judge leaves out, or numbers twice, counts as not
    covered, so a malformed answer can only lower recall.
    """
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1))
    user = f"Key points:\n{numbered}\n\nSummary:\n{summary_for_judge(summary)}"
    params = structured_params(judge, system=COVERAGE_SYSTEM, user=user, output=Coverage)
    result = await request_structured(gw, params, Coverage, stage="judge", paper_id=meta.id)
    votes: dict[int, list[bool]] = {}
    for judgment in result.judgments:
        votes.setdefault(judgment.point, []).append(judgment.covered)
    covered = [votes.get(i, []) == [True] for i in range(1, len(points) + 1)]
    if any(len(votes.get(i, [])) != 1 for i in range(1, len(points) + 1)):
        logger.warning("%s: judge answered %d of %d key points", meta.id, len(votes), len(points))
    return covered
