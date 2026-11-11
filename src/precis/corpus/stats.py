"""Corpus-wide numbers shown in the report, on the site and in the evaluation.

Faithfulness is counted per checked claim (the tldr and each finding of a
summary), because that is the unit the critic judges; a paper-level rate
would hide that most failing summaries fail on one finding out of several.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel

from precis.corpus.load import PaperView
from precis.models import ClaimCheck, UsageRow, Verdict, Verification


class VerdictCounts(BaseModel):
    """How many checked claims got each verdict."""

    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    contradicted: int = 0

    @classmethod
    def of(cls, checks: Iterable[ClaimCheck]) -> VerdictCounts:
        counts = Counter(c.verdict for c in checks)
        return cls(**{v.value: counts.get(v, 0) for v in Verdict})

    @property
    def total(self) -> int:
        return self.supported + self.partial + self.unsupported + self.contradicted

    def rate(self, verdict: Verdict) -> float:
        """Share of claims with ``verdict`` (0 when nothing was checked)."""
        return getattr(self, verdict.value) / self.total if self.total else 0.0

    def rates(self) -> dict[str, float]:
        return {v.value: self.rate(v) for v in Verdict}


def _checks(verifications: Iterable[Verification | None]) -> list[ClaimCheck]:
    return [c for v in verifications if v is not None for c in v.checks]


@dataclass(frozen=True)
class Faithfulness:
    """Verdicts before and after revision across the corpus."""

    papers: int
    verified: int
    """Papers whose first draft was verified."""
    flagged: int
    """Papers whose first draft had an unsupported or contradicted claim."""
    revised: int
    """Papers with a verified revision."""
    draft: VerdictCounts
    """Every verified first draft."""
    final: VerdictCounts
    """Every paper's final summary (the revision when there is one)."""
    revised_before: VerdictCounts
    """The revised papers only, before revision."""
    revised_after: VerdictCounts
    """The revised papers only, after revision."""


def faithfulness(papers: Sequence[PaperView]) -> Faithfulness:
    revised = [p for p in papers if p.revised and p.verification is not None]
    return Faithfulness(
        papers=len(papers),
        verified=sum(p.draft_verification is not None for p in papers),
        flagged=sum(
            p.draft_verification is not None and p.draft_verification.needs_revision()
            for p in papers
        ),
        revised=len(revised),
        draft=VerdictCounts.of(_checks(p.draft_verification for p in papers)),
        final=VerdictCounts.of(_checks(p.verification for p in papers)),
        revised_before=VerdictCounts.of(_checks(p.draft_verification for p in revised)),
        revised_after=VerdictCounts.of(_checks(p.verification for p in revised)),
    )


@dataclass(frozen=True)
class Spend:
    """Requests, tokens and cost summed over every stage and model."""

    calls: int
    cached_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_usd: float
    by_stage: list[UsageRow]


def spend(usage: Sequence[UsageRow]) -> Spend:
    rows = sorted(usage, key=lambda r: (-r.cost_usd, r.stage, r.model))
    return Spend(
        calls=sum(r.calls for r in rows),
        cached_calls=sum(r.cached_calls for r in rows),
        input_tokens=sum(r.input_tokens for r in rows),
        output_tokens=sum(r.output_tokens for r in rows),
        cache_read_input_tokens=sum(r.cache_read_input_tokens for r in rows),
        cache_creation_input_tokens=sum(r.cache_creation_input_tokens for r in rows),
        cost_usd=sum(r.cost_usd for r in rows),
        by_stage=rows,
    )


@dataclass(frozen=True)
class CorpusFacts:
    papers: int
    first: date | None
    last: date | None
    categories: list[tuple[str, int]]
    """Primary categories by paper count, largest first."""


def corpus_facts(papers: Sequence[PaperView]) -> CorpusFacts:
    dates = sorted(p.meta.published for p in papers)
    categories = Counter(p.meta.primary_category for p in papers)
    return CorpusFacts(
        papers=len(papers),
        first=dates[0] if dates else None,
        last=dates[-1] if dates else None,
        categories=sorted(categories.items(), key=lambda kv: (-kv[1], kv[0])),
    )
