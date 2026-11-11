"""What a pipeline run did, as counters for the live view and the run record."""

from __future__ import annotations

import statistics
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from precis.agents.trace import AgentTrace
from precis.models import RevisionResult, Stage, Verdict, Verification

PIPELINE_STAGES = (Stage.READ, Stage.VERIFY, Stage.REVISE)


@dataclass
class ClaimTally:
    """Checked claims and how many were supported, summed over verifications."""

    claims: int = 0
    supported: int = 0

    def add(self, verification: Verification) -> None:
        self.claims += len(verification.checks)
        self.supported += sum(c.verdict is Verdict.SUPPORTED for c in verification.checks)

    def summary(self) -> dict[str, Any]:
        rate = round(self.supported / self.claims, 4) if self.claims else None
        return {"claims": self.claims, "supported": self.supported, "supported_rate": rate}


@dataclass
class Faithfulness:
    """Per-claim verdict rates over every paper of the run that has a verification.

    Cached stages count as well as stages that ran, so the population is the
    run's papers, not whatever happened to run in this invocation. The revised
    subset is reported before and after revision on its own, because those
    papers start far below the rest.
    """

    papers: int = 0
    first_drafts: ClaimTally = field(default_factory=ClaimTally)
    final: ClaimTally = field(default_factory=ClaimTally)
    revised_papers: int = 0
    revised_before: ClaimTally = field(default_factory=ClaimTally)
    revised_after: ClaimTally = field(default_factory=ClaimTally)

    def add(self, first: Verification, revision: RevisionResult | None) -> None:
        self.papers += 1
        self.first_drafts.add(first)
        if revision is None:
            self.final.add(first)
            return
        self.final.add(revision.verification)
        self.revised_papers += 1
        self.revised_before.add(first)
        self.revised_after.add(revision.verification)

    def summary(self) -> dict[str, Any]:
        return {
            "papers": self.papers,
            "final": self.final.summary(),
            "first_drafts": self.first_drafts.summary(),
            "revised": {
                "papers": self.revised_papers,
                "before": self.revised_before.summary(),
                "after": self.revised_after.summary(),
            },
        }


@dataclass
class StageTally:
    """Outcomes of one stage in this run.

    ``done`` counts stages that ran and finished; ``cached`` those skipped
    because a ``done`` record with the same config hash existed; ``skipped``
    revisions that were not needed; ``interrupted`` stages left ``pending``
    by an interrupt or the budget.
    """

    running: int = 0
    done: int = 0
    cached: int = 0
    skipped: int = 0
    failed: int = 0
    refused: int = 0
    interrupted: int = 0
    traces: list[AgentTrace] = field(default_factory=list)

    @property
    def finished(self) -> int:
        return self.done + self.cached + self.skipped + self.failed + self.refused


@dataclass
class RunStats:
    """Counters the scheduler updates as papers move through their stages."""

    papers: int
    stages: tuple[Stage, ...]
    started: float = field(default_factory=time.monotonic)
    tally: dict[Stage, StageTally] = field(
        default_factory=lambda: {s: StageTally() for s in PIPELINE_STAGES}
    )
    papers_finished: int = 0
    papers_blocked: int = 0
    """Papers with no document, or whose earlier stage has not succeeded."""
    faithfulness: Faithfulness = field(default_factory=Faithfulness)
    interrupted: bool = False
    budget_reached: bool = False

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    def eta_s(self) -> float | None:
        """Remaining time at the rate papers have finished so far."""
        if self.papers_finished == 0:
            return None
        rate = self.papers_finished / max(self.elapsed_s, 1e-9)
        return (self.papers - self.papers_finished) / rate

    def summary(self, gateway: dict[str, Any]) -> dict[str, Any]:
        """JSON-ready statistics for ``Store.finish_run``."""
        stages = {stage.value: _stage_summary(self.tally[stage]) for stage in self.stages}
        return {
            "papers": self.papers,
            "papers_finished": self.papers_finished,
            "papers_blocked": self.papers_blocked,
            "interrupted": self.interrupted,
            "budget_reached": self.budget_reached,
            "elapsed_s": round(self.elapsed_s, 3),
            "stages": stages,
            "faithfulness": self.faithfulness.summary(),
            "gateway": gateway,
        }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.fmean(values), 4) if values else None


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.median(values), 4) if values else None


def _stage_summary(tally: StageTally) -> dict[str, Any]:
    traces = tally.traces
    tools: Counter[str] = Counter()
    for trace in traces:
        tools.update(trace.tool_calls)
    out: dict[str, Any] = {
        "done": tally.done,
        "cached": tally.cached,
        "skipped": tally.skipped,
        "failed": tally.failed,
        "refused": tally.refused,
        "interrupted": tally.interrupted,
    }
    if traces:
        out["agents"] = {
            "runs": len(traces),
            "requests_mean": _mean(t.requests for t in traces),
            "tokens_mean": _mean(t.tokens for t in traces),
            "tokens_median": _median(t.tokens for t in traces),
            "output_tokens_mean": _mean(t.output_tokens for t in traces),
            "fraction_read_mean": _mean(t.fraction_read for t in traces),
            "tool_calls": dict(sorted(tools.items())),
            "tool_errors": sum(t.tool_errors for t in traces),
            "rejected_submissions": sum(t.rejected_submissions for t in traces),
            "nudges": sum(t.nudges for t in traces),
            "budget_notices": sum(t.budget_notices for t in traces),
            "max_tokens_retries": sum(t.max_tokens_retries for t in traces),
            "quotes_not_found": sum(t.quotes_not_found for t in traces),
            "unchecked_claims": sum(t.unchecked_claims for t in traces),
        }
    return out
