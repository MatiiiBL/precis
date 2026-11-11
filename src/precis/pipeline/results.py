"""Reading a paper's results back from the store.

Per the stage-output table in docs/design.md: ``read`` holds the first
summary, ``verify`` its verification, and ``revise`` (when ``done``) a
``RevisionResult`` with the rewritten summary and its re-verification. The
final summary is the revised one when there is one.
"""

from __future__ import annotations

from dataclasses import dataclass

from precis.models import (
    PaperSummary,
    RevisionResult,
    Stage,
    StageStatus,
    Verification,
)
from precis.store import Store


@dataclass(frozen=True)
class PaperResult:
    summary: PaperSummary
    verification: Verification | None
    """Verification of ``summary``; ``None`` if it has not been verified yet."""
    first_verification: Verification | None
    """Verification of the first draft (equal to ``verification`` when not revised)."""

    @property
    def revised(self) -> bool:
        return self.summary.revision > 0


def _done_output(store: Store, paper_id: str, stage: Stage) -> dict[str, object] | None:
    record = store.get_stage(paper_id, stage)
    if record is None or record.status is not StageStatus.DONE:
        return None
    return record.output


def paper_result(store: Store, paper_id: str) -> PaperResult | None:
    """The final summary and verification of a paper, or ``None`` before ``read`` is done."""
    read = _done_output(store, paper_id, Stage.READ)
    if read is None:
        return None
    summary = PaperSummary.model_validate(read)
    verify = _done_output(store, paper_id, Stage.VERIFY)
    first = Verification.model_validate(verify) if verify is not None else None
    revise = _done_output(store, paper_id, Stage.REVISE)
    if revise is not None:
        revision = RevisionResult.model_validate(revise)
        return PaperResult(revision.summary, revision.verification, first)
    return PaperResult(summary, first, first)
