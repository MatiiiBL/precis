"""Read the finished per-paper results out of the store.

This is the only corpus module that knows how the pipeline lays out stage
outputs (docs/design.md, section 1): the read stage holds the first summary,
the verify stage its verification, and the revise stage, when it ran, a
``RevisionResult`` with the rewritten summary and its re-verification. A
stage being re-run keeps its last result in its record until a new one
replaces it, so a record's output counts whatever its status.
Everything downstream works on ``FinishedPaper`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from precis.models import (
    PaperMeta,
    PaperSummary,
    RevisionResult,
    Stage,
    StageRecord,
    Verification,
)
from precis.store import Store


@dataclass(frozen=True)
class FinishedPaper:
    """One summarized paper, with its summary before and after revision."""

    meta: PaperMeta
    summary: PaperSummary
    """The final summary: the revision when there is one, else the first draft."""
    verification: Verification | None
    """Verification of ``summary``; None when the paper was never verified."""
    draft: PaperSummary
    """The first summary (revision 0)."""
    draft_verification: Verification | None
    """Verification of ``draft``."""

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def revised(self) -> bool:
        return self.summary.revision > self.draft.revision


def _finished_output(record: StageRecord | None) -> dict[str, object] | None:
    """The stage's last result: done, or kept while a re-run is unfinished or failed."""
    return None if record is None else record.output


def paper_view(
    meta: PaperMeta,
    read: StageRecord,
    verify: StageRecord | None,
    revise: StageRecord | None,
) -> FinishedPaper:
    """Combine one paper's stage records into a ``FinishedPaper``.

    ``read`` must have a finished summary; a verify or revise record without
    a result is treated as absent, so a paper stopped between stages still
    shows its first summary.
    """
    read_out = _finished_output(read)
    if read_out is None:
        raise ValueError(f"{meta.id}: the read stage has no finished summary")
    draft = PaperSummary.model_validate(read_out)
    verify_out = _finished_output(verify)
    draft_verification = None if verify_out is None else Verification.model_validate(verify_out)
    revise_out = _finished_output(revise)
    if revise_out is None:
        return FinishedPaper(meta, draft, draft_verification, draft, draft_verification)
    revision = RevisionResult.model_validate(revise_out)
    return FinishedPaper(meta, revision.summary, revision.verification, draft, draft_verification)


def load_papers(store: Store) -> list[FinishedPaper]:
    """Every paper whose read stage has a finished summary, ordered by id."""
    views = []
    for meta in sorted(store.list_papers(), key=lambda m: m.id):
        read = store.get_stage(meta.id, Stage.READ)
        if read is None or read.output is None:
            continue
        views.append(
            paper_view(
                meta,
                read,
                store.get_stage(meta.id, Stage.VERIFY),
                store.get_stage(meta.id, Stage.REVISE),
            )
        )
    return views
