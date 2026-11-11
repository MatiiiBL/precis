"""Read the finished per-paper results out of the store.

This is the only corpus module that knows how the pipeline lays out stage
outputs (docs/design.md, section 1): the read stage holds the first summary,
the verify stage its verification, and the revise stage, when it ran, a
``RevisionResult`` with the rewritten summary and its re-verification.
Everything downstream works on ``PaperView`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from precis.models import (
    PaperMeta,
    PaperSummary,
    RevisionResult,
    Stage,
    StageRecord,
    StageStatus,
    Verification,
)
from precis.store import Store


@dataclass(frozen=True)
class PaperView:
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


def _done_output(record: StageRecord | None) -> dict[str, object] | None:
    if record is None or record.status is not StageStatus.DONE:
        return None
    return record.output


def paper_view(
    meta: PaperMeta,
    read: StageRecord,
    verify: StageRecord | None,
    revise: StageRecord | None,
) -> PaperView:
    """Combine one paper's stage records into a ``PaperView``.

    ``read`` must be done; a verify or revise record that is missing or not
    done is treated as absent, so a paper stopped between stages still shows
    its first summary.
    """
    read_out = _done_output(read)
    if read_out is None:
        raise ValueError(f"{meta.id}: the read stage has no finished summary")
    draft = PaperSummary.model_validate(read_out)
    verify_out = _done_output(verify)
    draft_verification = None if verify_out is None else Verification.model_validate(verify_out)
    revise_out = _done_output(revise)
    if revise_out is None:
        return PaperView(meta, draft, draft_verification, draft, draft_verification)
    revision = RevisionResult.model_validate(revise_out)
    return PaperView(meta, revision.summary, revision.verification, draft, draft_verification)


def load_papers(store: Store) -> list[PaperView]:
    """Every paper whose read stage is done, ordered by id."""
    views = []
    for paper_id in sorted(store.papers_with_status(Stage.READ, StageStatus.DONE)):
        meta = store.get_paper(paper_id)
        read = store.get_stage(paper_id, Stage.READ)
        if meta is None or read is None:
            continue
        views.append(
            paper_view(
                meta,
                read,
                store.get_stage(paper_id, Stage.VERIFY),
                store.get_stage(paper_id, Stage.REVISE),
            )
        )
    return views
