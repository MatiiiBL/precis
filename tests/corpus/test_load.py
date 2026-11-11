from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from corpus.factories import make_meta, make_summary, make_verification
from precis.corpus.load import load_papers, paper_view
from precis.models import (
    PaperMeta,
    RevisionResult,
    Stage,
    StageRecord,
    StageStatus,
    Verdict,
)


def _record(
    paper_id: str, stage: Stage, output: BaseModel | None, status: StageStatus = StageStatus.DONE
) -> StageRecord:
    return StageRecord(
        paper_id=paper_id,
        stage=stage,
        status=status,
        attempts=1,
        config_hash="h",
        run_id="r",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        output=None if output is None else output.model_dump(mode="json"),
    )


@dataclass
class MemoryStages:
    papers: dict[str, PaperMeta] = field(default_factory=dict)
    stages: dict[tuple[str, Stage], StageRecord] = field(default_factory=dict)

    def get_paper(self, paper_id: str) -> PaperMeta | None:
        return self.papers.get(paper_id)

    def get_stage(self, paper_id: str, stage: Stage) -> StageRecord | None:
        return self.stages.get((paper_id, stage))

    def papers_with_status(self, stage: Stage, status: StageStatus) -> list[str]:
        return [p for (p, s), r in self.stages.items() if s is stage and r.status is status]

    def add(self, record: StageRecord) -> None:
        self.stages[(record.paper_id, record.stage)] = record


def test_unrevised_paper_uses_the_first_draft() -> None:
    draft = make_summary("p1", ["caches", "prefetching"])
    check = make_verification(draft)
    read = _record("p1", Stage.READ, draft)
    verify = _record("p1", Stage.VERIFY, check)
    skipped = _record("p1", Stage.REVISE, None, StageStatus.SKIPPED)
    view = paper_view(make_meta("p1"), read, verify, skipped)
    assert view.summary == view.draft == draft
    assert view.verification == view.draft_verification == check
    assert not view.revised


def test_revised_paper_uses_the_revision_and_keeps_the_draft() -> None:
    draft = make_summary("p1", ["caches", "prefetching"])
    check = make_verification(draft, [Verdict.CONTRADICTED, Verdict.SUPPORTED, Verdict.SUPPORTED])
    revised = make_summary("p1", ["caches", "prefetching"], revision=1)
    recheck = make_verification(revised)
    view = paper_view(
        make_meta("p1"),
        _record("p1", Stage.READ, draft),
        _record("p1", Stage.VERIFY, check),
        _record("p1", Stage.REVISE, RevisionResult(summary=revised, verification=recheck)),
    )
    assert view.revised
    assert (view.summary, view.verification) == (revised, recheck)
    assert (view.draft, view.draft_verification) == (draft, check)


def test_unfinished_verification_counts_as_absent() -> None:
    draft = make_summary("p1", ["caches", "prefetching"])
    running = _record("p1", Stage.VERIFY, None, StageStatus.RUNNING)
    view = paper_view(make_meta("p1"), _record("p1", Stage.READ, draft), running, None)
    assert view.verification is None


def test_read_stage_must_be_done() -> None:
    failed = _record("p1", Stage.READ, None, StageStatus.FAILED)
    with pytest.raises(ValueError, match="no finished summary"):
        paper_view(make_meta("p1"), failed, None, None)


def test_load_papers_reads_finished_papers_in_id_order() -> None:
    store = MemoryStages()
    for paper_id in ("p3", "p1", "p2"):
        store.papers[paper_id] = make_meta(paper_id)
        store.add(_record(paper_id, Stage.READ, make_summary(paper_id, ["a", "b"])))
    store.add(_record("p2", Stage.READ, None, StageStatus.FAILED))
    store.add(_record("p4", Stage.READ, make_summary("p4", ["a", "b"])))  # no metadata
    assert [v.id for v in load_papers(store)] == ["p1", "p3"]
