import pytest

from corpus.factories import (
    fill_store,
    make_meta,
    make_summary,
    make_verification,
    make_view,
    stage_record,
)
from precis.corpus.load import load_papers, paper_view
from precis.models import RevisionResult, Stage, StageStatus, Verdict
from precis.store import Store


def test_unrevised_paper_uses_the_first_draft() -> None:
    draft = make_summary("p1", ["caches", "prefetching"])
    check = make_verification(draft)
    read = stage_record("p1", Stage.READ, draft)
    verify = stage_record("p1", Stage.VERIFY, check)
    skipped = stage_record("p1", Stage.REVISE, None, StageStatus.SKIPPED)
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
        stage_record("p1", Stage.READ, draft),
        stage_record("p1", Stage.VERIFY, check),
        stage_record("p1", Stage.REVISE, RevisionResult(summary=revised, verification=recheck)),
    )
    assert view.revised
    assert (view.summary, view.verification) == (revised, recheck)
    assert (view.draft, view.draft_verification) == (draft, check)


def test_unfinished_verification_counts_as_absent() -> None:
    draft = make_summary("p1", ["caches", "prefetching"])
    running = stage_record("p1", Stage.VERIFY, None, StageStatus.RUNNING)
    view = paper_view(make_meta("p1"), stage_record("p1", Stage.READ, draft), running, None)
    assert view.verification is None


def test_read_stage_must_be_done() -> None:
    failed = stage_record("p1", Stage.READ, None, StageStatus.FAILED)
    with pytest.raises(ValueError, match="no finished summary"):
        paper_view(make_meta("p1"), failed, None, None)


def test_load_papers_round_trips_through_the_store() -> None:
    papers = [
        make_view("p3", ["a", "b"]),
        make_view("p1", ["a", "b"], draft_verdicts=[Verdict.UNSUPPORTED] * 3, revised=True),
    ]
    with Store(":memory:") as store:
        fill_store(store, papers)
        store.upsert_paper(make_meta("p2"))
        store.set_stage(stage_record("p2", Stage.READ, None, StageStatus.FAILED))
        loaded = load_papers(store)
    assert [v.id for v in loaded] == ["p1", "p3"]
    assert loaded[0] == papers[1]
    assert loaded[1] == papers[0]
