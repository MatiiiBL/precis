"""The scheduler over the offline client: stages, resume, failures, budget, SIGINT."""

import os
import signal
from datetime import UTC, datetime
from typing import Any

import pytest

from precis.corpus.load import load_papers
from precis.llm.fake import FakeReply
from precis.models import RevisionResult, Stage, StageRecord, StageStatus

from .helpers import add_papers, brief_title, config, setup

ALL = (Stage.READ, Stage.VERIFY, Stage.REVISE)


def status(s: Any, paper_id: str, stage: Stage) -> StageStatus | None:
    record = s.store.get_stage(paper_id, stage)
    return None if record is None else record.status


async def test_full_run_reads_verifies_and_revises_what_failed() -> None:
    s = setup()
    ids = add_papers(s.store, 4)
    stats = await s.pipeline("run-1").run(ids, ALL)

    assert stats["papers_finished"] == 4
    assert stats["stages"]["read"]["done"] == 4
    assert stats["stages"]["verify"]["done"] == 4
    # The offline reader swaps two numbers in one of paper 0's findings.
    assert stats["stages"]["revise"]["done"] == 1
    assert stats["stages"]["revise"]["skipped"] == 3
    assert status(s, ids[0], Stage.REVISE) is StageStatus.DONE
    assert status(s, ids[1], Stage.REVISE) is StageStatus.SKIPPED

    revise = s.store.get_stage(ids[0], Stage.REVISE)
    assert revise is not None
    assert revise.output is not None
    revision = RevisionResult.model_validate(revise.output)
    assert revision.summary.revision == 1
    assert revision.verification.revision == 1
    # The corpus reads the revised summary as the paper's final one.
    result = {p.id: p for p in load_papers(s.store)}[ids[0]]
    assert result.revised
    assert result.draft_verification is not None
    assert result.draft_verification.needs_revision()
    assert result.verification is not None
    assert not result.verification.needs_revision()

    faithfulness = stats["faithfulness"]
    assert faithfulness["verified"] == 4
    assert faithfulness["revised"] == 1
    reader = stats["stages"]["read"]["agents"]
    assert reader["runs"] == 4
    # Paper 2 first submits a rounded number, which validation sends back.
    assert reader["rejected_submissions"] == 1
    assert reader["requests_mean"] == 2.25
    assert 0 < reader["fraction_read_mean"] <= 1

    [run] = s.store.runs()
    assert run["run_id"] == "run-1"
    assert run["finished_at"] is not None
    assert run["stats"]["papers"] == 4
    calls = {c.stage for c in s.store.calls(run_id="run-1")}
    assert calls == {"read", "verify", "revise"}


async def test_second_run_skips_everything_by_config_hash() -> None:
    s = setup()
    ids = add_papers(s.store, 3)
    await s.pipeline("run-1").run(ids, ALL)
    requests = len(s.fake.requests)
    stats = await s.pipeline("run-2").run(ids, ALL)
    assert len(s.fake.requests) == requests
    assert stats["stages"]["read"]["cached"] == 3
    assert stats["stages"]["verify"]["cached"] == 3
    assert stats["stages"]["revise"]["cached"] == 3
    assert stats["papers_finished"] == 3


async def test_a_changed_setting_reruns_only_the_stages_it_affects() -> None:
    s = setup()
    ids = add_papers(s.store, 2)
    await s.pipeline("run-1").run(ids, ALL)
    changed = config(agents={"critic": {"effort": "high"}})
    stats = await s.pipeline("run-2", config=changed).run(ids, ALL)
    assert stats["stages"]["read"]["cached"] == 2
    assert stats["stages"]["verify"]["done"] == 2


async def test_stages_can_run_separately() -> None:
    s = setup()
    ids = add_papers(s.store, 2)
    stats = await s.pipeline("run-1").run(ids, [Stage.READ])
    assert stats["stages"] == {"read": stats["stages"]["read"]}
    assert stats["papers_finished"] == 2
    assert stats["papers_blocked"] == 0
    assert status(s, ids[0], Stage.VERIFY) is None
    stats = await s.pipeline("run-2").run(ids, [Stage.VERIFY])
    assert stats["stages"]["verify"]["done"] == 2
    assert stats["papers_finished"] == 2
    assert status(s, ids[0], Stage.REVISE) is None


async def test_running_no_stages_is_an_error() -> None:
    s = setup()
    with pytest.raises(ValueError, match="no stages"):
        await s.pipeline("run-1").run(add_papers(s.store, 1), [])


async def test_missing_documents_and_missing_inputs_block_the_paper() -> None:
    s = setup()
    ids = add_papers(s.store, 2)
    stats = await s.pipeline("run-1").run([*ids, "2501.99999"], [Stage.VERIFY])
    assert stats["papers_blocked"] == 3  # no summaries yet, and one unknown paper
    assert stats["stages"]["verify"]["done"] == 0
    assert len(s.fake.requests) == 0


async def test_a_failed_stage_is_recorded_and_retried_next_run() -> None:
    s = setup()
    ids = add_papers(s.store, 2)
    s.fake.fail_next("bad_request")
    stats = await s.pipeline("run-1", config=config(pipeline={"max_papers_in_flight": 1})).run(
        ids, ALL
    )
    record = s.store.get_stage(ids[0], Stage.READ)
    assert record is not None
    assert record.status is StageStatus.FAILED
    assert record.error is not None
    assert record.error.startswith("GatewayError: request rejected")
    assert status(s, ids[0], Stage.VERIFY) is None
    assert stats["stages"]["read"]["failed"] == 1
    assert stats["papers_finished"] == 1

    await s.pipeline("run-2").run(ids, ALL)
    record = s.store.get_stage(ids[0], Stage.READ)
    assert record is not None
    assert record.status is StageStatus.DONE
    assert record.attempts == 2
    assert record.run_id == "run-2"


async def test_refusals_are_recorded_as_refused() -> None:
    def refuse(params: dict[str, Any]) -> FakeReply | None:
        return FakeReply.refusal("bio") if "Design1" in brief_title(params) else None

    s = setup(refuse)
    ids = add_papers(s.store, 2)
    stats = await s.pipeline("run-1").run(ids, ALL)
    record = s.store.get_stage(ids[1], Stage.READ)
    assert record is not None
    assert record.status is StageStatus.REFUSED
    assert record.error is not None
    assert "category: bio" in record.error
    assert stats["stages"]["read"]["refused"] == 1
    assert status(s, ids[0], Stage.REVISE) is StageStatus.DONE


async def test_budget_stops_admission_and_leaves_work_pending() -> None:
    # The first request of a stage reserves its whole max_tokens, so keep it
    # small enough for this budget to admit some papers before it stops.
    small = {"max_tokens": 2000}
    s = setup(
        cfg=config(
            budget={"max_cost_usd": 0.15},
            agents={"reader": small, "critic": small, "reviser": small},
            pipeline={"max_papers_in_flight": 2},
        )
    )
    ids = add_papers(s.store, 8)
    stats = await s.pipeline("run-1").run(ids, ALL)
    assert stats["budget_reached"] is True
    assert 0 < stats["papers_finished"] < 8
    counts = s.store.stage_counts()
    assert sum(counts[stage][StageStatus.FAILED] for stage in ALL) == 0
    assert sum(counts[stage][StageStatus.PENDING] for stage in ALL) >= 1
    assert status(s, ids[-1], Stage.READ) is None  # never admitted

    stats = await s.pipeline("run-2", config=config()).run(ids, ALL)
    assert stats["papers_finished"] == 8


async def test_sigint_cancels_in_flight_work_and_leaves_it_pending() -> None:
    sent = []

    def interrupt(params: dict[str, Any]) -> FakeReply | None:
        if not sent and "Design1" in brief_title(params):
            sent.append(True)
            os.kill(os.getpid(), signal.SIGINT)
        return None

    s = setup(interrupt, cfg=config(pipeline={"max_papers_in_flight": 1}))
    ids = add_papers(s.store, 4)
    stats = await s.pipeline("run-1").run(ids, ALL)
    assert sent
    assert stats["interrupted"] is True
    assert stats["papers_finished"] == 1
    assert status(s, ids[0], Stage.REVISE) is StageStatus.DONE
    # The signal is handled a few event-loop steps after it is sent, so the
    # stage in flight by then (read or verify of paper 1) is left pending.
    records = [s.store.get_stage(ids[1], stage) for stage in ALL]
    pending = [r for r in records if r is not None and r.status is StageStatus.PENDING]
    assert [r.error for r in pending] == ["interrupted"]
    assert all(r is None or r.status is not StageStatus.FAILED for r in records)
    assert status(s, ids[2], Stage.READ) is None  # never admitted
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    stats = await s.pipeline("run-2").run(ids, ALL)
    assert stats["papers_finished"] == 4
    assert stats["interrupted"] is False
    assert stats["stages"]["read"]["cached"] >= 1


async def test_running_stages_from_a_crash_are_reset_and_rerun() -> None:
    s = setup()
    ids = add_papers(s.store, 1)
    s.store.set_stage(
        StageRecord(
            paper_id=ids[0],
            stage=Stage.READ,
            status=StageStatus.RUNNING,
            attempts=1,
            config_hash="old",
            run_id="crashed",
            updated_at=datetime.now(UTC),
        )
    )
    await s.pipeline("run-1").run(ids, ALL)
    record = s.store.get_stage(ids[0], Stage.READ)
    assert record is not None
    assert record.status is StageStatus.DONE
    assert record.attempts == 2


async def test_a_new_summary_marks_later_stages_pending() -> None:
    s = setup()
    ids = add_papers(s.store, 1)
    await s.pipeline("run-1").run(ids, ALL)
    fulltext = config(agents={"strategy": "fulltext"})
    await s.pipeline("run-2", config=fulltext).run(ids, [Stage.READ])
    for stage in (Stage.VERIFY, Stage.REVISE):
        record = s.store.get_stage(ids[0], stage)
        assert record is not None
        assert record.status is StageStatus.PENDING
        assert record.output is None
        assert record.error == "read changed"
    [result] = load_papers(s.store)
    assert result.verification is None
    assert not result.revised
