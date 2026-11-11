import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from precis.config import FetchConfig
from precis.models import PaperMeta, Stage, StageRecord, StageStatus
from precis.sources import runner as runner_module
from precis.sources.fulltext import FullText
from precis.sources.http import FetchError
from precis.sources.runner import FetchRunner, Outcome, fetch_config_hash
from precis.store import Store
from precis.text.html import parse_html

from .helpers import make_pdf, paper_html, paper_pdf

HASH = fetch_config_hash(FetchConfig())
CC_BY = "https://creativecommons.org/licenses/by/4.0/"


def _meta(paper_id: str, version: int = 1, license_url: str | None = None) -> PaperMeta:
    return PaperMeta(
        id=paper_id,
        version=version,
        title=f"Paper {paper_id}",
        authors=["Ada Author"],
        abstract="An abstract.",
        categories=["cs.AR"],
        primary_category="cs.AR",
        published=date(2025, 7, 1),
        license=license_url,
        url=f"https://arxiv.org/abs/{paper_id}v{version}",
    )


class StubFetcher:
    """Stands in for FullTextFetcher: a scripted result or error per paper id."""

    def __init__(self, results: dict[str, FullText | Exception]) -> None:
        self.results = results
        self.calls: list[str] = []

    async def fetch(self, meta: PaperMeta) -> FullText:
        self.calls.append(meta.id)
        result = self.results[meta.id]
        if isinstance(result, Exception):
            raise result
        return result


def _full_text(paper_id: str, fallback: str | None = None) -> FullText:
    return FullText(parse_html(paper_html(), paper_id), CC_BY, fallback)


def _runner(store: Store, outcomes: list[Outcome] | None = None, config_hash: str = HASH) -> Any:
    on_outcome = outcomes.append if outcomes is not None else None
    return FetchRunner(store, config_hash=config_hash, run_id="r1", on_outcome=on_outcome)


def _running(paper_id: str, stage: Stage) -> StageRecord:
    """A stage a crashed process left running."""
    return StageRecord(
        paper_id=paper_id,
        stage=stage,
        status=StageStatus.RUNNING,
        attempts=1,
        config_hash="x",
        run_id="crashed",
        updated_at=datetime.now(UTC),
    )


def _stage(store: Store, paper_id: str) -> StageRecord:
    record = store.get_stage(paper_id, Stage.FETCH)
    assert record is not None
    return record


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


async def test_each_paper_is_recorded_as_it_finishes(store: Store) -> None:
    papers = [_meta("2507.00001"), _meta("2507.00002"), _meta("2507.00003")]
    for meta in papers:
        store.upsert_paper(meta)
    fetcher = StubFetcher(
        {
            "2507.00001": _full_text("2507.00001"),
            "2507.00002": _full_text("2507.00002", fallback="no HTML version (HTTP 404)"),
            "2507.00003": FetchError("no PDF (HTTP 404)"),
        }
    )
    outcomes: list[Outcome] = []
    runner = _runner(store, outcomes)
    await runner.fetch_arxiv(fetcher, papers)

    done = _stage(store, "2507.00001")
    assert (done.status, done.attempts, done.config_hash, done.run_id) == (
        StageStatus.DONE,
        1,
        HASH,
        "r1",
    )
    assert done.output is not None
    assert (done.output["version"], done.output["source_format"]) == (1, "html")
    failed = _stage(store, "2507.00003")
    assert (failed.status, failed.error) == (StageStatus.FAILED, "no PDF (HTTP 404)")
    assert store.get_document("2507.00001") is not None
    assert store.get_document("2507.00003") is None
    # The license learned from the full text is kept with the metadata.
    paper = store.get_paper("2507.00001")
    assert paper is not None
    assert paper.license == CC_BY
    assert sorted(o.status for o in outcomes) == ["failed", "html", "html"]
    assert runner.stats.fallbacks == {"2507.00002": "no HTML version (HTTP 404)"}
    assert runner.stats.as_dict()["counts"] == {"html": 2, "failed": 1}


async def test_a_second_run_skips_done_papers_and_retries_failures(store: Store) -> None:
    papers = [_meta("2507.00001"), _meta("2507.00002")]
    first = StubFetcher({"2507.00001": _full_text("2507.00001"), "2507.00002": FetchError("x")})
    await _runner(store).fetch_arxiv(first, papers)

    second = StubFetcher({"2507.00002": _full_text("2507.00002")})
    runner = _runner(store)
    await runner.fetch_arxiv(second, papers)
    assert second.calls == ["2507.00002"]
    assert runner.stats.counts == {"skipped": 1, "html": 1}
    retried = _stage(store, "2507.00002")
    assert (retried.status, retried.attempts, retried.error) == (StageStatus.DONE, 2, None)


async def test_a_new_version_or_new_settings_fetch_again(store: Store) -> None:
    fetcher = StubFetcher({"2507.00001": _full_text("2507.00001")})
    await _runner(store).fetch_arxiv(fetcher, [_meta("2507.00001", version=1)])
    runner = _runner(store)
    assert runner.is_done("2507.00001", 1)
    assert not runner.is_done("2507.00001", 2)
    assert not _runner(store, config_hash="other").is_done("2507.00001", 1)
    await runner.fetch_arxiv(fetcher, [_meta("2507.00001", version=2)])
    assert fetcher.calls == ["2507.00001", "2507.00001"]


def test_a_new_parser_version_means_papers_are_fetched_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = fetch_config_hash(FetchConfig())
    monkeypatch.setattr(runner_module, "PARSER_VERSION", runner_module.PARSER_VERSION + 1)
    assert fetch_config_hash(FetchConfig()) != before


async def test_an_unexpected_error_fails_one_paper_not_the_run(store: Store) -> None:
    papers = [_meta("2507.00001"), _meta("2507.00002")]
    fetcher = StubFetcher(
        {"2507.00001": RuntimeError("parser bug"), "2507.00002": _full_text("2507.00002")}
    )
    runner = _runner(store)
    await runner.fetch_arxiv(fetcher, papers)
    assert runner.stats.failures == {"2507.00001": "RuntimeError: parser bug"}
    assert runner.stats.counts["html"] == 1


async def test_a_document_the_store_cannot_write_fails_that_paper_only(store: Store) -> None:
    """The real fetch died on 2512.14151: store.put_document raised for one paper and
    the error escaped the per-paper handling, ending the whole command."""
    broken = parse_html(paper_html(), "2507.00001")
    lone_surrogate = chr(0xD835)
    sections = [
        s.model_copy(update={"text": f"{s.text} {lone_surrogate}"}) for s in broken.sections
    ]
    unstorable = FullText(broken.model_copy(update={"sections": sections}), None, None)
    papers = [_meta("2507.00001"), _meta("2507.00002")]
    fetcher = StubFetcher({"2507.00001": unstorable, "2507.00002": _full_text("2507.00002")})
    runner = _runner(store)
    await runner.fetch_arxiv(fetcher, papers)
    failed = _stage(store, "2507.00001")
    assert failed.status is StageStatus.FAILED
    assert "surrogates not allowed" in (failed.error or "")
    assert store.get_document("2507.00001") is None
    assert _stage(store, "2507.00002").status is StageStatus.DONE
    assert runner.stats.counts == {"failed": 1, "html": 1}


def test_a_crashed_fetch_is_reset_without_touching_other_stages(store: Store) -> None:
    """Review R9: a fetch must not reset the rows of a `precis run` in progress."""
    for paper_id, stage in (("2507.00001", Stage.FETCH), ("2507.00002", Stage.READ)):
        store.set_stage(_running(paper_id, stage))
    runner = _runner(store)
    assert runner.reset_interrupted() == 1
    assert _stage(store, "2507.00001").status is StageStatus.PENDING
    read = store.get_stage("2507.00002", Stage.READ)
    assert read is not None
    assert read.status is StageStatus.RUNNING
    assert runner.stats.as_dict()["reset"] == 1


async def test_an_interrupted_paper_goes_back_to_pending(store: Store) -> None:
    started = asyncio.Event()

    class Hanging:
        async def fetch(self, meta: PaperMeta) -> FullText:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    task = asyncio.create_task(_runner(store).fetch_arxiv(Hanging(), [_meta("2507.00001")]))
    await started.wait()
    assert _stage(store, "2507.00001").status is StageStatus.RUNNING
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    record = _stage(store, "2507.00001")
    assert (record.status, record.attempts) == (StageStatus.PENDING, 0)


async def test_metadata_only_marks_pending_without_touching_done_papers(store: Store) -> None:
    runner = _runner(store)
    runner.save_metadata(_meta("2507.00001"))
    runner.mark_pending("2507.00001")
    assert _stage(store, "2507.00001").status is StageStatus.PENDING

    fetcher = StubFetcher({"2507.00002": _full_text("2507.00002")})
    await runner.fetch_arxiv(fetcher, [_meta("2507.00002")])
    runner.mark_pending("2507.00002")
    assert runner.is_done("2507.00002", 1)


def test_search_results_do_not_erase_a_known_license(store: Store) -> None:
    store.upsert_paper(_meta("2507.00001", license_url=CC_BY))
    _runner(store).save_metadata(_meta("2507.00001"))
    paper = store.get_paper("2507.00001")
    assert paper is not None
    assert paper.license == CC_BY


async def test_local_pdfs_are_stored_and_skipped_when_done(store: Store, tmp_path: Path) -> None:
    paths = [tmp_path / "good.pdf", tmp_path / "stub.pdf"]
    paths[0].write_bytes(paper_pdf())
    paths[1].write_bytes(make_pdf([["1 Introduction", "Only a line."]]))
    runner = _runner(store)
    await runner.fetch_local(paths)
    assert runner.stats.counts == {"pdf": 1, "failed": 1}
    (paper,) = store.list_papers()
    assert paper.id.startswith("local:")
    assert store.get_document(paper.id) is not None
    assert next(iter(runner.stats.failures.values())).startswith("only")

    again = _runner(store)
    await again.fetch_local(paths)
    assert again.stats.counts == {"skipped": 1, "failed": 1}
