import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from precis import store as store_module
from precis.models import (
    Caption,
    Document,
    LLMCall,
    PaperMeta,
    Section,
    Stage,
    StageRecord,
    StageStatus,
    Topic,
)
from precis.store import BatchItem, SchemaError, Store


def _paper(pid: str, published: date, categories: list[str] | None = None) -> PaperMeta:
    cats = categories or ["cs.AR"]
    return PaperMeta(
        id=pid,
        title=f"Paper {pid}",
        authors=["A. Author"],
        abstract="An abstract.",
        categories=cats,
        primary_category=cats[0],
        published=published,
        url=f"https://arxiv.org/abs/{pid}",
    )


def _stage(pid: str, stage: Stage, status: StageStatus) -> StageRecord:
    return StageRecord(
        paper_id=pid,
        stage=stage,
        status=status,
        attempts=1,
        config_hash="h",
        run_id="run-1",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        output={"tldr": "x"} if status is StageStatus.DONE else None,
    )


def _call(call_id: str, **kw: object) -> LLMCall:
    fields: dict[str, object] = {
        "call_id": call_id,
        "run_id": "run-1",
        "paper_id": "p1",
        "stage": "read",
        "model": "claude-opus-5-5",
        "request_hash": "abc",
        "status": "ok",
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    fields.update(kw)
    return LLMCall.model_validate(fields)


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def test_papers_are_listed_newest_first_and_filtered_by_any_category(store: Store) -> None:
    store.upsert_paper(_paper("2401.00002", date(2024, 1, 2)))
    store.upsert_paper(_paper("2401.00001", date(2024, 1, 2), ["cs.LG", "cs.AR"]))
    store.upsert_paper(_paper("2312.00009", date(2023, 12, 30), ["cs.LG"]))

    assert [p.id for p in store.list_papers()] == ["2401.00001", "2401.00002", "2312.00009"]
    assert [p.id for p in store.list_papers(category="cs.AR")] == ["2401.00001", "2401.00002"]
    assert [p.id for p in store.list_papers(category="cs.LG", limit=1)] == ["2401.00001"]
    assert store.count_papers() == 3


def test_upsert_replaces_paper_and_its_categories(store: Store) -> None:
    store.upsert_paper(_paper("p", date(2024, 1, 1), ["cs.AR"]))
    updated = _paper("p", date(2024, 1, 1), ["cs.LG"]).model_copy(update={"title": "New"})
    store.upsert_paper(updated)

    assert store.get_paper("p") == updated
    assert store.list_papers(category="cs.AR") == []
    assert store.get_paper("missing") is None


def test_documents_round_trip(store: Store) -> None:
    doc = Document(
        paper_id="p",
        source_format="html",
        sections=[Section(id="s1", title="Intro", level=1, text="Hello.", n_tokens=2)],
        captions=[Caption(id="fig1", kind="figure", text="A plot.")],
        n_tokens=2,
        sha256="0" * 64,
    )
    store.put_document(doc)
    assert store.get_document("p") == doc
    assert store.get_document("q") is None


def test_stages_upsert_count_and_reset_after_crash(store: Store) -> None:
    store.set_stage(_stage("p1", Stage.READ, StageStatus.RUNNING))
    store.set_stage(_stage("p2", Stage.READ, StageStatus.RUNNING))
    store.set_stage(_stage("p1", Stage.READ, StageStatus.DONE))
    store.set_stage(_stage("p3", Stage.VERIFY, StageStatus.FAILED))

    counts = store.stage_counts()
    assert counts[Stage.READ][StageStatus.DONE] == 1
    assert counts[Stage.READ][StageStatus.RUNNING] == 1
    assert counts[Stage.VERIFY][StageStatus.FAILED] == 1
    assert counts[Stage.REVISE] == dict.fromkeys(StageStatus, 0)
    assert store.papers_with_status(Stage.READ, StageStatus.DONE) == ["p1"]

    assert store.reset_running() == 1
    record = store.get_stage("p2", Stage.READ)
    assert record is not None
    assert record.status is StageStatus.PENDING
    assert store.papers_with_status(Stage.READ, StageStatus.PENDING) == ["p2"]
    assert store.get_stage("p2", Stage.VERIFY) is None


def test_calls_filter_and_usage_is_aggregated_per_stage_and_model(store: Store) -> None:
    store.record_call(_call("c1", input_tokens=100, output_tokens=10, cost_usd=0.5))
    store.record_call(
        _call("c2", input_tokens=50, cache_read_input_tokens=1000, output_tokens=5, cost_usd=0.25)
    )
    store.record_call(_call("c3", status="cached", input_tokens=100, output_tokens=10))
    store.record_call(_call("c4", stage="verify", paper_id="p2", model="claude-haiku-5-5"))
    store.record_call(_call("c5", run_id="run-2", cost_usd=9.0))

    assert [c.call_id for c in store.calls(run_id="run-1", paper_id="p1")] == ["c1", "c2", "c3"]
    assert store.calls(paper_id="p2")[0].model == "claude-haiku-5-5"

    rows = {(r.stage, r.model): r for r in store.usage(run_id="run-1")}
    read = rows["read", "claude-opus-5-5"]
    assert (read.calls, read.cached_calls) == (3, 1)
    assert read.input_tokens == 250
    assert read.cache_read_input_tokens == 1000
    assert read.cost_usd == pytest.approx(0.75)
    assert set(rows) == {("read", "claude-opus-5-5"), ("verify", "claude-haiku-5-5")}
    assert sum(r.cost_usd for r in store.usage()) == pytest.approx(9.75)


def test_response_cache(store: Store) -> None:
    assert store.cache_get("h") is None
    store.cache_put("h", {"id": "msg_1", "content": []})
    assert store.cache_get("h") == {"id": "msg_1", "content": []}


def test_runs_and_artifacts(store: Store) -> None:
    store.start_run("r1", {"mode": "online"})
    store.finish_run("r1", {"papers": 3})
    store.start_run("r2", {"mode": "batch"})
    runs = store.runs()
    assert [r["run_id"] for r in runs] == ["r1", "r2"]
    assert runs[0]["stats"] == {"papers": 3}
    assert runs[1]["finished_at"] is None
    with pytest.raises(KeyError):
        store.finish_run("nope", {})

    topic = Topic(id=1, label="Caches", description="d", keywords=["k"], paper_ids=["p"])
    store.put_artifact("topic:1", topic)
    store.put_artifact("k", [1, 2])
    assert Topic.model_validate(store.get_artifact("topic:1")) == topic
    assert store.get_artifact("k") == [1, 2]
    assert store.get_artifact("missing") is None


def test_file_store_uses_wal_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "precis.db"
    with Store(path) as s:
        s.upsert_paper(_paper("p", date(2024, 1, 1)))
    with Store(path) as s:
        assert s.count_papers() == 1
        assert s.schema_version == len(store_module.MIGRATIONS)
    (mode,) = sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()
    assert mode == "wal"


def test_older_schema_is_migrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "precis.db"
    with Store(path) as s:
        s.put_artifact("k", 1)

    def add_column(db: sqlite3.Connection) -> None:
        db.execute("ALTER TABLE artifacts ADD COLUMN note TEXT")

    current = len(store_module.MIGRATIONS)
    monkeypatch.setattr(store_module, "MIGRATIONS", (*store_module.MIGRATIONS, add_column))
    with Store(path) as s:
        assert s.schema_version == current + 1
        assert s.get_artifact("k") == 1
    columns = [r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(artifacts)")]
    assert "note" in columns


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "precis.db"
    Store(path).close()
    db = sqlite3.connect(path)
    db.execute("UPDATE schema_version SET version = 99")
    db.commit()
    db.close()
    with pytest.raises(SchemaError):
        Store(path)


def test_reset_running_can_be_limited_to_some_stages(store: Store) -> None:
    store.set_stage(_stage("p1", Stage.FETCH, StageStatus.RUNNING))
    store.set_stage(_stage("p1", Stage.READ, StageStatus.RUNNING))
    store.set_stage(_stage("p2", Stage.VERIFY, StageStatus.RUNNING))

    assert store.reset_running([Stage.READ, Stage.VERIFY, Stage.REVISE]) == 2
    assert store.papers_with_status(Stage.FETCH, StageStatus.RUNNING) == ["p1"]
    assert store.reset_running([]) == 0
    assert store.reset_running() == 1
    assert store.papers_with_status(Stage.FETCH, StageStatus.PENDING) == ["p1"]


def test_batches_are_recorded_until_collected(store: Store) -> None:
    items = [
        BatchItem(custom_id="req-1", request_hash="h1", stage="read", paper_id="p1"),
        BatchItem(custom_id="req-2", request_hash="h2", stage="verify", paper_id=None),
    ]
    store.record_batch("msgbatch_1", items)
    store.record_batch("msgbatch_2", items[:1])
    assert store.open_batches() == [("msgbatch_1", items), ("msgbatch_2", items[:1])]

    store.close_batch("msgbatch_1")
    assert store.open_batches() == [("msgbatch_2", items[:1])]


def test_version_1_store_gains_the_batches_table(tmp_path: Path) -> None:
    path = tmp_path / "precis.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    db.execute("INSERT INTO schema_version VALUES (0)")
    store_module.MIGRATIONS[0](db)
    db.execute("UPDATE schema_version SET version = 1")
    db.commit()
    db.close()

    with Store(path) as s:
        assert s.schema_version == len(store_module.MIGRATIONS) >= 2
        assert s.open_batches() == []
