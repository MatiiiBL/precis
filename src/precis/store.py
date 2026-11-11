"""The SQLite store that holds everything a run produces.

One file in WAL mode keeps papers, their text, per-stage results, every LLM
call, the response cache, runs and corpus artifacts. Records go in as the
JSON of their pydantic model and come back validated against that model, so
a field that no longer fits the schema fails loudly instead of leaking a
half-parsed object into the pipeline. The columns next to the JSON are only
what the queries below filter, sort or aggregate on.

The pipeline runs on one asyncio event loop and calls the store from that
thread. Every write is a short statement committed immediately (autocommit
mode), so a crash loses at most the requests that were in flight; that is
cheap enough to do inline on the event loop.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from pydantic import BaseModel

from precis.models import (
    Document,
    LLMCall,
    PaperMeta,
    Stage,
    StageRecord,
    StageStatus,
    UsageRow,
)

type Migration = Callable[[sqlite3.Connection], None]


def _run_script(db: sqlite3.Connection, script: str) -> None:
    # ``executescript`` would commit the migration transaction half-way, so
    # run the statements one at a time inside it.
    for statement in script.split(";"):
        if statement.strip():
            db.execute(statement)


def _migrate_v1(db: sqlite3.Connection) -> None:
    _run_script(
        db,
        """
        CREATE TABLE papers (
            id        TEXT PRIMARY KEY,
            published TEXT NOT NULL,
            json      TEXT NOT NULL
        );
        CREATE INDEX papers_by_date ON papers (published DESC, id);

        CREATE TABLE paper_categories (
            paper_id TEXT NOT NULL,
            category TEXT NOT NULL,
            PRIMARY KEY (category, paper_id)
        ) WITHOUT ROWID;
        CREATE INDEX paper_categories_by_paper ON paper_categories (paper_id);

        CREATE TABLE documents (
            paper_id TEXT PRIMARY KEY,
            json     TEXT NOT NULL
        );

        CREATE TABLE stages (
            paper_id   TEXT NOT NULL,
            stage      TEXT NOT NULL,
            status     TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            json       TEXT NOT NULL,
            PRIMARY KEY (paper_id, stage)
        ) WITHOUT ROWID;
        CREATE INDEX stages_by_status ON stages (stage, status, paper_id);

        CREATE TABLE llm_calls (
            call_id    TEXT PRIMARY KEY,
            run_id     TEXT NOT NULL,
            paper_id   TEXT,
            stage      TEXT NOT NULL,
            model      TEXT NOT NULL,
            status     TEXT NOT NULL,
            started_at TEXT NOT NULL,
            input_tokens                INTEGER NOT NULL,
            output_tokens               INTEGER NOT NULL,
            cache_creation_input_tokens INTEGER NOT NULL,
            cache_read_input_tokens     INTEGER NOT NULL,
            cost_usd   REAL NOT NULL,
            json       TEXT NOT NULL
        );
        CREATE INDEX llm_calls_by_run ON llm_calls (run_id, stage, model);
        CREATE INDEX llm_calls_by_paper ON llm_calls (paper_id, started_at);

        CREATE TABLE response_cache (
            request_hash TEXT PRIMARY KEY,
            created_at   TEXT NOT NULL,
            json         TEXT NOT NULL
        );

        CREATE TABLE runs (
            run_id      TEXT PRIMARY KEY,
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            config      TEXT NOT NULL,
            stats       TEXT
        );

        CREATE TABLE artifacts (
            key        TEXT PRIMARY KEY,
            updated_at TEXT NOT NULL,
            json       TEXT NOT NULL
        );
        """,
    )


# MIGRATIONS[i] upgrades a store from schema version i to i + 1. Append only:
# a released migration never changes, because stores in the wild have run it.
MIGRATIONS: Sequence[Migration] = (_migrate_v1,)


class SchemaError(RuntimeError):
    """The store was written by a newer version of précis than this one."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dump(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, ensure_ascii=False)


class Store:
    """Typed access to the run database; see docs/design.md section 4."""

    def __init__(self, path: Path | str) -> None:
        """Open (creating and migrating if needed) the store at ``path``.

        ``":memory:"`` gives a private in-memory store, which is what tests use.
        """
        in_memory = str(path) == ":memory:"
        if not in_memory:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Autocommit: each statement is its own transaction unless wrapped in
        # ``_transaction``. That is what keeps a crash from losing more than
        # the in-flight work.
        self._db = sqlite3.connect(str(path), isolation_level=None)
        if not in_memory:
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = NORMAL")
        self._db.execute("PRAGMA busy_timeout = 5000")
        self._migrate()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone()
        if row is None:
            return 0
        (version,) = self._db.execute("SELECT version FROM schema_version").fetchone()
        return int(version)

    def _migrate(self) -> None:
        with self._transaction():
            version = self.schema_version
            if version > len(MIGRATIONS):
                raise SchemaError(
                    f"store schema version {version} is newer than this code "
                    f"(knows up to {len(MIGRATIONS)}); upgrade précis"
                )
            if version == len(MIGRATIONS):
                return
            if version == 0:
                self._db.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
                self._db.execute("INSERT INTO schema_version VALUES (0)")
            for migration in MIGRATIONS[version:]:
                migration(self._db)
            self._db.execute("UPDATE schema_version SET version = ?", (len(MIGRATIONS),))

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # ------------------------------------------------------------------
    # papers and text
    # ------------------------------------------------------------------

    def upsert_paper(self, meta: PaperMeta) -> None:
        with self._transaction():
            self._db.execute(
                "INSERT INTO papers (id, published, json) VALUES (?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET published = excluded.published, "
                "json = excluded.json",
                (meta.id, meta.published.isoformat(), meta.model_dump_json()),
            )
            self._db.execute("DELETE FROM paper_categories WHERE paper_id = ?", (meta.id,))
            self._db.executemany(
                "INSERT OR IGNORE INTO paper_categories (paper_id, category) VALUES (?, ?)",
                [(meta.id, c) for c in {meta.primary_category, *meta.categories}],
            )

    def get_paper(self, paper_id: str) -> PaperMeta | None:
        row = self._db.execute("SELECT json FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return None if row is None else PaperMeta.model_validate_json(row[0])

    def list_papers(
        self, *, category: str | None = None, limit: int | None = None
    ) -> list[PaperMeta]:
        """Papers newest first, then by id; ``category`` matches any listed category."""
        sql = "SELECT p.json FROM papers p"
        args: list[Any] = []
        if category is not None:
            sql += " JOIN paper_categories c ON c.paper_id = p.id AND c.category = ?"
            args.append(category)
        sql += " ORDER BY p.published DESC, p.id"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        return [PaperMeta.model_validate_json(r[0]) for r in self._db.execute(sql, args)]

    def count_papers(self) -> int:
        (n,) = self._db.execute("SELECT COUNT(*) FROM papers").fetchone()
        return int(n)

    def put_document(self, doc: Document) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO documents (paper_id, json) VALUES (?, ?)",
            (doc.paper_id, doc.model_dump_json()),
        )

    def get_document(self, paper_id: str) -> Document | None:
        row = self._db.execute(
            "SELECT json FROM documents WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return None if row is None else Document.model_validate_json(row[0])

    # ------------------------------------------------------------------
    # per-paper stages
    # ------------------------------------------------------------------

    def set_stage(self, record: StageRecord) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO stages (paper_id, stage, status, updated_at, json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.paper_id,
                record.stage.value,
                record.status.value,
                record.updated_at.isoformat(),
                record.model_dump_json(),
            ),
        )

    def get_stage(self, paper_id: str, stage: Stage) -> StageRecord | None:
        row = self._db.execute(
            "SELECT json FROM stages WHERE paper_id = ? AND stage = ?", (paper_id, stage.value)
        ).fetchone()
        return None if row is None else StageRecord.model_validate_json(row[0])

    def stage_counts(self) -> dict[Stage, dict[StageStatus, int]]:
        """Count of papers per stage and status; every pair is present, zero if unused."""
        counts = {stage: dict.fromkeys(StageStatus, 0) for stage in Stage}
        rows = self._db.execute("SELECT stage, status, COUNT(*) FROM stages GROUP BY stage, status")
        for stage, status, n in rows:
            counts[Stage(stage)][StageStatus(status)] = int(n)
        return counts

    def papers_with_status(self, stage: Stage, status: StageStatus) -> list[str]:
        rows = self._db.execute(
            "SELECT paper_id FROM stages WHERE stage = ? AND status = ? ORDER BY paper_id",
            (stage.value, status.value),
        )
        return [r[0] for r in rows]

    def reset_running(self, stages: Iterable[Stage] | None = None) -> int:
        """Mark stages left ``running`` by a crashed process as ``pending``; returns how many.

        ``stages`` limits the reset to the stages the calling command owns
        (``None`` means every stage), so ``precis run`` does not reset the
        ``fetch`` rows of a ``precis fetch`` that is running right now.
        """
        sql = "UPDATE stages SET status = ?, json = json_set(json, '$.status', ?) WHERE status = ?"
        args: list[Any] = [
            StageStatus.PENDING.value,
            StageStatus.PENDING.value,
            StageStatus.RUNNING.value,
        ]
        if stages is not None:
            names = [Stage(s).value for s in stages]
            if not names:
                return 0
            sql += f" AND stage IN ({', '.join('?' * len(names))})"
            args.extend(names)
        return self._db.execute(sql, args).rowcount

    # ------------------------------------------------------------------
    # LLM calls and the response cache
    # ------------------------------------------------------------------

    def record_call(self, call: LLMCall) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO llm_calls (call_id, run_id, paper_id, stage, model, status, "
            "started_at, input_tokens, output_tokens, cache_creation_input_tokens, "
            "cache_read_input_tokens, cost_usd, json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                call.call_id,
                call.run_id,
                call.paper_id,
                call.stage,
                call.model,
                call.status,
                call.started_at.isoformat(),
                call.input_tokens,
                call.output_tokens,
                call.cache_creation_input_tokens,
                call.cache_read_input_tokens,
                call.cost_usd,
                call.model_dump_json(),
            ),
        )

    def calls(self, *, run_id: str | None = None, paper_id: str | None = None) -> list[LLMCall]:
        """Calls in the order they started, optionally filtered by run and paper."""
        sql = "SELECT json FROM llm_calls WHERE 1 = 1"
        args: list[Any] = []
        if run_id is not None:
            sql += " AND run_id = ?"
            args.append(run_id)
        if paper_id is not None:
            sql += " AND paper_id = ?"
            args.append(paper_id)
        sql += " ORDER BY started_at, call_id"
        return [LLMCall.model_validate_json(r[0]) for r in self._db.execute(sql, args)]

    def usage(self, *, run_id: str | None = None) -> list[UsageRow]:
        """Token and cost totals per (stage, model), summed by SQLite rather than in Python."""
        sql = (
            "SELECT stage, model, COUNT(*), SUM(status = 'cached'), SUM(input_tokens), "
            "SUM(output_tokens), SUM(cache_creation_input_tokens), "
            "SUM(cache_read_input_tokens), SUM(cost_usd) FROM llm_calls"
        )
        args: list[Any] = []
        if run_id is not None:
            sql += " WHERE run_id = ?"
            args.append(run_id)
        sql += " GROUP BY stage, model ORDER BY stage, model"
        return [
            UsageRow(
                stage=stage,
                model=model,
                calls=calls,
                cached_calls=cached,
                input_tokens=inp,
                output_tokens=out,
                cache_creation_input_tokens=cache_write,
                cache_read_input_tokens=cache_read,
                cost_usd=cost,
            )
            for stage, model, calls, cached, inp, out, cache_write, cache_read, cost in (
                self._db.execute(sql, args)
            )
        ]

    def cache_get(self, request_hash: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT json FROM response_cache WHERE request_hash = ?", (request_hash,)
        ).fetchone()
        if row is None:
            return None
        response: dict[str, Any] = json.loads(row[0])
        return response

    def cache_put(self, request_hash: str, response: dict[str, Any]) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO response_cache (request_hash, created_at, json) "
            "VALUES (?, ?, ?)",
            (request_hash, _now(), _dump(response)),
        )

    def cache_delete(self, request_hash: str) -> None:
        """Drop a cached response (one that can no longer be read back, say)."""
        self._db.execute("DELETE FROM response_cache WHERE request_hash = ?", (request_hash,))

    # ------------------------------------------------------------------
    # runs and corpus artifacts
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, config: dict[str, Any]) -> None:
        self._db.execute(
            "INSERT INTO runs (run_id, started_at, config) VALUES (?, ?, ?)",
            (run_id, _now(), _dump(config)),
        )

    def finish_run(self, run_id: str, stats: dict[str, Any]) -> None:
        cursor = self._db.execute(
            "UPDATE runs SET finished_at = ?, stats = ? WHERE run_id = ?",
            (_now(), _dump(stats), run_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"unknown run {run_id!r}")

    def runs(self) -> list[dict[str, Any]]:
        """Every run, oldest first, with its config and (once finished) its stats."""
        rows = self._db.execute(
            "SELECT run_id, started_at, finished_at, config, stats FROM runs "
            "ORDER BY started_at, run_id"
        )
        return [
            {
                "run_id": run_id,
                "started_at": datetime.fromisoformat(started),
                "finished_at": None if finished is None else datetime.fromisoformat(finished),
                "config": json.loads(config),
                "stats": None if stats is None else json.loads(stats),
            }
            for run_id, started, finished, config, stats in rows
        ]

    def put_artifact(self, key: str, value: Any) -> None:
        """Store a pydantic model (as its JSON) or any JSON-serializable value under ``key``."""
        self._db.execute(
            "INSERT OR REPLACE INTO artifacts (key, updated_at, json) VALUES (?, ?, ?)",
            (key, _now(), _dump(value)),
        )

    def get_artifact(self, key: str) -> Any | None:
        """The parsed JSON stored under ``key``; validate it with the model you stored."""
        row = self._db.execute("SELECT json FROM artifacts WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])
