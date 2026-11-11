"""The fetch stage: full text for papers in the store, recorded per paper as it finishes.

A paper is skipped when its ``fetch`` stage is ``done`` with the current
``config_hash`` (parser version and preferred format) and its document is in
the store, so re-running a command resumes where it stopped and a parser
change re-parses everything from the raw files on disk without downloading
again. Each paper goes ``running`` -> ``done`` or ``failed`` (with the
reason); a paper interrupted by Ctrl-C goes back to ``pending``.

A few workers run at once. Downloads are still one every
``fetch.min_interval_s`` (they share one limiter); the workers exist so that
parsing one paper overlaps with downloading the next.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from precis.config import FetchConfig
from precis.models import Document, PaperMeta, Stage, StageRecord, StageStatus
from precis.sources.fulltext import FullTextFetcher
from precis.sources.http import FetchError
from precis.sources.local import LocalPaper, load_pdf, local_id
from precis.store import Store
from precis.text.document import ParseError
from precis.util import canonical_json, sha256_hex

logger = logging.getLogger(__name__)

# Part of every fetch stage's config hash: bump it when a parser change
# should re-parse papers that were already fetched.
PARSER_VERSION = 1
WORKERS = 4

OutcomeStatus = Literal["html", "pdf", "skipped", "failed"]


def fetch_config_hash(config: FetchConfig) -> str:
    """What determines a fetched document: the parsers and the preferred format."""
    return sha256_hex(canonical_json({"parser": PARSER_VERSION, "prefer": config.prefer}))[:16]


@dataclass(frozen=True)
class Outcome:
    """What happened to one paper; ``detail`` is the error or the reason for a PDF fallback."""

    paper_id: str
    status: OutcomeStatus
    detail: str | None = None
    n_tokens: int = 0


@dataclass
class FetchStats:
    counts: Counter[OutcomeStatus] = field(default_factory=Counter)
    failures: dict[str, str] = field(default_factory=dict)
    fallbacks: dict[str, str] = field(default_factory=dict)
    tokens: list[int] = field(default_factory=list)
    found: int = 0  # papers the search, id lookup or directory listing returned
    requests: int = 0  # HTTP requests made, retries included

    def add(self, outcome: Outcome) -> None:
        self.counts[outcome.status] += 1
        if outcome.status == "failed":
            self.failures[outcome.paper_id] = outcome.detail or ""
        elif outcome.status in ("html", "pdf"):
            self.tokens.append(outcome.n_tokens)
            if outcome.detail:
                self.fallbacks[outcome.paper_id] = outcome.detail

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "failures": self.failures,
            "fallbacks": self.fallbacks,
            "tokens": sorted(self.tokens),
            "found": self.found,
            "requests": self.requests,
        }


class FetchRunner:
    """Runs the fetch stage for arXiv papers or local PDFs against a ``Store``."""

    def __init__(
        self,
        store: Store,
        *,
        config_hash: str,
        run_id: str,
        workers: int = WORKERS,
        on_outcome: Callable[[Outcome], None] | None = None,
    ) -> None:
        self.store = store
        self.config_hash = config_hash
        self.run_id = run_id
        self.workers = workers
        self.stats = FetchStats()
        self._on_outcome = on_outcome

    def is_done(self, paper_id: str, version: int | None = None) -> bool:
        """Whether the paper's text is stored with the current settings (and arXiv version)."""
        record = self.store.get_stage(paper_id, Stage.FETCH)
        if record is None or record.status is not StageStatus.DONE:
            return False
        if record.config_hash != self.config_hash:
            return False
        if version is not None and (record.output or {}).get("version") != version:
            return False  # a newer version was posted since
        return self.store.get_document(paper_id) is not None

    def save_metadata(self, meta: PaperMeta) -> None:
        """Store search results without losing a license learned from the full text."""
        existing = self.store.get_paper(meta.id)
        if existing is not None and existing.license and not meta.license:
            meta = meta.model_copy(update={"license": existing.license})
        self.store.upsert_paper(meta)

    def mark_pending(self, paper_id: str) -> None:
        """Record that a paper's text is still to fetch (``--metadata-only``)."""
        if self.store.get_stage(paper_id, Stage.FETCH) is None:
            self._set(paper_id, StageStatus.PENDING, attempts=0)

    async def fetch_arxiv(self, fetcher: FullTextFetcher, papers: Sequence[PaperMeta]) -> None:
        """Fetch full text for ``papers`` (already in the store) that are not done."""
        todo = []
        for meta in papers:
            if self.is_done(meta.id, meta.version):
                self._report(Outcome(meta.id, "skipped"))
            else:
                todo.append(meta)
        await self._pool(todo, lambda meta: self._fetch_one(fetcher, meta))

    async def fetch_local(self, paths: Sequence[Path]) -> None:
        """Parse local PDFs into the store, skipping files already done."""
        await self._pool(list(paths), self._load_one)

    async def _pool(self, items: Sequence[Any], work: Callable[[Any], Awaitable[None]]) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)

        async def worker() -> None:
            while not queue.empty():
                await work(queue.get_nowait())

        async with asyncio.TaskGroup() as group:
            for _ in range(min(self.workers, len(items))):
                group.create_task(worker())

    async def _fetch_one(self, fetcher: FullTextFetcher, meta: PaperMeta) -> None:
        attempts = self._attempts(meta.id) + 1
        self._set(meta.id, StageStatus.RUNNING, attempts)
        try:
            full = await fetcher.fetch(meta)
        except asyncio.CancelledError:
            self._set(meta.id, StageStatus.PENDING, attempts - 1)
            raise
        except Exception as exc:
            self._fail(meta.id, attempts, exc)
            return
        if full.license and full.license != meta.license:
            self.store.upsert_paper(meta.model_copy(update={"license": full.license}))
        self._done(full.document, attempts, full.fallback_reason, meta.version)

    async def _load_one(self, path: Path) -> None:
        try:
            paper_id = local_id(await asyncio.to_thread(path.read_bytes))
        except OSError as exc:
            self._report(Outcome(str(path), "failed", f"{type(exc).__name__}: {exc}"))
            return
        if self.is_done(paper_id):
            self._report(Outcome(paper_id, "skipped"))
            return
        attempts = self._attempts(paper_id) + 1
        self._set(paper_id, StageStatus.RUNNING, attempts)
        try:
            paper: LocalPaper = await asyncio.to_thread(load_pdf, path)
        except asyncio.CancelledError:
            self._set(paper_id, StageStatus.PENDING, attempts - 1)
            raise
        except Exception as exc:
            self._fail(paper_id, attempts, exc, source=path)
            return
        self.store.upsert_paper(paper.meta)
        self._done(paper.document, attempts, None, None)

    def _done(
        self, doc: Document, attempts: int, fallback_reason: str | None, version: int | None
    ) -> None:
        # The document goes in first: a crash between the two writes leaves the
        # stage "running", which reset_running turns back into "pending".
        self.store.put_document(doc)
        output = {
            "version": version,
            "source_format": doc.source_format,
            "n_tokens": doc.n_tokens,
            "n_sections": len(doc.sections),
            "n_captions": len(doc.captions),
            "fallback_reason": fallback_reason,
        }
        self._set(doc.paper_id, StageStatus.DONE, attempts, output=output)
        status: OutcomeStatus = "html" if doc.source_format == "html" else "pdf"
        self._report(Outcome(doc.paper_id, status, fallback_reason, doc.n_tokens))

    def _fail(
        self, paper_id: str, attempts: int, exc: Exception, source: Path | None = None
    ) -> None:
        if isinstance(exc, FetchError | ParseError | OSError):
            reason = str(exc)
        else:
            # A parser bug on one odd paper must not stop a 2,000-paper run;
            # it is recorded with its type and logged with the traceback.
            logger.exception("unexpected error fetching %s", source or paper_id)
            reason = f"{type(exc).__name__}: {exc}"
        self._set(paper_id, StageStatus.FAILED, attempts, error=reason)
        self._report(Outcome(paper_id, "failed", reason))

    def _attempts(self, paper_id: str) -> int:
        record = self.store.get_stage(paper_id, Stage.FETCH)
        return record.attempts if record else 0

    def _set(
        self,
        paper_id: str,
        status: StageStatus,
        attempts: int,
        *,
        error: str | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        self.store.set_stage(
            StageRecord(
                paper_id=paper_id,
                stage=Stage.FETCH,
                status=status,
                attempts=attempts,
                config_hash=self.config_hash,
                run_id=self.run_id,
                error=error,
                updated_at=datetime.now(UTC),
                output=output,
            )
        )

    def _report(self, outcome: Outcome) -> None:
        self.stats.add(outcome)
        if self._on_outcome is not None:
            self._on_outcome(outcome)
