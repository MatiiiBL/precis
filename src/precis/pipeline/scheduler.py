"""``Pipeline``: runs thousands of papers through read, verify and revise.

At most ``pipeline.max_papers_in_flight`` papers are active at once; a
paper runs its stages in order, and each stage's result is stored the
moment it exists, so a crash or an interrupt loses only the requests in
flight. The gateway regulates the request rate underneath, so this layer
only bounds how much work (and memory) is open at a time.

Each stage is skipped when a ``done`` record with the same config hash
exists. A stage that fails is recorded with its error and stops that
paper's later stages; a refusal is recorded as ``refused``. When the budget
is reached, or on SIGINT, no new papers are admitted: on SIGINT the
in-flight work is cancelled, and in both cases unfinished stages are left
``pending`` for the next run.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from precis.agents import AgentRefused, run_critic, run_reviser, run_summarize
from precis.agents.tools import PassageIndex, paper_index
from precis.agents.trace import AgentResult
from precis.config import Config
from precis.llm import BudgetExceeded, Gateway, RefusalError
from precis.models import (
    Document,
    PaperMeta,
    PaperSummary,
    RevisionResult,
    Stage,
    StageRecord,
    StageStatus,
    Verification,
)
from precis.pipeline import hashing
from precis.pipeline.stats import PIPELINE_STAGES, RunStats
from precis.store import Store

log = logging.getLogger(__name__)

_DOWNSTREAM = {Stage.READ: (Stage.VERIFY, Stage.REVISE), Stage.VERIFY: (Stage.REVISE,)}


class _Blocked(Exception):
    """A stage cannot run because an earlier stage has no usable result."""


class _StopPaper(Exception):
    """A stage did not finish; the paper's later stages must wait."""


@dataclass
class _Paper:
    meta: PaperMeta
    doc: Document
    summary: PaperSummary | None = None
    verification: Verification | None = None
    revision: RevisionResult | None = None
    _index: PassageIndex | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return self.meta.id

    def index(self) -> PassageIndex:
        """The paper's search index, shared by its stages.

        Built on first use, so papers whose stages are all cached never build one.
        """
        if self._index is None:
            self._index = paper_index(self.doc)
        return self._index


class Pipeline:
    """Runs the per-paper stages over many papers, resumably."""

    def __init__(self, config: Config, store: Store, gateway: Gateway) -> None:
        self.config = config
        self.store = store
        self.gateway = gateway
        self.stats = RunStats(papers=0, stages=PIPELINE_STAGES)
        self._stopping = False
        self._workers: list[asyncio.Task[None]] = []

    @property
    def run_id(self) -> str:
        return self.gateway.run_id

    def interrupt(self) -> None:
        """Stop admitting papers and cancel in-flight work (the SIGINT handler)."""
        if not self.stats.interrupted:
            log.warning("interrupted: cancelling in-flight papers")
        self.stats.interrupted = True
        self._stopping = True
        for worker in self._workers:
            worker.cancel()

    async def run(
        self, paper_ids: Sequence[str], stages: Iterable[Stage] = PIPELINE_STAGES
    ) -> dict[str, Any]:
        """Run ``stages`` (in pipeline order) for ``paper_ids``; returns the run's statistics.

        The statistics are also stored with the run record.
        """
        wanted = tuple(s for s in PIPELINE_STAGES if s in set(stages))
        if not wanted:
            raise ValueError("no stages to run; choose from read, verify and revise")
        self.stats = RunStats(papers=len(paper_ids), stages=wanted)
        self._stopping = False
        # Only this command's stages: a concurrent `precis fetch` owns the fetch rows.
        repaired = self.store.reset_running(stages=PIPELINE_STAGES)
        if repaired:
            log.warning("reset %d stages left running by an earlier crash", repaired)
        self.store.start_run(
            self.run_id,
            {
                "stages": [s.value for s in wanted],
                "papers": len(paper_ids),
                "mode": self.gateway.mode,
                "config": self.config.model_dump(mode="json"),
            },
        )
        queue = deque(paper_ids)
        try:
            with self._sigint_handler():
                self._workers = [
                    asyncio.create_task(self._worker(queue, wanted))
                    for _ in range(min(self.config.pipeline.max_papers_in_flight, len(queue)))
                ]
                outcomes = await asyncio.gather(*self._workers, return_exceptions=True)
                self._workers = []
        finally:
            summary = self.stats.summary(_gateway_summary(self.gateway))
            self.store.finish_run(self.run_id, summary)
        # Stage errors are recorded per paper; anything else is a bug to surface.
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(
                outcome, asyncio.CancelledError
            ):
                raise outcome
        return summary

    @contextlib.contextmanager
    def _sigint_handler(self) -> Iterator[None]:
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self.interrupt)
        except (NotImplementedError, RuntimeError, ValueError):
            # No signal support here (not the main thread, or not Unix).
            yield
            return
        try:
            yield
        finally:
            loop.remove_signal_handler(signal.SIGINT)

    async def _worker(self, queue: deque[str], stages: tuple[Stage, ...]) -> None:
        while queue and not self._stopping:
            await self._paper(queue.popleft(), stages)

    # ------------------------------------------------------------------
    # one paper
    # ------------------------------------------------------------------

    async def _paper(self, paper_id: str, stages: tuple[Stage, ...]) -> None:
        meta = self.store.get_paper(paper_id)
        doc = self.store.get_document(paper_id) if meta is not None else None
        if meta is None or doc is None:
            reason = "not in the store" if meta is None else "no document; run precis fetch"
            log.info("%s: %s", paper_id, reason)
            self.stats.blocked[reason] += 1
            return
        paper = _Paper(meta, doc)
        # Stages before the last requested one supply its inputs; later ones are not touched.
        needed = PIPELINE_STAGES[: max(PIPELINE_STAGES.index(s) for s in stages) + 1]
        try:
            for stage in needed:
                await self._stage(paper, stage, run=stage in stages)
        except _Blocked as blocked:
            log.info("%s: %s", paper_id, blocked)
            self.stats.blocked[str(blocked)] += 1
        except _StopPaper:
            pass
        else:
            self.stats.papers_finished += 1
        if paper.verification is not None:
            self.stats.faithfulness.add(paper.verification, paper.revision)

    async def _stage(self, paper: _Paper, stage: Stage, *, run: bool) -> None:
        """Bring one stage up to date, or load its stored result when ``run`` is off.

        Raises ``_Blocked`` when an earlier stage's result is missing and
        ``_StopPaper`` when this stage did not finish.
        """
        record = self.store.get_stage(paper.id, stage)
        if not run:
            # A stage that was not asked for only supplies its stored result,
            # whatever settings produced it.
            if record is None or record.status is not StageStatus.DONE:
                raise _Blocked(f"{stage.value} not done; include it in --stages")
            self._load(paper, stage, record)
            return
        config_hash = self._config_hash(paper, stage)
        tally = self.stats.tally[stage]
        if (
            record is not None
            and record.config_hash == config_hash
            and record.status in (StageStatus.DONE, StageStatus.SKIPPED)
        ):
            self._load(paper, stage, record)
            tally.cached += 1
            return
        if stage is Stage.REVISE and not (
            self.config.agents.revise and paper.verification and paper.verification.needs_revision()
        ):
            self._record(paper.id, stage, StageStatus.SKIPPED, config_hash, attempts=0)
            tally.skipped += 1
            return
        await self._execute(paper, stage, config_hash, record)

    def _config_hash(self, paper: _Paper, stage: Stage) -> str:
        agents = self.config.agents
        if stage is Stage.READ:
            return hashing.read_hash(agents, paper.meta, paper.doc)
        if paper.summary is None:
            raise _Blocked("read not done; include it in --stages")
        if stage is Stage.VERIFY:
            return hashing.verify_hash(agents, paper.meta, paper.doc, paper.summary)
        if paper.verification is None:
            raise _Blocked("verify not done; include it in --stages")
        return hashing.revise_hash(agents, paper.meta, paper.doc, paper.summary, paper.verification)

    def _load(self, paper: _Paper, stage: Stage, record: StageRecord) -> None:
        if record.output is None:
            return
        if stage is Stage.READ:
            paper.summary = PaperSummary.model_validate(record.output)
        elif stage is Stage.VERIFY:
            paper.verification = Verification.model_validate(record.output)
        else:
            paper.revision = RevisionResult.model_validate(record.output)

    async def _execute(
        self, paper: _Paper, stage: Stage, config_hash: str, previous: StageRecord | None
    ) -> None:
        attempts = _attempt(previous, config_hash)
        # Requests are deterministic, so a retry with the plain cache would
        # replay the replies that made the stage fail. Each retry gets its own
        # cache scope; a crash inside it still resumes from that scope.
        salt = f"attempt-{attempts}" if attempts > 1 else ""
        tally = self.stats.tally[stage]
        self._record(paper.id, stage, StageStatus.RUNNING, config_hash, attempts)
        tally.running += 1
        try:
            with self.gateway.cache_scope(salt):
                output = await self._compute(paper, stage)
        except asyncio.CancelledError:
            self._record(paper.id, stage, StageStatus.PENDING, config_hash, attempts, "interrupted")
            tally.interrupted += 1
            raise
        except BudgetExceeded as e:
            self._stopping = True
            self.stats.budget_reached = True
            self._record(paper.id, stage, StageStatus.PENDING, config_hash, attempts, str(e))
            tally.interrupted += 1
            raise _StopPaper from e
        except (AgentRefused, RefusalError) as e:
            self._record(paper.id, stage, StageStatus.REFUSED, config_hash, attempts, str(e))
            tally.refused += 1
            raise _StopPaper from e
        except Exception as e:
            log.warning(
                "%s %s failed: %s",
                paper.id,
                stage.value,
                e,
                exc_info=log.isEnabledFor(logging.DEBUG),
            )
            error = f"{type(e).__name__}: {e}"
            self._record(paper.id, stage, StageStatus.FAILED, config_hash, attempts, error)
            tally.failed += 1
            raise _StopPaper from e
        finally:
            tally.running -= 1
        self._record(paper.id, stage, StageStatus.DONE, config_hash, attempts, output=output)
        tally.done += 1
        if previous is None or previous.output != output.model_dump(mode="json"):
            self._invalidate_downstream(paper.id, stage)

    async def _compute(self, paper: _Paper, stage: Stage) -> BaseModel:
        gw, agents = self.gateway, self.config.agents
        tally = self.stats.tally[stage]
        if stage is Stage.READ:
            read = await run_summarize(gw, paper.meta, paper.doc, agents, index=paper.index())
            tally.traces.append(read.trace)
            paper.summary = read.value
            return read.value
        assert paper.summary is not None
        if stage is Stage.VERIFY:
            verified = await run_critic(
                gw, paper.meta, paper.doc, paper.summary, agents, index=paper.index()
            )
            tally.traces.append(verified.trace)
            paper.verification = verified.value
            return verified.value
        assert paper.verification is not None
        revised = await run_reviser(
            gw,
            paper.meta,
            paper.doc,
            paper.summary,
            paper.verification,
            agents,
            index=paper.index(),
        )
        checked: AgentResult[Verification] = await run_critic(
            gw, paper.meta, paper.doc, revised.value, agents, stage="revise", index=paper.index()
        )
        tally.traces += [revised.trace, checked.trace]
        paper.revision = RevisionResult(summary=revised.value, verification=checked.value)
        return paper.revision

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------

    def _record(
        self,
        paper_id: str,
        stage: Stage,
        status: StageStatus,
        config_hash: str,
        attempts: int,
        error: str | None = None,
        *,
        output: BaseModel | None = None,
    ) -> None:
        self.store.set_stage(
            StageRecord(
                paper_id=paper_id,
                stage=stage,
                status=status,
                attempts=attempts,
                config_hash=config_hash,
                run_id=self.run_id,
                error=error,
                updated_at=datetime.now(UTC),
                output=output.model_dump(mode="json") if output is not None else None,
            )
        )

    def _invalidate_downstream(self, paper_id: str, stage: Stage) -> None:
        """A new result makes later stages' stored results stale: mark them pending."""
        for later in _DOWNSTREAM.get(stage, ()):
            record = self.store.get_stage(paper_id, later)
            if record is not None and record.status is not StageStatus.PENDING:
                self.store.set_stage(
                    record.model_copy(
                        update={
                            "status": StageStatus.PENDING,
                            "output": None,
                            "error": f"{stage.value} changed",
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )


def _attempt(previous: StageRecord | None, config_hash: str) -> int:
    """Which attempt at the stage's current inputs this execution is.

    New inputs start at 1. A stage that failed or was refused is retried as
    the next attempt; one left pending by an interrupt, the budget or a crash
    resumes the attempt it was in.
    """
    if previous is None or previous.config_hash != config_hash:
        return 1
    if previous.status in (StageStatus.FAILED, StageStatus.REFUSED):
        return previous.attempts + 1
    return max(previous.attempts, 1)


def _gateway_summary(gateway: Gateway) -> dict[str, Any]:
    stats = gateway.stats()
    return {
        "calls": stats.calls,
        "cached_calls": stats.cached_calls,
        "retries": stats.retries,
        "rate_limited": stats.rate_limited,
        "overloaded": stats.overloaded,
        "errors": stats.errors,
        "refusals": stats.refusals,
        "input_tokens": stats.input_tokens,
        "output_tokens": stats.output_tokens,
        "cache_creation_input_tokens": stats.cache_creation_input_tokens,
        "cache_read_input_tokens": stats.cache_read_input_tokens,
        "cost_usd": round(stats.cost_usd, 6),
    }
