"""Batch mode: serving ``Gateway.create`` calls through the Message Batches API.

Callers await ``BatchQueue.submit`` exactly as they would an online
request. Requests collect in a queue that is sent as one batch when it
reaches ``limits.batch_max_requests`` or ``limits.batch_max_wait_s`` after
its first request, whichever comes first; each batch is then polled with
exponential backoff, its results are streamed, and each result resolves the
caller waiting on its ``custom_id``. A multi-turn agent therefore runs
unchanged at the batch price: each turn is one request in some batch.

Items that come back ``errored`` with a server-side error, ``expired`` or
``canceled``, or that are missing from the results, go back into the queue
and are sent again in a later batch (not online: that would forfeit the
discount, and the caller has already accepted batch latency). Items
``errored`` with ``invalid_request_error`` would fail again, so they fail
the caller immediately. Every item is sent at most ``limits.max_attempts``
times.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import anthropic
from anthropic.types import Message
from anthropic.types.messages import MessageBatch, MessageBatchResult

from precis.config import LimitsConfig
from precis.llm.client import AsyncAnthropicLike
from precis.llm.clock import Clock
from precis.llm.errors import GatewayError

log = logging.getLogger(__name__)

_TRANSIENT = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


@dataclass(frozen=True)
class BatchResult:
    message: Message
    attempts: int
    latency_s: float
    batch_id: str


@dataclass
class _Item:
    custom_id: str
    params: dict[str, Any]
    future: asyncio.Future[BatchResult]
    queued_at: float
    attempts: int = 0


@dataclass
class _Counts:
    batches_sent: int = 0
    resubmitted: int = 0
    in_batches: int = 0
    running: set[asyncio.Task[None]] = field(default_factory=set)


class BatchQueue:
    """Collects requests into batches and routes each result back to its caller."""

    def __init__(
        self,
        client: AsyncAnthropicLike,
        clock: Clock,
        limits: LimitsConfig,
        rng: random.Random,
    ) -> None:
        self._client = client
        self._clock = clock
        self._limits = limits
        self._rng = rng
        self._queue: list[_Item] = []
        self._timer: asyncio.Task[None] | None = None
        self._counts = _Counts()
        self._next_id = 0

    @property
    def queued(self) -> int:
        """Requests waiting to be put in a batch."""
        return len(self._queue)

    @property
    def in_batches(self) -> int:
        """Requests in batches that have been sent and not yet ended."""
        return self._counts.in_batches

    @property
    def batches_pending(self) -> int:
        return len(self._counts.running)

    @property
    def batches_sent(self) -> int:
        return self._counts.batches_sent

    @property
    def resubmitted(self) -> int:
        return self._counts.resubmitted

    async def submit(self, params: dict[str, Any]) -> BatchResult:
        """Queue one request and wait for its result."""
        self._next_id += 1
        item = _Item(
            custom_id=f"req-{self._next_id:08d}",
            params=params,
            future=asyncio.get_running_loop().create_future(),
            queued_at=self._clock.now(),
        )
        self._enqueue(item)
        return await item.future

    async def aclose(self) -> None:
        """Send whatever is queued now and wait until every batch has finished."""
        while self._queue or self._counts.running:
            if self._queue:
                self._flush()
            await asyncio.gather(*self._counts.running, return_exceptions=True)

    def _enqueue(self, item: _Item) -> None:
        self._queue.append(item)
        if len(self._queue) >= self._limits.batch_max_requests:
            self._flush()
        elif self._timer is None:
            self._timer = asyncio.create_task(self._flush_later())

    async def _flush_later(self) -> None:
        await self._clock.sleep(self._limits.batch_max_wait_s)
        self._timer = None
        self._flush()

    def _flush(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        while self._queue:
            items = self._queue[: self._limits.batch_max_requests]
            del self._queue[: len(items)]
            items = [i for i in items if not i.future.done()]
            if not items:
                continue
            task = asyncio.create_task(self._run(items))
            self._counts.running.add(task)
            task.add_done_callback(self._counts.running.discard)

    async def _run(self, items: list[_Item]) -> None:
        pending = {item.custom_id: item for item in items}
        for item in items:
            item.attempts += 1
        self._counts.in_batches += len(items)
        batches = self._client.messages.batches
        requests = [{"custom_id": i.custom_id, "params": i.params} for i in items]
        try:
            batch = await self._with_retries(partial(batches.create, requests=requests))
            self._counts.batches_sent += 1
            log.info("sent batch %s with %d requests", batch.id, len(items))
            await self._wait_until_ended(batch)
            results = await self._with_retries(partial(batches.results, batch.id))
            retry: list[tuple[_Item, str]] = []
            async for entry in results:
                if entry.custom_id in pending:
                    item = pending.pop(entry.custom_id)
                    if not self._settle(item, entry.result, batch.id):
                        retry.append((item, f"batch result {entry.result.type}"))
            # Requeue together, once the whole result set is read, so the
            # retries share one batch instead of trickling into several.
            retry += [(item, "missing from the batch results") for item in pending.values()]
            pending.clear()
            for item, reason in retry:
                self._retry_or_fail(item, reason)
        except GatewayError as e:
            for item in pending.values():
                self._retry_or_fail(item, str(e))
        except BaseException as e:
            for item in pending.values():
                if item.future.done():
                    continue
                if isinstance(e, asyncio.CancelledError):
                    item.future.cancel()
                else:
                    item.future.set_exception(e)
            raise
        finally:
            self._counts.in_batches -= len(items)

    def _settle(self, item: _Item, result: MessageBatchResult, batch_id: str) -> bool:
        """Resolve the caller from its result; ``False`` if the item should be sent again."""
        if result.type == "succeeded":
            if not item.future.done():
                item.future.set_result(
                    BatchResult(
                        result.message,
                        attempts=item.attempts,
                        latency_s=self._clock.now() - item.queued_at,
                        batch_id=batch_id,
                    )
                )
            return True
        if result.type == "errored" and result.error.error.type == "invalid_request_error":
            if not item.future.done():
                item.future.set_exception(
                    GatewayError(
                        f"batch request rejected: {result.error.error.message}",
                        attempts=item.attempts,
                    )
                )
            return True
        return False

    def _retry_or_fail(self, item: _Item, reason: str) -> None:
        if item.future.done():
            return
        if item.attempts >= self._limits.max_attempts:
            item.future.set_exception(
                GatewayError(
                    f"batch request failed after {item.attempts} attempts: {reason}",
                    attempts=item.attempts,
                )
            )
            return
        self._counts.resubmitted += 1
        log.debug("resubmitting %s (%s)", item.custom_id, reason)
        self._enqueue(item)

    async def _wait_until_ended(self, batch: MessageBatch) -> None:
        delay = self._limits.batch_poll_s
        while batch.processing_status != "ended":
            await self._clock.sleep(delay)
            delay = min(delay * 2, self._limits.batch_poll_max_s)
            batch = await self._with_retries(
                partial(self._client.messages.batches.retrieve, batch.id)
            )

    async def _with_retries[T](self, call: Callable[[], Awaitable[T]]) -> T:
        """Retry a Batches API call on 429, 5xx and connection errors with backoff."""
        limits = self._limits
        for attempt in range(1, limits.max_attempts + 1):
            try:
                return await call()
            except _TRANSIENT as e:
                if attempt == limits.max_attempts:
                    raise GatewayError(f"Batches API call failed: {e}", attempts=attempt) from e
                ceiling = min(limits.backoff_max_s, limits.backoff_base_s * 2 ** (attempt - 1))
                await self._clock.sleep(self._rng.uniform(0, ceiling))
            except anthropic.APIStatusError as e:
                raise GatewayError(f"Batches API call rejected: {e}", attempts=attempt) from e
        raise AssertionError("unreachable")
