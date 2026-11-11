"""Batch mode: serving ``Gateway.create`` calls through the Message Batches API.

Callers await ``BatchQueue.submit`` exactly as they would an online
request. Requests collect in a queue that is sent as one batch when it
reaches ``limits.batch_max_requests`` or ``limits.batch_max_bytes``, or
``limits.batch_max_wait_s`` after its first request; each batch is polled
with exponential backoff, its results are streamed, and each result is
handed to the gateway (``on_result``: accounting and the response cache)
and then to the caller waiting on its ``custom_id``. A multi-turn agent
therefore runs unchanged at the batch price: each turn is one request in
some batch.

Results are paid for once a batch has been created, so they are never
thrown away:

- every batch is recorded in the store when it is created and closed only
  once its results have been read, and every succeeded result is filed
  (recorded and cached) whether or not its caller is still waiting;
- a gateway that starts with batches left open by an earlier run collects
  their results, and a request whose result is in one of them waits for it
  instead of being sent again;
- polling and results-stream errors are retried with long backoff against
  the same batch, which keeps processing on the server regardless;
- ``aclose`` stops waiting at once and leaves unfinished batches recorded
  for the next run, saying so in the log.

Items that come back ``errored`` with a server-side error, ``expired`` or
``canceled``, or that are missing from the results of a batch that ended,
are sent again in a later batch (not online: the caller has accepted batch
latency, and an online retry would forfeit the discount). Items ``errored``
with ``invalid_request_error`` fail the caller at once. A batch the API
rejects with a 4xx (too large, say) is split in half and each half sent on
its own, down to the single request that is at fault.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import anthropic
from anthropic.types import Message
from anthropic.types.messages import MessageBatchResult

from precis.config import LimitsConfig
from precis.llm.base import Reply
from precis.llm.client import AsyncAnthropicLike
from precis.llm.clock import Clock
from precis.llm.errors import GatewayError, RefusalError
from precis.store import BatchItem, Store
from precis.util import canonical_json

log = logging.getLogger(__name__)

type OnResult = Callable[[BatchItem, Message, int, float, str], Reply]
"""Files one succeeded result: (item, message, attempts, latency_s, batch_id) -> the reply.

Filing must be idempotent: a batch collected again after a kill files its
results again.
"""

_TRANSIENT = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)

# Bytes a request adds to a batch besides its params (custom_id, framing).
_ITEM_OVERHEAD = 100


class _Resend(Exception):
    """An earlier run's item did not succeed; the waiting caller should send its own."""


class _Rejected(Exception):
    """The Batches API refused to create a batch (a 4xx)."""


@dataclass(eq=False)
class _Item:
    ref: BatchItem
    params: dict[str, Any]
    future: asyncio.Future[Reply]
    queued_at: float
    size: int
    attempts: int = 0


class BatchQueue:
    """Collects requests into batches, files every result and routes it to its caller."""

    def __init__(
        self,
        client: AsyncAnthropicLike,
        clock: Clock,
        limits: LimitsConfig,
        rng: random.Random,
        store: Store,
        on_result: OnResult,
        *,
        offline: bool = False,
    ) -> None:
        self._client = client
        self._offline = offline
        self._clock = clock
        self._limits = limits
        self._rng = rng
        self._store = store
        self._on_result = on_result
        self._queue: list[_Item] = []
        self._queued_bytes = 0
        self._timer: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._waiting_for: dict[str, asyncio.Future[Reply]] = {}
        self._open: set[str] = set()
        self._in_batches = 0
        self._batches_sent = 0
        self._resubmitted = 0
        self._next_id = 0
        self._collecting = False

    @property
    def queued(self) -> int:
        """Requests waiting to be put in a batch."""
        return len(self._queue)

    @property
    def in_batches(self) -> int:
        """Requests in batches that have been sent and not yet ended."""
        return self._in_batches

    @property
    def batches_pending(self) -> int:
        """Batches sent (or being collected) whose results have not been read."""
        return len(self._open)

    @property
    def batches_sent(self) -> int:
        return self._batches_sent

    @property
    def resubmitted(self) -> int:
        return self._resubmitted

    def start_collecting(self) -> None:
        """Collect the results of batches earlier runs left open (once; needs a running loop)."""
        if self._collecting:
            return
        self._collecting = True
        open_batches = self._store.open_batches(offline=self._offline)
        if open_batches:
            log.info("collecting %d batches left open by earlier runs", len(open_batches))
        for batch_id, refs in open_batches:
            items = [self._new_item(ref, {}) for ref in refs]
            for item in items:
                item.attempts = 1
                self._waiting_for.setdefault(item.ref.request_hash, item.future)
            self._open.add(batch_id)
            self._spawn(self._collect(batch_id, items, requeue=False))

    async def submit(
        self, params: dict[str, Any], request_hash: str, stage: str, paper_id: str | None
    ) -> Reply:
        """Queue one request (or join the batch item already serving it) and wait for it."""
        existing = self._waiting_for.get(request_hash)
        if existing is not None:
            try:
                return await asyncio.shield(existing)
            except _Resend:
                pass
        self._next_id += 1
        ref = BatchItem(f"req-{self._next_id:08d}", request_hash, stage, paper_id)
        item = self._new_item(ref, params)
        self._waiting_for[request_hash] = item.future
        self._enqueue(item)
        return await asyncio.shield(item.future)

    async def aclose(self) -> None:
        """Stop without waiting: unsent requests are dropped, sent batches stay recorded."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        for item in self._queue:
            _fail(item, GatewayError("gateway closed before the request was sent"))
        self._queue.clear()
        self._queued_bytes = 0
        left = sorted(self._open)
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for future in self._waiting_for.values():
            if not future.done():
                future.set_exception(GatewayError("gateway closed while its batch was running"))
        if left:
            log.warning(
                "%d batches are still processing (%s); they stay recorded and the next run "
                "will collect their results",
                len(left),
                ", ".join(left),
            )

    # ------------------------------------------------------------------
    # queueing
    # ------------------------------------------------------------------

    def _new_item(self, ref: BatchItem, params: dict[str, Any]) -> _Item:
        future: asyncio.Future[Reply] = asyncio.get_running_loop().create_future()
        future.add_done_callback(partial(self._forget, ref.request_hash))
        size = len(canonical_json(params).encode()) + _ITEM_OVERHEAD
        return _Item(ref, params, future, self._clock.now(), size)

    def _forget(self, request_hash: str, future: asyncio.Future[Reply]) -> None:
        if self._waiting_for.get(request_hash) is future:
            del self._waiting_for[request_hash]
        if not future.cancelled():
            future.exception()  # retrieved: nobody may be waiting for an orphan's error

    def _enqueue(self, item: _Item) -> None:
        self._queue.append(item)
        self._queued_bytes += item.size
        if self._full():
            self._flush(everything=False)
        elif self._timer is None:
            self._timer = asyncio.create_task(self._flush_later())

    def _full(self) -> bool:
        limits = self._limits
        return (
            len(self._queue) >= limits.batch_max_requests
            or self._queued_bytes >= limits.batch_max_bytes
        )

    async def _flush_later(self) -> None:
        await self._clock.sleep(self._limits.batch_max_wait_s)
        self._timer = None
        self._flush(everything=True)

    def _flush(self, *, everything: bool) -> None:
        """Send full batches, or (``everything``) all that is queued; the rest keeps waiting."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        limits = self._limits
        while self._queue and (everything or self._full()):
            take, size = 0, 0
            for item in self._queue:
                if take and (
                    take >= limits.batch_max_requests or size + item.size > limits.batch_max_bytes
                ):
                    break
                take += 1
                size += item.size
            items = self._queue[:take]
            del self._queue[:take]
            self._queued_bytes -= size
            live = [i for i in items if not i.future.done()]
            if live:
                self._spawn(self._send(live))
        if self._queue:
            self._timer = asyncio.create_task(self._flush_later())

    def _spawn(self, work: Awaitable[None]) -> None:
        task: asyncio.Task[None] = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------------
    # one batch
    # ------------------------------------------------------------------

    async def _send(self, items: list[_Item]) -> None:
        requests = [{"custom_id": i.ref.custom_id, "params": i.params} for i in items]
        try:
            batch = await self._create(requests)
        except _Rejected as e:
            if len(items) > 1:
                half = len(items) // 2
                log.warning("batch of %d rejected (%s); sending it in halves", len(items), e)
                self._spawn(self._send(items[:half]))
                self._spawn(self._send(items[half:]))
            else:
                _fail(items[0], GatewayError(f"batch request rejected: {e}", attempts=1))
            return
        except GatewayError as e:
            for item in items:  # never sent, so nothing was bought; still an attempt
                item.attempts += 1
                self._retry_or_fail(item, str(e))
            return
        for item in items:
            item.attempts += 1
        self._store.record_batch(batch.id, [i.ref for i in items], offline=self._offline)
        self._batches_sent += 1
        self._open.add(batch.id)
        log.info("sent batch %s with %d requests", batch.id, len(items))
        await self._collect(batch.id, items, requeue=True)

    async def _create(self, requests: list[dict[str, Any]]) -> Any:
        limits = self._limits
        for attempt in range(1, limits.max_attempts + 1):
            try:
                return await self._client.messages.batches.create(requests=requests)
            except _TRANSIENT as e:
                if attempt == limits.max_attempts:
                    raise GatewayError(f"could not create a batch: {e}", attempts=attempt) from e
                await self._clock.sleep(self._backoff(attempt, limits.backoff_max_s))
            except anthropic.APIStatusError as e:
                raise _Rejected(str(e.api_message or e)) from e
        raise AssertionError("unreachable")

    async def _collect(self, batch_id: str, items: list[_Item], *, requeue: bool) -> None:
        """Wait for a batch, file its results, then retry or release what did not succeed."""
        pending = {item.ref.custom_id: item for item in items}
        self._in_batches += len(items)
        try:
            retry: list[tuple[_Item, str]] = []
            if await self._wait_until_ended(batch_id):
                await self._read_results(batch_id, pending, retry)
            retry += [(item, "missing from the batch results") for item in pending.values()]
            self._store.close_batch(batch_id)
            self._open.discard(batch_id)
        finally:
            self._in_batches -= len(items)
        # Requeue together, once the whole result set is read, so the retries
        # share one batch instead of trickling into several.
        for item, reason in retry:
            if requeue:
                self._retry_or_fail(item, reason)
            elif not item.future.done():
                item.future.set_exception(_Resend(reason))

    async def _wait_until_ended(self, batch_id: str) -> bool:
        """Poll until the batch ends (``True``) or the API says it does not exist (``False``).

        Connection errors, 429s and 5xx never give up on the batch: it keeps
        processing on the server, and sending its requests again would pay
        for them twice.
        """
        delay = self._limits.batch_poll_s
        failures = 0
        while True:
            try:
                batch = await self._client.messages.batches.retrieve(batch_id)
            except _TRANSIENT as e:
                failures += 1
                log.warning("polling batch %s failed (%s); retrying", batch_id, e)
                await self._clock.sleep(self._backoff(failures, self._limits.batch_poll_max_s))
                continue
            except anthropic.APIStatusError as e:
                log.warning(
                    "batch %s cannot be read (%s); its requests will be resent", batch_id, e
                )
                return False
            failures = 0
            if batch.processing_status == "ended":
                return True
            await self._clock.sleep(delay)
            delay = min(delay * 2, self._limits.batch_poll_max_s)

    async def _read_results(
        self, batch_id: str, pending: dict[str, _Item], retry: list[tuple[_Item, str]]
    ) -> None:
        """Stream the results, reopening the stream after an error and skipping what was read."""
        failures = 0
        while pending:
            try:
                async for entry in await self._client.messages.batches.results(batch_id):
                    item = pending.pop(entry.custom_id, None)
                    if item is not None and not self._settle(item, entry.result, batch_id):
                        retry.append((item, f"batch result {entry.result.type}"))
                return
            except _TRANSIENT as e:
                failures += 1
                log.warning("reading results of batch %s failed (%s); retrying", batch_id, e)
                await self._clock.sleep(self._backoff(failures, self._limits.batch_poll_max_s))
            except anthropic.APIStatusError as e:
                log.warning("results of batch %s cannot be read (%s)", batch_id, e)
                return

    def _settle(self, item: _Item, result: MessageBatchResult, batch_id: str) -> bool:
        """File one result and resolve its caller; ``False`` if the item should be sent again."""
        if result.type == "succeeded":
            latency = self._clock.now() - item.queued_at
            try:
                reply = self._on_result(item.ref, result.message, item.attempts, latency, batch_id)
            except RefusalError as e:
                _fail(item, e)
            else:
                if not item.future.done():
                    item.future.set_result(reply)
            return True
        if result.type == "errored" and result.error.error.type == "invalid_request_error":
            message = result.error.error.message
            _fail(item, GatewayError(f"batch request rejected: {message}", attempts=item.attempts))
            return True
        log.debug("batch %s: %s came back %s", batch_id, item.ref.custom_id, result.type)
        return False

    def _retry_or_fail(self, item: _Item, reason: str) -> None:
        if item.future.done():
            return
        if item.attempts >= self._limits.max_attempts:
            _fail(
                item,
                GatewayError(
                    f"batch request failed after {item.attempts} attempts: {reason}",
                    attempts=item.attempts,
                ),
            )
            return
        self._resubmitted += 1
        log.debug("resubmitting %s (%s)", item.ref.custom_id, reason)
        self._enqueue(item)

    def _backoff(self, failures: int, ceiling: float) -> float:
        """Full-jitter exponential backoff, capped at ``ceiling`` seconds."""
        base = self._limits.backoff_base_s
        return self._rng.uniform(0, min(ceiling, base * 2 ** (failures - 1)))


def _fail(item: _Item, error: Exception) -> None:
    if not item.future.done():
        item.future.set_exception(error)
