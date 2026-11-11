"""Client-side admission control: token buckets and an AIMD concurrency window.

The Messages API limits each model by requests, input tokens and output
tokens per minute, enforced with token buckets that refill continuously.
``Admission`` mirrors those three buckets on the client so requests wait
here instead of collecting 429s, and bounds concurrency with a window that,
like TCP's, starts by doubling (slow start), then increases additively and
halves when the server signals overload (429 or 529).

The bucket and window classes are synchronous and read time from a
``Clock``, so their invariants can be checked exhaustively (see the
property tests); ``Admission`` adds the asyncio waiting on top.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from precis.llm.clock import Clock

SECONDS_PER_MINUTE = 60.0


class TokenBucket:
    """A continuously refilling bucket holding at most ``burst_s`` seconds of tokens.

    ``rate_per_min`` of ``None`` means unknown: the bucket admits everything
    until a rate is configured or learned. The level may go negative when a
    reservation is corrected upward after the response; later requests then
    wait for the debt to refill, which is how the bucket stays honest when
    the estimate was low.
    """

    def __init__(self, clock: Clock, rate_per_min: float | None, *, burst_s: float) -> None:
        self._clock = clock
        self._burst_s = burst_s
        self._configured = rate_per_min
        self._rate: float | None = None
        self._level = 0.0
        self._last = clock.now()
        self._set_rate(rate_per_min)

    @property
    def rate_per_min(self) -> float | None:
        return self._rate

    @property
    def capacity(self) -> float:
        if self._rate is None:
            return math.inf
        return max(1.0, self._rate * self._burst_s / SECONDS_PER_MINUTE)

    @property
    def level(self) -> float:
        self._refill()
        return self._level

    def _set_rate(self, rate: float | None) -> None:
        self._refill()
        was_unknown = self._rate is None
        self._rate = rate
        if rate is not None:
            self._level = self.capacity if was_unknown else min(self._level, self.capacity)

    def learn(self, server_limit: float) -> None:
        """Adopt the server's limit, or keep the configured one if it is lower."""
        rate = server_limit if self._configured is None else min(self._configured, server_limit)
        if rate != self._rate:
            self._set_rate(rate)

    def _refill(self) -> None:
        now = self._clock.now()
        if self._rate is not None:
            refill = (now - self._last) * self._rate / SECONDS_PER_MINUTE
            self._level = min(self.capacity, self._level + refill)
        self._last = now

    def wait_time(self, n: float) -> float:
        """Seconds until ``n`` tokens can be taken (0 if they can be taken now).

        A request larger than the whole bucket is let through when the bucket
        is full, otherwise it could never be admitted.
        """
        if self._rate is None:
            return 0.0
        self._refill()
        need = min(n, self.capacity)
        if self._level >= need:
            return 0.0
        return (need - self._level) * SECONDS_PER_MINUTE / self._rate

    def take(self, n: float) -> None:
        self._refill()
        if self._rate is not None:
            self._level -= n

    def credit(self, n: float) -> None:
        """Return ``n`` tokens (or charge ``-n`` more) after the real usage is known."""
        self._refill()
        if self._rate is not None:
            self._level = min(self.capacity, self._level + n)

    def clamp(self, remaining: float) -> None:
        """Lower the level to the server's ``remaining`` when the server knows better."""
        self._refill()
        if self._rate is not None:
            self._level = min(self._level, remaining)


class Outcome(Enum):
    SUCCESS = "success"
    OVERLOAD = "overload"  # 429 or 529: the server asked us to slow down
    FAILURE = "failure"  # anything else; says nothing about capacity


class AIMDWindow:
    """Concurrency limit with TCP's slow start, then additive increase and halving.

    It starts in *slow start*: every success adds a slot, which doubles the
    limit per window of successes, so a run reaches a high ceiling in a few
    round trips instead of thousands of requests. The first overload (429
    or 529) halves the limit and ends slow start for good; from then on the
    limit grows by one per window of ``floor(limit)`` successes, roughly one
    round trip of the whole in-flight set, as in TCP congestion avoidance.
    All requests that were in flight when an overload arrived belong to the
    same congestion event, so only the first of their overload signals
    halves the limit; the epoch number tells them apart.
    """

    def __init__(
        self, initial: int, maximum: int, minimum: int = 1, *, slow_start: bool = True
    ) -> None:
        if not 1 <= minimum <= maximum:
            raise ValueError("need 1 <= minimum <= maximum")
        self.minimum = minimum
        self.maximum = maximum
        self.limit = float(min(max(initial, minimum), maximum))
        self.slow_start = slow_start
        self.in_flight = 0
        self.epoch = 0
        self._successes = 0

    def can_admit(self) -> bool:
        return self.in_flight < math.floor(self.limit)

    def admit(self) -> int:
        """Take a slot (the caller checked ``can_admit``); returns the current epoch."""
        self.in_flight += 1
        return self.epoch

    def release(self, epoch: int, outcome: Outcome) -> None:
        self.in_flight -= 1
        if outcome is Outcome.SUCCESS:
            if self.slow_start:
                self.limit = min(self.maximum, self.limit + 1)
                return
            self._successes += 1
            if self._successes >= math.floor(self.limit):
                self._successes = 0
                self.limit = min(self.maximum, self.limit + 1)
        elif outcome is Outcome.OVERLOAD and epoch == self.epoch:
            self.limit = max(float(self.minimum), float(math.floor(self.limit / 2)))
            self.slow_start = False
            self.epoch += 1
            self._successes = 0

    def set_maximum(self, maximum: int) -> None:
        self.maximum = max(self.minimum, maximum)
        self.limit = min(self.limit, self.maximum)


@dataclass(frozen=True)
class Ticket:
    """What ``Admission.acquire`` handed out, needed to give it back."""

    epoch: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Usage:
    """Tokens the server actually counted against the per-minute limits.

    Input counts uncached input plus cache writes: cache reads do not count
    toward input-token limits on current models.
    """

    input_tokens: int
    output_tokens: int


_HEADER_PREFIX = "anthropic-ratelimit-"


class Admission:
    """Waits until a request may be sent: a concurrency slot, then bucket capacity.

    Slots are handed out first-come first-served; a request that holds a
    slot then waits for all three buckets at once, again in arrival order,
    so a large request is not starved by a stream of small ones.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        initial_concurrency: int,
        max_concurrency: int,
        rpm: float | None,
        itpm: float | None,
        otpm: float | None,
        burst_s: float,
        slow_start: bool = True,
    ) -> None:
        self._clock = clock
        self.window = AIMDWindow(initial_concurrency, max_concurrency, slow_start=slow_start)
        self.requests = TokenBucket(clock, rpm, burst_s=burst_s)
        self.input_tokens = TokenBucket(clock, itpm, burst_s=burst_s)
        self.output_tokens = TokenBucket(clock, otpm, burst_s=burst_s)
        self._slot_waiters: deque[asyncio.Future[int]] = deque()
        self._bucket_lock = asyncio.Lock()
        self._paused_until = -math.inf
        self.queued = 0

    @property
    def in_flight(self) -> int:
        return self.window.in_flight

    @property
    def concurrency_limit(self) -> int:
        return math.floor(self.window.limit)

    async def acquire(self, input_tokens: int, output_tokens: int) -> Ticket:
        """Wait for a slot and for bucket capacity, then reserve the estimates."""
        self.queued += 1
        try:
            epoch = await self._acquire_slot()
            try:
                await self._acquire_tokens(input_tokens, output_tokens)
            except BaseException:
                self.window.release(epoch, Outcome.FAILURE)
                self._wake()
                raise
        finally:
            self.queued -= 1
        return Ticket(epoch, input_tokens, output_tokens)

    def release(self, ticket: Ticket, outcome: Outcome, usage: Usage | None = None) -> None:
        """Free the slot and correct the reservation with what the server counted.

        Without usage (the request failed) the token reservations are
        refunded; the request itself stays counted, because the server
        counts failed requests too.
        """
        used = usage or Usage(0, 0)
        self.input_tokens.credit(ticket.input_tokens - used.input_tokens)
        self.output_tokens.credit(ticket.output_tokens - used.output_tokens)
        self.window.release(ticket.epoch, outcome)
        self._wake()

    def pause(self, seconds: float) -> None:
        """Admit nothing for ``seconds`` (a 429's ``retry-after`` applies to everyone)."""
        self._paused_until = max(self._paused_until, self._clock.now() + seconds)

    def observe(self, headers: Mapping[str, str]) -> None:
        """Learn limits from ``anthropic-ratelimit-*`` response headers."""
        for name, bucket in (
            ("requests", self.requests),
            ("input-tokens", self.input_tokens),
            ("output-tokens", self.output_tokens),
        ):
            limit = _number(headers.get(f"{_HEADER_PREFIX}{name}-limit"))
            if limit is not None and limit > 0:
                bucket.learn(limit)
            remaining = _number(headers.get(f"{_HEADER_PREFIX}{name}-remaining"))
            if remaining is not None:
                bucket.clamp(remaining)

    async def _acquire_slot(self) -> int:
        if not self._slot_waiters and self.window.can_admit():
            return self.window.admit()
        waiter: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._slot_waiters.append(waiter)
        try:
            return await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                # The slot was handed over just as we were cancelled.
                self.window.release(waiter.result(), Outcome.FAILURE)
                self._wake()
            raise

    def _wake(self) -> None:
        while self._slot_waiters and self.window.can_admit():
            waiter = self._slot_waiters.popleft()
            if not waiter.done():
                waiter.set_result(self.window.admit())

    async def _acquire_tokens(self, input_tokens: int, output_tokens: int) -> None:
        async with self._bucket_lock:
            while True:
                wait = max(
                    self._paused_until - self._clock.now(),
                    self.requests.wait_time(1),
                    self.input_tokens.wait_time(input_tokens),
                    self.output_tokens.wait_time(output_tokens),
                )
                if wait <= 0:
                    break
                await self._clock.sleep(wait)
            self.requests.take(1)
            self.input_tokens.take(input_tokens)
            self.output_tokens.take(output_tokens)


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
