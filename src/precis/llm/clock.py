"""Time as a dependency.

Every time-dependent part of the runtime (token buckets, the AIMD window,
retry backoff, batch polling, the fake API's latency and rate limits) reads
time through a ``Clock`` instead of calling ``time`` or ``asyncio.sleep``
directly. Production uses ``RealClock``; simulations use ``ScaledClock`` to
run an hour of traffic in a few seconds with the same code; unit tests that
step through one request at a time use ``ManualClock``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Seconds on a monotonic timeline; only differences are meaningful."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the calling task for ``seconds`` of this clock's time."""
        ...


class RealClock:
    """Wall time: ``time.monotonic`` and ``asyncio.sleep``."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class ScaledClock:
    """Simulated time that runs ``speed`` times faster than wall time.

    ``ScaledClock(600)`` turns a simulated minute into 0.1 s, so every
    component keeps its real-world parameters (60-second rate-limit windows,
    10-second polls) while a test finishes quickly. The price is resolution:
    a millisecond of scheduling jitter becomes ``speed`` milliseconds of
    simulated time, so pick the slowest speed that keeps the run short.
    """

    def __init__(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = speed
        self._origin = time.monotonic()

    def now(self) -> float:
        return (time.monotonic() - self._origin) * self.speed

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds) / self.speed)


class ManualClock:
    """Time that only moves when told to, for step-by-step unit tests.

    ``sleep`` advances the clock by the requested amount and yields once,
    so a single task observes exact durations (``slept`` records them).
    With several tasks sleeping at once the durations add up instead of
    overlapping; use ``ScaledClock`` for concurrent simulations.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.t = start
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += max(0.0, seconds)

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)
        await asyncio.sleep(0)
