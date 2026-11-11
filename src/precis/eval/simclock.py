"""Virtual time for the offline benchmarks.

``precis.llm.ScaledClock`` runs simulated time a fixed factor faster than
wall time, which works while the simulation is cheap compared with the
time it simulates. A 2,000-paper run against a 4,000 requests/minute limit
is not: at any useful speed-up the event loop would have to process tens of
thousands of requests per wall second, it falls behind, and the CPU time it
spends turns into simulated latency, so the benchmark measures the
benchmark.

``VirtualClock`` removes wall time altogether. Time stands still while any
task can run, and jumps to the earliest pending wake-up the moment every
task is waiting on the clock (the "autojump" of trio's ``MockClock``). The
result is independent of the machine's speed and, since asyncio schedules
ready tasks in FIFO order, exactly reproducible for the same seeds.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import Coroutine
from typing import Any


class VirtualClock:
    """A ``precis.llm.Clock`` whose time advances only when the event loop is idle.

    Run the simulation with ``await clock.run(coro)``; outside ``run`` a
    sleeping task would wait forever, because nothing advances the clock.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._order = itertools.count()
        self.jumps = 0

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self._now + seconds, next(self._order), future))
        await future

    async def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` to completion, advancing time whenever nothing else can run."""
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(coro)
        while not task.done():
            await asyncio.sleep(0)
            if not _idle(loop):
                continue
            self._drop_cancelled()
            if self._sleepers:
                self._wake_next()
            elif not task.done():
                # Every task waits on something other than this clock (a thread,
                # real I/O): give the selector a moment instead of spinning.
                await asyncio.sleep(0.001)
        return task.result()

    def _drop_cancelled(self) -> None:
        while self._sleepers and self._sleepers[0][2].done():
            heapq.heappop(self._sleepers)

    def _wake_next(self) -> None:
        wake_at = self._sleepers[0][0]
        self._now = max(self._now, wake_at)
        self.jumps += 1
        while self._sleepers and self._sleepers[0][0] <= self._now:
            _, _, future = heapq.heappop(self._sleepers)
            if not future.done():
                future.set_result(None)


def _idle(loop: asyncio.AbstractEventLoop) -> bool:
    """Whether no callback is ready to run, i.e. every task is waiting.

    CPython's event loops keep ready callbacks in ``_ready``; it is not
    public API, but it is the only exact test. On a loop without it every
    turn counts as idle, which can move time on before a chain of wake-ups
    has settled.
    """
    ready = getattr(loop, "_ready", None)
    return ready is None or len(ready) == 0
