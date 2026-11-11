import asyncio
import random
import time

import pytest

from precis.eval.simclock import VirtualClock


async def test_sleeps_overlap_and_take_no_wall_time() -> None:
    clock = VirtualClock()
    finished: list[tuple[str, float]] = []

    async def worker(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        finished.append((name, clock.now()))

    async def main() -> str:
        await asyncio.gather(worker("slow", 3600), worker("fast", 60), worker("now", 0))
        return "done"

    started = time.perf_counter()
    assert await clock.run(main()) == "done"
    assert time.perf_counter() - started < 1
    assert finished == [("now", 0), ("fast", 60), ("slow", 3600)]
    assert clock.now() == 3600


async def test_chains_of_wakeups_settle_before_time_moves() -> None:
    clock = VirtualClock()
    event = asyncio.Event()
    seen: list[float] = []

    async def setter() -> None:
        await clock.sleep(5)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        event.set()

    async def waiter() -> None:
        await event.wait()
        seen.append(clock.now())
        await clock.sleep(1)
        seen.append(clock.now())

    async def main() -> None:
        await asyncio.gather(setter(), waiter(), clock.sleep(100))

    await clock.run(main())
    assert seen == [5, 6]


async def test_many_tasks_run_fast_and_deterministically() -> None:
    async def simulate() -> list[float]:
        clock = VirtualClock()
        rng = random.Random(1)
        ends: list[float] = []

        async def worker() -> None:
            for _ in range(5):
                await clock.sleep(rng.expovariate(1 / 10))
            ends.append(clock.now())

        await clock.run(asyncio.gather(*(worker() for _ in range(2000))))
        return ends

    started = time.perf_counter()
    first = await simulate()
    assert time.perf_counter() - started < 5
    assert first == await simulate()


async def test_errors_and_cancellation_propagate() -> None:
    clock = VirtualClock()

    async def boom() -> None:
        await clock.sleep(10)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await clock.run(boom())

    async def cancelled_sleeper() -> float:
        task = asyncio.ensure_future(clock.sleep(1000))
        await clock.sleep(1)
        task.cancel()
        await clock.sleep(1)
        return clock.now()

    clock = VirtualClock()
    assert await clock.run(cancelled_sleeper()) == 2


async def test_tiny_waits_still_move_time_forward() -> None:
    # At t = 1e6 s the float spacing is ~1e-10 s: a wait of 1e-12 s would leave
    # the clock where it was, and a caller polling for a deficit would spin.
    clock = VirtualClock(start=1e6)

    async def wait_for_deficit() -> int:
        target, polls = 1e6 + 0.01, 0
        while clock.now() < target:
            await clock.sleep(1e-12)
            polls += 1
        return polls

    assert await clock.run(wait_for_deficit()) == 10


async def test_waiting_on_a_thread_is_not_a_deadlock() -> None:
    clock = VirtualClock()

    async def main() -> int:
        await clock.sleep(3)
        return await asyncio.to_thread(lambda: 7)

    assert await clock.run(main()) == 7
    assert clock.now() == 3
