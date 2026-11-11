"""End-to-end behaviour of the gateway against the fake's server-side limits.

These run in simulated time (``ScaledClock``) for a few simulated minutes
each, so their assertions leave room for scheduling jitter.
"""

from precis.llm.fake import Faults, ServerLimits

from .simulate import simulate


async def test_learns_the_rate_limit_and_converges_without_sustained_429s() -> None:
    limits = ServerLimits(rpm=300, itpm=1_000_000, otpm=200_000)
    r = await simulate(
        duration_s=240,
        clients=48,
        speed=60,
        limits=limits,
        gateway_limits={"initial_concurrency": 16, "max_concurrency": 48},
    )

    stats = r.gateway.stats()
    assert stats.rpm_limit == 300  # learned from the response headers
    steady = r.rate_per_min(90, 240)
    assert 0.85 * 300 <= steady <= 1.10 * 300
    total = len(r.fake.events)
    assert r.count("429") <= 0.02 * total
    assert r.count("429", 120, 240) == 0
    assert r.failures == 0


async def test_aimd_settles_near_the_servers_capacity() -> None:
    r = await simulate(
        duration_s=300,
        clients=48,
        speed=60,
        faults=Faults(overload_above=12),
        gateway_limits={"initial_concurrency": 4, "max_concurrency": 48},
    )
    late = [s for s in r.samples if s.t >= 120]
    mean_limit = sum(s.limit for s in late) / len(late)
    # The sawtooth oscillates between about half the capacity and just above it.
    assert 6 <= mean_limit <= 14
    assert max(s.limit for s in late) <= 16
    ok, overloaded = r.count("ok", 120), r.count("529", 120)
    assert overloaded <= 0.1 * (ok + overloaded)
    assert r.failures == 0
