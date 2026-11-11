"""End-to-end behaviour of the gateway against the fake's server-side limits.

These run in simulated time (``ScaledClock``) for a few simulated minutes
each, so their assertions leave room for scheduling jitter.
"""

from precis.llm.fake import Faults, ServerLimits

from .simulate import simulate


async def test_slow_start_reaches_the_rate_limit_quickly_without_429s() -> None:
    limits = ServerLimits(rpm=300, itpm=1_000_000, otpm=200_000)
    r = await simulate(
        duration_s=240,
        clients=48,
        speed=60,
        limits=limits,
        gateway_limits={"initial_concurrency": 8, "max_concurrency": 48},
    )

    stats = r.gateway.stats()
    assert stats.rpm_limit == 300  # learned from the response headers
    # About 23 requests in flight sustain 300 RPM at ~4.5 s each. Doubling
    # per window gets there in a few round trips; one slot per window from 8
    # took about 70 s.
    reached = next(s.t for s in r.samples if s.limit >= 23)
    assert reached < 30
    steady = r.rate_per_min(60, 240)
    assert 0.85 * 300 <= steady <= 1.10 * 300
    # The token buckets, not the window, hold the rate: growing fast does not
    # overshoot the server's limits.
    assert r.count("429") == 0
    assert r.failures == 0


async def test_slow_start_stops_at_the_first_overload_then_aimd_settles() -> None:
    r = await simulate(
        duration_s=300,
        clients=48,
        speed=60,
        faults=Faults(overload_above=12),
        gateway_limits={"initial_concurrency": 4, "max_concurrency": 48},
    )
    first_overload = next(e.t for e in r.fake.events if e.outcome == "529")
    assert first_overload < 30  # doubling from 4 finds the capacity within a few round trips
    # Slow start overshoots the capacity by at most one doubling, once.
    assert max(s.limit for s in r.samples) <= 2 * 12
    ok, overloaded = r.count("ok"), r.count("529")
    assert overloaded <= 0.05 * (ok + overloaded)

    late = [s for s in r.samples if s.t >= 120]
    mean_limit = sum(s.limit for s in late) / len(late)
    # The sawtooth oscillates between about half the capacity and just above it.
    assert 6 <= mean_limit <= 14
    assert max(s.limit for s in late) <= 16
    assert r.failures == 0
