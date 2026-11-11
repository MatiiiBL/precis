import asyncio
import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from precis.llm.clock import ManualClock
from precis.llm.limiter import Admission, AIMDWindow, Outcome, TokenBucket, Usage

# ----------------------------------------------------------------------
# token bucket
# ----------------------------------------------------------------------


@settings(max_examples=150, deadline=None)
@given(
    rate=st.integers(min_value=10, max_value=100_000),
    burst_s=st.floats(min_value=1.0, max_value=60.0),
    requests=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=5.0),  # gap before the request, seconds
            st.floats(min_value=0.01, max_value=1.0),  # size as a fraction of the burst
            st.floats(min_value=0.5, max_value=2.0),  # actual / reserved
        ),
        min_size=1,
        max_size=60,
    ),
)
def test_bucket_never_admits_more_than_burst_plus_rate_over_any_window(
    rate: int, burst_s: float, requests: list[tuple[float, float, float]]
) -> None:
    # Usage is only known after a request is admitted, so a request that used
    # more than it reserved can overshoot by that difference; the bucket then
    # makes the next request wait off the debt. Over any window the actual
    # usage is therefore at most one burst, plus the refill over the window,
    # plus the overshoot of the window's last request.
    clock = ManualClock()
    bucket = TokenBucket(clock, rate, burst_s=burst_s)
    admitted: list[tuple[float, float, float]] = []
    for gap, fraction, actual_ratio in requests:
        clock.advance(gap)
        reserved = fraction * bucket.capacity
        clock.advance(bucket.wait_time(reserved))
        bucket.take(reserved)
        actual = reserved * actual_ratio
        bucket.credit(reserved - actual)  # the correction once usage is known
        admitted.append((clock.now(), actual, max(0.0, actual - reserved)))

    per_second = rate / 60
    for i, (start, _, _) in enumerate(admitted):
        total = 0.0
        for end, amount, overshoot in admitted[i:]:
            total += amount
            bound = bucket.capacity + per_second * (end - start) + overshoot
            assert total <= bound * (1 + 1e-9) + 1e-9


def test_bucket_lets_an_oversized_request_through_when_full() -> None:
    clock = ManualClock()
    bucket = TokenBucket(clock, 600, burst_s=10)  # capacity 100
    assert bucket.wait_time(500) == 0
    bucket.take(500)
    assert bucket.level == -400
    # The debt has to refill before anything else goes: 500 tokens at 10/s.
    assert bucket.wait_time(1) == pytest.approx(40.1)


def test_unknown_rate_admits_everything_until_learned() -> None:
    clock = ManualClock()
    bucket = TokenBucket(clock, None, burst_s=10)
    assert bucket.wait_time(10**9) == 0
    bucket.learn(60)
    assert bucket.rate_per_min == 60
    assert bucket.capacity == 10
    bucket.take(10)
    assert bucket.wait_time(1) == pytest.approx(1.0)


def test_learning_never_raises_a_configured_limit() -> None:
    bucket = TokenBucket(ManualClock(), 100, burst_s=60)
    bucket.learn(1000)
    assert bucket.rate_per_min == 100
    bucket.learn(50)
    assert bucket.rate_per_min == 50


# ----------------------------------------------------------------------
# AIMD window
# ----------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(
    initial=st.integers(1, 32),
    maximum=st.integers(1, 64),
    ops=st.lists(st.tuples(st.booleans(), st.sampled_from(list(Outcome))), max_size=300),
)
def test_window_never_admits_beyond_its_limit(
    initial: int, maximum: int, ops: list[tuple[bool, Outcome]]
) -> None:
    window = AIMDWindow(initial, maximum)
    holders: list[int] = []
    for admit, outcome in ops:
        if admit and window.can_admit():
            assert window.in_flight < math.floor(window.limit)
            holders.append(window.admit())
        elif holders:
            window.release(holders.pop(0), outcome)
        assert window.in_flight == len(holders) <= maximum
        assert 1 <= window.limit <= maximum


def test_window_halves_once_per_congestion_event_and_grows_back() -> None:
    window = AIMDWindow(initial=8, maximum=10)
    epochs = [window.admit() for _ in range(8)]
    for epoch in epochs[:3]:
        window.release(epoch, Outcome.OVERLOAD)
    assert window.limit == 4  # three overloads from one burst count once
    for epoch in epochs[3:]:
        window.release(epoch, Outcome.SUCCESS)
    # Additive increase: +1 per floor(limit) successes, up to the maximum.
    successes = 0
    while window.limit < 10:
        window.release(window.admit(), Outcome.SUCCESS)
        successes += 1
    assert successes == 4 + 5 + 6 + 7 + 8 + 9 - 5  # the five above counted toward 4 -> 5
    window.release(window.admit(), Outcome.SUCCESS)
    assert window.limit == 10


# ----------------------------------------------------------------------
# Admission (async)
# ----------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 10**6),
    tasks=st.integers(1, 80),
    max_concurrency=st.integers(1, 16),
)
def test_admission_never_exceeds_the_concurrency_limit(
    seed: int, tasks: int, max_concurrency: int
) -> None:
    rng = random.Random(seed)
    admission = Admission(
        ManualClock(),
        initial_concurrency=max(1, max_concurrency // 2),
        max_concurrency=max_concurrency,
        rpm=None,
        itpm=None,
        otpm=None,
        burst_s=10,
    )
    window = admission.window
    real_admit = window.admit
    peak = 0

    def checked_admit() -> int:
        nonlocal peak
        assert window.in_flight < math.floor(window.limit) <= max_concurrency
        peak = max(peak, window.in_flight + 1)
        return real_admit()

    window.admit = checked_admit  # type: ignore[method-assign]

    async def worker() -> None:
        ticket = await admission.acquire(10, 10)
        for _ in range(rng.randint(0, 5)):
            await asyncio.sleep(0)
        outcome = rng.choice([Outcome.SUCCESS] * 4 + [Outcome.OVERLOAD, Outcome.FAILURE])
        admission.release(ticket, outcome, Usage(10, 10))

    async def run() -> None:
        await asyncio.gather(*(worker() for _ in range(tasks)))

    asyncio.run(run())
    assert admission.in_flight == 0
    assert admission.queued == 0
    assert peak <= max_concurrency


async def test_admission_waits_for_bucket_capacity_in_arrival_order() -> None:
    clock = ManualClock()
    admission = Admission(
        clock,
        initial_concurrency=4,
        max_concurrency=4,
        rpm=60,
        itpm=None,
        otpm=None,
        burst_s=2,  # two requests' worth: then one per second
    )
    order: list[int] = []

    async def go(i: int) -> None:
        ticket = await admission.acquire(1, 1)
        order.append(i)
        admission.release(ticket, Outcome.SUCCESS, Usage(1, 1))

    await asyncio.gather(*(go(i) for i in range(5)))
    assert order == [0, 1, 2, 3, 4]
    assert clock.now() == pytest.approx(3.0)


async def test_reservations_are_corrected_from_usage() -> None:
    clock = ManualClock()
    admission = Admission(
        clock,
        initial_concurrency=1,
        max_concurrency=1,
        rpm=None,
        itpm=6000,
        otpm=None,
        burst_s=10,  # 1,000 input tokens
    )
    ticket = await admission.acquire(800, 100)
    admission.release(ticket, Outcome.SUCCESS, Usage(input_tokens=200, output_tokens=100))
    assert admission.input_tokens.level == pytest.approx(800)
    ticket = await admission.acquire(100, 100)
    admission.release(ticket, Outcome.SUCCESS, Usage(input_tokens=1500, output_tokens=100))
    # 800 - 100 reserved, then 1,400 more charged: the next request waits off the debt.
    assert admission.input_tokens.level == pytest.approx(-700)


def test_limits_are_learned_from_response_headers() -> None:
    admission = Admission(
        ManualClock(),
        initial_concurrency=1,
        max_concurrency=1,
        rpm=None,
        itpm=None,
        otpm=5000,
        burst_s=60,
    )
    admission.observe(
        {
            "anthropic-ratelimit-requests-limit": "4000",
            "anthropic-ratelimit-requests-remaining": "3000",
            "anthropic-ratelimit-input-tokens-limit": "2000000",
            "anthropic-ratelimit-input-tokens-remaining": "2000000",
            "anthropic-ratelimit-output-tokens-limit": "400000",
            "anthropic-ratelimit-output-tokens-remaining": "not a number",
        }
    )
    assert admission.requests.rate_per_min == 4000
    assert admission.requests.level == pytest.approx(3000)
    assert admission.input_tokens.rate_per_min == 2_000_000
    assert admission.output_tokens.rate_per_min == 5000  # configured lower than the server's


async def test_pause_holds_every_request() -> None:
    clock = ManualClock()
    admission = Admission(
        clock,
        initial_concurrency=2,
        max_concurrency=2,
        rpm=None,
        itpm=None,
        otpm=None,
        burst_s=10,
    )
    admission.pause(12)
    await admission.acquire(1, 1)
    assert clock.now() == pytest.approx(12)
