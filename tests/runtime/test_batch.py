import asyncio
import contextlib

import pytest
from anthropic.types import Message

from precis.llm import Gateway, GatewayError, Reply
from precis.llm.clock import ManualClock
from precis.llm.fake import FakeAnthropic, Faults
from precis.llm.gateway import request_hash_of
from precis.llm.pricing import cost
from precis.store import Store
from precis.util import canonical_json

from .helpers import config, echo, request


def batch_gateway(
    fake: FakeAnthropic, store: Store, clock: ManualClock, **limits: float
) -> Gateway:
    cfg = config(
        limits={"batch_max_requests": 100, "batch_max_wait_s": 30, "batch_poll_s": 10, **limits},
        pipeline={"fallbacks": True},
    )
    return Gateway(fake, store, run_id="batch-run", config=cfg, clock=clock, mode="batch")


async def test_200_concurrent_callers_are_served_by_a_few_batches(
    store: Store, clock: ManualClock
) -> None:
    faults = Faults(batch_errored=0.08, batch_expired=0.04)
    fake = FakeAnthropic([echo], clock=clock, seed=3, faults=faults, batch_latency_s=60)
    gw = batch_gateway(fake, store, clock)

    replies = await asyncio.gather(
        *(gw.create(request(f"question {i}"), stage="read", paper_id=f"p{i}") for i in range(200))
    )
    await gw.aclose()

    for i, reply in enumerate(replies):
        assert isinstance(reply, Message)
        assert reply.content[0].type == "text"
        assert reply.content[0].text == f"echo: question {i}"
    sizes = fake.batch_sizes
    assert sizes[:2] == [100, 100]
    # Errored and expired items went back into later, smaller batches.
    assert 3 <= len(sizes) <= 6
    assert sum(sizes) > 200
    # Fallbacks are on, but the fake rejects batches that carry them, so
    # getting here also shows the gateway left them out of batch requests.

    calls = store.calls(run_id="batch-run")
    assert len(calls) == 200
    assert all(c.batch and c.status == "ok" for c in calls)
    assert any(c.attempts > 1 for c in calls)
    first = replies[0]
    assert calls[0].cost_usd == pytest.approx(cost(first.usage, first.model, batch=True))
    stats = gw.stats()
    assert (stats.calls, stats.batch_calls, stats.errors) == (200, 200, 0)
    assert (stats.queued, stats.in_flight, stats.batches_pending) == (0, 0, 0)


async def test_a_lone_request_is_sent_after_the_wait(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=5)
    gw = batch_gateway(fake, store, clock)
    reply = await gw.create(request("alone"), stage="read")
    assert reply.content[0].type == "text"
    assert fake.batch_sizes == [1]
    assert clock.now() >= 30


async def test_multi_turn_callers_work_unchanged(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=5)
    gw = batch_gateway(fake, store, clock, batch_max_requests=2)

    async def agent(name: str) -> str:
        messages = [{"role": "user", "content": f"{name} turn 1"}]
        first = await gw.create(request(messages=messages), stage="read")
        messages += [
            {"role": "assistant", "content": [b.model_dump() for b in first.content]},
            {"role": "user", "content": f"{name} turn 2"},
        ]
        second = await gw.create(request(messages=messages), stage="read")
        block = second.content[0]
        assert block.type == "text"
        return block.text

    assert await asyncio.gather(agent("a"), agent("b")) == ["echo: a turn 2", "echo: b turn 2"]
    assert fake.batch_sizes == [2, 2]


async def test_invalid_batch_items_fail_without_retry(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=5)
    gw = batch_gateway(fake, store, clock, batch_max_requests=2)

    good, bad = await asyncio.gather(
        gw.create(request("fine"), stage="read"),
        gw.create(request("bad", temperature=0.3), stage="read"),
        return_exceptions=True,
    )
    assert isinstance(good, Message)
    assert isinstance(bad, GatewayError)
    assert bad.attempts == 1
    assert fake.batch_sizes == [2]
    assert sorted(c.status for c in store.calls()) == ["error", "ok"]


async def test_items_that_keep_failing_give_up(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=5, faults=Faults(batch_expired=1.0))
    gw = batch_gateway(fake, store, clock, max_attempts=3)
    with pytest.raises(GatewayError) as excinfo:
        await gw.create(request(), stage="read")
    assert excinfo.value.attempts == 3
    assert fake.batch_sizes == [1, 1, 1]


# ----------------------------------------------------------------------
# durability (review R3) and size limits (review R8)
# ----------------------------------------------------------------------


async def _cancel_once_sent(fake: FakeAnthropic, caller: asyncio.Task[Reply]) -> None:
    """Cancel a caller as soon as its request is in a batch, as an interrupt would."""
    for _ in range(10_000):
        if fake.batch_sizes:
            break
        await asyncio.sleep(0)
    assert fake.batch_sizes
    caller.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await caller


async def test_an_interrupted_batch_is_collected_by_the_next_run(
    store: Store, clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    # A batch that takes "forever": closing must not wait for it.
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=10**9)
    first = batch_gateway(fake, store, clock)
    caller = asyncio.create_task(first.create(request("billed"), stage="read", paper_id="p1"))
    await _cancel_once_sent(fake, caller)
    await asyncio.wait_for(first.aclose(), timeout=5)
    assert len(store.open_batches()) == 1
    assert "collect" in caplog.text

    clock.advance(10**9)
    second = batch_gateway(fake, store, clock)
    reply = await second.create(request("billed"), stage="read", paper_id="p1")
    await second.aclose()

    assert reply.content[0].type == "text"
    assert fake.batch_sizes == [1]  # collected, not bought again
    assert store.open_batches() == []
    [call] = store.calls()
    assert (call.status, call.batch, call.paper_id) == ("ok", True, "p1")


async def test_results_nobody_waits_for_are_cached(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=60)
    first = batch_gateway(fake, store, clock)
    caller = asyncio.create_task(first.create(request("orphan"), stage="read"))
    await _cancel_once_sent(fake, caller)
    clock.advance(60)
    second = batch_gateway(fake, store, clock)
    await second.create(request("something else"), stage="read")  # starts collection
    await second.aclose()
    assert store.cache_get(request_hash_of(request("orphan"))) is not None


async def test_a_polling_outage_keeps_waiting_for_the_same_batch(
    store: Store, clock: ManualClock
) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=600)
    fake.fail_batch_api("retrieve", times=20)
    gw = batch_gateway(fake, store, clock)
    reply = await gw.create(request("patient"), stage="read")
    assert reply.content[0].type == "text"
    assert fake.batch_sizes == [1]


async def test_errors_while_streaming_results_are_retried(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=60)
    fake.fail_batch_api("results_stream")
    gw = batch_gateway(fake, store, clock, batch_max_requests=3)
    replies = await asyncio.gather(*(gw.create(request(f"q{i}"), stage="read") for i in range(3)))
    assert [r.content[0].type for r in replies] == ["text"] * 3
    assert fake.batch_sizes == [3]


def _request_bytes(text: str) -> int:
    return len(canonical_json(request(text)).encode())


async def test_batches_are_flushed_by_size(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=60)
    texts = [f"q{i} " + "x" * 5000 for i in range(4)]
    limit = int(_request_bytes(texts[0]) * 2.5)  # room for two requests, not three
    gw = batch_gateway(fake, store, clock, batch_max_bytes=limit)
    await asyncio.gather(*(gw.create(request(text), stage="read") for text in texts))
    assert fake.batch_sizes == [2, 2]


async def test_a_rejected_batch_is_split_in_half(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=60)
    fake.max_batch_bytes = int((_request_bytes("q0") + 40) * 2.5)
    gw = batch_gateway(fake, store, clock, batch_max_requests=4)
    replies = await asyncio.gather(*(gw.create(request(f"q{i}"), stage="read") for i in range(4)))
    assert len(replies) == 4
    assert fake.batch_sizes == [2, 2]
