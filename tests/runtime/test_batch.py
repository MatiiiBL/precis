import asyncio

import pytest
from anthropic.types import Message

from precis.llm import Gateway, GatewayError
from precis.llm.clock import ManualClock
from precis.llm.fake import FakeAnthropic, Faults
from precis.llm.pricing import cost
from precis.store import Store

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
