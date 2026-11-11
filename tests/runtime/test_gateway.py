from typing import Any

import pytest
from anthropic.types import Message
from anthropic.types.beta import BetaMessage

from precis.llm import (
    BudgetExceeded,
    Gateway,
    GatewayError,
    GatewayStats,
    RefusalError,
    UnpricedModelError,
)
from precis.llm.clock import ManualClock
from precis.llm.fake import FakeAnthropic, FakeReply, Faults
from precis.llm.pricing import cost
from precis.store import Store

from .helpers import config, echo, request


def gateway(
    fake: FakeAnthropic, store: Store, clock: ManualClock, **sections: dict[str, Any]
) -> Gateway:
    sections.setdefault("pipeline", {"fallbacks": False})
    return Gateway(fake, store, run_id="run-1", config=config(**sections), clock=clock)


async def test_reply_is_recorded_with_its_cost(
    fake: FakeAnthropic, store: Store, clock: ManualClock
) -> None:
    gw = gateway(fake, store, clock)
    reply = await gw.create(request("hello"), stage="read", paper_id="p1")

    assert isinstance(reply, Message)
    assert reply.content[0].type == "text"
    [call] = store.calls(run_id="run-1")
    assert call.status == "ok"
    assert (call.paper_id, call.stage, call.model) == ("p1", "read", "claude-opus-5-5")
    assert call.cost_usd == pytest.approx(cost(reply.usage, "claude-opus-5-5"))
    assert call.cost_usd > 0
    assert call.request_id is not None
    assert gw.spent_usd == pytest.approx(call.cost_usd)
    stats = gw.stats()
    assert isinstance(stats, GatewayStats)
    assert (stats.calls, stats.in_flight, stats.errors) == (1, 0, 0)
    assert stats.output_tokens == reply.usage.output_tokens
    assert gw.run_id == "run-1"


async def test_cache_hit_costs_nothing_and_is_recorded(
    fake: FakeAnthropic, store: Store, clock: ManualClock
) -> None:
    gw = gateway(fake, store, clock)
    first = await gw.create(request("same"), stage="read")
    second = await gw.create(request("same"), stage="read")

    assert second == first
    assert len(fake.requests) == 1
    calls = store.calls()
    assert [c.status for c in calls] == ["ok", "cached"]
    assert calls[1].cost_usd == 0
    assert gw.stats().cached_calls == 1
    assert store.usage()[0].cached_calls == 1


async def test_cache_can_be_turned_off(
    fake: FakeAnthropic, store: Store, clock: ManualClock
) -> None:
    gw = gateway(fake, store, clock, pipeline={"use_cache": False, "fallbacks": False})
    await gw.create(request("same"), stage="read")
    await gw.create(request("same"), stage="read")
    assert len(fake.requests) == 2


async def test_429_waits_for_retry_after_and_halves_concurrency(
    store: Store, clock: ManualClock
) -> None:
    fake = FakeAnthropic([echo], clock=clock, faults=Faults(retry_after_s=7))
    fake.fail_next("rate_limit")
    gw = gateway(fake, store, clock, limits={"initial_concurrency": 8})

    await gw.create(request(), stage="read")

    assert len(fake.requests) == 2
    assert any(7 <= s <= 8 for s in clock.slept)
    stats = gw.stats()
    assert (stats.rate_limited, stats.retries, stats.calls) == (1, 1, 1)
    assert stats.concurrency_limit == 4
    assert store.calls()[0].attempts == 2


async def test_529_backs_off_and_decreases_concurrency(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock)
    fake.fail_next("overloaded", "overloaded")
    gw = gateway(fake, store, clock, limits={"initial_concurrency": 8, "backoff_base_s": 2})

    await gw.create(request(), stage="read")

    stats = gw.stats()
    assert stats.overloaded == 2
    assert stats.retries == 2
    # Sequential overloads are separate congestion events: 8 -> 4 -> 2.
    assert stats.concurrency_limit == 2
    backoffs = [s for s in clock.slept if s > fake.latency.error_s]
    assert all(s <= 4 for s in backoffs)


async def test_timeouts_and_server_errors_are_retried(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock)
    fake.fail_next("timeout", "server_error")
    gw = gateway(fake, store, clock)

    await gw.create(request(), stage="read")

    stats = gw.stats()
    assert (stats.timeouts, stats.server_errors, stats.retries) == (1, 1, 2)
    # Neither says anything about capacity: concurrency is untouched.
    assert stats.concurrency_limit == 8


async def test_400_fails_immediately(fake: FakeAnthropic, store: Store, clock: ManualClock) -> None:
    fake.fail_next("bad_request")
    gw = gateway(fake, store, clock)

    with pytest.raises(GatewayError) as excinfo:
        await gw.create(request(), stage="read", paper_id="p1")

    assert excinfo.value.attempts == 1
    assert len(fake.requests) == 1
    assert gw.stats().errors == 1
    [call] = store.calls()
    assert (call.status, call.cost_usd) == ("error", 0)


async def test_invalid_request_shape_is_not_retried(
    fake: FakeAnthropic, store: Store, clock: ManualClock
) -> None:
    gw = gateway(fake, store, clock)
    with pytest.raises(GatewayError):
        await gw.create(request(temperature=0.5), stage="read")
    assert len(fake.requests) == 1


async def test_retries_give_up_after_max_attempts(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock)
    fake.fail_next(*["server_error"] * 3)
    gw = gateway(fake, store, clock, limits={"max_attempts": 3})

    with pytest.raises(GatewayError) as excinfo:
        await gw.create(request(), stage="read")
    assert excinfo.value.attempts == 3
    assert len(fake.requests) == 3


async def test_budget_stops_new_requests(
    fake: FakeAnthropic, store: Store, clock: ManualClock
) -> None:
    gw = gateway(fake, store, clock, budget={"max_cost_usd": 0.01})
    await gw.create(request("one", max_tokens=50), stage="read")
    assert 0 < gw.spent_usd < 0.01

    async def spend_until_stopped() -> None:
        for i in range(1000):
            await gw.create(request(f"more {i}", max_tokens=50), stage="read")

    with pytest.raises(BudgetExceeded):
        await spend_until_stopped()
    assert 1 < len(fake.requests) < 1000
    assert gw.spent_usd <= 0.01
    sent = len(fake.requests)
    with pytest.raises(BudgetExceeded):
        await gw.create(request("after", max_tokens=50), stage="read")
    assert len(fake.requests) == sent


async def test_budget_counts_the_estimate_of_the_next_request(
    fake: FakeAnthropic, store: Store, clock: ManualClock
) -> None:
    gw = gateway(fake, store, clock, budget={"max_cost_usd": 0.001})
    # 2,000 output tokens of claude-opus-5-5 could cost up to $0.04.
    with pytest.raises(BudgetExceeded):
        await gw.create(request(max_tokens=2000), stage="read")
    assert fake.requests == []


async def test_refusal_raises_with_category(store: Store, clock: ManualClock) -> None:
    fake = FakeAnthropic([lambda p: FakeReply.refusal("bio")], clock=clock)
    gw = gateway(fake, store, clock)

    with pytest.raises(RefusalError) as excinfo:
        await gw.create(request(), stage="read")

    assert excinfo.value.category == "bio"
    [call] = store.calls()
    assert call.status == "refusal"
    assert gw.stats().refusals == 1
    # A refusal is not cached: the next attempt asks again.
    with pytest.raises(RefusalError):
        await gw.create(request(), stage="read")
    assert len(fake.requests) == 2


async def test_fallback_serves_refusals_and_is_priced_by_serving_model(
    store: Store, clock: ManualClock
) -> None:
    def responder(params: dict[str, Any]) -> FakeReply:
        if params["model"] == "claude-opus-5-5":
            return FakeReply.refusal("cyber")
        return FakeReply.text("served by the fallback")

    fake = FakeAnthropic([responder], clock=clock)
    gw = gateway(fake, store, clock, pipeline={"fallbacks": True})

    reply = await gw.create(request(), stage="read")

    assert isinstance(reply, BetaMessage)
    assert reply.model == "claude-opus-5"
    assert reply.stop_reason == "end_turn"
    sent = fake.requests[0]
    assert sent["fallbacks"] == "default"
    [call] = store.calls()
    assert call.model == "claude-opus-5"
    assert call.cost_usd == pytest.approx(cost(reply.usage, "claude-opus-5"))


async def test_fallbacks_are_not_sent_for_models_without_them(
    fake: FakeAnthropic, store: Store, clock: ManualClock
) -> None:
    gw = gateway(fake, store, clock, pipeline={"fallbacks": True})
    reply = await gw.create(request(model="claude-haiku-5-5"), stage="judge")
    assert isinstance(reply, Message)
    assert "fallbacks" not in fake.requests[0]


def test_unpriced_model_is_a_configuration_error(
    fake: FakeAnthropic, store: Store, clock: ManualClock
) -> None:
    with pytest.raises(UnpricedModelError, match="claude-unknown"):
        gateway(fake, store, clock, agents={"reader": {"model": "claude-unknown"}})
