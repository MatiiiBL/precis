from typing import Any

import anthropic
import pytest
from anthropic.types import Message

from precis.llm.clock import ManualClock
from precis.llm.fake import FakeAnthropic, FakeReply, Faults, LatencyModel, ServerLimits
from precis.util import estimate_tokens

from .helpers import echo, request

LONG = "word " * 1000  # ~1,250 tokens: above the 512-token cacheable minimum


def _message(**usage: int) -> Message:
    return Message.model_validate(
        {
            "id": "msg_scripted",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "scripted " * 40}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, **usage},
        }
    )


async def test_reply_is_a_real_message_with_estimated_usage(fake: FakeAnthropic) -> None:
    reply = await fake.messages.create(**request("x" * 400))
    assert isinstance(reply, Message)
    assert reply.usage.input_tokens == estimate_tokens("x" * 400)
    assert reply.usage.output_tokens == estimate_tokens("echo: " + "x" * 400)
    assert reply.id.startswith("msg_fake_")


async def test_complete_message_keeps_given_usage_or_gets_estimated(clock: ManualClock) -> None:
    given = _message(input_tokens=11, output_tokens=7)
    fake = FakeAnthropic([lambda p: FakeReply(message=given)], clock=clock)
    assert (await fake.messages.create(**request())).usage.output_tokens == 7

    fake = FakeAnthropic([lambda p: FakeReply(message=_message())], clock=clock)
    reply = await fake.messages.create(**request())
    assert reply.id == "msg_scripted"
    assert reply.usage.output_tokens == estimate_tokens("scripted " * 40)
    assert reply.usage.input_tokens > 0


async def test_tool_use_blocks_get_ids_and_output_is_capped(clock: ManualClock) -> None:
    fake = FakeAnthropic(
        [lambda p: FakeReply.tool_use("search_paper", {"query": "cache " * 200})], clock=clock
    )
    reply = await fake.messages.create(**request(max_tokens=20))
    block = reply.content[0]
    assert block.type == "tool_use"
    assert block.id.startswith("toolu_fake_")
    assert reply.stop_reason == "max_tokens"
    assert reply.usage.output_tokens == 20


async def test_first_matching_responder_wins_and_unmatched_requests_fail(
    clock: ManualClock,
) -> None:
    def only_haiku(params: dict[str, Any]) -> FakeReply | None:
        return FakeReply.text("haiku") if params["model"] == "claude-haiku-5-5" else None

    fake = FakeAnthropic([only_haiku, echo], clock=clock)
    reply = await fake.messages.create(**request(model="claude-haiku-5-5"))
    assert reply.content[0].type == "text"
    assert reply.content[0].text == "haiku"

    with pytest.raises(LookupError):
        await FakeAnthropic([only_haiku], clock=clock).messages.create(**request())


async def test_prompt_caching_reads_the_prefix_written_before(fake: FakeAnthropic) -> None:
    system = [{"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}}]
    first = await fake.messages.create(**request("question one", system=system))
    second = await fake.messages.create(**request("question two", system=system))

    prefix = estimate_tokens(LONG)
    assert first.usage.cache_creation_input_tokens == prefix
    assert first.usage.cache_read_input_tokens == 0
    assert second.usage.cache_read_input_tokens == prefix
    assert second.usage.cache_creation_input_tokens == 0
    assert second.usage.input_tokens == estimate_tokens("question two")


async def test_automatic_caching_grows_with_the_conversation(fake: FakeAnthropic) -> None:
    turn1 = [{"role": "user", "content": LONG}]
    first = await fake.messages.create(
        **request(messages=turn1, cache_control={"type": "ephemeral"})
    )
    turn2 = [*turn1, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    second = await fake.messages.create(
        **request(messages=turn2, cache_control={"type": "ephemeral"})
    )
    assert first.usage.cache_creation_input_tokens == estimate_tokens(LONG)
    assert second.usage.cache_read_input_tokens == estimate_tokens(LONG)
    assert second.usage.cache_creation_input_tokens == estimate_tokens("ok") + estimate_tokens(
        "more"
    )


async def test_cache_entries_expire(fake: FakeAnthropic, clock: ManualClock) -> None:
    system = [{"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}}]
    await fake.messages.create(**request(system=system))
    clock.advance(301)
    again = await fake.messages.create(**request(system=system))
    assert again.usage.cache_read_input_tokens == 0


async def test_short_prefixes_are_not_cached(fake: FakeAnthropic) -> None:
    system = [{"type": "text", "text": "short", "cache_control": {"type": "ephemeral"}}]
    reply = await fake.messages.create(**request(system=system))
    assert reply.usage.cache_creation_input_tokens == 0


async def test_server_limits_answer_429_with_headers(clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, limits=ServerLimits(rpm=2, itpm=10**6, otpm=10**6))
    raw = await fake.messages.with_raw_response.create(**request())
    assert raw.headers["anthropic-ratelimit-requests-limit"] == "2"
    assert raw.headers["anthropic-ratelimit-requests-remaining"] == "1"
    assert "anthropic-ratelimit-input-tokens-reset" in raw.headers
    await fake.messages.create(**request())

    with pytest.raises(anthropic.RateLimitError) as excinfo:
        await fake.messages.create(**request())
    headers = excinfo.value.response.headers
    assert headers["retry-after"] == "30"
    assert headers["anthropic-ratelimit-requests-remaining"] == "0"

    clock.advance(30)
    await fake.messages.create(**request())


async def test_input_tokens_limit(clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, limits=ServerLimits(rpm=100, itpm=600, otpm=10**6))
    await fake.messages.create(**request("x" * 2000))  # 500 tokens
    with pytest.raises(anthropic.RateLimitError):
        await fake.messages.create(**request("x" * 2000))


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("rate_limit", anthropic.RateLimitError),
        ("overloaded", anthropic.OverloadedError),
        ("server_error", anthropic.InternalServerError),
        ("timeout", anthropic.APITimeoutError),
        ("bad_request", anthropic.BadRequestError),
    ],
)
async def test_scripted_faults_raise_sdk_errors(
    fake: FakeAnthropic, kind: Any, error: type[Exception]
) -> None:
    fake.fail_next(kind)
    with pytest.raises(error):
        await fake.messages.create(**request())
    await fake.messages.create(**request())


@pytest.mark.parametrize(
    "extra",
    [
        {"temperature": 0.2},
        {"thinking": {"type": "enabled", "budget_tokens": 2000}},
        {"thinking": {"type": "disabled"}},
        {"tool_choice": {"type": "any"}},
    ],
)
async def test_rejects_request_shapes_current_models_reject(
    fake: FakeAnthropic, extra: dict[str, Any]
) -> None:
    with pytest.raises(anthropic.BadRequestError):
        await fake.messages.create(**request(**extra))


async def test_unknown_model_is_404(fake: FakeAnthropic) -> None:
    with pytest.raises(anthropic.NotFoundError):
        await fake.messages.create(**request(model="claude-nope"))


async def test_random_faults_are_reproducible_under_a_seed(clock: ManualClock) -> None:
    async def outcomes(seed: int) -> list[str]:
        fake = FakeAnthropic([echo], clock=clock, seed=seed, faults=Faults(server_error=0.3))
        result = []
        for _ in range(30):
            try:
                await fake.messages.create(**request())
                result.append("ok")
            except anthropic.InternalServerError:
                result.append("500")
        return result

    first = await outcomes(7)
    assert first == await outcomes(7)
    assert "500" in first
    assert "ok" in first


async def test_latency_model_uses_the_clock(clock: ManualClock) -> None:
    latency = LatencyModel(first_token_s=1.0, seconds_per_output_token=0.5, jitter=0)
    fake = FakeAnthropic([lambda p: FakeReply.text("abcd" * 4)], clock=clock, latency=latency)
    await fake.messages.create(**request())
    assert clock.now() == pytest.approx(1.0 + 4 * 0.5)


async def test_beta_fallback_reports_the_switch(clock: ManualClock) -> None:
    def responder(params: dict[str, Any]) -> FakeReply:
        if params["model"] == "claude-opus-5-5":
            return FakeReply.refusal("cyber")
        return FakeReply.text("fine")

    fake = FakeAnthropic([responder], clock=clock)
    reply = await fake.beta.messages.create(
        **request(fallbacks="default", betas=["server-side-fallback-2026-07-01"])
    )
    assert reply.model == "claude-opus-5"
    assert reply.content[0].type == "fallback"
    iterations = reply.usage.iterations or []
    assert [i.type for i in iterations] == ["message", "fallback_message"]

    refused = await fake.beta.messages.create(
        **request(model="claude-opus-5-5", betas=["server-side-fallback-2026-07-01"])
    )
    assert refused.stop_reason == "refusal"


async def test_batches_end_after_their_latency(clock: ManualClock) -> None:
    fake = FakeAnthropic([echo], clock=clock, batch_latency_s=60)
    batch = await fake.messages.batches.create(
        requests=[
            {"custom_id": "a", "params": request("one")},
            {"custom_id": "b", "params": request("two", temperature=1.0)},
        ]
    )
    assert batch.processing_status == "in_progress"
    assert batch.request_counts.processing == 2

    clock.advance(60)
    batch = await fake.messages.batches.retrieve(batch.id)
    assert batch.processing_status == "ended"
    results = {r.custom_id: r.result async for r in await fake.messages.batches.results(batch.id)}
    assert results["a"].type == "succeeded"
    assert results["b"].type == "errored"
    assert results["b"].error.error.type == "invalid_request_error"
    assert (batch.request_counts.succeeded, batch.request_counts.errored) == (1, 1)
