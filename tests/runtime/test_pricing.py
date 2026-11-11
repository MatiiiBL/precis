import pytest
from anthropic.types import Usage
from anthropic.types.beta import BetaMessage

from precis.config import DEFAULT_MODEL
from precis.llm.pricing import (
    BATCH_DISCOUNT,
    PRICES,
    UnpricedModelError,
    check_priced,
    cost,
    estimate_cost,
    reply_cost,
)

REQUIRED = [
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-2",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-haiku-4-5",
]


def usage(inp: int = 0, out: int = 0, write: int = 0, read: int = 0, write_1h: int = 0) -> Usage:
    return Usage.model_validate(
        {
            "input_tokens": inp,
            "output_tokens": out,
            "cache_creation_input_tokens": write,
            "cache_read_input_tokens": read,
            "cache_creation": {
                "ephemeral_5m_input_tokens": write - write_1h,
                "ephemeral_1h_input_tokens": write_1h,
            },
        }
    )


def test_current_models_are_priced() -> None:
    check_priced([*REQUIRED, DEFAULT_MODEL])
    with pytest.raises(UnpricedModelError, match="claude-x"):
        check_priced(["claude-opus-5-5", "claude-x"])


def test_input_and_output_prices() -> None:
    # Claude Opus 5.5: $4 / $20 per million tokens.
    assert cost(usage(inp=1_000_000), "claude-opus-5-5") == pytest.approx(4.0)
    assert cost(usage(out=1_000_000), "claude-opus-5-5") == pytest.approx(20.0)


def test_cache_reads_and_writes_are_priced_separately() -> None:
    # Claude Opus 5.5: 5-minute writes $5, 1-hour writes $8, reads $0.20 per MTok.
    u = usage(inp=100_000, write=200_000, write_1h=50_000, read=1_000_000, out=10_000)
    expected = (100_000 * 4 + 150_000 * 5 + 50_000 * 8 + 1_000_000 * 0.20 + 10_000 * 20) / 1e6
    assert cost(u, "claude-opus-5-5") == pytest.approx(expected)
    # Cache reads are 0.05x input on Opus 5.5 but 0.1x on Opus 5.
    assert cost(usage(read=1_000_000), "claude-opus-5") == pytest.approx(0.50)


def test_batch_discount_halves_everything() -> None:
    u = usage(inp=10_000, out=2_000, write=5_000, read=50_000)
    assert BATCH_DISCOUNT == 0.5
    assert cost(u, "claude-sonnet-5-5", batch=True) == pytest.approx(
        cost(u, "claude-sonnet-5-5") / 2
    )


def test_haiku_5_5_long_prompts_pay_the_higher_tier() -> None:
    short = cost(usage(inp=100_000, out=1_000), "claude-haiku-5-5")
    assert short == pytest.approx((100_000 * 0.10 + 1_000 * 0.50) / 1e6)
    # Cache reads count toward the prompt length that selects the tier.
    long = cost(usage(inp=50_000, read=60_000, out=1_000), "claude-haiku-5-5")
    assert long == pytest.approx((50_000 * 0.50 + 60_000 * 0.05 + 1_000 * 2.50) / 1e6)


def test_price_follows_the_serving_model() -> None:
    u = usage(inp=1_000, out=1_000)
    assert cost(u, "claude-opus-5") != cost(u, "claude-opus-5-5")
    with pytest.raises(UnpricedModelError):
        cost(u, "claude-unknown")


def test_estimate_prices_expected_cache_reads() -> None:
    assert estimate_cost("claude-opus-5-5", 1_000, 1_000) == pytest.approx(
        cost(usage(inp=1_000, out=1_000), "claude-opus-5-5")
    )
    assert set(REQUIRED) <= set(PRICES)


def test_reply_cost_prices_every_iteration_at_its_own_model() -> None:
    def iteration(kind: str, model: str | None, inp: int, out: int) -> dict[str, object]:
        entry: dict[str, object] = {
            "type": kind,
            "input_tokens": inp,
            "output_tokens": out,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        if model is not None:
            entry["model"] = model
        return entry

    def reply(iterations: list[dict[str, object]] | None) -> BetaMessage:
        return BetaMessage.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 500, "iterations": iterations},
            }
        )

    fallback = reply(
        [
            iteration("message", "claude-opus-5-5", 1000, 200),
            iteration("fallback_message", "claude-opus-5", 1000, 500),
        ]
    )
    declined = (1000 * 4 + 200 * 20) / 1e6
    served = (1000 * 5 + 500 * 25) / 1e6
    assert reply_cost(fallback) == pytest.approx(declined + served)
    assert reply_cost(fallback, batch=True) == pytest.approx((declined + served) / 2)
    # Without iterations, or for entries with no model, the serving model prices it.
    assert reply_cost(reply(None)) == pytest.approx(served)
    assert reply_cost(reply([iteration("message", None, 1000, 500)])) == pytest.approx(served)
