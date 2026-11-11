"""Prices of Claude models and the cost of a response.

Source: Anthropic's pricing page,
https://platform.claude.com/docs/en/about-claude/pricing (model pricing,
prompt caching, batch processing and long context pricing sections),
retrieved 2026-11-11. Prices are USD per million tokens on the Claude API
with global routing; data residency and fast mode multipliers are not
modelled because the pipeline uses neither.

Cache reads are 0.1x the input price on most models, 0.05x on Claude
Opus 5.5 and Claude Sonnet 5.5 and 0.025x on Claude Fable 5.2 / 5.1; cache
writes are 1.25x (5-minute TTL) and 2x (1-hour TTL). The Batch API halves
every one of these, and the discounts stack. Claude Haiku 5.5 is the one
current model priced by prompt length: a request whose prompt (uncached
input, cache writes and cache reads together) is over 100,000 tokens pays
the higher tier for all of its tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from anthropic.types.beta import BetaUsage

from precis.llm.base import Reply

BATCH_DISCOUNT = 0.5
"""Multiplier applied to every token price for Message Batches API requests."""


@dataclass(frozen=True)
class Price:
    """USD per million tokens for each kind of token a response bills."""

    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


@dataclass(frozen=True)
class ModelPricing:
    """A model's prices, with an optional higher tier for long prompts."""

    base: Price
    long_prompt: Price | None = None
    long_prompt_threshold: int = 100_000

    def for_prompt(self, prompt_tokens: int) -> Price:
        if self.long_prompt is not None and prompt_tokens > self.long_prompt_threshold:
            return self.long_prompt
        return self.base


def _p(inp: float, out: float, write_5m: float, write_1h: float, read: float) -> ModelPricing:
    return ModelPricing(Price(inp, out, write_5m, write_1h, read))


PRICES: dict[str, ModelPricing] = {
    "claude-fable-5-2": _p(8.00, 40.00, 10.00, 16.00, 0.20),
    "claude-fable-5-1": _p(10.00, 50.00, 12.50, 20.00, 0.25),
    "claude-fable-5": _p(10.00, 50.00, 12.50, 20.00, 1.00),
    "claude-opus-5-5": _p(4.00, 20.00, 5.00, 8.00, 0.20),
    "claude-opus-5": _p(5.00, 25.00, 6.25, 10.00, 0.50),
    "claude-opus-4-8": _p(5.00, 25.00, 6.25, 10.00, 0.50),
    "claude-opus-4-7": _p(5.00, 25.00, 6.25, 10.00, 0.50),
    "claude-opus-4-6": _p(5.00, 25.00, 6.25, 10.00, 0.50),
    "claude-sonnet-5-5": _p(2.00, 10.00, 2.50, 4.00, 0.10),
    "claude-sonnet-5": _p(2.00, 10.00, 2.50, 4.00, 0.20),
    "claude-sonnet-4-6": _p(3.00, 15.00, 3.75, 6.00, 0.30),
    "claude-haiku-5-5": ModelPricing(
        base=Price(0.10, 0.50, 0.125, 0.20, 0.01),
        long_prompt=Price(0.50, 2.50, 0.625, 1.00, 0.05),
    ),
    "claude-haiku-4-5": _p(1.00, 5.00, 1.25, 2.00, 0.10),
}


class UnpricedModelError(KeyError):
    """A model with no entry in ``PRICES``: a configuration error, never a free call."""


class CacheCreationLike(Protocol):
    @property
    def ephemeral_1h_input_tokens(self) -> int: ...


class UsageLike(Protocol):
    """The ``usage`` fields of ``Message`` and ``BetaMessage`` that are billed."""

    @property
    def input_tokens(self) -> int: ...
    @property
    def output_tokens(self) -> int: ...
    @property
    def cache_creation_input_tokens(self) -> int | None: ...
    @property
    def cache_read_input_tokens(self) -> int | None: ...
    @property
    def cache_creation(self) -> CacheCreationLike | None: ...


def pricing(model: str) -> ModelPricing:
    try:
        return PRICES[model]
    except KeyError:
        raise UnpricedModelError(
            f"no price for model {model!r}; add it to precis.llm.pricing.PRICES"
        ) from None


def check_priced(models: set[str] | list[str] | tuple[str, ...]) -> None:
    """Raise ``UnpricedModelError`` naming every model in ``models`` that has no price."""
    missing = sorted(m for m in models if m not in PRICES)
    if missing:
        raise UnpricedModelError(
            f"no price for model(s) {', '.join(missing)}; add them to precis.llm.pricing.PRICES"
        )


def cost(usage: UsageLike, model: str, *, batch: bool = False) -> float:
    """USD cost of one response.

    ``model`` must be the model that served the response (``reply.model``),
    which after a server-side fallback differs from the one requested.
    Cache writes are priced at the 5-minute rate except the part that
    ``usage.cache_creation`` reports as written to the 1-hour cache.
    """
    cache_write = usage.cache_creation_input_tokens or 0
    cache_read = usage.cache_read_input_tokens or 0
    breakdown = usage.cache_creation
    write_1h = min(breakdown.ephemeral_1h_input_tokens, cache_write) if breakdown else 0
    prompt_tokens = usage.input_tokens + cache_write + cache_read
    price = pricing(model).for_prompt(prompt_tokens)
    usd = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + (cache_write - write_1h) * price.cache_write_5m
        + write_1h * price.cache_write_1h
        + cache_read * price.cache_read
    ) / 1_000_000
    return usd * BATCH_DISCOUNT if batch else usd


def reply_cost(reply: Reply, *, batch: bool = False) -> float:
    """USD cost of a reply, every attempt behind it included.

    When ``usage.iterations`` is present (a server-side fallback, a server
    tool loop) each entry is priced at its own model and they are summed;
    top-level ``usage`` covers only the attempt that produced the returned
    message, so pricing it alone would leave out a declined attempt.
    Entries without a model ran on the model that served the reply.
    """
    usage = reply.usage
    if isinstance(usage, BetaUsage) and usage.iterations:
        return sum(
            cost(entry, getattr(entry, "model", None) or reply.model, batch=batch)
            for entry in usage.iterations
        )
    return cost(usage, reply.model, batch=batch)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int = 0,
    batch: bool = False,
) -> float:
    """Expected cost of a request before it is sent, from token estimates."""
    price = pricing(model).for_prompt(input_tokens + cache_read_tokens)
    usd = (
        input_tokens * price.input
        + cache_read_tokens * price.cache_read
        + output_tokens * price.output
    ) / 1_000_000
    return usd * BATCH_DISCOUNT if batch else usd
