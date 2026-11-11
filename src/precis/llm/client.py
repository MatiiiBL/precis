"""The slice of the Anthropic SDK client that the gateway uses, as a protocol.

``anthropic.AsyncAnthropic`` and ``precis.llm.fake.FakeAnthropic`` both
satisfy ``AsyncAnthropicLike`` (mypy checks it), so the gateway is typed
against exactly what it calls and nothing else.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Iterable, Mapping
from typing import Any, Protocol

import anthropic
from anthropic.types import Message
from anthropic.types.beta import BetaMessage
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from precis.config import Config


class RawResponse[T](Protocol):
    """``with_raw_response.create``'s result: headers, and the parsed message."""

    @property
    def headers(self) -> Mapping[str, str]: ...

    async def parse(self) -> T: ...


class RawMessagesLike(Protocol):
    def create(self, *args: Any, **params: Any) -> Awaitable[RawResponse[Message]]: ...


class RawBetaMessagesLike(Protocol):
    def create(self, *args: Any, **params: Any) -> Awaitable[RawResponse[BetaMessage]]: ...


class BatchesLike(Protocol):
    async def create(self, *, requests: Iterable[Any]) -> MessageBatch: ...

    async def retrieve(self, message_batch_id: str) -> MessageBatch: ...

    async def results(
        self, message_batch_id: str
    ) -> AsyncIterable[MessageBatchIndividualResponse]: ...


class MessagesLike(Protocol):
    @property
    def with_raw_response(self) -> RawMessagesLike: ...

    @property
    def batches(self) -> BatchesLike: ...


class BetaMessagesLike(Protocol):
    @property
    def with_raw_response(self) -> RawBetaMessagesLike: ...


class BetaLike(Protocol):
    @property
    def messages(self) -> BetaMessagesLike: ...


class AsyncAnthropicLike(Protocol):
    @property
    def messages(self) -> MessagesLike: ...

    @property
    def beta(self) -> BetaLike: ...


def make_client(config: Config) -> AsyncAnthropicLike:
    """The real async SDK client, with the SDK's own retries turned off.

    The gateway retries every failed attempt itself so that each one goes
    through admission control and shows up in the accounting; SDK retries
    would happen behind its back. Credentials come from the environment as
    usual (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or a profile).
    """
    return anthropic.AsyncAnthropic(max_retries=0, timeout=config.limits.request_timeout_s)
