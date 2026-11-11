"""The slice of the Anthropic SDK client that the gateway uses, as a protocol.

``anthropic.AsyncAnthropic`` and ``precis.llm.fake.FakeAnthropic`` both
satisfy ``AsyncAnthropicLike`` (mypy checks it), so the gateway is typed
against exactly what it calls and nothing else.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable, Mapping
from types import TracebackType
from typing import Any, Protocol

import anthropic
from anthropic.types import Message
from anthropic.types.beta import BetaMessage
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from precis.config import Config


class ResponseLike(Protocol):
    @property
    def headers(self) -> Mapping[str, str]: ...


class MessageStreamLike[T](Protocol):
    """An open streamed response: its HTTP headers, and the message once it ends."""

    @property
    def response(self) -> ResponseLike: ...

    async def get_final_message(self) -> T: ...


class MessageStreamManagerLike[T](Protocol):
    """``messages.stream(...)``: entering the block sends the request."""

    async def __aenter__(self) -> MessageStreamLike[T]: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
        /,
    ) -> object: ...


class BatchesLike(Protocol):
    async def create(self, *, requests: Iterable[Any]) -> MessageBatch: ...

    async def retrieve(self, message_batch_id: str) -> MessageBatch: ...

    async def results(
        self, message_batch_id: str
    ) -> AsyncIterable[MessageBatchIndividualResponse]: ...


class MessagesLike(Protocol):
    def stream(self, *args: Any, **params: Any) -> MessageStreamManagerLike[Message]: ...

    @property
    def batches(self) -> BatchesLike: ...


class BetaMessagesLike(Protocol):
    def stream(self, *args: Any, **params: Any) -> MessageStreamManagerLike[BetaMessage]: ...


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
    would happen behind its back. Requests are streamed, so the timeout
    bounds the wait for each chunk, not the whole generation. Credentials
    come from the environment as usual (``ANTHROPIC_API_KEY``,
    ``ANTHROPIC_AUTH_TOKEN`` or a profile).
    """
    return anthropic.AsyncAnthropic(max_retries=0, timeout=config.limits.request_timeout_s)
