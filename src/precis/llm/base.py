"""The interface that agent and corpus code use to reach a model.

Code that writes prompts depends only on this protocol, never on the SDK
client or on ``Gateway`` directly, so it can be tested against a scripted
stand-in and run unchanged online, through batches or offline.
"""

from __future__ import annotations

from typing import Any, Protocol

from anthropic.types import Message
from anthropic.types.beta import BetaMessage

# Requests sent with server-side fallbacks go through the beta endpoint and
# come back as BetaMessage; both expose the same content blocks, stop_reason
# and usage fields that callers read.
type Reply = Message | BetaMessage


class ModelGateway(Protocol):
    async def create(
        self, params: dict[str, Any], *, stage: str, paper_id: str | None = None
    ) -> Reply:
        """Send one Messages API request.

        ``params`` are the keyword arguments of ``client.messages.create``;
        ``stage`` and ``paper_id`` label the request for accounting.
        """
        ...
