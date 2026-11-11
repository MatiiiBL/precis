"""The client behind ``--offline``: a ``FakeAnthropic`` that plays every role.

It collects the deterministic responders of every package that makes
requests: the per-paper agents (``precis.agents.offline``: reader, critic,
reviser, and the fulltext and mapreduce strategies), the corpus steps
(``precis.corpus.offline``: topic labels, syntheses, the report
introduction) and the evaluation judge (``precis.eval.offline``).
``precis run``, the corpus commands and ``precis eval`` all use it. A request
no responder recognizes raises ``LookupError`` in the fake, which surfaces as
a failed stage rather than a silent wrong answer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from anthropic.types import Message

from precis.agents import offline as agents_offline
from precis.corpus import offline as corpus_offline
from precis.eval import offline as eval_offline
from precis.llm import Clock
from precis.llm.fake import FakeAnthropic, FakeReply, Responder


def from_messages(responder: Callable[[dict[str, Any]], Message | None]) -> Responder:
    """Adapt a responder that builds a whole ``Message`` to the fake's form."""

    def respond(params: dict[str, Any]) -> FakeReply | None:
        message = responder(params)
        return None if message is None else FakeReply(message=message)

    return respond


RESPONDERS: list[Responder] = [
    *(from_messages(r) for r in agents_offline.RESPONDERS),
    *corpus_offline.RESPONDERS,
    *eval_offline.RESPONDERS,
]
"""Every package's offline responders, in the order they are tried."""


def offline_client(*, clock: Clock | None = None, seed: int = 0) -> FakeAnthropic:
    """A client that answers every request ``precis`` makes, without the network."""
    return FakeAnthropic(RESPONDERS, clock=clock, seed=seed)
