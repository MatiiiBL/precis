"""The client behind ``--offline``: a ``FakeAnthropic`` that plays every role.

It collects the deterministic responders of every package that makes
requests: the per-paper agents (``precis.agents.offline``: reader, critic,
reviser, and the fulltext and mapreduce strategies), the corpus steps
(``precis.corpus.offline``: topic labels, syntheses, the report
introduction) and the evaluation judge (``precis.eval.offline``).
``precis run``, the corpus commands and ``precis eval`` all use it. A request
no responder recognizes raises ``LookupError`` in the fake, which surfaces as
a failed stage rather than a silent wrong answer.

Offline results share the store with real ones but never pass for them: the
gateway caches offline replies under their own keys and records offline
calls apart from spend, and offline stages hash apart, so a run against the
API redoes them. The other way round is refused (``offline_conflict``): an
offline ``run``, ``topics`` or ``report`` would replace results that were
paid for.
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
from precis.store import Store


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


def offline_conflict(store: Store) -> str | None:
    """Why an offline command must not write its results into ``store``, or ``None``.

    An offline run replaces the stage records, and offline ``topics`` and
    ``report`` the corpus artifacts, that a run against the API produced.
    """
    paid = store.paid_calls()
    if not paid:
        return None
    return (
        f"this store holds results paid for with the API ({paid:,} requests), and offline "
        "results would replace them. Copy the store and run offline on the copy "
        "(precis --store <copy> ...)."
    )


def spend_phrase(usd: float, *, offline: bool, digits: int = 2) -> str:
    """How a command reports its cost: offline, nothing was spent."""
    amount = f"${usd:.{digits}f}"
    return f"{amount} at API prices (offline: nothing spent)" if offline else f"spent {amount}"
