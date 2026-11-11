"""Exceptions raised by ``Gateway.create``.

Callers catch ``GatewayError`` to mark a stage failed; ``RefusalError`` and
``BudgetExceeded`` are subclasses because they call for different handling
(mark the stage ``refused``; stop admitting new papers).
"""

from __future__ import annotations

from precis.llm.base import Reply


class GatewayError(Exception):
    """A request failed: a non-retryable error, or retries ran out.

    ``attempts`` is how many times the request was sent before giving up.
    """

    def __init__(self, message: str, *, attempts: int = 0) -> None:
        super().__init__(message)
        self.attempts = attempts


class RefusalError(GatewayError):
    """The model (and any fallback model) declined the request (``stop_reason == "refusal"``).

    ``category`` is ``stop_details.category`` when the API gave one
    (``"cyber"``, ``"bio"``, ``"reasoning_extraction"``, ...), else ``None``.
    """

    def __init__(self, reply: Reply) -> None:
        details = reply.stop_details
        self.reply = reply
        self.category: str | None = details.category if details is not None else None
        super().__init__(f"{reply.model} refused the request (category: {self.category})")


class BudgetExceeded(GatewayError):
    """Sending this request would take spend past ``budget.max_cost_usd``."""
