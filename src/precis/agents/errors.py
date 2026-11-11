"""Ways an agent run can end without a usable result.

The scheduler records an ``AgentRefused`` as a ``refused`` stage and every
other ``AgentError`` as ``failed``, with the exception text as the error.
"""

from __future__ import annotations


class AgentError(Exception):
    """The agent stopped without producing a valid result."""


class AgentRefused(AgentError):
    """The model declined the request (``stop_reason == "refusal"``).

    The gateway raises its own refusal error when server-side fallbacks also
    decline; this one covers replies that reach the agent with the refusal
    still in them, for example when fallbacks are switched off.
    """

    def __init__(self, category: str | None, explanation: str | None = None) -> None:
        self.category = category
        self.explanation = explanation
        detail = f" ({category})" if category else ""
        super().__init__(
            f"model refused the request{detail}" + (f": {explanation}" if explanation else "")
        )
