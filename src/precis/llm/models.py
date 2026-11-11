"""Model facts the runtime depends on (current Claude models, November 2026).

Source: the Anthropic model migration guide's refusal sections. Server-side
fallbacks (``fallbacks="default"`` under the ``server-side-fallback-2026-07-01``
beta) are available for the models below; the API picks the target by
refusal category, so ``FALLBACK_TARGETS`` lists every model a request may be
served by, which is what pricing needs to know in advance. Claude Haiku 5.5
has no server-side fallback, and the Batches API rejects the parameter.
"""

from __future__ import annotations

FALLBACK_BETA = "server-side-fallback-2026-07-01"

FALLBACK_TARGETS: dict[str, tuple[str, ...]] = {
    "claude-fable-5-2": ("claude-opus-4-8", "claude-opus-5"),
    "claude-fable-5-1": ("claude-opus-4-8", "claude-opus-5"),
    "claude-opus-5-5": ("claude-opus-5", "claude-opus-4-8"),
    "claude-opus-5": ("claude-opus-4-8",),
    "claude-sonnet-5-5": ("claude-sonnet-5",),
}


def supports_fallbacks(model: str) -> bool:
    return model in FALLBACK_TARGETS
