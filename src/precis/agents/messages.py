"""Building requests and reading replies, shared by every agent and strategy.

All requests share one shape: the role's system prompt as a cached block
(identical for every paper, so its cache entry is reused across papers),
adaptive thinking with the configured effort, and no ``temperature`` or
forced ``tool_choice``, which current models reject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anthropic.types import ToolUseBlock
from anthropic.types.beta import BetaToolUseBlock

from precis.agents.errors import AgentError, AgentRefused
from precis.agents.prompts import Prompt
from precis.agents.trace import AgentTrace
from precis.config import AgentConfig, CacheTTL
from precis.llm import ModelGateway, Reply


def cache_control(ttl: CacheTTL) -> dict[str, str]:
    """A cache breakpoint; the 5-minute lifetime is the API default and is left implicit."""
    return {"type": "ephemeral", "ttl": "1h"} if ttl == "1h" else {"type": "ephemeral"}


def base_params(
    config: AgentConfig, system: Prompt, messages: list[dict[str, Any]], *, ttl: CacheTTL
) -> dict[str, Any]:
    """Parameters every request starts from; callers add tools or an output format."""
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": [{"type": "text", "text": system.text, "cache_control": cache_control(ttl)}],
        "messages": list(messages),
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": config.effort},
    }


def assistant_turn(reply: Reply) -> dict[str, Any]:
    """The reply as the next request's assistant message, every block included.

    Thinking blocks must go back exactly as they came, so nothing is dropped
    or edited. Blocks are serialized to plain JSON (``None`` fields omitted)
    rather than passed as SDK objects, so that a reply served live and the
    same reply restored from the response cache produce identical request
    bytes, and therefore identical request hashes, on the next turn.
    """
    return {
        "role": "assistant",
        "content": [
            block.to_dict(mode="json", exclude_unset=False, exclude_none=True)
            for block in reply.content
        ],
    }


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


def tool_calls(reply: Reply) -> list[ToolCall]:
    return [
        ToolCall(block.id, block.name, dict(block.input))
        for block in reply.content
        if isinstance(block, ToolUseBlock | BetaToolUseBlock)
    ]


def reply_text(reply: Reply) -> str:
    return "".join(block.text for block in reply.content if block.type == "text")


def tool_result(call_id: str, text: str, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": call_id, "content": text}
    if is_error:
        result["is_error"] = True
    return result


async def send(
    gw: ModelGateway,
    params: dict[str, Any],
    *,
    stage: str,
    paper_id: str,
    trace: AgentTrace,
) -> Reply:
    """Send one request, handling the stop reasons no caller can work with.

    A refusal raises ``AgentRefused``. A reply cut off at ``max_tokens`` is
    discarded, because a truncated tool call or JSON document is unusable,
    and the request is sent once more with twice the limit; thinking counts
    toward ``max_tokens``, so an unusually long deliberation is the usual
    cause. A second truncation raises ``AgentError``.
    """
    reply = await gw.create(params, stage=stage, paper_id=paper_id)
    trace.add(reply)
    if reply.stop_reason == "max_tokens":
        trace.max_tokens_retries += 1
        params = {**params, "max_tokens": params["max_tokens"] * 2}
        reply = await gw.create(params, stage=stage, paper_id=paper_id)
        trace.add(reply)
    if reply.stop_reason == "refusal":
        details = reply.stop_details
        raise AgentRefused(
            details.category if details else None, details.explanation if details else None
        )
    if reply.stop_reason == "max_tokens":
        raise AgentError(f"the reply was cut off at max_tokens={params['max_tokens']} twice")
    return reply
