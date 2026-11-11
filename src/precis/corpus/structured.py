"""Single-shot requests whose answer is a JSON object of a known shape.

Naming a topic, writing a synthesis and judging a summary are each one
request with no tools, so they use structured outputs
(``output_config.format``) with the same strict schemas
(``precis.agents.schema.strict_schema``) and the same sending rules
(``precis.agents.messages.send``: a reply cut off at ``max_tokens`` is
retried once with twice the limit, a refusal raises) as the agents.

No ``cache_control`` is sent. Every one of these requests has a unique body
after a short system prompt (below the minimum cacheable prefix), so a cache
breakpoint would only add the cache-write premium without a later read.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from precis.agents.messages import reply_text, send
from precis.agents.schema import strict_schema
from precis.agents.trace import AgentTrace
from precis.config import AgentConfig
from precis.llm.base import ModelGateway


class StructuredOutputError(RuntimeError):
    """The model's reply did not contain a valid object of the requested shape."""


def structured_params[T: BaseModel](
    agent: AgentConfig, *, system: str, user: str, output: type[T]
) -> dict[str, Any]:
    """Build the ``messages.create`` arguments for one structured-output request.

    Thinking is adaptive and ``agent.effort`` sets its depth. The prompt text
    must be deterministic, because the gateway caches responses by a hash of
    these parameters.
    """
    return {
        "model": agent.model,
        "max_tokens": agent.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": agent.effort,
            "format": {"type": "json_schema", "schema": strict_schema(output)},
        },
    }


async def request_structured[T: BaseModel](
    gw: ModelGateway,
    params: dict[str, Any],
    output: type[T],
    *,
    stage: str,
    paper_id: str | None = None,
    trace: AgentTrace | None = None,
) -> T:
    """Send ``params`` and parse the reply's text as ``output``.

    ``trace``, when given, accumulates the usage of every request sent.
    Raises ``StructuredOutputError`` when the reply does not validate;
    ``AgentError`` (truncated twice), ``AgentRefused`` and gateway errors
    propagate.
    """
    reply = await send(gw, params, stage=stage, paper_id=paper_id, trace=trace or AgentTrace())
    if reply.stop_reason not in ("end_turn", "stop_sequence"):
        raise StructuredOutputError(f"{stage} reply stopped with {reply.stop_reason!r}")
    try:
        return output.model_validate(json.loads(reply_text(reply)))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise StructuredOutputError(f"{stage} reply does not match {output.__name__}") from exc
