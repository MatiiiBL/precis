"""Single-shot requests whose answer is a JSON object of a known shape.

Naming a topic, writing a synthesis and judging a summary are each one
request with no tools, so they use structured outputs
(``output_config.format``): the API constrains the reply to the schema, and
the reply is still validated here because a response cut off by
``max_tokens`` or a refusal carries no complete object.

No ``cache_control`` is sent. Every one of these requests has a unique body
after a short system prompt (below the minimum cacheable prefix), so a cache
breakpoint would only add the cache-write premium without a later read.
"""

from __future__ import annotations

from typing import Any

from anthropic import transform_schema
from pydantic import BaseModel, ValidationError

from precis.config import AgentConfig
from precis.llm.base import ModelGateway


class StructuredOutputError(RuntimeError):
    """The model's reply did not contain a complete object of the requested shape."""


def structured_params[T: BaseModel](
    agent: AgentConfig, *, system: str, user: str, output: type[T]
) -> dict[str, Any]:
    """Build the ``messages.create`` arguments for one structured-output request.

    Thinking stays at the model's adaptive default; ``agent.effort`` sets its
    depth. The prompt text must be deterministic, because the gateway caches
    responses by a hash of these parameters.
    """
    return {
        "model": agent.model,
        "max_tokens": agent.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "effort": agent.effort,
            "format": {"type": "json_schema", "schema": transform_schema(output)},
        },
    }


async def request_structured[T: BaseModel](
    gw: ModelGateway,
    params: dict[str, Any],
    output: type[T],
    *,
    stage: str,
    paper_id: str | None = None,
) -> T:
    """Send ``params`` and parse the reply's text as ``output``.

    Raises ``StructuredOutputError`` when the reply stopped early or does not
    validate; gateway errors (refusals, budget, exhausted retries) propagate.
    """
    reply = await gw.create(params, stage=stage, paper_id=paper_id)
    if reply.stop_reason != "end_turn":
        raise StructuredOutputError(
            f"{stage} request stopped with {reply.stop_reason!r} before a complete answer"
        )
    text = "".join(block.text for block in reply.content if block.type == "text")
    try:
        return output.model_validate_json(text)
    except ValidationError as exc:
        raise StructuredOutputError(f"{stage} reply does not match {output.__name__}") from exc
