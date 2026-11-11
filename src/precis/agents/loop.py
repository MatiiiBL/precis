"""The tool loop the reader, critic and reviser run on.

The loop is written by hand rather than with the SDK's tool runner because
the requests go through ``ModelGateway`` (admission control, accounting,
batching), and because the loop enforces things the runner does not: a turn
and token budget announced to the model before it runs out, a submit tool
whose validation errors go back to the model, and one nudge when the model
ends its turn without submitting.

The conversation is append-only. Assistant turns go back exactly as they
came (thinking included), budget notices are mid-conversation system
messages (a text block after the tool results on models that reject those)
rather than edits to the system prompt, and the tool set never changes;
together with top-level automatic caching this lets every turn read the
previous turn's prefix from the prompt cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from precis.agents.errors import AgentError
from precis.agents.messages import (
    ToolCall,
    assistant_turn,
    base_params,
    cache_control,
    harness_note,
    send,
    tool_calls,
    tool_result,
)
from precis.agents.prompts import Prompt
from precis.agents.submissions import SubmissionRejected, SubmitTool
from precis.agents.tools import READING_TOOL_NAMES, READING_TOOLS, ReadingTools, ToolOutput
from precis.agents.trace import AgentResult, AgentTrace
from precis.config import AgentConfig, CacheTTL
from precis.llm import ModelGateway

log = logging.getLogger(__name__)

# The budget notice goes out when this share of the token budget is spent,
# or when this many turns are left.
NOTICE_FRACTION = 0.8
NOTICE_TURNS_LEFT = 2
# Requests allowed after the budget is used up, to submit or fix a rejected
# submission, before the run fails.
GRACE_TURNS = 2


@dataclass(frozen=True)
class AgentSpec[T]:
    """One role: its prompt, its first message and how it finishes."""

    stage: str
    system: Prompt
    brief: str
    submit: SubmitTool[T]
    finish_hint: str
    """What to do when the budget runs out, e.g. "Call submit_summary with ..."."""
    cache_ttl: CacheTTL


async def run_agent[T](
    gw: ModelGateway,
    spec: AgentSpec[T],
    tools: ReadingTools,
    config: AgentConfig,
    *,
    paper_id: str,
) -> AgentResult[T]:
    """Run ``spec`` until its submit tool accepts a result.

    Raises ``AgentRefused`` when the model declines and ``AgentError`` when
    the run ends without a valid submission: two turns without a tool call,
    or no valid submission within the budget's grace turns.
    """
    trace = AgentTrace()
    definitions = [*READING_TOOLS, spec.submit.definition]
    messages: list[dict[str, Any]] = [{"role": "user", "content": spec.brief}]
    exhausted_after: int | None = None

    while True:
        if exhausted_after is None and _exhausted(trace, config):
            exhausted_after = trace.requests
        if exhausted_after is not None and trace.requests >= exhausted_after + GRACE_TURNS:
            raise AgentError(
                f"no valid {spec.submit.name} call within the budget ({trace.requests} requests, "
                f"{trace.budget_tokens:,} budget tokens)"
            )
        if (
            trace.budget_notices == 0
            and messages[-1]["role"] == "user"
            and _nearly_spent(trace, config)
        ):
            harness_note(config.model, messages, _budget_notice(trace, config, spec))
            trace.budget_notices += 1

        params = base_params(config, spec.system, messages, ttl=spec.cache_ttl)
        params["tools"] = definitions
        params["cache_control"] = cache_control(spec.cache_ttl)
        reply = await send(gw, params, stage=spec.stage, paper_id=paper_id, trace=trace)

        if reply.stop_reason not in ("tool_use", "end_turn", "stop_sequence", "pause_turn"):
            raise AgentError(f"unexpected stop reason {reply.stop_reason!r}")
        messages.append(assistant_turn(reply))
        if reply.stop_reason == "pause_turn":
            continue

        calls = tool_calls(reply)
        if not calls:
            if trace.nudges:
                raise AgentError(
                    f"the model ended its turn twice without calling {spec.submit.name}"
                )
            trace.nudges += 1
            messages.append({"role": "user", "content": _nudge(spec)})
            continue

        results = []
        for call in calls:
            if call.name == spec.submit.name:
                try:
                    value = spec.submit.parse(call.input)
                except SubmissionRejected as rejected:
                    trace.rejected_submissions += 1
                    results.append(tool_result(call.id, str(rejected), is_error=True))
                    continue
                _finish(trace, tools)
                return AgentResult(value, trace)
            output = _use_tool(call, tools, spec, budget_left=exhausted_after is None)
            trace.tool_calls[call.name] += 1
            trace.tool_errors += int(output.is_error)
            results.append(tool_result(call.id, output.text, is_error=output.is_error))
        messages.append({"role": "user", "content": results})


def _use_tool(
    call: ToolCall, tools: ReadingTools, spec: AgentSpec[Any], *, budget_left: bool
) -> ToolOutput:
    if not budget_left and call.name in READING_TOOL_NAMES:
        return ToolOutput(
            f"The reading budget is used up, so {call.name} is no longer available. "
            f"{spec.finish_hint}",
            is_error=True,
        )
    return tools.run(call.name, call.input)


def budget_line(config: AgentConfig) -> str:
    """The budget, stated in the first message so the model can plan its reading."""
    return (
        f"Budget: at most {config.max_turns} turns and about {config.token_budget:,} tokens "
        "(input and output; each turn re-reads the conversation from a cache, which counts "
        "at a tenth). Open what you need rather than everything."
    )


def _exhausted(trace: AgentTrace, config: AgentConfig) -> bool:
    return trace.requests >= config.max_turns or trace.budget_tokens >= config.token_budget


def _nearly_spent(trace: AgentTrace, config: AgentConfig) -> bool:
    return (
        config.max_turns - trace.requests <= NOTICE_TURNS_LEFT
        or trace.budget_tokens >= NOTICE_FRACTION * config.token_budget
    )


def _budget_notice(trace: AgentTrace, config: AgentConfig, spec: AgentSpec[Any]) -> str:
    return (
        f"Budget notice: you have used {trace.requests} of {config.max_turns} turns and about "
        f"{trace.budget_tokens:,} of {config.token_budget:,} tokens. Stop reading. "
        f"{spec.finish_hint}"
    )


def _nudge(spec: AgentSpec[Any]) -> str:
    name = spec.submit.name
    return (
        f"You ended your turn without calling {name}, and nothing is recorded until you do. "
        f"Call {name} now."
    )


def _finish(trace: AgentTrace, tools: ReadingTools) -> None:
    trace.tokens_read = tools.log.tokens_read
    trace.fraction_read = tools.log.fraction_of(tools.view.doc)
    log.debug(
        "agent finished: %d requests, %d tokens, %.0f%% of the body read",
        trace.requests,
        trace.tokens,
        100 * trace.fraction_read,
    )
