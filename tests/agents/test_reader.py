"""Scripted conversations with the reader agent."""

from typing import Any

import pytest

from precis.agents import AgentError, AgentRefused
from precis.agents.reader import run_reader
from precis.config import AgentConfig
from precis.models import PaperSummary
from precis.util import canonical_json

from .helpers import (
    ScriptedGateway,
    agents_config,
    call,
    last_user_blocks,
    make_doc,
    make_meta,
    message,
    text_turn,
    tool_turn,
    valid_draft,
)

SUBMIT = call("submit_summary", valid_draft(), "toolu_submit")


async def read(gw: ScriptedGateway, **config: Any) -> Any:
    doc = make_doc()
    return await run_reader(gw, make_meta(), doc, agents_config(**config))


async def test_happy_path_reads_then_submits() -> None:
    gw = ScriptedGateway(
        [
            tool_turn(
                call("read_section", {"section_id": "s1", "page": 1}, "toolu_a"),
                call("read_section", {"section_id": "s4.2", "page": 1}, "toolu_b"),
            ),
            tool_turn(SUBMIT),
        ]
    )
    result = await read(gw)
    summary = result.value
    assert isinstance(summary, PaperSummary)
    assert summary.paper_id == "2406.01234"
    assert summary.revision == 0
    assert summary.keywords == ["cache replacement", "last-level cache", "reuse prediction"]
    assert gw.stages == ["read", "read"]

    trace = result.trace
    assert trace.requests == 2
    assert trace.tool_calls == {"read_section": 2}
    assert trace.input_tokens == 200
    assert 0 < trace.fraction_read < 1

    second = gw.requests[1]
    # The assistant turn goes back unchanged, thinking block and signature included.
    assert second["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig-1"},
            call("read_section", {"section_id": "s1", "page": 1}, "toolu_a"),
            call("read_section", {"section_id": "s4.2", "page": 1}, "toolu_b"),
        ],
    }
    # Both results come back in one user message, in call order.
    results = last_user_blocks(second)
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert results[0]["content"].startswith("[s1] Introduction (page 1 of 3)")
    assert "is_error" not in results[0]


async def test_request_shape() -> None:
    gw = ScriptedGateway([tool_turn(SUBMIT)])
    await read(gw)
    params = gw.requests[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "medium"}
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in params["tools"]] == [
        "read_section",
        "search_paper",
        "read_captions",
        "submit_summary",
    ]
    assert all(t["strict"] for t in params["tools"])
    for absent in ("temperature", "top_p", "top_k", "tool_choice"):
        assert absent not in params
    brief = params["messages"][0]["content"]
    assert "Title: ReuseNet" in brief
    assert "We present ReuseNet" in brief
    assert "<outline>\nBody:" in brief


async def test_one_hour_cache_lifetime() -> None:
    gw = ScriptedGateway([tool_turn(SUBMIT)])
    await read(gw, cache_ttl="1h")
    params = gw.requests[0]
    one_hour = {"type": "ephemeral", "ttl": "1h"}
    assert params["cache_control"] == one_hour
    assert params["system"][0]["cache_control"] == one_hour


async def test_requests_are_deterministic() -> None:
    first, second = ScriptedGateway([tool_turn(SUBMIT)]), ScriptedGateway([tool_turn(SUBMIT)])
    await read(first)
    await read(second)
    assert canonical_json(first.requests) == canonical_json(second.requests)


async def test_abstract_can_be_withheld() -> None:
    gw = ScriptedGateway([tool_turn(SUBMIT)])
    await read(gw, include_abstract=False)
    brief = gw.requests[0]["messages"][0]["content"]
    assert "We present ReuseNet" not in brief
    assert "Abstract: withheld" in brief


async def test_tool_errors_go_back_to_the_model() -> None:
    gw = ScriptedGateway(
        [
            tool_turn(call("read_section", {"section_id": "s9", "page": 1})),
            tool_turn(SUBMIT),
        ]
    )
    result = await read(gw)
    [error] = last_user_blocks(gw.requests[1])
    assert error["is_error"] is True
    assert "There is no section or caption with id 's9'" in error["content"]
    assert result.trace.tool_errors == 1


async def test_rejected_submission_is_explained_and_retried() -> None:
    bad = valid_draft(
        tldr=" ".join(["word"] * 45),
        findings=[
            {"claim": "ReuseNet reduces LLC misses by 21% over LRU.", "section_ids": ["s4.2"]},
            {"claim": "ReuseNet improves IPC.", "section_ids": ["s7"]},
        ],
        keywords=["cache"],
    )
    gw = ScriptedGateway([tool_turn(call("submit_summary", bad, "toolu_bad")), tool_turn(SUBMIT)])
    result = await read(gw)
    [error] = last_user_blocks(gw.requests[1])
    assert error["is_error"] is True
    text = error["content"]
    assert "tldr has 45 words; the limit is 40." in text
    assert "finding 1 cites 's7', which is not an id in this paper" in text
    assert "keywords has 1 items; give 3 to 8." in text
    assert "finding 0 states 21, which does not appear anywhere in the paper." in text
    assert result.trace.rejected_submissions == 1
    assert result.value.findings[0].claim.startswith("ReuseNet reduces LLC misses by 18.4%")


async def test_schema_violations_are_rejected_too() -> None:
    gw = ScriptedGateway([tool_turn(call("submit_summary", {"tldr": "x"})), tool_turn(SUBMIT)])
    await read(gw)
    [error] = last_user_blocks(gw.requests[1])
    assert "problem: Field required" in error["content"]


async def test_one_nudge_when_the_model_stops_without_submitting() -> None:
    gw = ScriptedGateway([text_turn("Here is my summary: ..."), tool_turn(SUBMIT)])
    result = await read(gw)
    nudge = gw.requests[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert "without calling submit_summary" in nudge["content"]
    assert result.trace.nudges == 1


async def test_stopping_twice_without_submitting_fails() -> None:
    gw = ScriptedGateway([text_turn("Done."), text_turn("Done.")])
    with pytest.raises(AgentError, match="twice without calling submit_summary"):
        await read(gw)


def reading_turn(n: int, usage: tuple[int, int] = (100, 20)) -> Any:
    return tool_turn(call("search_paper", {"query": f"query {n}"}, f"toolu_{n}"), usage=usage)


async def test_budget_notice_is_a_system_message_after_the_tool_results() -> None:
    gw = ScriptedGateway([reading_turn(1), reading_turn(2), tool_turn(SUBMIT)])
    result = await read(gw, reader=AgentConfig(max_turns=4))
    first, second, third = gw.requests
    assert all(m["role"] != "system" for m in first["messages"] + second["messages"])
    assert third["messages"][-2]["role"] == "user"
    notice = third["messages"][-1]
    assert notice["role"] == "system"
    assert notice["content"].startswith("Budget notice: you have used 2 of 4 turns")
    assert "Call submit_summary now" in notice["content"]
    assert result.trace.budget_notices == 1


async def test_token_budget_triggers_the_notice() -> None:
    gw = ScriptedGateway([reading_turn(1, usage=(900, 50)), tool_turn(SUBMIT)])
    await read(gw, reader=AgentConfig(token_budget=1000))
    notice = gw.requests[1]["messages"][-1]
    assert notice["role"] == "system"
    assert "about 950 of 1,000 tokens" in notice["content"]


async def test_reading_stops_when_the_budget_is_spent_and_the_run_fails_after_grace() -> None:
    gw = ScriptedGateway([reading_turn(i) for i in range(1, 5)])
    with pytest.raises(AgentError, match="no valid submit_summary call within the budget"):
        await read(gw, reader=AgentConfig(max_turns=2))
    assert len(gw.requests) == 4
    [refused] = last_user_blocks(gw.requests[3])
    assert refused["is_error"] is True
    assert "reading budget is used up" in refused["content"]


async def test_submission_after_the_budget_is_spent_is_accepted() -> None:
    gw = ScriptedGateway([reading_turn(1), reading_turn(2), tool_turn(SUBMIT)])
    result = await read(gw, reader=AgentConfig(max_turns=2))
    assert result.value.tldr.startswith("ReuseNet")


async def test_refusal_raises() -> None:
    refusal = message([], "refusal", stop_details={"type": "refusal", "category": "cyber"})
    gw = ScriptedGateway([refusal])
    with pytest.raises(AgentRefused) as raised:
        await read(gw)
    assert raised.value.category == "cyber"


async def test_max_tokens_is_retried_once_with_a_larger_limit() -> None:
    truncated = message([{"type": "thinking", "thinking": "", "signature": "s"}], "max_tokens")
    gw = ScriptedGateway([truncated, tool_turn(SUBMIT)])
    result = await read(gw)
    assert gw.requests[1]["max_tokens"] == 2 * gw.requests[0]["max_tokens"]
    # The truncated reply is dropped, not appended.
    assert gw.requests[1]["messages"] == gw.requests[0]["messages"]
    assert result.trace.max_tokens_retries == 1


async def test_max_tokens_twice_fails() -> None:
    truncated = message([], "max_tokens")
    gw = ScriptedGateway([truncated, truncated])
    with pytest.raises(AgentError, match="max_tokens"):
        await read(gw)


async def test_unexpected_stop_reason_fails() -> None:
    gw = ScriptedGateway([message([], "model_context_window_exceeded")])
    with pytest.raises(AgentError, match="model_context_window_exceeded"):
        await read(gw)


async def test_pause_turn_is_resumed_without_a_user_message() -> None:
    paused = message([{"type": "text", "text": "Reading..."}], "pause_turn")
    gw = ScriptedGateway([paused, tool_turn(SUBMIT)])
    await read(gw)
    assert gw.requests[1]["messages"][-1]["role"] == "assistant"
