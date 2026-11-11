"""The loop behaviours every agent shares, run against each role."""

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from precis.agents import AgentError, AgentRefused, run_critic, run_reader, run_reviser
from precis.config import AgentConfig
from precis.models import ClaimCheck, PaperSummary, Verdict, Verification

from .helpers import (
    KeywordIndex,
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

SUMMARY = PaperSummary(paper_id="2406.01234", **valid_draft())
VERIFICATION = Verification(
    paper_id="2406.01234",
    revision=0,
    checks=[
        ClaimCheck(target="tldr", verdict=Verdict.SUPPORTED, evidence="x", note="ok"),
        ClaimCheck(target="finding:0", verdict=Verdict.UNSUPPORTED, evidence="", note="Not found."),
        ClaimCheck(target="finding:1", verdict=Verdict.SUPPORTED, evidence="y", note="ok"),
    ],
)
CHECKS = {
    "checks": [
        {
            "target": target,
            "verdict": "supported",
            "evidence": "ReuseNet reduces LLC misses by 18.4% on average over LRU.",
            "section_id": "s4.2",
            "note": "Stated in the results.",
        }
        for target in ("tldr", "finding:0", "finding:1")
    ]
}

Run = Callable[[ScriptedGateway, dict[str, Any]], Awaitable[Any]]


async def reader(gw: ScriptedGateway, overrides: dict[str, Any]) -> Any:
    doc = make_doc()
    config = agents_config(reader=AgentConfig(**overrides))
    return await run_reader(gw, make_meta(), doc, config, index=KeywordIndex(doc))


async def critic(gw: ScriptedGateway, overrides: dict[str, Any]) -> Any:
    doc = make_doc()
    config = agents_config(critic=AgentConfig(**overrides))
    return await run_critic(gw, make_meta(), doc, SUMMARY, config, index=KeywordIndex(doc))


async def reviser(gw: ScriptedGateway, overrides: dict[str, Any]) -> Any:
    doc = make_doc()
    config = agents_config(reviser=AgentConfig(**overrides))
    return await run_reviser(
        gw, make_meta(), doc, SUMMARY, VERIFICATION, config, index=KeywordIndex(doc)
    )


ROLES = [
    pytest.param(reader, "submit_summary", valid_draft(), id="reader"),
    pytest.param(critic, "submit_verification", CHECKS, id="critic"),
    pytest.param(reviser, "submit_summary", valid_draft(), id="reviser"),
]


def submit(name: str, payload: dict[str, Any]) -> Any:
    return tool_turn(call(name, payload, "toolu_submit"))


@pytest.mark.parametrize(("run", "name", "payload"), ROLES)
async def test_tool_error_then_submit(run: Run, name: str, payload: dict[str, Any]) -> None:
    gw = ScriptedGateway(
        [tool_turn(call("read_section", {"section_id": "s1", "page": 9})), submit(name, payload)]
    )
    result = await run(gw, {})
    [error] = last_user_blocks(gw.requests[1])
    assert error["is_error"] is True
    assert result.trace.tool_errors == 1


@pytest.mark.parametrize(("run", "name", "payload"), ROLES)
async def test_rejected_submission_then_fix(run: Run, name: str, payload: dict[str, Any]) -> None:
    gw = ScriptedGateway([submit(name, {}), submit(name, payload)])
    result = await run(gw, {})
    [error] = last_user_blocks(gw.requests[1])
    assert error["is_error"] is True
    assert "Field required" in error["content"]
    assert result.trace.rejected_submissions == 1


@pytest.mark.parametrize(("run", "name", "payload"), ROLES)
async def test_nudge_then_submit(run: Run, name: str, payload: dict[str, Any]) -> None:
    gw = ScriptedGateway([text_turn("All done."), submit(name, payload)])
    result = await run(gw, {})
    assert f"without calling {name}" in gw.requests[1]["messages"][-1]["content"]
    assert result.trace.nudges == 1


@pytest.mark.parametrize(("run", "name", "payload"), ROLES)
async def test_budget_notice_names_the_submit_tool(
    run: Run, name: str, payload: dict[str, Any]
) -> None:
    search = tool_turn(call("search_paper", {"query": "misses"}))
    gw = ScriptedGateway([search, submit(name, payload)])
    await run(gw, {"max_turns": 3})
    notice = gw.requests[1]["messages"][-1]
    assert notice["role"] == "system"
    assert f"Call {name} now" in notice["content"]


@pytest.mark.parametrize(("run", "name", "payload"), ROLES)
async def test_refusal(run: Run, name: str, payload: dict[str, Any]) -> None:
    refusal = message([], "refusal", stop_details={"type": "refusal", "category": "cyber"})
    with pytest.raises(AgentRefused):
        await run(ScriptedGateway([refusal]), {})


@pytest.mark.parametrize(("run", "name", "payload"), ROLES)
async def test_max_tokens(run: Run, name: str, payload: dict[str, Any]) -> None:
    truncated = message([], "max_tokens")
    gw = ScriptedGateway([truncated, submit(name, payload)])
    result = await run(gw, {"max_tokens": 1000})
    assert [r["max_tokens"] for r in gw.requests] == [1000, 2000]
    assert result.trace.max_tokens_retries == 1
    with pytest.raises(AgentError):
        await run(ScriptedGateway([truncated, truncated]), {})
