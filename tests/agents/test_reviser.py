"""Scripted conversations with the reviser."""

import json

from precis.agents.reviser import run_reviser
from precis.models import ClaimCheck, PaperSummary, Verdict, Verification

from .helpers import (
    ScriptedGateway,
    agents_config,
    call,
    make_doc,
    make_meta,
    tool_turn,
    valid_draft,
)

WRONG = "ReuseNet reduces LLC misses by 41.3% on average over LRU."


def failing() -> tuple[PaperSummary, Verification]:
    draft = valid_draft()
    draft["findings"][0]["claim"] = WRONG
    summary = PaperSummary(paper_id="2406.01234", **draft)
    checks = [
        ClaimCheck(target="tldr", verdict=Verdict.SUPPORTED, evidence="x", note="ok"),
        ClaimCheck(
            target="finding:0",
            verdict=Verdict.CONTRADICTED,
            evidence="ReuseNet reduces LLC misses by 18.4% on average over LRU.",
            section_id="s4.2",
            note="The paper reports 18.4%, not 41.3%.",
        ),
        ClaimCheck(target="finding:1", verdict=Verdict.PARTIAL, evidence="y", note="Scope."),
    ]
    return summary, Verification(paper_id="2406.01234", revision=0, checks=checks)


async def test_reviser_sees_the_failures_and_returns_revision_one() -> None:
    summary, verification = failing()
    gw = ScriptedGateway(
        [
            tool_turn(call("read_section", {"section_id": "s4.2", "page": 1})),
            tool_turn(call("submit_summary", valid_draft(), "toolu_s")),
        ]
    )
    doc = make_doc()
    result = await run_reviser(gw, make_meta(), doc, summary, verification, agents_config())
    revised = result.value
    assert revised.revision == 1
    assert revised.paper_id == "2406.01234"
    assert revised.findings[0].claim != WRONG
    assert gw.stages == ["revise", "revise"]

    brief = gw.requests[0]["messages"][0]["content"]
    assert "[finding:0] contradicted: The paper reports 18.4%, not 41.3%." in brief
    assert "Verifier's quote (s4.2): <paper_text>ReuseNet reduces LLC misses by 18.4%" in brief
    assert "[finding:1] partial: Scope." in brief
    assert "[tldr]" not in brief.split("<failed_checks>")[1]
    assert "Claims that passed: tldr." in brief
    in_brief = json.loads(brief.split("<summary>\n")[1].split("\n</summary>")[0])
    assert in_brief["findings"][0]["claim"] == WRONG
    assert "paper_id" not in in_brief
