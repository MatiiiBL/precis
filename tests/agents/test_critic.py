"""Scripted conversations with the critic, and the quote check."""

from typing import Any

import pytest

from precis.agents.critic import QUOTE_NOT_FOUND, quotes_not_found, run_critic
from precis.models import PaperSummary, Verdict

from .helpers import (
    KeywordIndex,
    ScriptedGateway,
    agents_config,
    call,
    last_user_blocks,
    make_doc,
    make_meta,
    tool_turn,
    valid_draft,
)

TLDR_QUOTE = "ReuseNet reduces LLC misses by 18.4% over LRU on SPEC CPU2017."
FINDING_QUOTE = "ReuseNet reduces LLC misses by 18.4% on average over LRU."
IPC_QUOTE = "It improves IPC by 6.2% over LRU and by 1.9% over Hawkeye."


def summary() -> PaperSummary:
    return PaperSummary(paper_id="2406.01234", **valid_draft())


def check(
    target: str, verdict: str, evidence: str, section_id: str | None = "s4.2"
) -> dict[str, Any]:
    return {
        "target": target,
        "verdict": verdict,
        "evidence": evidence,
        "section_id": section_id,
        "note": f"{target} checked.",
    }


def submit(*checks: dict[str, Any]) -> Any:
    return tool_turn(call("submit_verification", {"checks": list(checks)}, "toolu_v"))


async def verify(gw: ScriptedGateway, **kwargs: Any) -> Any:
    doc = make_doc()
    return await run_critic(
        gw, make_meta(), doc, summary(), agents_config(), index=KeywordIndex(doc), **kwargs
    )


GOOD = (
    check("tldr", "supported", TLDR_QUOTE, "s1"),
    check("finding:0", "supported", FINDING_QUOTE),
    check("finding:1", "supported", IPC_QUOTE),
)


async def test_brief_lists_every_claim_with_its_citations() -> None:
    gw = ScriptedGateway([submit(*GOOD)])
    await verify(gw)
    brief = gw.requests[0]["messages"][0]["content"]
    assert "[tldr] ReuseNet predicts line reuse" in brief
    assert "[finding:1] ReuseNet improves IPC by 6.2% over LRU. (cites: s4.2)" in brief
    assert "for each of: tldr, finding:0, finding:1." in brief
    assert "We present ReuseNet" not in brief  # the abstract is not evidence
    assert [t["name"] for t in gw.requests[0]["tools"]][-1] == "submit_verification"


async def test_verification_keeps_checks_in_claim_order() -> None:
    gw = ScriptedGateway(
        [
            tool_turn(call("read_section", {"section_id": "s4.2", "page": 1})),
            submit(GOOD[2], GOOD[0], GOOD[1]),
        ]
    )
    result = await verify(gw)
    verification = result.value
    assert verification.paper_id == "2406.01234"
    assert verification.revision == 0
    assert [c.target for c in verification.checks] == ["tldr", "finding:0", "finding:1"]
    assert verification.supported_fraction() == 1.0
    assert not verification.needs_revision()
    assert result.trace.quotes_not_found == 0
    assert gw.stages == ["verify", "verify"]


async def test_supported_verdict_with_an_invented_quote_is_downgraded() -> None:
    invented = check("finding:1", "supported", "ReuseNet improves IPC by 6.2% in every workload.")
    contradicted = check("finding:0", "contradicted", "This quote is not in the paper either.")
    gw = ScriptedGateway([submit(GOOD[0], contradicted, invented)])
    result = await verify(gw)
    checks = result.value.checks
    assert checks[2].verdict is Verdict.PARTIAL
    assert checks[2].note == f"{QUOTE_NOT_FOUND} finding:1 checked."
    # Only supported verdicts are downgraded; a contradiction still triggers revision.
    assert checks[1].verdict is Verdict.CONTRADICTED
    assert result.trace.quotes_not_found == 1
    assert quotes_not_found(result.value) == 1
    assert result.value.needs_revision()


async def test_invalid_submissions_are_explained() -> None:
    bad = submit(
        check("tldr", "supported", ""),
        check("finding:0", "supported", FINDING_QUOTE, "s9"),
        check("finding:0", "partial", FINDING_QUOTE),
        check("finding:7", "unsupported", ""),
    )
    gw = ScriptedGateway([bad, submit(*GOOD)])
    result = await verify(gw)
    [error] = last_user_blocks(gw.requests[1])
    assert error["is_error"] is True
    text = error["content"]
    assert "tldr: a supported verdict needs a verbatim quote." in text
    assert "finding:0: section_id 's9' is not an id in this paper." in text
    assert "finding:0 is checked more than once" in text
    assert "'finding:7' is not a claim" in text
    assert "no check for finding:1." in text
    assert result.trace.rejected_submissions == 1


async def test_unsupported_verdict_may_have_no_quote() -> None:
    gw = ScriptedGateway([submit(GOOD[0], GOOD[1], check("finding:1", "unsupported", "", None))])
    result = await verify(gw)
    assert result.value.checks[2].verdict is Verdict.UNSUPPORTED
    assert result.value.needs_revision()


async def test_stage_label_can_be_overridden() -> None:
    gw = ScriptedGateway([submit(*GOOD)])
    await verify(gw, stage="revise")
    assert gw.stages == ["revise"]


@pytest.mark.parametrize("target", ["tldr ", " finding:0"])
async def test_targets_are_trimmed(target: str) -> None:
    padded = [dict(c) for c in GOOD]
    index = 0 if target.strip() == "tldr" else 1
    padded[index]["target"] = target
    gw = ScriptedGateway([submit(*padded)])
    result = await verify(gw)
    assert [c.target for c in result.value.checks] == ["tldr", "finding:0", "finding:1"]
