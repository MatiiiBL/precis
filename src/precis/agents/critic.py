"""The critic: checks a summary's claims against the paper, with quotes.

The critic must return one ``ClaimCheck`` for the tldr and for each finding,
each with a verbatim quote as evidence. Quotes are then looked up in the
paper: a model can produce a fluent "quote" that the paper does not contain,
and a ``supported`` verdict resting on one is downgraded to ``partial`` with a
marker in its note, so the record shows why and the rate can be measured.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from precis.agents.loop import AgentSpec, run_agent
from precis.agents.paper import PaperView
from precis.agents.prompts import load_prompt
from precis.agents.schema import strict_schema
from precis.agents.submissions import SubmitTool, describe_validation_error, rejection
from precis.agents.tools import PassageIndex, ReadingTools, paper_index
from precis.agents.trace import AgentResult
from precis.config import AgentsConfig
from precis.llm import ModelGateway
from precis.models import (
    ClaimCheck,
    Document,
    PaperMeta,
    PaperSummary,
    Record,
    Verdict,
    Verification,
)

SUBMIT_VERIFICATION_NAME = "submit_verification"

QUOTE_NOT_FOUND = "[quote not found in the paper]"
"""Prefix of the note of a check downgraded because its quote is not in the paper."""

CRITIC_FINISH = (
    "Call submit_verification now with a check for every claim; give unsupported, with an "
    "empty quote, for claims you have not been able to check."
)


class VerificationDraft(Record):
    """What the critic submits; the pipeline adds the paper id and revision."""

    checks: list[ClaimCheck]


CHECK_DESCRIPTIONS = {
    "checks": "One check per claim, in the order the claims are listed.",
    "checks.target": '"tldr" or "finding:<i>", exactly as the claim is labelled.',
    "checks.verdict": "supported, partial, unsupported or contradicted.",
    "checks.evidence": "Verbatim quote from the paper; empty when there is none.",
    "checks.section_id": "Id of the section or caption the quote is from; null without a quote.",
    "checks.note": "Why the verdict was given, in one sentence.",
}

CHECKS_SCHEMA = strict_schema(VerificationDraft, descriptions=CHECK_DESCRIPTIONS)


def claim_targets(summary: PaperSummary) -> list[tuple[str, str]]:
    """The claims a verification covers, as (target, text), in order."""
    return [("tldr", summary.tldr)] + [
        (f"finding:{i}", f.claim) for i, f in enumerate(summary.findings)
    ]


def render_claims(summary: PaperSummary) -> str:
    lines = [f"[tldr] {summary.tldr}"]
    for i, finding in enumerate(summary.findings):
        cited = ", ".join(finding.section_ids) or "none"
        lines.append(f"[finding:{i}] {finding.claim} (cites: {cited})")
    return "\n".join(lines)


def critic_brief(view: PaperView, summary: PaperSummary) -> str:
    targets = ", ".join(t for t, _ in claim_targets(summary))
    return (
        f"<paper>\n{view.header(include_abstract=False)}\n</paper>\n\n"
        f"<outline>\n{view.outline()}\n</outline>\n\n"
        f"<claims>\n{render_claims(summary)}\n</claims>\n\n"
        f"Check every claim against the paper, then call submit_verification with one check "
        f"for each of: {targets}."
    )


def parse_checks(
    view: PaperView, summary: PaperSummary, data: Mapping[str, Any]
) -> list[ClaimCheck]:
    """Validate a submission: exactly one well-formed check per claim, in order."""
    try:
        draft = VerificationDraft.model_validate(data)
    except ValidationError as e:
        raise rejection(describe_validation_error(e), SUBMIT_VERIFICATION_NAME) from e
    expected = [target for target, _ in claim_targets(summary)]
    by_target: dict[str, ClaimCheck] = {}
    problems = []
    for check in draft.checks:
        target = check.target.strip()
        if target not in expected:
            problems.append(f"{target!r} is not a claim; targets are {', '.join(expected)}.")
        elif target in by_target:
            problems.append(f"{target} is checked more than once; give one check per claim.")
        else:
            by_target[target] = check.model_copy(update={"target": target})
        section_id = check.section_id
        if section_id and section_id not in view.sections and section_id not in view.captions:
            problems.append(f"{target}: section_id {section_id!r} is not an id in this paper.")
        if check.verdict is not Verdict.UNSUPPORTED and not check.evidence.strip():
            problems.append(f"{target}: a {check.verdict.value} verdict needs a verbatim quote.")
    missing = [t for t in expected if t not in by_target]
    if missing:
        problems.append(f"no check for {', '.join(missing)}.")
    if problems:
        raise rejection(problems, SUBMIT_VERIFICATION_NAME)
    return [by_target[t] for t in expected]


def check_quotes(view: PaperView, checks: list[ClaimCheck]) -> tuple[list[ClaimCheck], int]:
    """Downgrade ``supported`` checks whose quote is not in the paper.

    Returns the checks and how many were downgraded.
    """
    out = []
    downgraded = 0
    for check in checks:
        if check.verdict is Verdict.SUPPORTED and not view.contains_quote(check.evidence):
            check = check.model_copy(
                update={"verdict": Verdict.PARTIAL, "note": f"{QUOTE_NOT_FOUND} {check.note}"}
            )
            downgraded += 1
        out.append(check)
    return out, downgraded


def quotes_not_found(verification: Verification) -> int:
    """How many checks of a stored verification were downgraded by ``check_quotes``."""
    return sum(c.note.startswith(QUOTE_NOT_FOUND) for c in verification.checks)


def verification_tool(view: PaperView, summary: PaperSummary) -> SubmitTool[list[ClaimCheck]]:
    return SubmitTool(
        name=SUBMIT_VERIFICATION_NAME,
        description=(
            "Submit your verdicts: one check for the tldr and one for each finding, in order, "
            "each with a verbatim quote from the paper as evidence. A valid submission ends "
            "your work; if it is rejected, fix what the error names and submit all checks again."
        ),
        input_schema=CHECKS_SCHEMA,
        parse=lambda data: parse_checks(view, summary, data),
    )


async def run_critic(
    gw: ModelGateway,
    meta: PaperMeta,
    doc: Document,
    summary: PaperSummary,
    config: AgentsConfig,
    *,
    stage: str = "verify",
    index: PassageIndex | None = None,
) -> AgentResult[Verification]:
    """``verify_summary`` with its run statistics.

    ``stage`` labels the requests for accounting; the revise stage passes
    ``"revise"`` when it re-verifies the rewritten summary.
    """
    view = PaperView(meta, doc, page_tokens=config.page_tokens)
    tools = ReadingTools(view, index or paper_index(doc), search_results=config.search_results)
    spec = AgentSpec(
        stage=stage,
        system=load_prompt("critic"),
        brief=critic_brief(view, summary),
        submit=verification_tool(view, summary),
        finish_hint=CRITIC_FINISH,
        cache_ttl=config.cache_ttl,
    )
    result = await run_agent(gw, spec, tools, config.critic, paper_id=meta.id)
    checks, downgraded = check_quotes(view, result.value)
    result.trace.quotes_not_found = downgraded
    verification = Verification(paper_id=meta.id, revision=summary.revision, checks=checks)
    return AgentResult(verification, result.trace)


async def verify_summary(
    gw: ModelGateway,
    meta: PaperMeta,
    doc: Document,
    summary: PaperSummary,
    config: AgentsConfig,
) -> Verification:
    """Check the tldr and every finding of ``summary`` against the paper."""
    return (await run_critic(gw, meta, doc, summary, config)).value
