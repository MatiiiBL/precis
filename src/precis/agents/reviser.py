"""The reviser: rewrites a summary whose claims failed verification.

The reviser gets the same reading tools as the reader, so it corrects a claim
from the paper's text rather than from the critic's note alone; the critic
can be wrong too, and the rewritten summary is verified again afterwards.
"""

from __future__ import annotations

from precis.agents.loop import AgentSpec, run_agent
from precis.agents.paper import PaperView
from precis.agents.prompts import load_prompt
from precis.agents.submissions import summary_tool
from precis.agents.tools import PassageIndex, ReadingTools, paper_index
from precis.agents.trace import AgentResult
from precis.config import AgentsConfig
from precis.llm import ModelGateway
from precis.models import (
    ClaimCheck,
    Document,
    PaperMeta,
    PaperSummary,
    SummaryDraft,
    Verdict,
    Verification,
)

REVISER_FINISH = (
    "Call submit_summary now with the corrected summary, dropping any failed claim you have "
    "not been able to confirm in the paper."
)


def _render_check(check: ClaimCheck) -> str:
    lines = [f"[{check.target}] {check.verdict.value}: {check.note}"]
    if check.evidence:
        where = f" ({check.section_id})" if check.section_id else ""
        lines.append(f'  Verifier\'s quote{where}: "{check.evidence}"')
    return "\n".join(lines)


def reviser_brief(view: PaperView, summary: PaperSummary, verification: Verification) -> str:
    draft = SummaryDraft.model_validate(summary.model_dump(include=set(SummaryDraft.model_fields)))
    failed = [c for c in verification.checks if c.verdict is not Verdict.SUPPORTED]
    passed = [c.target for c in verification.checks if c.verdict is Verdict.SUPPORTED]
    checks = "\n".join(_render_check(c) for c in failed)
    return (
        f"<paper>\n{view.header(include_abstract=False)}\n</paper>\n\n"
        f"<outline>\n{view.outline()}\n</outline>\n\n"
        f"<summary>\n{draft.model_dump_json(indent=2)}\n</summary>\n\n"
        f"<failed_checks>\n{checks}\n</failed_checks>\n\n"
        f"Claims that passed: {', '.join(passed) or 'none'}. "
        "Findings are numbered from 0 in the order of the summary. Correct the failed claims "
        "from the paper, then call submit_summary with the complete corrected summary."
    )


async def run_reviser(
    gw: ModelGateway,
    meta: PaperMeta,
    doc: Document,
    summary: PaperSummary,
    verification: Verification,
    config: AgentsConfig,
    *,
    index: PassageIndex | None = None,
) -> AgentResult[PaperSummary]:
    """``revise_summary`` with its run statistics."""
    view = PaperView(meta, doc, page_tokens=config.page_tokens)
    tools = ReadingTools(view, index or paper_index(doc), search_results=config.search_results)
    spec = AgentSpec(
        stage="revise",
        system=load_prompt("reviser"),
        brief=reviser_brief(view, summary, verification),
        submit=summary_tool(view),
        finish_hint=REVISER_FINISH,
    )
    result = await run_agent(gw, spec, tools, config.reviser, paper_id=meta.id)
    revised = PaperSummary(
        paper_id=meta.id, revision=summary.revision + 1, **result.value.model_dump()
    )
    return AgentResult(revised, result.trace)


async def revise_summary(
    gw: ModelGateway,
    meta: PaperMeta,
    doc: Document,
    summary: PaperSummary,
    verification: Verification,
    config: AgentsConfig,
) -> PaperSummary:
    """Rewrite the claims of ``summary`` that ``verification`` did not support."""
    return (await run_reviser(gw, meta, doc, summary, verification, config)).value
