"""The reader: an agent that navigates a paper with tools and summarizes it.

The reader starts from the outline instead of the full text, so it spends
tokens on the parts of a paper that carry its results (usually the
introduction, the method and the evaluation) rather than on related work,
proofs and appendices.
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
from precis.models import Document, PaperMeta, PaperSummary

READER_FINISH = (
    "Call submit_summary now with the best summary that the text you have read supports, "
    "leaving out anything you have not read."
)


def reader_brief(view: PaperView, *, include_abstract: bool) -> str:
    """The reader's first message: who wrote what, and where everything is."""
    return (
        f"<paper>\n{view.header(include_abstract=include_abstract)}\n</paper>\n\n"
        f"<outline>\n{view.outline()}\n</outline>\n\n"
        "Read the parts of this paper you need with the tools, then call submit_summary."
    )


async def run_reader(
    gw: ModelGateway,
    meta: PaperMeta,
    doc: Document,
    config: AgentsConfig,
    *,
    index: PassageIndex | None = None,
) -> AgentResult[PaperSummary]:
    """``read_paper`` with its run statistics, and an optional prebuilt index."""
    view = PaperView(meta, doc, page_tokens=config.page_tokens)
    tools = ReadingTools(view, index or paper_index(doc), search_results=config.search_results)
    spec = AgentSpec(
        stage="read",
        system=load_prompt("reader"),
        brief=reader_brief(view, include_abstract=config.include_abstract),
        submit=summary_tool(view),
        finish_hint=READER_FINISH,
    )
    result = await run_agent(gw, spec, tools, config.reader, paper_id=meta.id)
    summary = PaperSummary(paper_id=meta.id, revision=0, **result.value.model_dump())
    return AgentResult(summary, result.trace)


async def read_paper(
    gw: ModelGateway, meta: PaperMeta, doc: Document, config: AgentsConfig
) -> PaperSummary:
    """Summarize a paper with the reading agent (the ``agentic`` strategy)."""
    return (await run_reader(gw, meta, doc, config)).value
