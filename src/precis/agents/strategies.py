"""Baseline summarization strategies for the ablation, and the dispatcher.

``fulltext`` sends the whole body in one request; ``mapreduce`` takes notes
on chunks of the body in parallel and writes the summary from the notes.
Both return the summary through structured output (``output_config.format``)
instead of a tool, so each step is a single request that also runs through
the Message Batches API. Their summaries pass the same validation as the
reader's; a failure gets one repair turn with the problems listed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Iterable
from typing import Any

from precis.agents.errors import AgentError
from precis.agents.messages import assistant_turn, base_params, reply_text, send
from precis.agents.paper import PaperView, paper_text
from precis.agents.prompts import Prompt, load_prompt
from precis.agents.reader import run_reader
from precis.agents.submissions import SUMMARY_SCHEMA, SubmissionRejected, parse_summary
from precis.agents.tools import PassageIndex
from precis.agents.trace import AgentResult, AgentTrace
from precis.config import AgentConfig, AgentsConfig, CacheTTL
from precis.llm import ModelGateway, Reply
from precis.models import Document, PaperMeta, PaperSummary, SummaryDraft
from precis.util import estimate_tokens

_TEXT_STOPS = ("end_turn", "stop_sequence")


def _check_text_reply(reply: Reply) -> str:
    if reply.stop_reason not in _TEXT_STOPS:
        raise AgentError(f"unexpected stop reason {reply.stop_reason!r}")
    return reply_text(reply)


async def structured_summary(
    gw: ModelGateway,
    view: PaperView,
    config: AgentConfig,
    system: Prompt,
    user: str,
    trace: AgentTrace,
    *,
    ttl: CacheTTL,
) -> SummaryDraft:
    """One structured-output request for a ``SummaryDraft``, plus one repair turn."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    for attempt in range(2):
        params = base_params(config, system, messages, ttl=ttl)
        params["output_config"] = {
            **params["output_config"],
            "format": {"type": "json_schema", "schema": SUMMARY_SCHEMA},
        }
        reply = await send(gw, params, stage="read", paper_id=view.meta.id, trace=trace)
        text = _check_text_reply(reply)
        try:
            return parse_summary(view, json.loads(text))
        except (json.JSONDecodeError, SubmissionRejected) as e:
            trace.rejected_submissions += 1
            if attempt:
                raise AgentError(f"the summary failed validation twice: {e}") from e
            messages += [assistant_turn(reply), {"role": "user", "content": str(e)}]
    raise AssertionError("unreachable")


# -- fulltext ----------------------------------------------------------------


def fulltext_message(view: PaperView, *, include_abstract: bool) -> str:
    return (
        f"<paper>\n{view.header(include_abstract=include_abstract)}\n</paper>\n\n"
        f"<body>\n{paper_text(view.body())}\n</body>\n\n"
        "Write the summary of this paper as JSON."
    )


async def run_fulltext(
    gw: ModelGateway, meta: PaperMeta, doc: Document, config: AgentsConfig
) -> AgentResult[PaperSummary]:
    """Summarize from the whole body in one request."""
    view = PaperView(meta, doc, page_tokens=config.page_tokens)
    trace = AgentTrace()
    user = fulltext_message(view, include_abstract=config.include_abstract)
    draft = await structured_summary(
        gw, view, config.reader, load_prompt("fulltext"), user, trace, ttl=config.cache_ttl
    )
    trace.tokens_read, trace.fraction_read = doc.n_tokens, 1.0
    return AgentResult(PaperSummary(paper_id=meta.id, **draft.model_dump()), trace)


# -- mapreduce ---------------------------------------------------------------


def chunk_body(view: PaperView, chunk_tokens: int) -> list[list[str]]:
    """Group the body into chunks of at most about ``chunk_tokens`` tokens.

    Chunks break between sections; a section longer than a chunk is split
    between its pages, and the captions come last. Each piece starts with its
    section id so that notes can cite it.
    """
    pieces: list[str] = []
    for section in view.doc.sections:
        pages = view.pages(section.id)
        for number, page in enumerate(pages, 1):
            more = f" (part {number} of {len(pages)})" if len(pages) > 1 else ""
            pieces.append(f"[{section.id}] {section.title}{more}\n\n{page}")
    pieces += [f"[{c.id}] ({c.kind}) {c.text.strip()}" for c in view.doc.captions]
    chunks: list[list[str]] = []
    size = 0
    for piece in pieces:
        cost = estimate_tokens(piece)
        if not chunks or size + cost > chunk_tokens:
            chunks.append([])
            size = 0
        chunks[-1].append(piece)
        size += cost
    return chunks


def notes_message(view: PaperView, chunk: list[str], number: int, total: int) -> str:
    return (
        f"Paper: {paper_text(view.meta.title, inline=True)}\n"
        f"This is part {number} of {total} of the body.\n\n"
        f"<part>\n{paper_text('\n\n'.join(chunk))}\n</part>\n\n"
        "Write your notes on this part."
    )


def combine_message(view: PaperView, notes: list[str], *, include_abstract: bool) -> str:
    """The notes are model text, but they carry the paper's words, so they are paper text too."""
    parts = "\n\n".join(
        f"## Notes on part {i} of {len(notes)}\n{paper_text(text)}"
        for i, text in enumerate(notes, 1)
    )
    return (
        f"<paper>\n{view.header(include_abstract=include_abstract)}\n</paper>\n\n"
        f"<outline>\n{view.outline()}\n</outline>\n\n"
        f"<notes>\n{parts}\n</notes>\n\n"
        "Write the summary of this paper as JSON."
    )


async def gather_all[T](aws: Iterable[Awaitable[T]]) -> list[T]:
    """Like ``asyncio.gather``, but cancels the rest when one fails.

    The first exception propagates unwrapped, so callers can tell a refusal
    or a spent budget from other failures.
    """
    tasks = [asyncio.ensure_future(aw) for aw in aws]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def run_mapreduce(
    gw: ModelGateway, meta: PaperMeta, doc: Document, config: AgentsConfig
) -> AgentResult[PaperSummary]:
    """Take notes on chunks of the body in parallel, then combine them."""
    view = PaperView(meta, doc, page_tokens=config.page_tokens)
    trace = AgentTrace()
    chunks = chunk_body(view, config.chunk_tokens)
    notes_prompt = load_prompt("mapreduce_notes")

    async def take_notes(number: int, chunk: list[str]) -> str:
        user = notes_message(view, chunk, number, len(chunks))
        messages = [{"role": "user", "content": user}]
        params = base_params(config.reader, notes_prompt, messages, ttl=config.cache_ttl)
        reply = await send(gw, params, stage="read", paper_id=meta.id, trace=trace)
        return _check_text_reply(reply)

    notes = await gather_all(take_notes(i, chunk) for i, chunk in enumerate(chunks, 1))
    user = combine_message(view, notes, include_abstract=config.include_abstract)
    combine = load_prompt("mapreduce_combine")
    draft = await structured_summary(
        gw, view, config.reader, combine, user, trace, ttl=config.cache_ttl
    )
    trace.tokens_read, trace.fraction_read = doc.n_tokens, 1.0
    return AgentResult(PaperSummary(paper_id=meta.id, **draft.model_dump()), trace)


# -- dispatch ----------------------------------------------------------------


async def run_summarize(
    gw: ModelGateway,
    meta: PaperMeta,
    doc: Document,
    config: AgentsConfig,
    *,
    index: PassageIndex | None = None,
) -> AgentResult[PaperSummary]:
    """``summarize`` with its run statistics; ``index`` is used by the agentic reader."""
    if config.strategy == "fulltext":
        return await run_fulltext(gw, meta, doc, config)
    if config.strategy == "mapreduce":
        return await run_mapreduce(gw, meta, doc, config)
    return await run_reader(gw, meta, doc, config, index=index)


async def summarize(
    gw: ModelGateway, meta: PaperMeta, doc: Document, config: AgentsConfig
) -> PaperSummary:
    """Summarize a paper with the strategy ``config.strategy`` names."""
    return (await run_summarize(gw, meta, doc, config)).value
