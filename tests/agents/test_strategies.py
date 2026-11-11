"""The fulltext and mapreduce baselines, and strategy dispatch."""

import asyncio
import json
from typing import Any

import pytest

from precis.agents import AgentError, AgentRefused, run_summarize
from precis.agents.paper import PaperView
from precis.agents.strategies import chunk_body, gather_all, run_fulltext, run_mapreduce
from precis.llm.fake import FakeAnthropic
from precis.util import estimate_tokens

from .helpers import (
    KeywordIndex,
    ScriptedGateway,
    agents_config,
    call,
    make_doc,
    make_meta,
    message,
    text_turn,
    tool_turn,
    valid_draft,
)


def json_turn(draft: dict[str, Any]) -> Any:
    return text_turn(json.dumps(draft))


async def test_fulltext_sends_the_body_once_with_structured_output() -> None:
    gw = ScriptedGateway([json_turn(valid_draft())])
    result = await run_fulltext(gw, make_meta(), make_doc(), agents_config())
    assert result.value.paper_id == "2406.01234"
    assert result.trace.requests == 1
    assert result.trace.fraction_read == 1.0
    [params] = gw.requests
    assert "tools" not in params
    assert "cache_control" not in params  # a one-off prompt: caching it only costs
    assert params["output_config"]["effort"] == "medium"
    assert params["output_config"]["format"]["type"] == "json_schema"
    body = params["messages"][0]["content"]
    for sid in (
        "[s1] Introduction",
        "[s3] Design of ReuseNet",
        "[s5] Conclusion",
        "[tab1] (table)",
    ):
        assert sid in body


async def test_fulltext_gets_one_repair_turn() -> None:
    bad = valid_draft(findings=[{"claim": "ReuseNet works.", "section_ids": ["s99"]}])
    gw = ScriptedGateway([json_turn(bad), json_turn(valid_draft())])
    result = await run_fulltext(gw, make_meta(), make_doc(), agents_config())
    repair = gw.requests[1]["messages"]
    assert repair[1]["role"] == "assistant"
    assert "findings has 1 items" in repair[2]["content"]
    assert result.trace.rejected_submissions == 1

    gw = ScriptedGateway([json_turn(bad), text_turn("not json")])
    with pytest.raises(AgentError, match="failed validation twice"):
        await run_fulltext(gw, make_meta(), make_doc(), agents_config())


async def test_fulltext_request_is_accepted_by_the_batches_api() -> None:
    gw = ScriptedGateway([json_turn(valid_draft())])
    await run_fulltext(gw, make_meta(), make_doc(), agents_config())
    fake = FakeAnthropic()
    batch = await fake.messages.batches.create(
        requests=[{"custom_id": "p1", "params": gw.requests[0]}]
    )
    assert batch.processing_status == "in_progress"


def test_chunks_cover_the_body_within_the_size_limit() -> None:
    view = PaperView(make_meta(), make_doc(), page_tokens=40)
    chunks = chunk_body(view, 120)
    pieces = [p for chunk in chunks for p in chunk]
    assert len(chunks) > 1
    assert all(sum(estimate_tokens(p) for p in chunk) <= 120 for chunk in chunks)
    assert pieces[0].startswith("[s1] Introduction (part 1 of 3)")
    assert any(p.startswith("[s3] Design of ReuseNet (part 2 of 2)") for p in pieces)
    assert pieces[-1].startswith("[tab1] (table)")
    assert not any(p.startswith("[s2] ") for p in pieces)  # heading only


async def test_mapreduce_takes_notes_in_parallel_then_combines() -> None:
    def notes(params: dict[str, Any]) -> Any:
        part = params["messages"][0]["content"]
        return text_turn(f"[s4.2] notes on {part.splitlines()[1]}")

    doc = make_doc()
    view = PaperView(make_meta(), doc, page_tokens=40)
    n_chunks = len(chunk_body(view, 120))
    gw = ScriptedGateway([notes] * n_chunks + [json_turn(valid_draft())])
    result = await run_mapreduce(gw, make_meta(), doc, agents_config(chunk_tokens=120))
    assert result.value.findings[0].section_ids == ["s4.2"]
    assert result.trace.requests == n_chunks + 1
    combine = gw.requests[-1]
    assert combine["output_config"]["format"]["type"] == "json_schema"
    assert f"## Notes on part {n_chunks} of {n_chunks}" in combine["messages"][0]["content"]
    assert all("format" not in r["output_config"] for r in gw.requests[:-1])


async def test_a_failed_chunk_cancels_the_others_and_propagates_unwrapped() -> None:
    cancelled = []

    async def slow() -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return "late"

    async def refuse() -> str:
        raise AgentRefused("bio")

    with pytest.raises(AgentRefused):
        await gather_all([slow(), refuse()])
    assert cancelled == [True]


async def test_dispatch_follows_the_configured_strategy() -> None:
    doc = make_doc()
    gw = ScriptedGateway([json_turn(valid_draft())])
    await run_summarize(gw, make_meta(), doc, agents_config(strategy="fulltext"))
    assert "tools" not in gw.requests[0]

    gw = ScriptedGateway([tool_turn(call("submit_summary", valid_draft()))])
    await run_summarize(gw, make_meta(), doc, agents_config(), index=KeywordIndex(doc))
    assert gw.requests[0]["tools"][0]["name"] == "read_section"

    refusal = message([], "refusal", stop_details={"type": "refusal", "category": None})
    with pytest.raises(AgentRefused):
        await run_summarize(
            ScriptedGateway([refusal]), make_meta(), doc, agents_config(strategy="mapreduce")
        )
