import re
from datetime import UTC, datetime
from typing import Any

import pytest

from corpus.factories import ScriptedGateway, make_view
from precis.config import AgentConfig
from precis.corpus.load import FinishedPaper
from precis.corpus.synthesis import (
    MERGE_SYSTEM,
    SYNTHESIS_SYSTEM,
    cited_ids,
    clean_citations,
    compact_entry,
    deal,
    synthesize_topic,
    write_report,
)
from precis.models import Topic

TAG = re.compile(r"\[arXiv:([^\]]+)\]")


def _papers(n: int, prefix: str = "2507.") -> list[FinishedPaper]:
    return [make_view(f"{prefix}{i:05d}", ["cache replacement", "prefetching"]) for i in range(n)]


def _topic(papers: list[FinishedPaper], topic_id: int = 1) -> Topic:
    return Topic(
        id=topic_id,
        label="Cache Replacement",
        description="Papers on caches.",
        keywords=["cache"],
        paper_ids=[p.id for p in papers],
    )


def _echo(params: dict[str, Any], stage: str) -> dict[str, Any]:
    """Cite the first two papers in the prompt and one paper that is not there."""
    ids = list(dict.fromkeys(TAG.findall(params["messages"][0]["content"])))
    if stage == "introduce":
        return {"title": " A Survey ", "introduction": " Intro. "}
    first, second = ids[0], ids[min(1, len(ids) - 1)]
    return {
        "overview": f"Work [arXiv:{first}] and [arXiv:{second}] differ [arXiv:9999.99999].",
        "trends": [f"More caching [arXiv:{first}, arXiv:9999.99999].", "[arXiv:0000.00000]"],
        "open_problems": ["Scaling is open."],
        "key_papers": [f"arXiv:{second}", "9999.99999", first, second],
    }


def test_clean_citations_keeps_valid_and_counts_removed() -> None:
    text = "A [arXiv:1] b [arXiv:2]. C [arXiv:1; arXiv:3, 2]."
    cleaned, kept, removed = clean_citations(text, {"1", "3"})
    assert cleaned == "A [arXiv:1] b. C [arXiv:1] [arXiv:3]."
    assert (kept, removed) == (3, 2)
    assert cited_ids(text) == ["1", "2", "1", "3", "2"]


def test_deal_fills_round_robin_chunks_within_budget() -> None:
    items = ["x" * 400] * 10  # 100 tokens each
    chunks = deal(items, 350)
    assert len(chunks) == 4
    assert chunks[0] == [0, 4, 8]
    assert sorted(i for c in chunks for i in c) == list(range(10))
    assert deal(items, 5000) == [list(range(10))]


def test_deal_refuses_an_entry_larger_than_the_budget() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        deal(["x" * 4000], 500)


async def test_small_topic_takes_one_request_and_loses_bad_citations() -> None:
    papers = _papers(5)
    gateway = ScriptedGateway(_echo)
    synthesis, stats = await synthesize_topic(
        gateway, _topic(papers), papers, AgentConfig(effort="high"), budget=60000
    )
    assert gateway.stages() == ["synthesize"]
    prompt = gateway.calls[0].user_text
    assert all(compact_entry(p) in prompt for p in papers)
    assert gateway.calls[0].params["system"] == SYNTHESIS_SYSTEM
    first, second = papers[0].id, papers[1].id
    assert synthesis.overview == f"Work [arXiv:{first}] and [arXiv:{second}] differ."
    assert synthesis.trends == [f"More caching [arXiv:{first}]."]
    assert synthesis.key_papers == [second, first]
    assert (stats.citations, stats.invalid_citations, stats.invalid_key_papers) == (3, 3, 1)
    assert (stats.requests, stats.levels, stats.papers) == (1, 1, 5)


async def test_key_papers_fall_back_to_the_most_central() -> None:
    papers = _papers(4)

    def script(params: dict[str, Any], stage: str) -> dict[str, Any]:
        return {"overview": "O.", "trends": [], "open_problems": [], "key_papers": ["nope"]}

    synthesis, _ = await synthesize_topic(
        ScriptedGateway(script), _topic(papers), papers, AgentConfig(), budget=60000
    )
    assert synthesis.key_papers == [p.id for p in papers[:3]]


async def test_large_topic_is_summarized_hierarchically_without_dropping_papers() -> None:
    papers = _papers(40)
    entry_tokens = max(len(compact_entry(p)) for p in papers) // 4 + 1
    gateway = ScriptedGateway(_echo)
    synthesis, stats = await synthesize_topic(
        gateway, _topic(papers), papers, AgentConfig(), budget=entry_tokens * 12
    )
    partials = [c for c in gateway.calls if c.params["system"] == SYNTHESIS_SYSTEM]
    merges = [c for c in gateway.calls if c.params["system"] == MERGE_SYSTEM]
    assert len(partials) == 4
    assert len(merges) == 1
    seen = [pid for c in partials for pid in TAG.findall(c.user_text)]
    assert sorted(seen) == sorted(p.id for p in papers)  # every paper exactly once
    assert "part 1 of 4" in partials[0].user_text
    assert (stats.requests, stats.levels) == (5, 2)
    assert set(cited_ids(synthesis.overview)) <= {p.id for p in papers}


async def test_merges_recurse_when_partials_do_not_fit_together() -> None:
    papers = _papers(30)
    entry_tokens = max(len(compact_entry(p)) for p in papers) // 4 + 1
    gateway = ScriptedGateway(_echo)
    _, stats = await synthesize_topic(
        gateway, _topic(papers), papers, AgentConfig(), budget=entry_tokens * 5
    )
    assert stats.levels >= 3
    assert gateway.calls[-1].params["system"] == MERGE_SYSTEM
    assert [c.params["system"] for c in gateway.calls].count(MERGE_SYSTEM) >= 2


async def test_drafts_too_long_to_merge_are_an_error_not_a_cut() -> None:
    papers = _papers(30)
    entry_tokens = max(len(compact_entry(p)) for p in papers) // 4 + 1
    with pytest.raises(ValueError, match="too long to merge"):
        await synthesize_topic(
            ScriptedGateway(_echo), _topic(papers), papers, AgentConfig(), budget=entry_tokens * 2
        )


async def test_write_report_synthesizes_topics_then_the_introduction() -> None:
    first, second = _papers(3, "2507."), _papers(3, "2508.")
    topics = [_topic(first, 1), _topic(second, 2)]
    gateway = ScriptedGateway(_echo)
    now = datetime(2026, 11, 1, tzinfo=UTC)
    report, stats = await write_report(
        gateway, topics, first + second, AgentConfig(), budget=60000, now=lambda: now
    )
    assert gateway.stages() == ["synthesize", "synthesize", "introduce"]
    intro_prompt = gateway.calls[-1].user_text
    assert "Papers: 6" in intro_prompt
    assert "1. Cache Replacement (3 papers)" in intro_prompt
    assert "[arXiv:" not in intro_prompt
    assert report.title == "A Survey"
    assert report.introduction == "Intro."
    assert report.generated_at == now
    assert report.n_papers == 6
    assert [s.topic_id for s in report.topics] == [1, 2]
    assert stats.invalid_citations == 6
    # A citation to the other topic's paper counts as invalid too.
    assert not set(cited_ids(report.topics[0].overview)) & {p.id for p in second}
