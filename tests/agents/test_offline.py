"""The offline responders, end to end through the real gateway and fake client."""

from typing import Any

import pytest

from precis.agents import run_critic, run_reviser, run_summarize
from precis.agents.trace import AgentTrace
from precis.config import Config
from precis.llm import Gateway, ManualClock
from precis.models import PaperMeta, PaperSummary, Verdict
from precis.offline import offline_client
from precis.store import Store

from .helpers import KeywordIndex, agents_config, make_doc, make_meta

# The offline reader makes a deliberate mistake on half of the titles (by
# hash): it swaps two numbers in a finding, or rounds one in its first
# submission. These three titles fall in each bucket.
SWAPPED = "ReuseNet: Learned Reuse Prediction for Last-Level Caches"
ROUNDED = SWAPPED + " Revisited"
FAITHFUL = SWAPPED + " (extended)"


@pytest.fixture
def gateway() -> Gateway:
    clock = ManualClock()
    return Gateway(
        offline_client(clock=clock), Store(":memory:"), run_id="t", config=Config(), clock=clock
    )


async def summarize(gw: Gateway, meta: PaperMeta, **config: Any) -> tuple[PaperSummary, AgentTrace]:
    doc = make_doc()
    result = await run_summarize(gw, meta, doc, agents_config(**config), index=KeywordIndex(doc))
    return result.value, result.trace


async def test_reader_reads_a_few_sections_and_submits_a_valid_summary(gateway: Gateway) -> None:
    summary, trace = await summarize(gateway, make_meta(title=FAITHFUL))
    assert trace.requests == 2
    assert trace.tool_calls["read_section"] == 3
    assert 0 < trace.fraction_read < 1
    assert summary.findings
    assert all(f.section_ids for f in summary.findings)
    stats = gateway.stats()
    assert stats.calls == 2
    assert stats.cache_read_input_tokens > 0  # the second turn reads the first from cache


async def test_a_rounded_number_is_rejected_and_corrected(gateway: Gateway) -> None:
    summary, trace = await summarize(gateway, make_meta(title=ROUNDED))
    assert trace.requests == 3
    assert trace.rejected_submissions == 1
    assert "18.4%" in summary.findings[0].claim


async def test_verify_and_revise_repair_swapped_numbers(gateway: Gateway) -> None:
    meta = make_meta(title=SWAPPED)
    doc = make_doc()
    config = agents_config()
    summary, _ = await summarize(gateway, meta)
    assert any("by 1.9% over LRU and by 6.2% over Hawkeye" in f.claim for f in summary.findings)
    first = (await run_critic(gateway, meta, doc, summary, config, index=KeywordIndex(doc))).value
    assert first.needs_revision()
    assert any(c.verdict is Verdict.CONTRADICTED for c in first.checks)

    revised = (
        await run_reviser(gateway, meta, doc, summary, first, config, index=KeywordIndex(doc))
    ).value
    assert revised.revision == 1
    second = (
        await run_critic(
            gateway, meta, doc, revised, config, stage="revise", index=KeywordIndex(doc)
        )
    ).value
    assert second.revision == 1
    assert not second.needs_revision()
    assert second.supported_fraction() > first.supported_fraction()


async def test_faithful_summary_passes_verification(gateway: Gateway) -> None:
    meta = make_meta(title=FAITHFUL)
    doc = make_doc()
    summary, _ = await summarize(gateway, meta)
    result = await run_critic(gateway, meta, doc, summary, agents_config(), index=KeywordIndex(doc))
    assert not result.value.needs_revision()
    assert result.trace.quotes_not_found == 0


@pytest.mark.parametrize("strategy", ["fulltext", "mapreduce"])
async def test_baselines_run_offline(gateway: Gateway, strategy: str) -> None:
    summary, trace = await summarize(gateway, make_meta(), strategy=strategy, chunk_tokens=150)
    assert summary.findings
    assert trace.fraction_read == 1.0
    if strategy == "mapreduce":
        assert trace.requests > 2


async def test_offline_runs_are_deterministic() -> None:
    def gateway() -> Gateway:
        clock = ManualClock()
        return Gateway(
            offline_client(clock=clock), Store(":memory:"), run_id="t", config=Config(), clock=clock
        )

    first, _ = await summarize(gateway(), make_meta())
    second, _ = await summarize(gateway(), make_meta())
    assert first == second
