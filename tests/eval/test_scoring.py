from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from corpus.factories import ScriptedGateway, make_meta, make_summary
from precis.config import AgentConfig
from precis.eval.abstract import abstract_markdown, score_against_abstracts
from precis.eval.judge import COVERAGE_SYSTEM, EXTRACT_SYSTEM, coverage
from precis.eval.metrics import bootstrap, percentile, rouge, summary_text
from precis.eval.results import markdown_table, write_results


def test_rouge_is_higher_for_overlapping_text() -> None:
    reference = "We propose a learned cache replacement policy that improves hit rate by 7%."
    close = rouge("A learned cache replacement policy improves the hit rate by 7%.", reference)
    far = rouge("Qubit routing with surface codes reduces gate errors.", reference)
    assert close.rouge1 > 0.6 > far.rouge1
    assert close.rougeL > far.rougeL
    assert rouge(reference, reference).rouge2 == pytest.approx(1.0)


def test_summary_text_uses_abstract_like_fields() -> None:
    summary = make_summary("p", ["caches", "prefetching"])
    text = summary_text(summary)
    assert summary.tldr in text
    assert summary.findings[1].claim in text
    assert summary.limitations[0] not in text


def test_bootstrap_interval_contains_the_mean() -> None:
    estimate = bootstrap([0.0, 0.5, 1.0, 1.0, 0.5] * 10, seed=1)
    assert estimate.low < estimate.mean < estimate.high
    assert estimate.n == 50
    assert bootstrap([]).n == 0
    assert bootstrap([0.4, 0.6], seed=3) == bootstrap([0.4, 0.6], seed=3)


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 95) == 0.0


def _judge(params: dict[str, Any], stage: str) -> dict[str, Any]:
    assert stage == "judge"
    user = params["messages"][0]["content"]
    if params["system"] == EXTRACT_SYSTEM:
        assert "Abstract:" in user
        return {"key_points": ["Point one.", "Point two.", " ", "Point three.", "Point four."]}
    assert params["system"] == COVERAGE_SYSTEM
    assert "Abstract" not in user  # the judge sees key points and the summary only
    return {
        "judgments": [
            {"point": 1, "covered": True, "reason": "r"},
            {"point": 2, "covered": False, "reason": "r"},
            {"point": 3, "covered": True, "reason": "r"},
            {"point": 4, "covered": True, "reason": "r"},
        ]
    }


async def test_key_point_recall_and_rouge_against_abstracts() -> None:
    items = [
        (
            make_meta(f"p{i}", abstract=f"We study {w} and improve it by {i}%."),
            make_summary(f"p{i}", [w, "simulation"]),
        )
        for i, w in enumerate(["cache replacement", "qubit routing", "network flow control"])
    ]
    gateway = ScriptedGateway(_judge)
    result = await score_against_abstracts(gateway, items, AgentConfig(effort="low"))
    assert result.papers == 3
    assert result.scores[0].key_points == [
        "Point one.",
        "Point two.",
        "Point three.",
        "Point four.",
    ]
    assert result.recall.mean == pytest.approx(0.75)
    assert result.rouge.rouge1 > result.rouge_mismatched.rouge1
    assert sorted(c.paper_id for c in gateway.calls) == sorted(["p0", "p1", "p2"] * 2)
    table = abstract_markdown(result, model="claude-opus-5-5")
    assert "| Key-point recall | 0.750 [0.750, 0.750] |  |" in table


async def test_malformed_coverage_answers_only_lower_recall() -> None:
    def script(params: dict[str, Any], stage: str) -> dict[str, Any]:
        return {
            "judgments": [
                {"point": 1, "covered": True, "reason": "r"},
                {"point": 1, "covered": True, "reason": "again"},
                {"point": 3, "covered": True, "reason": "r"},
                {"point": 9, "covered": True, "reason": "unknown"},
            ]
        }

    covered = await coverage(
        ScriptedGateway(script),
        make_meta("p"),
        ["a", "b", "c"],
        make_summary("p", ["x"]),
        AgentConfig(),
    )
    assert covered == [False, False, True]


def test_markdown_table_formats_and_aligns() -> None:
    table = markdown_table(["Name", "n", "Rate"], [["a|b", 1200, 0.5], ["c", 3, 12.25]])
    assert table.splitlines() == [
        "| Name | n | Rate |",
        "| --- | ---: | ---: |",
        "| a\\|b | 1,200 | 0.500 |",
        "| c | 3 | 12.2 |",
    ]


def test_write_results(tmp_path: Path) -> None:
    class Result(BaseModel):
        value: int

    paths = write_results(tmp_path / "results", "demo", Result(value=3), "# Demo")
    assert [p.name for p in paths] == ["demo.json", "demo.md"]
    assert Result.model_validate_json(paths[0].read_text()) == Result(value=3)
    assert paths[1].read_text() == "# Demo\n"
