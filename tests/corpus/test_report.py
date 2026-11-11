import re
from datetime import UTC, datetime

from corpus.factories import make_meta, make_view
from precis.corpus.markup import (
    arxiv_links,
    link_citations,
    markdown_html,
    page_name,
    slugify,
)
from precis.corpus.report import render_report
from precis.corpus.stats import VerdictCounts, corpus_facts, faithfulness, spend
from precis.corpus.synthesis import ReportStats, SynthesisStats
from precis.models import CorpusReport, Topic, TopicSynthesis, UsageRow, Verdict

S, P, U, C = Verdict.SUPPORTED, Verdict.PARTIAL, Verdict.UNSUPPORTED, Verdict.CONTRADICTED


def test_slugify_matches_github_anchors() -> None:
    assert slugify("1. Cache Replacement & Prefetching") == "1-cache-replacement--prefetching"
    assert slugify("About this report") == "about-this-report"


def test_page_names_and_links() -> None:
    assert page_name("2406.01234") == "2406.01234"
    assert page_name("local:ab12/x") == "local_ab12_x"
    assert arxiv_links(make_meta("2406.01234"))["pdf"] == "https://arxiv.org/pdf/2406.01234v1"
    assert arxiv_links(make_meta("local:ab12")) == {}


def test_link_citations_links_known_papers_only() -> None:
    def href(paper_id: str) -> str | None:
        return f"p/{paper_id}" if paper_id == "1" else None

    assert link_citations("A [arXiv:1]; B [arXiv:2].", href) == "A [arXiv:1](p/1); B [arXiv:2]."
    assert link_citations("[arXiv:1, arXiv:2]", href) == "[arXiv:1](p/1) [arXiv:2]"


def test_markdown_rendering_escapes_raw_html() -> None:
    html = markdown_html("Hello **world** <script>alert(1)</script>")
    assert "<strong>world</strong>" in html
    assert "<script>" not in html


def test_faithfulness_counts_claims_before_and_after_revision() -> None:
    papers = [
        make_view("a", ["x", "y"], draft_verdicts=[S, S, S]),
        make_view("b", ["x", "y"], draft_verdicts=[S, C, U], revised=True),
        make_view("c", ["x", "y"], draft_verdicts=[P, S, S]),
    ]
    faith = faithfulness(papers)
    assert (faith.papers, faith.verified, faith.flagged, faith.revised) == (3, 3, 1, 1)
    assert faith.draft == VerdictCounts(supported=6, partial=1, unsupported=1, contradicted=1)
    assert faith.final == VerdictCounts(supported=8, partial=1)
    assert faith.revised_before.rate(S) == 1 / 3
    assert faith.revised_after.rate(S) == 1.0
    assert VerdictCounts().rate(S) == 0.0


def test_spend_and_facts() -> None:
    row = UsageRow(
        stage="read",
        model="m",
        calls=3,
        cached_calls=1,
        input_tokens=100,
        output_tokens=10,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=50,
        cost_usd=0.5,
    )
    total = spend([row, row.model_copy(update={"stage": "verify", "cost_usd": 1.0})])
    assert (total.calls, total.cost_usd) == (6, 1.5)
    assert [r.stage for r in total.by_stage] == ["verify", "read"]
    facts = corpus_facts([make_view("a", ["x"]), make_view("b", ["y"])])
    assert facts.categories == [("cs.AR", 2)]
    assert corpus_facts([]).first is None


def _report() -> tuple[CorpusReport, list[Topic], list]:
    papers = [make_view(f"2507.0000{i}", ["caches", "prefetching"]) for i in range(4)]
    topics = [
        Topic(id=1, label="Caches & Memory", description="D1.", keywords=["cache"],
              paper_ids=["2507.00000", "2507.00001"]),
        Topic(id=2, label="Prefetching", description="D2.", keywords=["prefetch"],
              paper_ids=["2507.00002", "2507.00003"]),
    ]  # fmt: skip
    synthesis = TopicSynthesis(
        topic_id=1,
        overview="Caches improve [arXiv:2507.00000].",
        trends=["Learned policies [arXiv:2507.00001]."],
        open_problems=["Security."],
        key_papers=["2507.00001", "2507.00000"],
    )
    report = CorpusReport(
        title="Recent Computer Architecture",
        generated_at=datetime(2026, 11, 1, tzinfo=UTC),
        n_papers=4,
        introduction="This corpus covers caches.",
        topics=[synthesis],
    )
    return report, topics, papers


def test_report_has_contents_sections_and_linked_citations() -> None:
    report, topics, papers = _report()
    stats = ReportStats(
        topics=[SynthesisStats(topic_id=1, papers=2, requests=1, levels=1, citations=2,
                               invalid_citations=1, invalid_key_papers=0)]
    )  # fmt: skip
    text = render_report(report, topics, papers, stats=stats, faith=faithfulness(papers))
    assert text.startswith("# Recent Computer Architecture\n")
    headings = re.findall(r"^## (.+)$", text, flags=re.M)
    assert headings == [
        "Contents",
        "Introduction",
        "1. Caches & Memory",
        "2. Prefetching",
        "About this report",
    ]
    for heading in headings[1:]:
        assert f"(#{slugify(heading)})" in text
    assert "[arXiv:2507.00000](https://arxiv.org/abs/2507.00000v1)" in text
    key = text.split("**Key papers**")[1]
    assert key.index("2507.00001") < key.index("2507.00000")
    assert "*No synthesis was written for this topic.*" in text
    assert "1 removed" in text
    assert "100.0% of first drafts" in text
    assert text.endswith(".\n")
