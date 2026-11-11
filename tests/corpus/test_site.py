import json
import re
from datetime import UTC, datetime
from pathlib import Path

from corpus.factories import make_view, themed_corpus
from precis.corpus.load import PaperView
from precis.corpus.site import SiteContent, build_site
from precis.corpus.stats import spend
from precis.models import CorpusReport, Topic, TopicSynthesis, UsageRow, Verdict

S, C, U = Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.UNSUPPORTED


def _content(papers: list[PaperView]) -> SiteContent:
    half = len(papers) // 2
    topics = [
        Topic(id=1, label="Cache Replacement", description="Caches.", keywords=["cache"],
              paper_ids=[p.id for p in papers[:half]]),
        Topic(id=2, label="Quantum <Routing>", description="Qubits.", keywords=["qubit"],
              paper_ids=[p.id for p in papers[half:]]),
    ]  # fmt: skip
    first = papers[0].id
    report = CorpusReport(
        title="Architecture in 2025",
        generated_at=datetime(2026, 11, 1, tzinfo=UTC),
        n_papers=len(papers),
        introduction="An **introduction**.",
        topics=[
            TopicSynthesis(
                topic_id=1,
                overview=f"Caches matter [arXiv:{first}]. <script>alert(1)</script>",
                trends=[f"Learning [arXiv:{first}]."],
                open_problems=["Security."],
                key_papers=[first],
            )
        ],
    )
    usage = [
        UsageRow(stage="read", model="claude-opus-5-5", calls=10, cached_calls=0,
                 input_tokens=1000, output_tokens=100, cache_creation_input_tokens=0,
                 cache_read_input_tokens=500, cost_usd=1.25),
    ]  # fmt: skip
    return SiteContent(papers=papers, topics=topics, report=report, cost=spend(usage))


def test_site_has_every_page_and_asset(tmp_path: Path) -> None:
    papers, _ = themed_corpus(5)
    build = build_site(_content(papers), tmp_path)
    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "search.html").is_file()
    assert (tmp_path / "assets" / "style.css").stat().st_size > 1000
    assert sorted(p.name for p in (tmp_path / "topics").iterdir()) == ["1.html", "2.html"]
    assert len(list((tmp_path / "papers").iterdir())) == len(papers)
    assert build.pages == 1 + 2 + len(papers) + 1


def test_pages_reference_only_local_resources(tmp_path: Path) -> None:
    papers, _ = themed_corpus(3)
    build_site(_content(papers), tmp_path)
    for page in tmp_path.rglob("*.html"):
        html = page.read_text()
        for url in re.findall(r'(?:src|href)="([^"]+)"', html):
            if url.startswith("https://arxiv.org/"):
                continue
            assert "://" not in url, (page, url)
            target = (page.parent / url.split("?")[0]).resolve()
            assert target.exists(), (page, url)


def test_overview_shows_stats_topics_and_spend(tmp_path: Path) -> None:
    papers = [
        make_view("2507.00001", ["a", "b"], draft_verdicts=[S, C, U], revised=True),
        make_view("2507.00002", ["a", "b"]),
    ]
    build_site(_content(papers), tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert "<title>Architecture in 2025</title>" in html
    assert "<strong>introduction</strong>" in html
    assert "Quantum &lt;Routing&gt;" in html
    assert "$1.25" in html
    assert "1 of 2 first drafts had an unsupported or contradicted claim" in html
    assert "from 33% to 100%" in html


def test_topic_page_links_citations_and_escapes_model_html(tmp_path: Path) -> None:
    papers, _ = themed_corpus(3)
    build_site(_content(papers), tmp_path)
    html = (tmp_path / "topics" / "1.html").read_text()
    first = papers[0].id
    assert f'<a href="../papers/{first}.html">arXiv:{first}</a>' in html
    assert "<script>alert" not in html
    assert html.count('class="paper-row"') == 6 + 1  # all papers plus one key paper
    other = (tmp_path / "topics" / "2.html").read_text()
    assert "Overview" in other  # breadcrumb only: no synthesis for topic 2
    assert "<h2>Overview</h2>" not in other


def test_paper_page_shows_verdicts_evidence_and_revision(tmp_path: Path) -> None:
    revised = make_view("2507.00001", ["cache replacement", "prefetching"],
                        draft_verdicts=[S, C, U], revised=True)  # fmt: skip
    local = make_view("local:ab12", ["qubit routing", "surface code"])
    build_site(_content([revised, local]), tmp_path)
    html = (tmp_path / "papers" / "2507.00001.html").read_text()
    assert html.count('class="badge v-supported"') == 3
    assert "2 claims were not supported by the paper in the first draft" in html
    assert "<blockquote>quote</blockquote>" in html
    assert 'href="https://arxiv.org/pdf/2507.00001v1"' in html
    assert 'href="../topics/1.html"' in html
    local_html = (tmp_path / "papers" / "local_ab12.html").read_text()
    assert "arxiv.org" not in local_html


def test_search_index_covers_every_paper(tmp_path: Path) -> None:
    papers, _ = themed_corpus(4)
    build = build_site(_content(papers), tmp_path)
    script = (tmp_path / "assets" / "search-index.js").read_text()
    assert script.startswith("window.PRECIS_INDEX=")
    entries = json.loads(script.removeprefix("window.PRECIS_INDEX=").rstrip().rstrip(";"))
    assert [e["id"] for e in entries] == [p.id for p in papers]
    assert all((tmp_path / e["p"]).is_file() for e in entries)
    assert entries[0]["c"] == "Cache Replacement"
    assert build.index_bytes == len(script.encode()) - len("window.PRECIS_INDEX=;\n")


def test_rebuild_removes_stale_pages(tmp_path: Path) -> None:
    papers, _ = themed_corpus(3)
    build_site(_content(papers), tmp_path)
    build_site(_content(papers[:4]), tmp_path)
    assert len(list((tmp_path / "papers").iterdir())) == 4


def test_site_without_report_or_cost(tmp_path: Path) -> None:
    papers, _ = themed_corpus(2)
    content = _content(papers)
    bare = SiteContent(papers=content.papers, topics=content.topics)
    build_site(bare, tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert "<title>Précis</title>" in html
    assert "Model usage" not in html
