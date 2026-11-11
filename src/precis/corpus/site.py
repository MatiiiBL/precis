"""Render the corpus as a static website.

Layout of ``out``::

    index.html               overview: corpus facts, faithfulness, spend, topic cards
    topics/<id>.html         one page per topic: synthesis and its papers
    papers/<id>.html         one page per paper: summary with verification verdicts
    search.html              client-side search
    assets/style.css         one stylesheet, light and dark
    assets/search-index.js   the search index (JSON assigned to a global)

The site has no external scripts, fonts or styles and works from a file
URL. Pages carry no JavaScript except the search page; every other page
has a plain form that opens it. The index is a JSON document wrapped in a
one-line script rather than a ``.json`` file because browsers refuse
``fetch`` on ``file://`` pages, and it is loaded only by the search page,
so paper pages stay small. Building 2,500 paper pages takes a few seconds
and pages are written only for the papers given, with stale pages from a
previous build removed.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from precis.corpus.load import PaperView
from precis.corpus.markup import (
    arxiv_links,
    inline_html,
    link_citations,
    markdown_html,
    page_name,
)
from precis.corpus.stats import Faithfulness, Spend, corpus_facts, faithfulness
from precis.models import ClaimCheck, CorpusReport, Topic, TopicSynthesis, Verdict

logger = logging.getLogger(__name__)

VERDICT_LABELS = {
    Verdict.SUPPORTED: "Supported",
    Verdict.PARTIAL: "Partly supported",
    Verdict.UNSUPPORTED: "Unsupported",
    Verdict.CONTRADICTED: "Contradicted",
}


@dataclass(frozen=True)
class SiteContent:
    """Everything the site shows. ``report`` and ``cost`` may be missing."""

    papers: Sequence[PaperView]
    topics: Sequence[Topic]
    report: CorpusReport | None = None
    cost: Spend | None = None
    title: str = "Précis"


@dataclass(frozen=True)
class SiteBuild:
    pages: int
    bytes: int
    seconds: float
    index_bytes: int


@dataclass(frozen=True)
class _Claim:
    """A claim of the summary with the critic's check, ready for a template."""

    text: str
    check: ClaimCheck | None


def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("precis.corpus", "templates"),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["markdown"] = markdown_html
    env.filters["inline"] = inline_html
    env.filters["percent"] = lambda x: f"{100 * x:.0f}%"
    env.globals["verdict_labels"] = {v.value: label for v, label in VERDICT_LABELS.items()}
    env.globals["page_name"] = page_name
    env.globals["faith_of"] = lambda paper: faithfulness([paper]).final
    return env


def _claims(paper: PaperView) -> tuple[_Claim, list[_Claim]]:
    checks = {c.target: c for c in paper.verification.checks} if paper.verification else {}
    tldr = _Claim(paper.summary.tldr, checks.get("tldr"))
    findings = [
        _Claim(f.claim, checks.get(f"finding:{i}")) for i, f in enumerate(paper.summary.findings)
    ]
    return tldr, findings


def _search_index(content: SiteContent, topic_by_paper: dict[str, Topic]) -> list[dict[str, str]]:
    entries = []
    for paper in content.papers:
        authors = paper.meta.authors
        topic = topic_by_paper.get(paper.id)
        entries.append(
            {
                "id": paper.id,
                "t": paper.meta.title,
                "a": authors[0] + (" et al." if len(authors) > 1 else "") if authors else "",
                "s": paper.summary.tldr,
                "k": ", ".join(paper.summary.keywords),
                "c": topic.label if topic else "",
                "p": f"papers/{page_name(paper.id)}.html",
                "d": paper.meta.published.isoformat(),
            }
        )
    return entries


class _Writer:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.pages = 0
        self.bytes = 0
        self.written: set[Path] = set()

    def write(self, relative: str, text: str) -> None:
        path = self.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode()
        path.write_bytes(data)
        self.written.add(path)
        self.bytes += len(data)
        if relative.endswith(".html"):
            self.pages += 1

    def remove_stale(self, directory: str) -> None:
        for path in (self.out / directory).glob("*.html"):
            if path not in self.written:
                path.unlink()


def build_site(content: SiteContent, out: Path) -> SiteBuild:
    """Write the site for ``content`` into ``out`` (created if needed)."""
    started = time.perf_counter()
    env = _environment()
    writer = _Writer(out)
    by_id = {p.id: p for p in content.papers}
    topic_by_paper = {pid: t for t in content.topics for pid in t.paper_ids}
    syntheses: dict[int, TopicSynthesis] = (
        {s.topic_id: s for s in content.report.topics} if content.report else {}
    )
    faith: Faithfulness = faithfulness(content.papers)
    site_title = content.report.title if content.report else content.title

    def page(root: str) -> dict[str, Any]:
        def cite(text: str) -> str:
            def href(paper_id: str) -> str | None:
                return f"{root}papers/{page_name(paper_id)}.html" if paper_id in by_id else None

            return link_citations(text, href)

        return {"root": root, "site_title": site_title, "cite": cite}

    writer.write(
        "index.html",
        env.get_template("index.html").render(
            **page(""),
            facts=corpus_facts(content.papers),
            faith=faith,
            cost=content.cost,
            report=content.report,
            topics=content.topics,
        ),
    )
    topic_template = env.get_template("topic.html")
    for topic in content.topics:
        papers = [by_id[pid] for pid in topic.paper_ids if pid in by_id]
        synthesis = syntheses.get(topic.id)
        writer.write(
            f"topics/{topic.id}.html",
            topic_template.render(
                **page("../"),
                topic=topic,
                papers=papers,
                synthesis=synthesis,
                key_papers=[by_id[pid] for pid in synthesis.key_papers if pid in by_id]
                if synthesis
                else [],
            ),
        )
    paper_template = env.get_template("paper.html")
    for paper in content.papers:
        tldr, findings = _claims(paper)
        writer.write(
            f"papers/{page_name(paper.id)}.html",
            paper_template.render(
                **page("../"),
                paper=paper,
                topic=topic_by_paper.get(paper.id),
                links=arxiv_links(paper.meta),
                tldr=tldr,
                findings=findings,
                counts=faithfulness([paper]),
            ),
        )
    writer.write("search.html", env.get_template("search.html").render(**page("")))

    index = json.dumps(
        _search_index(content, topic_by_paper), ensure_ascii=False, separators=(",", ":")
    )
    writer.write("assets/search-index.js", f"window.PRECIS_INDEX={index};\n")
    stylesheet = resources.files("precis.corpus").joinpath("static", "style.css")
    with resources.as_file(stylesheet) as css:
        (out / "assets").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(css, out / "assets" / "style.css")
    writer.remove_stale("papers")
    writer.remove_stale("topics")
    seconds = time.perf_counter() - started
    logger.info("site: %d pages, %.1f MB in %.2fs", writer.pages, writer.bytes / 1e6, seconds)
    return SiteBuild(writer.pages, writer.bytes, seconds, len(index.encode()))
