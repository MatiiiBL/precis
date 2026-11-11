"""Render a ``CorpusReport`` as one Markdown document.

The document is meant to read well on GitHub: a table of contents whose
links use GitHub's heading anchors, one numbered section per topic, citation
tags turned into links to the papers, and a closing section on how the
report was made and how faithful the underlying summaries are.
"""

from __future__ import annotations

from collections.abc import Sequence

from precis.corpus.load import FinishedPaper
from precis.corpus.markup import arxiv_links, count, link_citations, slugify
from precis.corpus.stats import Faithfulness, Spend, corpus_facts
from precis.corpus.synthesis import ReportStats
from precis.models import CorpusReport, Topic, Verdict


def _percent(x: float) -> str:
    return f"{100 * x:.1f}%"


def _topic_heading(topic: Topic) -> str:
    return f"{topic.id}. {topic.label}"


def render_report(
    report: CorpusReport,
    topics: Sequence[Topic],
    papers: Sequence[FinishedPaper],
    *,
    stats: ReportStats | None = None,
    faith: Faithfulness | None = None,
    cost: Spend | None = None,
) -> str:
    """The report as Markdown; ``stats``, ``faith`` and ``cost`` fill the closing section."""
    by_id = {p.id: p for p in papers}
    syntheses = {s.topic_id: s for s in report.topics}

    def href(paper_id: str) -> str | None:
        paper = by_id.get(paper_id)
        return arxiv_links(paper.meta).get("abs") if paper else None

    facts = corpus_facts(papers)
    span = f"{facts.first} to {facts.last}" if facts.first else "no dates"
    out = [
        f"# {report.title}",
        "",
        f"*{count(report.n_papers, 'paper')} published {span}, in "
        f"{count(len(topics), 'topic')}. Generated {report.generated_at:%Y-%m-%d}.*",
        "",
        "## Contents",
        "",
        "- [Introduction](#introduction)",
    ]
    for topic in topics:
        heading = _topic_heading(topic)
        out.append(f"- [{heading}](#{slugify(heading)}) ({count(len(topic.paper_ids), 'paper')})")
    out += ["- [About this report](#about-this-report)", "", "## Introduction", ""]
    out += [report.introduction.strip(), ""]

    for topic in topics:
        size = count(len(topic.paper_ids), "paper")
        out += [f"## {_topic_heading(topic)}", ""]
        out += [f"*{size}. Keywords: {', '.join(topic.keywords)}.*", ""]
        out += [topic.description, ""]
        synthesis = syntheses.get(topic.id)
        if synthesis is None:
            out += ["*No synthesis was written for this topic.*", ""]
            continue
        out += [link_citations(synthesis.overview, href), ""]
        if synthesis.trends:
            out += ["**Trends**", ""]
            out += [f"- {link_citations(t, href)}" for t in synthesis.trends] + [""]
        if synthesis.open_problems:
            out += ["**Open problems**", ""]
            out += [f"- {link_citations(p, href)}" for p in synthesis.open_problems] + [""]
        key = [by_id[pid] for pid in synthesis.key_papers if pid in by_id]
        if key:
            out += ["**Key papers**", ""]
            for paper in key:
                link = href(paper.id)
                title = f"[{paper.meta.title}]({link})" if link else paper.meta.title
                out.append(f"- {title}: {paper.summary.tldr}")
            out.append("")

    out += ["## About this report", ""]
    out += [
        "Each paper was read and summarized by an LLM agent, and every claim in the "
        "summary (the TL;DR and each finding) was checked against the paper's text by "
        "a second agent; summaries with unsupported or contradicted claims were "
        "rewritten once and checked again. Topics come from TF-IDF and k-means over "
        "the summaries, and each topic section was written from its papers' TL;DRs "
        "and findings.",
        "",
    ]
    if faith is not None and faith.final.total:
        out.append(
            f"- Claims supported by the paper: {_percent(faith.draft.rate(Verdict.SUPPORTED))} "
            f"of first drafts, {_percent(faith.final.rate(Verdict.SUPPORTED))} after revision "
            f"({faith.revised} of {faith.verified} summaries revised)."
        )
    if stats is not None:
        out.append(
            f"- Citations: {stats.citations} kept; {stats.invalid_citations} removed because "
            "they named a paper outside the topic."
        )
        if stats.failed:
            missing = ", ".join(str(f.topic_id) for f in stats.failed)
            out.append(f"- No synthesis could be written for topic(s) {missing}; see the logs.")
    if cost is not None:
        out.append(
            f"- Model usage: {cost.calls:,} requests, {cost.input_tokens:,} input and "
            f"{cost.output_tokens:,} output tokens, ${cost.cost_usd:,.2f}."
        )
    return "\n".join(out).rstrip() + "\n"
