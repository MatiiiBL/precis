"""Write one synthesis per topic and the report's introduction.

A topic's synthesis is written from compact summaries of its papers (tldr
and findings, about 150 tokens each), tagged with the citation the model
must use. A topic whose compact summaries exceed
``corpus.synthesis_input_tokens`` is summarized hierarchically instead of
being cut: its papers are dealt round-robin, in centrality order, into the
fewest chunks that fit, each chunk gets a partial synthesis, and the
partials are merged (in groups, recursively, if they do not fit either).
Every paper is read by some request, and every chunk is a cross-section of
the topic rather than its most or least central slice.

Citations are checked against the topic: a citation of a paper outside it
is a hallucination or a cross-topic leak, and is removed and counted.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from precis.config import AgentConfig
from precis.corpus.load import PaperView
from precis.corpus.structured import request_structured, structured_params
from precis.llm.base import ModelGateway
from precis.models import CorpusReport, Topic, TopicSynthesis
from precis.util import estimate_tokens

# "[arXiv:2406.01234]"; grouped forms such as "[arXiv:2406.01234, arXiv:2407.0042]"
# are accepted and split, since models produce them despite instructions.
CITATION = re.compile(r"\s*\[arXiv:\s*([^\]]+)\]")


def citation(paper_id: str) -> str:
    return f"[arXiv:{paper_id}]"


def cited_ids(text: str) -> list[str]:
    """Paper ids cited in ``text``, in order of appearance (with repeats)."""
    return [pid for match in CITATION.finditer(text) for pid in _split_ids(match.group(1))]


def _split_ids(group: str) -> list[str]:
    ids = []
    for part in re.split(r"[,;]", group):
        token = part.strip()
        token = token.removeprefix("arXiv:").strip()
        if token:
            ids.append(token)
    return ids


def clean_citations(text: str, allowed: set[str]) -> tuple[str, int, int]:
    """Drop citations of papers not in ``allowed``.

    Returns the cleaned text, the number of valid citations kept and the
    number removed. Grouped citations are rewritten one id per bracket.
    """
    kept = removed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal kept, removed
        ids = _split_ids(match.group(1))
        valid = [pid for pid in ids if pid in allowed]
        kept += len(valid)
        removed += len(ids) - len(valid)
        if not valid:
            return ""
        lead = match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip())]
        return lead + " ".join(citation(pid) for pid in valid)

    return CITATION.sub(replace, text), kept, removed


# --------------------------------------------------------------------------
# What the model writes
# --------------------------------------------------------------------------


class SynthesisDraft(BaseModel):
    """One topic section, before its citations are checked."""

    overview: str = Field(
        description="3-5 paragraphs of Markdown prose, no headings, citing papers by their tags."
    )
    trends: list[str] = Field(description="3-6 one-sentence trends, with citations.")
    open_problems: list[str] = Field(description="2-5 open problems or gaps, with citations.")
    key_papers: list[str] = Field(
        description="Ids (without the arXiv: prefix) of the 3-6 most representative papers, "
        "most representative first."
    )


class IntroductionDraft(BaseModel):
    title: str = Field(description="A specific title for the report, at most 12 words.")
    introduction: str = Field(description="2-4 paragraphs of Markdown prose, no headings.")


SYNTHESIS_SYSTEM = """\
You write one section of a survey of recent research papers. You are given a \
topic's name and description and compact summaries of its papers (TL;DR and key \
findings), each tagged with its citation, such as [arXiv:2406.01234].

Write:
- overview: 3-5 paragraphs of Markdown prose without headings that synthesize the \
topic. Group related papers, compare their approaches and results, and give \
concrete numbers when the summaries do. Support statements about specific papers \
with their citation tags, copied exactly; put one tag in each pair of brackets and \
cite only papers from the list.
- trends: 3-6 one-sentence trends across the papers, with citations.
- open_problems: 2-5 open problems or gaps that the papers' findings and \
limitations point to, with citations.
- key_papers: the ids of the 3-6 papers that best represent the topic, most \
representative first, without the "arXiv:" prefix.

Write for researchers in the field: precise, neutral and specific. Do not \
speculate beyond what the summaries say."""

PARTIAL_NOTE = """\
This topic has too many papers for one request, so you see part {part} of {parts} \
of its papers, a cross-section of the whole topic. Write the section for these \
papers only; a later step merges the parts."""

MERGE_SYSTEM = """\
You merge partial drafts of one section of a survey of recent research papers. \
Each draft was written from a different subset of the same topic's papers. \
Combine them into one section with the same parts (overview, trends, \
open_problems, key_papers): remove repetition, connect related findings across \
drafts, and keep the most important points.

Keep citation tags exactly as they appear in the drafts, such as \
[arXiv:2406.01234], one tag in each pair of brackets, and do not cite papers that \
no draft cites. The overview is 3-5 paragraphs of Markdown prose without \
headings. key_papers lists the ids of the 3-6 most representative papers, most \
representative first, without the "arXiv:" prefix."""

INTRO_SYSTEM = """\
You write the title and introduction of a report that surveys a corpus of recent \
research papers. The report has one section per topic; you are given statistics \
about the corpus and each topic's name, size, description and trends.

The introduction is 2-4 paragraphs of Markdown prose without headings: the scope \
of the corpus, the main themes and how they relate, and what stands out across \
topics. Refer to topics by name. Do not cite individual papers. The title is \
specific to this corpus, at most 12 words."""


def compact_entry(paper: PaperView) -> str:
    """A paper as the synthesizer sees it: citation tag, title, tldr and findings."""
    lines = [f"{citation(paper.id)} {paper.meta.title}", f"TL;DR: {paper.summary.tldr}"]
    lines += [f"- {finding.claim}" for finding in paper.summary.findings]
    return "\n".join(lines)


def _topic_header(topic: Topic) -> str:
    return f"Topic: {topic.label}\nDescription: {topic.description}"


def topic_prompt(topic: Topic, entries: Sequence[str], part: tuple[int, int] | None) -> str:
    parts = [_topic_header(topic)]
    if part is not None:
        parts.append(PARTIAL_NOTE.format(part=part[0], parts=part[1]))
    parts.append(f"Papers ({len(entries)}):\n\n" + "\n\n".join(entries))
    return "\n\n".join(parts)


def draft_text(draft: SynthesisDraft) -> str:
    """A partial draft as the merge step sees it."""
    lines = [draft.overview, "", "Trends:"]
    lines += [f"- {t}" for t in draft.trends]
    lines += ["", "Open problems:"]
    lines += [f"- {p}" for p in draft.open_problems]
    lines += ["", "Key papers: " + ", ".join(draft.key_papers)]
    return "\n".join(lines)


def merge_prompt(topic: Topic, drafts: Sequence[SynthesisDraft]) -> str:
    blocks = [f"Draft {i}:\n{draft_text(d)}" for i, d in enumerate(drafts, start=1)]
    return _topic_header(topic) + "\n\n" + "\n\n".join(blocks)


def deal(items: Sequence[str], budget: int) -> list[list[int]]:
    """Split ``items`` into the fewest round-robin chunks whose tokens fit ``budget``.

    Returns indices into ``items``. Raises ``ValueError`` when a single item is
    larger than the budget, since no split could fit it without cutting it.
    """
    sizes = [estimate_tokens(item) for item in items]
    if sizes and max(sizes) > budget:
        raise ValueError(f"an entry of {max(sizes)} tokens exceeds the {budget}-token budget")
    n = max(1, math.ceil(sum(sizes) / budget))
    while True:
        chunks = [list(range(start, len(items), n)) for start in range(n)]
        if all(sum(sizes[i] for i in chunk) <= budget for chunk in chunks):
            return [chunk for chunk in chunks if chunk]
        n += 1


def group(items: Sequence[str], budget: int) -> list[list[int]]:
    """Pack ``items`` greedily into consecutive groups whose tokens fit ``budget``.

    An item larger than the budget gets a group of its own; the caller
    detects a level that cannot shrink.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for i, item in enumerate(items):
        size = estimate_tokens(item)
        if current and used + size > budget:
            groups.append(current)
            current, used = [], 0
        current.append(i)
        used += size
    if current:
        groups.append(current)
    return groups


# --------------------------------------------------------------------------
# Running the requests
# --------------------------------------------------------------------------


class SynthesisStats(BaseModel):
    """How one topic's synthesis was produced and how many citations were dropped."""

    topic_id: int
    papers: int
    requests: int
    levels: int = Field(description="1 = one request; 2 = chunks then a merge; ...")
    citations: int
    invalid_citations: int
    invalid_key_papers: int


class ReportStats(BaseModel):
    topics: list[SynthesisStats]

    @property
    def invalid_citations(self) -> int:
        return sum(t.invalid_citations for t in self.topics)

    @property
    def citations(self) -> int:
        return sum(t.citations for t in self.topics)


REPORT_ARTIFACT = "corpus/report"
REPORT_STATS_ARTIFACT = "corpus/report-stats"


@dataclass
class _Synthesizer:
    gw: ModelGateway
    agent: AgentConfig
    budget: int
    requests: int = 0

    async def _ask(self, system: str, user: str) -> SynthesisDraft:
        self.requests += 1
        params = structured_params(self.agent, system=system, user=user, output=SynthesisDraft)
        return await request_structured(self.gw, params, SynthesisDraft, stage="synthesize")

    async def run(self, topic: Topic, papers: Sequence[PaperView]) -> tuple[SynthesisDraft, int]:
        entries = [compact_entry(p) for p in papers]
        chunks = deal(entries, self.budget)
        if len(chunks) == 1:
            return await self._ask(SYNTHESIS_SYSTEM, topic_prompt(topic, entries, None)), 1
        drafts = await asyncio.gather(
            *(
                self._ask(
                    SYNTHESIS_SYSTEM,
                    topic_prompt(topic, [entries[i] for i in chunk], (n, len(chunks))),
                )
                for n, chunk in enumerate(chunks, start=1)
            )
        )
        levels = 2
        while True:
            texts = [merge_prompt(topic, [d]) for d in drafts]
            groups = group(texts, self.budget)
            if len(groups) == 1:
                return await self._ask(MERGE_SYSTEM, merge_prompt(topic, drafts)), levels
            if all(len(g) == 1 for g in groups):
                raise ValueError(
                    f"topic {topic.id}: partial drafts are too long to merge within "
                    f"{self.budget} tokens"
                )
            merges = [merge_prompt(topic, [drafts[i] for i in g]) for g in groups]
            drafts = await asyncio.gather(*(self._ask(MERGE_SYSTEM, m) for m in merges))
            levels += 1


def _finish(
    topic: Topic, draft: SynthesisDraft, requests: int, levels: int
) -> tuple[TopicSynthesis, SynthesisStats]:
    allowed = set(topic.paper_ids)
    kept = removed = 0

    def check(text: str) -> str:
        nonlocal kept, removed
        cleaned, k, r = clean_citations(text, allowed)
        kept += k
        removed += r
        return cleaned.strip()

    overview = check(draft.overview)
    trends = [t for t in (check(t) for t in draft.trends) if t]
    open_problems = [p for p in (check(p) for p in draft.open_problems) if p]
    key_papers: list[str] = []
    invalid_key = 0
    for raw in draft.key_papers:
        paper_id = raw.strip().removeprefix("arXiv:").strip()
        if paper_id not in allowed:
            invalid_key += 1
        elif paper_id not in key_papers:
            key_papers.append(paper_id)
    if not key_papers:
        key_papers = topic.paper_ids[:3]
    synthesis = TopicSynthesis(
        topic_id=topic.id,
        overview=overview,
        trends=trends,
        open_problems=open_problems,
        key_papers=key_papers,
    )
    stats = SynthesisStats(
        topic_id=topic.id,
        papers=len(topic.paper_ids),
        requests=requests,
        levels=levels,
        citations=kept,
        invalid_citations=removed,
        invalid_key_papers=invalid_key,
    )
    return synthesis, stats


async def synthesize_topic(
    gw: ModelGateway,
    topic: Topic,
    papers: Sequence[PaperView],
    agent: AgentConfig,
    *,
    budget: int,
) -> tuple[TopicSynthesis, SynthesisStats]:
    """Write ``topic``'s synthesis from ``papers`` (given in centrality order)."""
    synthesizer = _Synthesizer(gw, agent, budget)
    draft, levels = await synthesizer.run(topic, papers)
    return _finish(topic, draft, synthesizer.requests, levels)


def intro_prompt(
    topics: Sequence[Topic], syntheses: Sequence[TopicSynthesis], papers: Sequence[PaperView]
) -> str:
    categories = Counter(p.meta.primary_category for p in papers)
    top = sorted(categories.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    lines = [f"Papers: {len(papers)}"]
    if papers:
        dates = sorted(p.meta.published for p in papers)
        lines.append(f"Published: {dates[0].isoformat()} to {dates[-1].isoformat()}")
    lines += ["Primary categories: " + ", ".join(f"{c} ({n})" for c, n in top), "", "Topics:"]
    by_id = {s.topic_id: s for s in syntheses}
    for topic in topics:
        lines += ["", f"{topic.id}. {topic.label} ({len(topic.paper_ids)} papers)"]
        lines.append(f"Description: {topic.description}")
        synthesis = by_id.get(topic.id)
        if synthesis is not None:
            lines += [f"- {_strip_citations(t)}" for t in synthesis.trends]
    return "\n".join(lines)


def _strip_citations(text: str) -> str:
    return CITATION.sub("", text).strip()


async def write_report(
    gw: ModelGateway,
    topics: Sequence[Topic],
    papers: Sequence[PaperView],
    agent: AgentConfig,
    *,
    budget: int,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[CorpusReport, ReportStats]:
    """Synthesize every topic (concurrently), then write the introduction."""
    by_id = {p.id: p for p in papers}
    results = await asyncio.gather(
        *(
            synthesize_topic(
                gw,
                topic,
                [by_id[pid] for pid in topic.paper_ids if pid in by_id],
                agent,
                budget=budget,
            )
            for topic in topics
        )
    )
    syntheses = [synthesis for synthesis, _ in results]
    params: dict[str, Any] = structured_params(
        agent,
        system=INTRO_SYSTEM,
        user=intro_prompt(topics, syntheses, papers),
        output=IntroductionDraft,
    )
    intro = await request_structured(gw, params, IntroductionDraft, stage="introduce")
    report = CorpusReport(
        title=intro.title.strip(),
        generated_at=now(),
        n_papers=len(papers),
        introduction=intro.introduction.strip(),
        topics=syntheses,
    )
    return report, ReportStats(topics=[stats for _, stats in results])
