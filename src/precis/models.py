"""Records shared by every pipeline stage.

These models are the contract between ingestion, the agents, the corpus
stages and the store: each stage reads and writes only these types, and the
store keeps them as JSON. A field change here changes what is on disk, so it
needs a matching migration in ``precis.store``.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    """Base for every stored record: unknown fields are an error, not ignored."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Papers and their text
# --------------------------------------------------------------------------


class PaperMeta(Record):
    """Bibliographic metadata, as returned by the arXiv API."""

    id: str = Field(
        description='arXiv id without version ("2406.01234"), or "local:<sha256[:16]>".'
    )
    version: int | None = None
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    primary_category: str
    published: date
    updated: date | None = None
    license: str | None = None
    url: str = Field(description="Landing page (arXiv abs URL or file path).")


class Section(Record):
    """One section or subsection of a paper's body text."""

    id: str = Field(description='Stable within the document: "s1", "s2.3", "a1" for appendices.')
    title: str
    level: int = Field(ge=1, le=4, description="1 = section, 2 = subsection, ...")
    text: str
    n_tokens: int = Field(ge=0, description="precis.util.estimate_tokens(text).")


class Caption(Record):
    """A figure or table caption (tables include their cell text when available)."""

    id: str = Field(description='"fig1", "tab2", ...')
    kind: Literal["figure", "table"]
    text: str


class Document(Record):
    """A paper's full text, split into sections.

    The abstract lives in ``PaperMeta`` and the bibliography is dropped, so an
    agent that is not shown the abstract cannot read it from here either.
    """

    paper_id: str
    source_format: Literal["html", "pdf", "text"]
    sections: list[Section]
    captions: list[Caption]
    n_tokens: int = Field(ge=0, description="Sum of the section token estimates.")
    sha256: str = Field(description="Hash of the raw downloaded file.")


# --------------------------------------------------------------------------
# Per-paper outputs
# --------------------------------------------------------------------------


class Finding(Record):
    """One key result, tied to the sections that support it."""

    claim: str = Field(
        description="One specific sentence; use the paper's own numbers when it gives them."
    )
    section_ids: list[str] = Field(description="Ids of the sections that contain the evidence.")


class SummaryDraft(Record):
    """What a summarizing agent writes. The pipeline wraps it in ``PaperSummary``."""

    tldr: str = Field(description="At most 40 words: what the paper does and its headline result.")
    problem: str = Field(description="The problem and why it matters, in 1-3 sentences.")
    approach: str = Field(description="How the paper attacks it, in 2-4 sentences.")
    findings: list[Finding] = Field(description="2-6 key results.")
    contributions: list[str]
    limitations: list[str] = Field(description="Stated by the authors or evident from the setup.")
    methods: list[str] = Field(description="Named techniques, models or algorithms used.")
    datasets: list[str] = Field(description="Datasets, benchmarks or workloads evaluated on.")
    keywords: list[str] = Field(description="3-8 lowercase topic keywords.")


class PaperSummary(SummaryDraft):
    paper_id: str
    revision: int = Field(default=0, ge=0, description="0 = first draft, 1 = after verification.")


class Verdict(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ClaimCheck(Record):
    target: str = Field(description='"tldr" or "finding:<index into PaperSummary.findings>".')
    verdict: Verdict
    evidence: str = Field(description="Verbatim quote from the paper; empty when there is none.")
    section_id: str | None = None
    note: str = Field(description="Why the verdict was given, in one sentence.")


class Verification(Record):
    paper_id: str
    revision: int = Field(ge=0, description="The summary revision that was checked.")
    checks: list[ClaimCheck]

    def needs_revision(self) -> bool:
        return any(c.verdict in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED) for c in self.checks)

    def supported_fraction(self) -> float:
        if not self.checks:
            return 1.0
        return sum(c.verdict is Verdict.SUPPORTED for c in self.checks) / len(self.checks)


class RevisionResult(Record):
    """Output of the revise stage: the rewritten summary and its re-verification.

    The first draft and its verification stay in the read and verify stage
    records, so faithfulness can be compared before and after revision.
    """

    summary: PaperSummary = Field(description="The rewritten summary (revision 1).")
    verification: Verification = Field(description="Verification of the rewritten summary.")


# --------------------------------------------------------------------------
# Corpus outputs
# --------------------------------------------------------------------------


class Topic(Record):
    id: int
    label: str
    description: str
    keywords: list[str]
    paper_ids: list[str]


class TopicSynthesis(Record):
    topic_id: int
    overview: str = Field(description="Markdown; cites papers as [arXiv:2406.01234].")
    trends: list[str]
    open_problems: list[str]
    key_papers: list[str] = Field(description="Paper ids, most representative first.")


class CorpusReport(Record):
    title: str
    generated_at: datetime
    n_papers: int
    introduction: str
    topics: list[TopicSynthesis]


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


class Stage(StrEnum):
    """Per-paper stages, in pipeline order."""

    FETCH = "fetch"
    READ = "read"
    VERIFY = "verify"
    REVISE = "revise"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    REFUSED = "refused"
    SKIPPED = "skipped"


class StageRecord(Record):
    paper_id: str
    stage: Stage
    status: StageStatus
    attempts: int = Field(ge=0)
    config_hash: str = Field(
        description="Hash of model, prompt version and stage settings; a change re-runs the stage."
    )
    run_id: str
    error: str | None = None
    updated_at: datetime
    output: dict[str, Any] | None = Field(
        default=None, description="JSON of the stage's output record (PaperSummary, ...)."
    )


class LLMCall(Record):
    """One Messages API request (or one cache hit standing in for it)."""

    call_id: str
    run_id: str
    paper_id: str | None
    stage: str = Field(description='Stage or corpus step: "read", "verify", "label", "judge", ...')
    model: str = Field(description="The model that served the response.")
    request_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    attempts: int = 1
    status: Literal["ok", "cached", "error", "refusal"]
    stop_reason: str | None = None
    request_id: str | None = None
    batch: bool = False
    started_at: datetime


class UsageRow(Record):
    """Aggregated usage for one (stage, model) pair."""

    stage: str
    model: str
    calls: int
    cached_calls: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float
