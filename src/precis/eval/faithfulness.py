"""How faithful are the summaries, and what did revision change?

Everything here is computed from stored results; no request is made. The
critic's verdicts are counted per claim before and after revision, and its
evidence is audited: every check must quote the paper verbatim, so a quote
that cannot be found in the paper's text is a sign that the critic itself
hallucinated, and a verdict resting on it is not trustworthy.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from precis.corpus.load import FinishedPaper
from precis.corpus.stats import VerdictCounts, faithfulness
from precis.eval.results import markdown_table
from precis.models import ClaimCheck, Document, PaperSummary, Verdict

_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")
# Curly double and single quotes, by code point so they are unambiguous here.
_STRAIGHT_QUOTES = str.maketrans(
    {chr(0x201C): '"', chr(0x201D): '"', chr(0x2018): "'", chr(0x2019): "'"}
)


def normalize(text: str) -> str:
    """Fold case, Unicode forms and whitespace, so a quote matches despite
    re-wrapped lines, ligatures or curly quotes."""
    text = unicodedata.normalize("NFKC", text).casefold().translate(_STRAIGHT_QUOTES)
    return " ".join(text.split())


def document_text(doc: Document) -> str:
    parts = [s.title + " " + s.text for s in doc.sections] + [c.text for c in doc.captions]
    return normalize(" ".join(parts))


def quote_found(quote: str, normalized_text: str) -> bool:
    """Whether every fragment of ``quote`` (split at ellipses) occurs in the text."""
    fragments = [normalize(f).strip("\"' ") for f in _ELLIPSIS.split(quote)]
    fragments = [f for f in fragments if f]
    return bool(fragments) and all(f in normalized_text for f in fragments)


class QuoteAudit(BaseModel):
    checks: int
    """Checks whose paper text was available."""
    empty: int
    """Checks without a quote (allowed only when there is no evidence)."""
    not_found: int
    """Quotes that do not occur in the paper."""
    not_found_supported: int
    """Of those, checks that still said "supported"."""

    @property
    def not_found_rate(self) -> float:
        quoted = self.checks - self.empty
        return self.not_found / quoted if quoted else 0.0


def audit_quotes(papers: Sequence[FinishedPaper], documents: Mapping[str, Document]) -> QuoteAudit:
    """Audit the evidence of every check (first drafts and revisions)."""
    checks = empty = missing = missing_supported = 0
    for paper in papers:
        doc = documents.get(paper.id)
        if doc is None:
            continue
        text = document_text(doc)
        verifications = [paper.draft_verification]
        if paper.revised:
            verifications.append(paper.verification)
        for verification in verifications:
            for check in verification.checks if verification else []:
                checks += 1
                if not check.evidence.strip():
                    empty += 1
                elif not quote_found(check.evidence, text):
                    missing += 1
                    missing_supported += check.verdict is Verdict.SUPPORTED
    return QuoteAudit(
        checks=checks, empty=empty, not_found=missing, not_found_supported=missing_supported
    )


class RevisionExample(BaseModel):
    paper_id: str
    title: str
    target: str
    verdict: Verdict
    claim: str
    note: str
    revised_claim: str | None
    revised_verdict: Verdict | None


def _claim(summary: PaperSummary, target: str) -> str | None:
    if target == "tldr":
        return summary.tldr
    index = int(target.removeprefix("finding:")) if target.startswith("finding:") else -1
    return summary.findings[index].claim if 0 <= index < len(summary.findings) else None


def revision_examples(papers: Sequence[FinishedPaper], *, limit: int = 5) -> list[RevisionExample]:
    """Failed claims from revised papers, with what the revision says instead.

    A revision may reorder or drop findings, so the revised claim shown is
    the one at the same position, which is usually but not always the
    rewrite of the failed claim.
    """
    examples = []
    for paper in sorted(papers, key=lambda p: p.id):
        if not paper.revised or paper.draft_verification is None:
            continue
        after: dict[str, ClaimCheck] = (
            {c.target: c for c in paper.verification.checks} if paper.verification else {}
        )
        for check in paper.draft_verification.checks:
            if check.verdict not in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED):
                continue
            claim = _claim(paper.draft, check.target)
            if claim is None:
                continue
            revised_check = after.get(check.target)
            examples.append(
                RevisionExample(
                    paper_id=paper.id,
                    title=paper.meta.title,
                    target=check.target,
                    verdict=check.verdict,
                    claim=claim,
                    note=check.note,
                    revised_claim=_claim(paper.summary, check.target),
                    revised_verdict=revised_check.verdict if revised_check else None,
                )
            )
            break
        if len(examples) == limit:
            break
    return examples


class FaithfulnessEval(BaseModel):
    papers: int
    verified: int
    flagged: int
    revised: int
    draft: VerdictCounts
    final: VerdictCounts
    revised_before: VerdictCounts
    revised_after: VerdictCounts
    quotes: QuoteAudit
    examples: list[RevisionExample]


def evaluate_faithfulness(
    papers: Sequence[FinishedPaper], documents: Mapping[str, Document]
) -> FaithfulnessEval:
    faith = faithfulness(papers)
    return FaithfulnessEval(
        papers=faith.papers,
        verified=faith.verified,
        flagged=faith.flagged,
        revised=faith.revised,
        draft=faith.draft,
        final=faith.final,
        revised_before=faith.revised_before,
        revised_after=faith.revised_after,
        quotes=audit_quotes(papers, documents),
        examples=revision_examples(papers),
    )


def faithfulness_markdown(result: FaithfulnessEval) -> str:
    def row(name: str, counts: VerdictCounts) -> list[object]:
        rates = counts.rates()
        return [name, counts.total, *(f"{100 * rates[v.value]:.1f}%" for v in Verdict)]

    headers = ["Claims", "n", *(v.value.capitalize() for v in Verdict)]
    table = markdown_table(
        headers,
        [
            row("All first drafts", result.draft),
            row("All final summaries", result.final),
            row("Revised papers, before", result.revised_before),
            row("Revised papers, after", result.revised_after),
        ],
    )
    quotes = result.quotes
    lines = [
        "# Faithfulness of the summaries",
        "",
        f"{result.verified} of {result.papers} summaries verified; {result.flagged} first drafts "
        f"had an unsupported or contradicted claim and {result.revised} were revised.",
        "",
        table,
        f"Evidence quotes: {quotes.checks - quotes.empty} quoted checks, "
        f"{quotes.not_found} not found verbatim in the paper "
        f"({100 * quotes.not_found_rate:.1f}%; {quotes.not_found_supported} of them marked "
        f"supported), {quotes.empty} checks without a quote.",
    ]
    if result.examples:
        lines += ["", "## Examples of revised claims", ""]
        for ex in result.examples:
            lines.append(f"- **{ex.title}** (`{ex.paper_id}`, {ex.target}, {ex.verdict.value})")
            lines.append(f"  - Draft: {ex.claim}")
            lines.append(f"  - Critic: {ex.note}")
            if ex.revised_claim is not None:
                after = ex.revised_verdict.value if ex.revised_verdict else "not checked"
                lines.append(f"  - Revised ({after}): {ex.revised_claim}")
    return "\n".join(lines) + "\n"
