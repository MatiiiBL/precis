"""PDF to ``Document`` with pypdf, for papers without a usable HTML rendering.

A PDF has no structure to read, only positioned glyphs, so the parser works
on the text lines pypdf extracts and rebuilds the structure with heuristics
tuned on IEEE and ACM two-column papers:

1. Page furniture goes first: lines repeated at the top or bottom of many
   pages (running headers and footers), page numbers, and publisher
   boilerplate (ACM permission blocks, IEEE copyright lines).
2. Everything from the last "References"/"Bibliography" heading on is
   dropped, except an appendix that follows it.
3. Headings are found by pattern (``1 Introduction``, ``2.3 Title``,
   ``II. BACKGROUND``, ``A. Subsection``) and then filtered by numbering: the
   longest run of consecutively numbered candidates wins, so figure labels,
   list items and stray numbers that merely look like headings are ignored.
   Papers without numbered headings fall back to standalone all-caps lines
   and common section names; papers without any are cut into parts.
4. Text before the first heading (title, authors, abstract) is dropped.
   Captions ("Fig. 3: ...", "TABLE II ...") are moved out of the text, and
   runs of short non-prose lines (figure labels, table cells, equations
   mangled by extraction) are removed.
5. Lines are joined into paragraphs and words hyphenated across line breaks
   are rejoined, using the document's own vocabulary to decide between
   "archi-tecture" (join) and "column-major" (keep the hyphen).
"""

from __future__ import annotations

import io
import logging
import math
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Literal

from pypdf import PdfReader

from precis.models import Caption, Document
from precis.text.document import ParseError, SectionDraft, build_document, split_into_parts
from precis.text.normalize import clean_chars, squash
from precis.util import estimate_tokens, sha256_hex

logger = logging.getLogger(__name__)

# Fraction of pages whose text may fail to extract before the PDF is rejected.
MAX_FAILED_PAGES = 0.25

# An outline with fewer than DOMINANCE_MAX_SECTIONS sections, one of which
# holds over DOMINANCE_SHARE of a paper of at least DOMINANCE_MIN_TOKENS, is
# rejected (see _one_section_dominates). Short papers are exempt: a 2,000-token
# extended abstract can legitimately be one long section.
DOMINANCE_SHARE = 0.8
DOMINANCE_MAX_SECTIONS = 6
DOMINANCE_MIN_TOKENS = 3000

_FURNITURE_ZONE = 2  # lines at the top and bottom of a page checked for headers
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlc]{1,6})(?:\s+of\s+\d{1,4})?$", re.I)
_BOILERPLATE = re.compile(
    r"^(?:©|\(c\)\s|copyright\s+(?:©\s*)?\d{4})"
    r"|^acm isbn\b"
    r"|^https?://doi\.org/\S+$"
    r"|^\d{3}-\d+-\d{4}-\d{4}-\d/\d{2}/\$\d+\.\d{2}"
    r"|^arxiv:\d{4}\.\d{4,5}(?:v\d+)?\s+\[",
    re.I,
)
_PERMISSION_START = re.compile(r"^permission to make digital or hard copies", re.I)
_PERMISSION_END = re.compile(r"acm isbn|\$15\.00|https?://doi\.org/", re.I)

_REFERENCES = re.compile(
    r"^(?:(?:\d{1,2}|[IVX]{1,5})\.?\s+)?(?:references|bibliography|r eferences)[.:]?$", re.I
)
_APPENDIX = re.compile(r"^(?:[A-H]\.?\s+)?(?:appendix|appendices)\b", re.I)
_DROP_HEADINGS = re.compile(
    r"^(?:abstract|acknowledge?ments?|a cknowledgments?|a cknowledgements?"
    r"|index terms|keywords|ccs concepts|acm reference format)$",
    re.I,
)
_ABSTRACT_START = re.compile(r"^abstract\b\W*", re.I)
_FRONT_MATTER_END = re.compile(
    r"^(?:index terms|keywords|ccs concepts|acm reference format)\b", re.I
)

_ARABIC = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+(?P<title>\S.*)$")
_ROMAN = re.compile(r"^(?P<num>[IVX]{1,5})\.\s+(?P<title>\S.*)$")
_LETTER = re.compile(r"^(?P<num>[A-H])\.\s+(?P<title>\S.*)$")
_APPENDIX_NUMBERED = re.compile(r"^(?P<num>[A-H](?:\.\d{1,2}){0,2})\.?\s+(?P<title>\S.*)$")
_LONE_NUMBER = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,2})\.?$")
_IEEE_RUN_IN = re.compile(r"^(?P<num>\d{1,2})\)\s+(?P<title>[A-Z][^:]{1,80}):\s*(?P<rest>.*)$")
_RUN_IN = re.compile(r"^(?P<title>[^.]{2,80}?)\.\s?(?P<rest>[A-Z(\[](?:\S+\s+){3,}.*)$")
_CONNECTORS = frozenset({"and", "or", "of", "for", "the", "with", "in", "on", "to", "via", "a"})
_NUMBERED_REFERENCE = re.compile(r"^\[\d{1,3}\]")
# Table-of-contents entries: a title ending in a page number, the strong
# forms numbered ("2.1 Title 5") or with dot leaders ("Title . . . . 5").
_TOC_ENTRY = re.compile(r"[A-Za-z].*\s\d{1,3}$")
# Entries that count towards a block need two real words, so table rows that
# end in numbers ("3.1 56 X 56 X 128 3 X 3 X 128 X 256") never form one.
_TOC_WORDS = re.compile(r"[A-Za-z]{3,}\W.*[A-Za-z]{3,}")
_TOC_NUMBERED = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}|[A-H]|[IVX]{1,5})\.?\s+\S")
_TOC_LEADERS = re.compile(r"(?:\.\s?){3,}\d{1,3}$")
_CONTENTS = re.compile(r"^(?:table of )?contents$", re.I)
# Reference-list lines that look like lettered headings: "A. Smith, B. Jones, 2021".
_AUTHOR_LIKE = re.compile(r"\b(?:19|20)\d{2}\b|et al|\b[A-Z]\.\s?[A-Z]\.|^[A-Z][a-z]+,")
_UNITS = frozenset({"GHz", "MHz", "KHz", "KB", "MB", "GB", "TB", "Gb", "Mb", "W", "V", "X", "K"})
_COMMON_SECTIONS = frozenset(
    {
        "introduction",
        "background",
        "motivation",
        "related work",
        "related works",
        "methodology",
        "method",
        "methods",
        "design",
        "implementation",
        "evaluation",
        "experiments",
        "experimental setup",
        "experimental results",
        "results",
        "discussion",
        "limitations",
        "future work",
        "conclusion",
        "conclusions",
        "summary",
    }
)

_CAPTION = re.compile(
    r"^(?P<label>(?:Fig\.|Figure|FIGURE|Fig)\s*\d+|(?:Table|TABLE)\s+(?:[IVXLC]+|\d+))"
    r"(?:\s*[:.|]\s*(?P<rest>.*)|\s*$)"
)
_SENTENCE_END = re.compile(r"[.!?:][\"')\]\d]*$")
_LIST_ITEM = re.compile(r"^(?:[•●◦▪∙·-]|\(?\d{1,2}[.)]|\([a-z]\))\s")
# Word's PDF export spaces out hyphens in compounds: "safety -critical".
_SPACED_HYPHEN = re.compile(r"(?<=\w) -(?=\w)")
_WORD = re.compile(r"[A-Za-z]{2,}")
_HYPHEN_END = re.compile(r"([A-Za-z]+)-$")
_LEADING_WORD = re.compile(r"^([A-Za-z]+)")
_SPACE_BEFORE_PUNCT = re.compile(r"(?<=\w) ([.,;:)])(?=\s|$)")


@dataclass
class PdfFrontMatter:
    """What a PDF says about itself, for papers that have no arXiv metadata."""

    title: str | None
    authors: list[str]
    abstract: str
    license: str | None
    created: date | None


@dataclass
class _Heading:
    index: int  # first line of the heading
    title: str
    level: int
    span: int = 1  # lines the heading occupies
    rest: str = ""  # body text that shared the line with a run-in heading
    appendix: bool = False
    drop: bool = False


@dataclass(frozen=True)
class _Candidate:
    """A line that looks like a numbered heading, before numbering is checked."""

    index: int
    key: tuple[int, ...]  # (2,) for "2 Background", (2, 3) for "2.3 ..."
    title: str
    span: int = 1
    rest: str = ""
    runs_on: bool = False  # the next line continues it in lowercase, like a sentence

    @property
    def weight(self) -> float:
        """How heading-like the title is: Title Case or ALL CAPS beats a sentence.

        Numbered list items ("1. We propose ...") and sentences that wrapped
        after a number ("... shown in Table" / "3. ICE models energy via ...,
        validated across" / "multiple platforms ...") form chains too; the
        weight makes the chain through real headings win.
        """
        words = [w for w in self.title.split() if len(w) >= 4 and w[0].isalpha()]
        if not words or _all_caps(self.title):
            weight = 1.0
        else:
            capitalized = sum(w[0].isupper() for w in words) / len(words)
            weight = 1.0 if capitalized >= 0.5 else 0.7
        return weight / 2 if self.runs_on else weight

    def heading(self, level: int) -> _Heading:
        return _Heading(self.index, self.title, level, span=self.span, rest=self.rest)


@dataclass
class _Lines:
    """The extracted lines of a PDF, with the statistics the heuristics need."""

    lines: list[str]
    median_len: float
    vocabulary: set[str] = field(default_factory=set)


def parse_pdf(data: bytes, paper_id: str) -> Document:
    """Parse a PDF into a ``Document``; raises ``ParseError``."""
    document, _ = _parse(_open(data), data, paper_id)
    return document


def read_pdf(data: bytes, paper_id: str) -> tuple[Document, PdfFrontMatter]:
    """``parse_pdf`` plus what the PDF says about itself, for papers without arXiv
    metadata (``precis.sources.local``).

    The title comes from the document information when it looks like a real
    title (not "Microsoft Word - draft.docx"), otherwise from the largest
    text on the first page, which is where a paper prints its title.
    """
    reader = _open(data)
    document, front_lines = _parse(reader, data, paper_id)
    metadata: dict[str, Any] = dict(reader.metadata or {})
    title = _metadata_title(metadata.get("/Title"))
    if title is None and reader.pages:
        title = _largest_text(reader.pages[0])
    license_url = metadata.get("/License")
    front = PdfFrontMatter(
        title=title,
        authors=_metadata_authors(metadata.get("/Author")),
        abstract=_abstract(front_lines),
        license=str(license_url) if license_url else None,
        created=_creation_date(reader),
    )
    return document, front


def _parse(reader: PdfReader, data: bytes, paper_id: str) -> tuple[Document, list[str]]:
    """The document, and the front-matter lines that were cut from it."""
    lines = _clean_lines(_page_lines(reader))
    if not lines:
        raise ParseError("the PDF has no extractable text (scanned images?)")
    median = _median_len(lines)
    lines = _drop_contents(lines, median)
    doc_lines = _Lines(lines, median, _vocabulary(lines))
    body, tail = _split_references(lines)
    headings = _find_headings(body, doc_lines)
    drafts, captions = _structure(body, tail, headings, doc_lines)
    if headings and _one_section_dominates(drafts):
        headings, drafts, captions = _restructure(paper_id, body, tail, doc_lines)
    front_end = headings[0].index if headings else len(body) - len(_after_abstract(body))
    front = body[:front_end]
    return build_document(paper_id, "pdf", drafts, captions, sha256_hex(data)), front


def _one_section_dominates(drafts: list[SectionDraft]) -> bool:
    """A few sections, one of which holds nearly the whole paper: the headings are wrong.

    Real papers spread their text over their sections; an outline like
    "Conclusion: 46,000 tokens" means the chosen headings were not the
    paper's (a table of contents, figure labels) and most were missed.
    """
    sizes = [estimate_tokens(d.text) for d in drafts]
    total = sum(sizes)
    return (
        total >= DOMINANCE_MIN_TOKENS
        and sum(1 for s in sizes if s) < DOMINANCE_MAX_SECTIONS
        and max(sizes) > DOMINANCE_SHARE * total
    )


def _restructure(
    paper_id: str, body: list[str], tail: list[str], doc: _Lines
) -> tuple[list[_Heading], list[SectionDraft], list[Caption]]:
    """Second try when numbered headings gave a lopsided outline: unnumbered
    headings if they do better, else parts, since no outline beats a wrong one."""
    headings = _unnumbered_headings(body, doc)
    if headings:
        drafts, captions = _structure(body, tail, headings, doc)
        if not _one_section_dominates(drafts):
            logger.warning("%s: numbered headings were lopsided; used unnumbered ones", paper_id)
            return headings, drafts, captions
    logger.warning("%s: no trustworthy headings; split into parts", paper_id)
    drafts, captions = _structure(body, tail, [], doc)
    return [], drafts, captions


def _drop_contents(lines: list[str], median_len: float) -> list[str]:
    """Remove table-of-contents blocks, which otherwise pass for headings.

    A block is a run of entries ending in a page number (often after dot
    leaders) with no running text between them, at least three of which are
    numbered or dot-led, so a table whose rows end in numbers does not count
    unless it looks like a table of contents too. The "Contents" line above
    a block goes with it.
    """
    doc = _Lines(lines, median_len)
    drop: set[int] = set()
    i = 0
    while i < len(lines):
        if not _TOC_ENTRY.search(lines[i]):
            i += 1
            continue
        strong, last, j = 0, i, i
        while j < len(lines):
            line = lines[j]
            if _TOC_ENTRY.search(line):
                strong += _strong_toc_entry(line)
                last = j
            elif _is_prose(line, doc):
                break
            j += 1
        if strong >= 3:
            start = i - 1 if i > 0 and _CONTENTS.match(lines[i - 1]) else i
            drop.update(range(start, last + 1))
        i = last + 1
    return [line for n, line in enumerate(lines) if n not in drop]


def _strong_toc_entry(line: str) -> bool:
    numbered_or_dotted = bool(_TOC_NUMBERED.match(line) or _TOC_LEADERS.search(line))
    return numbered_or_dotted and bool(_TOC_WORDS.search(line))


# -- reading -------------------------------------------------------------------


def _open(data: bytes) -> PdfReader:
    if not data.startswith(b"%PDF"):
        raise ParseError("not a PDF file")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        _ = len(reader.pages)
    except Exception as exc:  # pypdf raises many types on malformed input
        raise ParseError(f"unreadable PDF: {exc}") from exc
    return reader


def _page_lines(reader: PdfReader) -> list[list[str]]:
    pages: list[list[str]] = []
    failed = 0
    for number, page in enumerate(reader.pages, 1):
        try:
            # Upright text only: drops the rotated arXiv stamp in the margin
            # and rotated axis labels inside figures.
            text = page.extract_text(orientations=(0,))
        except Exception as exc:  # one broken page should not lose the paper
            logger.warning("page %d: text extraction failed: %s", number, exc)
            failed += 1
            text = ""
        # Whole page first, so a surrogate pair is recombined before any split.
        text = clean_chars(text)
        pages.append([line for line in (squash(raw) for raw in text.splitlines()) if line])
    if pages and failed / len(pages) > MAX_FAILED_PAGES:
        raise ParseError(f"text extraction failed on {failed} of {len(pages)} pages")
    return pages


def _clean_lines(pages: list[list[str]]) -> list[str]:
    """Drop page furniture and boilerplate; return the remaining lines in order."""
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_furniture_key(line) for line in _zone(page)})
    threshold = max(2, math.ceil(0.3 * len(pages)))
    repeated = {key for key, n in counts.items() if n >= threshold} if len(pages) >= 3 else set()

    lines: list[str] = []
    for page in pages:
        zone = set(range(min(_FURNITURE_ZONE, len(page))))
        zone |= set(range(max(0, len(page) - _FURNITURE_ZONE), len(page)))
        for i, line in enumerate(page):
            if i in zone and (_furniture_key(line) in repeated or _PAGE_NUMBER.match(line)):
                continue
            lines.append(line)
    return _drop_boilerplate(lines)


def _zone(page: list[str]) -> list[str]:
    return page[:_FURNITURE_ZONE] + page[-_FURNITURE_ZONE:]


def _furniture_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower())


def _drop_boilerplate(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _PERMISSION_START.match(line):
            end = next(
                (j for j in range(i, min(i + 15, len(lines))) if _PERMISSION_END.search(lines[j])),
                i,
            )
            i = end + 1
            continue
        if not _BOILERPLATE.match(line):
            out.append(line)
        i += 1
    return out


def _median_len(lines: Sequence[str]) -> float:
    long_lines = [len(line) for line in lines if len(line.split()) >= 5]
    return float(statistics.median(long_lines)) if long_lines else 60.0


def _vocabulary(lines: list[str]) -> set[str]:
    text = " ".join(lines).lower()
    return set(re.findall(r"[a-z]+(?:-[a-z]+)*", text))


def _creation_date(reader: PdfReader) -> date | None:
    try:
        created = reader.metadata.creation_date if reader.metadata else None
    except (ValueError, TypeError):  # malformed "D:..." strings are common
        return None
    return created.date() if created else None


def _metadata_authors(value: object) -> list[str]:
    """``/Author`` as a list: arXiv writes "A; B; C", other tools "A, B and C"."""
    if not isinstance(value, str) or not value.strip():
        return []
    separator = r";" if ";" in value else r",|\band\b"
    return [name for name in (squash(n) for n in re.split(separator, value)) if name]


def _metadata_title(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    title = squash(value)
    lowered = title.lower()
    if (
        len(title.split()) < 2
        or lowered.startswith(("microsoft word", "untitled"))
        or lowered.endswith((".pdf", ".dvi", ".tex", ".doc", ".docx"))
    ):
        return None
    return title


def _largest_text(page: Any) -> str | None:
    """The text drawn in the largest font on a page: the title, on a paper's first page."""
    pieces: list[tuple[float, str]] = []

    def visit(text: str, cm: list[float], tm: list[float], _font: Any, size: float) -> None:
        scale = math.sqrt(abs(tm[0] * tm[3] - tm[1] * tm[2]))
        scale *= math.sqrt(abs(cm[0] * cm[3] - cm[1] * cm[2]))
        if text.strip():
            pieces.append((size * scale, text))

    try:
        page.extract_text(orientations=(0,), visitor_text=visit)
    except Exception as exc:
        logger.warning("title extraction failed: %s", exc)
        return None
    if not pieces:
        return None
    largest = max(size for size, _ in pieces)
    title = squash(" ".join(clean_chars(t) for size, t in pieces if size >= largest - 0.5))
    return title[:300] or None


# -- structure -----------------------------------------------------------------


def _split_references(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split at the last references heading: (body, everything after it)."""
    for i in range(len(lines) - 1, -1, -1):
        if _REFERENCES.match(lines[i]):
            return lines[:i], lines[i + 1 :]
    return lines, []


def _structure(
    body: list[str], tail: list[str], headings: list[_Heading], doc: _Lines
) -> tuple[list[SectionDraft], list[Caption]]:
    appendix_lines: list[str] = []
    if headings:
        last_top = max(h.index for h in headings if h.level == 1)
        start = _appendix_start(body, last_top + 1, {h.index for h in headings}, doc)
        if start is not None:
            appendix_lines = body[start:]
            body = body[:start]
            headings = [h for h in headings if h.index < start]
    tail_start = _appendix_start(tail, _references_end(tail), set(), doc)
    if tail_start is not None:
        appendix_lines += tail[tail_start:]
    captions: list[Caption] = []
    counts: dict[str, int] = {"figure": 0, "table": 0}
    drafts: list[SectionDraft] = []
    if headings:
        drafts.extend(_sections(body, headings, doc, captions, counts))
    else:
        paragraphs = _paragraphs(_text_lines(_after_abstract(body), doc, captions, counts), doc)
        drafts.extend(split_into_parts(paragraphs))
    if appendix_lines:
        drafts.extend(_appendix(appendix_lines, doc, captions, counts))
    return drafts, captions


def _sections(
    lines: list[str],
    headings: list[_Heading],
    doc: _Lines,
    captions: list[Caption],
    counts: dict[str, int],
) -> list[SectionDraft]:
    drafts: list[SectionDraft] = []
    bounds = [h.index for h in headings[1:]] + [len(lines)]
    for heading, end in zip(headings, bounds, strict=True):
        if heading.drop:
            continue
        own = ([heading.rest] if heading.rest else []) + lines[heading.index + heading.span : end]
        content = _text_lines(own, doc, captions, counts)
        drafts.append(
            SectionDraft(
                title=heading.title,
                level=heading.level,
                paragraphs=_paragraphs(content, doc),
                appendix=heading.appendix,
            )
        )
    return drafts


def _references_end(tail: list[str]) -> int:
    """Index after the last numbered reference ("[42] ..."), or 0 for author-year lists."""
    numbered = [i for i, line in enumerate(tail) if _NUMBERED_REFERENCE.match(line)]
    return numbered[-1] + 1 if numbered else 0


def _appendix_start(lines: list[str], start: int, taken: set[int], doc: _Lines) -> int | None:
    """Where an appendix begins at or after ``start``, or ``None``.

    An appendix announces itself with the word ("Appendix", "APPENDIX B")
    or with lettered headings counting from A ("A Allocation Policies").
    """
    for i in range(start, len(lines)):
        line = lines[i]
        if _APPENDIX.match(line) and len(line.split()) <= 10 and not line.endswith((".", ",")):
            return i
    lettered = [c for c in _lettered_candidates(lines, doc, start) if c.index not in taken]
    chain = _longest_chain(lettered)
    return chain[0].index if chain and chain[0].key == (1,) else None


def _lettered_candidates(lines: list[str], doc: _Lines, start: int = 0) -> list[_Candidate]:
    candidates = []
    for i in range(start, len(lines)):
        match = _APPENDIX_NUMBERED.match(lines[i])
        if match is None or _AUTHOR_LIKE.search(match["title"]):
            continue
        parts = match["num"].split(".")
        key = (ord(parts[0]) - ord("A") + 1, *(int(p) for p in parts[1:]))
        candidate = _candidate(lines, i, key, match["title"], doc)
        # "A Hopfield neural network is a nonlinear dynamical system, whereby
        # ..." starts with the article, not appendix A: a long title or one
        # that runs on into the next line is a sentence.
        if candidate is not None and len(candidate.title) <= 90 and not candidate.runs_on:
            candidates.append(candidate)
    return candidates


def _appendix(
    lines: list[str], doc: _Lines, captions: list[Caption], counts: dict[str, int]
) -> list[SectionDraft]:
    """Appendix sections: lettered headings (``A``, ``A.1``) when present, else one section."""
    headings = _numbered_chain(_lettered_candidates(lines, doc))
    if not headings or headings[0].index > 1:
        title = lines[0] if _APPENDIX.match(lines[0]) else "Appendix"
        for heading in headings:
            heading.level += 1
        headings.insert(0, _Heading(index=0, title=title, level=1))
    for heading in headings:
        heading.appendix = True
    return _sections(lines, headings, doc, captions, counts)


def _find_headings(lines: list[str], doc: _Lines) -> list[_Heading]:
    """Headings in reading order, chosen by consistent numbering (see module docstring)."""
    arabic: list[_Candidate] = []
    roman: list[_Candidate] = []
    for i, line in enumerate(lines):
        candidate: _Candidate | None = None
        if m := _ARABIC.match(line):
            if m["title"].split()[0] not in _UNITS:
                candidate = _candidate(lines, i, _arabic_key(m["num"]), m["title"], doc)
        elif (m := _LONE_NUMBER.match(line)) and _lone_number_heading(lines, i, doc):
            # "3" alone with the title on the next line; axis ticks are
            # excluded by requiring prose after the title.
            title_line = _candidate(lines, i + 1, _arabic_key(m["num"]), lines[i + 1], doc)
            if title_line is not None:
                candidate = replace(title_line, index=i, span=title_line.span + 1)
        if candidate is not None:
            arabic.append(candidate)
        candidate = None
        if m := _ROMAN.match(line):
            title = _small_caps(m["title"])
            candidate = _candidate(lines, i, (_roman(m["num"]),), title, doc)
        elif m := _LETTER.match(line):
            letter = ord(m["num"]) - ord("A") + 1
            candidate = _candidate(lines, i, (0, letter), m["title"], doc)
        elif (m := _IEEE_RUN_IN.match(line)) and _ieee_run_in(m):
            candidate = _Candidate(i, (0, 0, int(m["num"])), m["title"], rest=m["rest"])
        if candidate is not None:
            roman.append(candidate)

    by_arabic = _numbered_chain(arabic)
    by_roman = _roman_chain(roman)
    numbered = max(by_arabic, by_roman, key=lambda hs: sum(h.level == 1 for h in hs))
    if sum(h.level == 1 for h in numbered) < 2:
        return _unnumbered_headings(lines, doc)
    return _finish_headings(lines, numbered)


def _unnumbered_headings(lines: list[str], doc: _Lines) -> list[_Heading]:
    return _finish_headings(lines, _unnumbered(lines, doc))


def _finish_headings(lines: list[str], headings: list[_Heading]) -> list[_Heading]:
    """Mark Abstract/Acknowledgments-style headings, and add the ones the chain missed."""
    for heading in headings:
        heading.drop = bool(_DROP_HEADINGS.match(heading.title.rstrip(":")))
    return _with_drop_headings(lines, headings)


def _arabic_key(number: str) -> tuple[int, ...]:
    return tuple(int(part) for part in number.split("."))


def _candidate(
    lines: list[str], i: int, key: tuple[int, ...], title: str, doc: _Lines
) -> _Candidate | None:
    """A heading candidate at line ``i``, after joining wrapped lines and splitting a run-in.

    ACM's numbered run-in headings put the paragraph on the same line
    ("4.1.1 Memory Requirements. Working set size is ..."); the title is
    what precedes the first full stop and the rest, when it reads as a
    sentence rather than more title ("vs. SPEC CPU17, DCPerf"), is body text.
    """
    title, span = _continue_title(title, lines, i, doc)
    rest = ""
    run_in = _RUN_IN.match(title)
    if run_in and sum(w.islower() for w in _WORD.findall(run_in["rest"])) >= 2:
        title, rest = run_in["title"], run_in["rest"]
    if not _valid_title(title):
        return None
    runs_on = not rest and i + span < len(lines) and lines[i + span][:1].islower()
    return _Candidate(i, key, title, span, rest, runs_on)


def _ieee_run_in(match: re.Match[str]) -> bool:
    """``1) Computing Elements:The M100 NPU ...`` is a heading when the text after
    the colon starts a sentence; a colon-labelled list item continues one."""
    rest = match["rest"]
    return _valid_title(match["title"]) and (not rest or rest[0].isupper())


def _lone_number_heading(lines: list[str], i: int, doc: _Lines) -> bool:
    previous_has_letters = i == 0 or any(c.isalpha() for c in lines[i - 1])
    return previous_has_letters and i + 2 < len(lines) and _is_prose(lines[i + 2], doc)


def _numbered_chain(candidates: list[_Candidate], *, nested_keys: bool = True) -> list[_Heading]:
    """Top-level headings by best consecutive chain, then each level within its parent.

    A candidate's level is the length of its key. With ``nested_keys`` a
    child must carry its parent's number ("2.3" under "2"); IEEE numbering
    restarts per parent ("B." under "III."), which ``_roman_chain`` handles
    by passing ``False``.
    """
    chosen = _longest_chain([c for c in candidates if len(c.key) == 1])
    for depth in (2, 3):
        outline = sorted(chosen, key=lambda c: c.index)
        for n, parent in enumerate(outline):
            if len(parent.key) != depth - 1:
                continue
            end = next((p.index for p in outline[n + 1 :] if len(p.key) < depth), math.inf)
            kids = [
                c
                for c in candidates
                if len(c.key) == depth
                and parent.index < c.index < end
                and (not nested_keys or c.key[:-1] == parent.key)
            ]
            chosen.extend(_longest_chain(kids))
    chosen.sort(key=lambda c: c.index)
    return [c.heading(level=len(c.key)) for c in chosen]


def _roman_chain(candidates: list[_Candidate]) -> list[_Heading]:
    """IEEE style: ``II. TITLE`` sections, ``A. Title`` subsections inside them, and
    ``1) Title:`` run-in subsubsections inside those."""
    return _numbered_chain(candidates, nested_keys=False)


def _longest_chain(items: list[_Candidate]) -> list[_Candidate]:
    """The best in-order run of numbers counting up by one from 1 (or 2).

    Scores add each heading's ``weight``; one missed heading (a step of two)
    is tolerated at half credit, so a heading the extractor mangled does not
    discard the rest of the paper.
    """
    score: list[float] = []
    previous: list[int] = []
    for n, item in enumerate(items):
        value = item.key[-1]
        best, prev = (item.weight, -1) if value <= 2 else (-math.inf, -1)
        for m in range(n):
            step = value - items[m].key[-1]
            if step in (1, 2) and score[m] > -math.inf:
                candidate = score[m] + item.weight * (1.0 if step == 1 else 0.5)
                if candidate > best:
                    best, prev = candidate, m
        score.append(best)
        previous.append(prev)
    if not items or max(score) == -math.inf:
        return []
    end = max(range(len(items)), key=lambda n: score[n])
    chain = []
    while end != -1:
        chain.append(items[end])
        end = previous[end]
    return chain[::-1]


def _unnumbered(lines: list[str], doc: _Lines) -> list[_Heading]:
    """Standalone section lines for papers without numbers, followed by running text.

    A common section name ("Introduction", "RESULTS") is enough on its own;
    any other all-caps line must also follow the end of a sentence, so a
    figure label such as "DRAM" next to a paragraph is not taken for one.
    """
    headings: list[_Heading] = []
    for i, line in enumerate(lines):
        if not _valid_title(line) or i + 1 >= len(lines) or not _is_prose(lines[i + 1], doc):
            continue
        common = line.lower().rstrip(":") in _COMMON_SECTIONS
        letters = sum(c.isalpha() for c in line)
        caps = _all_caps(line) and letters >= 6 and len(line.split()) <= 8
        after_sentence = i == 0 or bool(_SENTENCE_END.search(lines[i - 1]))
        if common or (caps and after_sentence):
            headings.append(_Heading(index=i, title=line.rstrip(":"), level=1))
    return headings


def _with_drop_headings(lines: list[str], headings: list[_Heading]) -> list[_Heading]:
    """Add Abstract/Acknowledgments/Keywords headings so their text is cut out."""
    taken = {h.index for h in headings}
    first = headings[0].index if headings else len(lines)
    extra = [
        _Heading(index=i, title=line, level=1, drop=True)
        for i, line in enumerate(lines)
        if i not in taken and i > first and _DROP_HEADINGS.match(line.rstrip(":"))
    ]
    return sorted([*headings, *extra], key=lambda h: h.index)


def _valid_title(title: str) -> bool:
    title = title.strip()
    # Long enough for a title that wrapped onto two lines; a line of running
    # text is ruled out by the other checks and by the numbering chain.
    if not 2 <= len(title) <= 120 or len(title.split()) > 16:
        return False
    if not title[0].isupper() or title.endswith((",", ";", ".")):
        return False
    if any(marker in title for marker in ("et al", "http", "@", "[")):
        return False
    non_space = [c for c in title if not c.isspace()]
    return sum(c.isalpha() for c in non_space) >= 0.6 * len(non_space)


def _continue_title(title: str, lines: list[str], i: int, doc: _Lines) -> tuple[str, int]:
    """Join a heading that wrapped onto the next line: (title, lines used).

    A wrap is recognizable when the title breaks inside a hyphenated word
    ("Matrix-Vector Mul-" / "tiplication"), when an all-caps title continues
    with another short all-caps line, when the next line is a fragment of at
    most three lowercase words ("... without fault tolerant" / "technique"),
    when the title ends on a comma or a connector word, or when the next line
    is a short Title Case fragment with running text after it ("... Signal
    Processing and Deep" / "Learning").
    """
    if i + 1 >= len(lines):
        return title, 1
    nxt = lines[i + 1]
    words = nxt.split()
    if title.endswith("-") and len(words) <= 8 and nxt[:1].isalpha():
        return (title[:-1] + nxt if nxt[:1].islower() else title + nxt), 2
    if _all_caps(title) and _all_caps(nxt) and len(words) <= 8 and not _ARABIC.match(nxt):
        return f"{title} {nxt}", 2
    if len(words) <= 3 and nxt[:1].islower() and not _SENTENCE_END.search(nxt):
        return f"{title} {nxt}", 2
    # "3 SPEC CPU26 vs. SPEC CPU17, DCPerf, MLPerf," / "Agentic AI, GEMM"
    dangling = title.endswith(",") or title.rsplit(None, 1)[-1].lower() in _CONNECTORS
    if dangling and len(words) <= 8 and nxt[:1].isupper() and not _SENTENCE_END.search(nxt):
        return f"{title} {nxt}", 2
    # A Title Case fragment only continues a Title Case title that ran to the
    # end of its line; a short title followed by a capitalized line is a
    # heading followed by a sub-heading or an algorithm name ("FFT Engine" /
    # "Theory of Operation"). The fragment's length cap keeps out run-in
    # paragraphs that lost their spaces ("UltraAcceleratorLink. UALinkisan...").
    wrapped = len(lines[i]) >= 0.5 * doc.median_len and _title_case(title) and not _all_caps(title)
    fragment = (
        len(words) <= 4 and len(nxt) <= 40 and _title_case(nxt) and not _SENTENCE_END.search(nxt)
    )
    prose_follows = i + 2 < len(lines) and _is_prose(lines[i + 2], doc)
    if wrapped and fragment and prose_follows and not _starts_heading(nxt):
        return f"{title} {nxt}", 2
    return title, 1


def _title_case(text: str) -> bool:
    """Starts with a capital and capitalizes every word of four letters or more."""
    words = [w for w in text.split() if len(w) >= 4 and w[0].isalpha()]
    return text[:1].isupper() and all(w[0].isupper() for w in words)


def _starts_heading(line: str) -> bool:
    patterns = (_ARABIC, _ROMAN, _LETTER, _LONE_NUMBER, _IEEE_RUN_IN, _APPENDIX_NUMBERED)
    return any(p.match(line) for p in patterns)


def _all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def _small_caps(title: str) -> str:
    """Repair IEEE small-caps headings that extract as ``I NTRODUCTION``."""
    title = re.sub(r"\b([A-Z]) (?=[A-Z]{2,}\b)", r"\1", title)
    return re.sub(r"(?<=\w) -(?=\w)", "-", title)


def _roman(numeral: str) -> int:
    values = {"I": 1, "V": 5, "X": 10}
    total = 0
    for n, char in enumerate(numeral):
        value = values[char]
        total += -value if n + 1 < len(numeral) and values[numeral[n + 1]] > value else value
    return total


# -- text ----------------------------------------------------------------------


def _text_lines(
    lines: list[str], doc: _Lines, captions: list[Caption], counts: dict[str, int]
) -> list[str]:
    """Body lines of one section: captions moved out, figure and table debris removed."""
    kept: list[str] = []
    i = 0
    while i < len(lines):
        match = _CAPTION.match(lines[i])
        if match:
            i = _take_caption(lines, i, match, doc, captions, counts)
            continue
        kept.append(lines[i])
        i += 1
    return _drop_debris(kept, doc)


def _take_caption(
    lines: list[str],
    start: int,
    match: re.Match[str],
    doc: _Lines,
    captions: list[Caption],
    counts: dict[str, int],
) -> int:
    """Record the caption starting at ``start``; return the index after it."""
    kind: Literal["figure", "table"] = (
        "table" if match["label"].lower().startswith("tab") else "figure"
    )
    parts = [lines[start]]
    i = start + 1
    title_on_next_line = not (match["rest"] or "").strip()
    while i < len(lines) and len(parts) < 6:
        line = lines[i]
        if title_on_next_line:
            parts.append(line)
            i += 1
            title_on_next_line = False
            if _all_caps(line):
                break
            continue
        # Without font information a caption's end is a guess; stopping at the
        # first sentence end keeps body text out of captions (a long caption's
        # tail lands in the body instead, which loses nothing).
        last = parts[-1]
        short = len(last) < 0.8 * doc.median_len
        if _SENTENCE_END.search(last) or short or not _is_prose(line, doc):
            break
        parts.append(line)
        i += 1
    counts[kind] += 1
    prefix = "fig" if kind == "figure" else "tab"
    captions.append(
        Caption(id=f"{prefix}{counts[kind]}", kind=kind, text=_join_lines(parts, doc.vocabulary))
    )
    return i


def _drop_debris(lines: list[str], doc: _Lines) -> list[str]:
    """Remove runs of three or more non-prose lines, and lines with no letters at all."""
    kept: list[str] = []
    run: list[str] = []
    for line in [*lines, ""]:
        if line and not _is_prose(line, doc) and not _ends_paragraph(line, doc):
            run.append(line)
            continue
        if len(run) < 3:
            kept.extend(r for r in run if any(c.isalpha() for c in r))
        run = []
        if line:
            kept.append(line)
    return kept


def _is_prose(line: str, doc: _Lines) -> bool:
    """Running text has lowercase function words; labels, table cells and math do not."""
    lower = sum(1 for w in _WORD.findall(line) if w.islower())
    return lower >= 3 and len(line) >= 0.4 * doc.median_len


def _ends_paragraph(line: str, doc: _Lines) -> bool:
    return bool(_SENTENCE_END.search(line)) and len(line) < 0.8 * doc.median_len


def _paragraphs(lines: list[str], doc: _Lines) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and _LIST_ITEM.match(line):
            paragraphs.append(_join_lines(current, doc.vocabulary))
            current = []
        current.append(line)
        if _ends_paragraph(line, doc):
            paragraphs.append(_join_lines(current, doc.vocabulary))
            current = []
    if current:
        paragraphs.append(_join_lines(current, doc.vocabulary))
    return [p for p in paragraphs if p]


def _join_lines(lines: list[str], vocabulary: set[str]) -> str:
    text = ""
    for line in lines:
        if len(text) > 1 and text.endswith("-") and text[-2].isalnum() and line[:1].isalnum():
            head = _HYPHEN_END.search(text)
            lead = _LEADING_WORD.match(line)
            drop = bool(head and lead and _rejoin(head[1], lead[1], vocabulary))
            text = text[:-1] + ("" if drop else "-") + line
        else:
            text = f"{text} {line}" if text else line
    return _SPACED_HYPHEN.sub("-", _SPACE_BEFORE_PUNCT.sub(r"\1", squash(text)))


def _rejoin(head: str, tail: str, vocabulary: set[str]) -> bool:
    """Whether ``head-`` + ``tail`` across a line break is one word split by hyphenation."""
    joined = (head + tail).lower()
    if joined in vocabulary:
        return True
    if f"{head}-{tail}".lower() in vocabulary:
        return False
    # Compounds with acronyms or numbers ("CKKS-friendly", "64-bit") keep the hyphen.
    return head[-1].islower() and tail[0].islower()


def _after_abstract(lines: list[str]) -> list[str]:
    """For heading-less papers: drop the title block and abstract when they are recognizable,
    with the keyword line and a lone "Introduction" that may follow them."""
    for i, line in enumerate(lines[:80]):
        if _ABSTRACT_START.match(line):
            doc = _Lines(lines, _median_len(lines))
            for j in range(i, len(lines)):
                if _ends_paragraph(lines[j], doc):
                    k = j + 1
                    while k < len(lines) and (
                        _FRONT_MATTER_END.match(lines[k])
                        or lines[k].lower().rstrip(":") in _COMMON_SECTIONS
                    ):
                        k += 1
                    return lines[k:]
    return lines


def _abstract(front: list[str]) -> str:
    for i, line in enumerate(front):
        if _ABSTRACT_START.match(line):
            body = [_ABSTRACT_START.sub("", line, count=1)]
            for nxt in front[i + 1 :]:
                if _FRONT_MATTER_END.match(nxt):
                    break
                body.append(nxt)
            return _join_lines([b for b in body if b], _vocabulary(front))
    return ""
