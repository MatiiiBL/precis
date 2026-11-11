"""A paper as the agents see it: header, outline, pages, and text matching.

Everything an agent is shown about a paper is rendered here, deterministically,
so that the same paper always produces byte-identical prompts: the gateway's
response cache and the API's prompt cache both key on exact request bytes.

Paper text is untrusted: papers on arXiv have carried hidden instructions
aimed at AI reviewers. Every span of it that reaches a model is wrapped by
``paper_text`` in ``<paper_text>`` tags, which the system prompts declare to
be material to work on and never instructions.
"""

from __future__ import annotations

import re
import unicodedata
from functools import cached_property

from precis.models import Caption, Document, PaperMeta, Section
from precis.util import estimate_tokens

_MAX_AUTHORS = 6

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[$\\])")

# Typographic quotes (U+2018/9, U+201C/D) and dashes (hyphen, non-breaking
# hyphen, en and em dash, minus sign) that a model may swap for ASCII ones.
_QUOTE_CHARS = str.maketrans(
    {
        0x2018: "'",
        0x2019: "'",
        0x201C: '"',
        0x201D: '"',
        0x2010: "-",
        0x2011: "-",
        0x2013: "-",
        0x2014: "-",
        0x2212: "-",
    }
)
_WHITESPACE = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\[\.\.\.\]|\.\.\.")
_EDGE_PUNCTUATION = " \"'.,;:()[]"

# A number not glued to a word ("L2", "x86") or to another number ("1.2.3").
_NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")
# Integers this small are often spelled out ("three datasets"), so they are
# not required to appear as digits in the paper.
_SMALL_INTEGER = 12


PAPER_TEXT_TAG = "paper_text"
_TAG_IN_TEXT = re.compile(r"<\s*/?\s*paper_text\s*>", re.IGNORECASE)


def paper_text(text: str, *, inline: bool = False) -> str:
    """Wrap text that comes from a paper in ``<paper_text>`` tags.

    Tags of the same name inside the text are defused (``</paper_text>``
    becomes ``</paper-text>``), so the paper cannot end the block early and
    smuggle text outside it. ``inline`` keeps the tags on the same line, for
    short spans such as a quote.
    """
    defused = _TAG_IN_TEXT.sub(lambda m: m.group().replace("_", "-"), text.strip())
    if inline:
        return f"<{PAPER_TEXT_TAG}>{defused}</{PAPER_TEXT_TAG}>"
    return f"<{PAPER_TEXT_TAG}>\n{defused}\n</{PAPER_TEXT_TAG}>"


def normalize_text(text: str) -> str:
    """Fold the differences a faithful quote may still have from its source.

    Unicode compatibility forms (ligatures, non-breaking spaces), typographic
    quotes and dashes, runs of whitespace and letter case are normalized; the
    words themselves are not.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_QUOTE_CHARS)
    return _WHITESPACE.sub(" ", folded).strip().casefold()


def canonical_number(raw: str) -> str:
    """``"1,024"`` -> ``"1024"``, ``"1.50"`` -> ``"1.5"``, ``"07"`` -> ``"7"``."""
    whole, _, fraction = raw.replace(",", "").partition(".")
    fraction = fraction.rstrip("0")
    return str(int(whole)) + (f".{fraction}" if fraction else "")


def numbers_in(text: str) -> list[str]:
    """Numbers in ``text`` that must be traceable to the paper, canonicalized.

    Small integers are skipped because papers often spell them out.
    """
    found = []
    for match in _NUMBER.finditer(text):
        number = canonical_number(match.group())
        if "." in number or int(number) > _SMALL_INTEGER:
            found.append(number)
    return found


def split_sentences(text: str) -> list[str]:
    """Sentences of ``text``, split at end punctuation followed by a capital, digit or math."""
    return [
        s.strip()
        for paragraph in _PARAGRAPH.split(text)
        for s in _SENTENCE_END.split(" ".join(paragraph.split()))
        if s.strip()
    ]


def split_pages(text: str, page_tokens: int) -> list[str]:
    """Split ``text`` into pages of at most about ``page_tokens`` tokens.

    Pages break between paragraphs, then between sentences, and only cut
    inside a sentence that is longer than a page by itself. An empty text has
    no pages.
    """
    pieces: list[str] = []
    for paragraph in (p.strip() for p in _PARAGRAPH.split(text)):
        if not paragraph:
            continue
        if estimate_tokens(paragraph) <= page_tokens:
            pieces.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            pieces.extend(_hard_wrap(sentence, page_tokens))
    pages: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        cost = estimate_tokens(piece)
        if current and size + cost > page_tokens:
            pages.append("\n\n".join(current))
            current, size = [], 0
        current.append(piece)
        size += cost
    if current:
        pages.append("\n\n".join(current))
    return pages


def _hard_wrap(sentence: str, page_tokens: int) -> list[str]:
    limit = page_tokens * 4
    return [sentence[i : i + limit] for i in range(0, len(sentence), limit)]


class PaperView:
    """Read-only view of one paper, shared by the tools and the prompts."""

    def __init__(self, meta: PaperMeta, doc: Document, *, page_tokens: int) -> None:
        self.meta = meta
        self.doc = doc
        self.page_tokens = page_tokens
        self.sections: dict[str, Section] = {s.id: s for s in doc.sections}
        self.captions: dict[str, Caption] = {c.id: c for c in doc.captions}
        self._order = {s.id: i for i, s in enumerate(doc.sections)}
        self._pages: dict[str, list[str]] = {}

    # -- lookup ------------------------------------------------------------

    def known_ids(self) -> list[str]:
        """Section ids in outline order, then caption ids."""
        return [*self.sections, *self.captions]

    def pages(self, section_id: str) -> list[str]:
        if section_id not in self._pages:
            text = self.sections[section_id].text
            self._pages[section_id] = split_pages(text, self.page_tokens)
        return self._pages[section_id]

    def subsections(self, section_id: str) -> list[Section]:
        """Direct children of a section, from the flat, level-annotated list."""
        start = self._order[section_id]
        level = self.doc.sections[start].level
        children = []
        for section in self.doc.sections[start + 1 :]:
            if section.level <= level:
                break
            if section.level == level + 1:
                children.append(section)
        return children

    def next_section(self, section_id: str) -> Section | None:
        index = self._order[section_id] + 1
        return self.doc.sections[index] if index < len(self.doc.sections) else None

    # -- rendering ---------------------------------------------------------

    def header(self, *, include_abstract: bool) -> str:
        """Title, authors, categories and (optionally) the abstract, as paper text."""
        meta = self.meta
        authors = ", ".join(meta.authors[:_MAX_AUTHORS])
        if len(meta.authors) > _MAX_AUTHORS:
            authors += f", et al. ({len(meta.authors)} authors)"
        others = [c for c in meta.categories if c != meta.primary_category]
        categories = meta.primary_category + (f" (also {', '.join(others)})" if others else "")
        lines = [f"Title: {meta.title}", f"Authors: {authors or 'unknown'}"]
        lines.append(f"Categories: {categories}")
        if include_abstract:
            lines += ["Abstract:", meta.abstract.strip()]
        text = paper_text("\n".join(lines))
        if not include_abstract:
            text += "\nThe abstract is withheld; work from the body text."
        return text

    def outline(self) -> str:
        """One line per section with its id, title and size, indented by level."""
        lines = [
            f"Body: {self.doc.n_tokens:,} tokens in {len(self.doc.sections)} sections. "
            f"read_section returns pages of about {self.page_tokens:,} tokens."
        ]
        for section in self.doc.sections:
            n_pages = len(self.pages(section.id))
            if n_pages == 0:
                size = "heading only"
            else:
                size = f"{section.n_tokens:,} tokens, {n_pages} page{'s' if n_pages > 1 else ''}"
            indent = "  " * (section.level - 1)
            lines.append(f"{indent}{section.id}  {section.title}  ({size})")
        figures = sum(c.kind == "figure" for c in self.doc.captions)
        tables = len(self.doc.captions) - figures
        lines.append(f"Captions: {figures} figures, {tables} tables (read_captions).")
        return "\n".join(lines)

    def body(self) -> str:
        """The whole body with ``[id] Title`` headings, then every caption."""
        parts = [f"[{s.id}] {s.title}\n\n{s.text.strip()}".rstrip() for s in self.doc.sections]
        parts += [f"[{c.id}] ({c.kind}) {c.text.strip()}" for c in self.doc.captions]
        return "\n\n".join(parts)

    # -- matching ----------------------------------------------------------

    @cached_property
    def _normalized_text(self) -> str:
        texts = [s.text for s in self.doc.sections] + [c.text for c in self.doc.captions]
        return normalize_text("\n".join(texts))

    @cached_property
    def numbers(self) -> frozenset[str]:
        """Every number in the title, abstract, body and captions."""
        texts = [self.meta.title, self.meta.abstract]
        texts += [s.text for s in self.doc.sections] + [c.text for c in self.doc.captions]
        return frozenset(n for text in texts for n in numbers_in(text))

    def contains_quote(self, quote: str) -> bool:
        """Whether ``quote`` occurs in the body or captions.

        Whitespace, case, typographic quotes and dashes are normalized, edge
        punctuation is ignored, and "..." splits the quote into fragments that
        must each occur.
        """
        fragments = [
            fragment.strip(_EDGE_PUNCTUATION) for fragment in _ELLIPSIS.split(normalize_text(quote))
        ]
        fragments = [f for f in fragments if f]
        return bool(fragments) and all(f in self._normalized_text for f in fragments)
