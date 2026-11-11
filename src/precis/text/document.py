"""Assembling parsed sections into a ``Document``.

Both parsers produce a flat, ordered list of ``SectionDraft`` (title, level,
paragraphs, appendix flag); this module turns it into the stored record. Ids
are assigned by position (``s1``, ``s1.2``, ``a1``), which matches the
paper's own numbering whenever the paper numbers its sections normally and
stays stable when it does not.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

from precis.models import Caption, Document, Section
from precis.text.normalize import join_paragraphs, squash
from precis.util import estimate_tokens

MAX_LEVEL = 4

# Documents without detectable headings are cut into parts of about this many
# tokens, so a reader agent can page through them instead of facing one
# section the size of the whole paper.
PART_TOKENS = 1500

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
# A run-in acknowledgment, at the start of a paragraph or after a sentence:
# "Acknowledgment This work is sponsored by ...", "Acknowledgments. We thank ...".
_ACKNOWLEDGMENT = re.compile(r"(?:^|(?<=[.!?]\s))(?i:acknowledge?ments?)\b\W*(?=[A-Z])")


class ParseError(Exception):
    """A downloaded file could not be turned into a usable ``Document``."""


@dataclass
class SectionDraft:
    """A section as a parser sees it, before ids and token counts exist."""

    title: str
    level: int
    paragraphs: list[str] = field(default_factory=list)
    appendix: bool = False

    @property
    def text(self) -> str:
        return join_paragraphs(self.paragraphs)


def build_document(
    paper_id: str,
    source_format: Literal["html", "pdf", "text"],
    drafts: Iterable[SectionDraft],
    captions: list[Caption],
    sha256: str,
) -> Document:
    """Drop run-in acknowledgments and empty leaf sections, assign ids and levels, count tokens."""
    cleaned = [
        replace(d, paragraphs=[p for p in map(strip_acknowledgment, d.paragraphs) if p])
        for d in drafts
    ]
    sections = _assign_ids(_prune_empty(cleaned))
    return Document(
        paper_id=paper_id,
        source_format=source_format,
        sections=sections,
        captions=captions,
        n_tokens=sum(s.n_tokens for s in sections),
        sha256=sha256,
    )


def strip_acknowledgment(paragraph: str) -> str:
    """Cut a run-in acknowledgment and everything after it in the paragraph.

    Acknowledgments under their own heading are dropped by the parsers; this
    catches the ones written as a bold lead-in to the last paragraph, whose
    grant numbers would otherwise read as results of the paper.
    """
    match = _ACKNOWLEDGMENT.search(paragraph)
    return paragraph[: match.start()].rstrip() if match else paragraph


def _prune_empty(drafts: list[SectionDraft]) -> list[SectionDraft]:
    """Drop sections with no text unless they head a deeper section."""
    kept: list[SectionDraft] = []
    for draft in reversed(drafts):
        heads_child = bool(kept) and kept[-1].level > draft.level
        if draft.text or heads_child:
            kept.append(draft)
    kept.reverse()
    return kept


def _assign_ids(drafts: list[SectionDraft]) -> list[Section]:
    if not drafts:
        return []
    base = min(d.level for d in drafts)
    counters: dict[bool, list[int]] = {False: [], True: []}
    sections: list[Section] = []
    for draft in drafts:
        stack = counters[draft.appendix]
        # A level may not skip its parent (a \subsubsection straight after a
        # \section becomes a subsection), so every id has a real parent.
        level = max(1, min(draft.level - base + 1, len(stack) + 1, MAX_LEVEL))
        if len(stack) >= level:
            del stack[level:]
            stack[level - 1] += 1
        else:
            stack.append(1)
        prefix = "a" if draft.appendix else "s"
        text = draft.text
        sections.append(
            Section(
                id=prefix + ".".join(str(n) for n in stack),
                title=squash(draft.title) or f"Section {len(sections) + 1}",
                level=level,
                text=text,
                n_tokens=estimate_tokens(text),
            )
        )
    return sections


def split_into_parts(paragraphs: list[str], target_tokens: int = PART_TOKENS) -> list[SectionDraft]:
    """Cut heading-less text into sections titled "Part k" of about ``target_tokens``.

    Cuts fall between paragraphs when possible, then between sentences, and
    only split inside a sentence when one sentence alone exceeds the target.
    """
    # Each piece remembers its paragraph, so pieces of one paragraph that land
    # in the same part are glued back together.
    pieces = [
        (index, piece)
        for index, paragraph in enumerate(paragraphs)
        for piece in _pieces(squash(paragraph), target_tokens)
    ]
    parts: list[list[tuple[int, str]]] = []
    size = 0
    for index, piece in pieces:
        n = estimate_tokens(piece)
        if parts and size + n <= target_tokens:
            parts[-1].append((index, piece))
            size += n
        else:
            parts.append([(index, piece)])
            size = n
    drafts = []
    for number, chunk in enumerate(parts, 1):
        merged: list[str] = []
        for i, (index, piece) in enumerate(chunk):
            if i and chunk[i - 1][0] == index:
                merged[-1] = f"{merged[-1]} {piece}"
            else:
                merged.append(piece)
        drafts.append(SectionDraft(title=f"Part {number}", level=1, paragraphs=merged))
    return drafts


def _pieces(paragraph: str, target_tokens: int) -> list[str]:
    """A paragraph, or its sentences (or word runs) when it is longer than a part."""
    if not paragraph:
        return []
    if estimate_tokens(paragraph) <= target_tokens:
        return [paragraph]
    out: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        if estimate_tokens(sentence) <= target_tokens:
            out.append(sentence)
            continue
        words = sentence.split()
        step = max(1, len(words) * target_tokens // estimate_tokens(sentence))
        out.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
    return out
