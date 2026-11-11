"""The reading tools the reader, critic and reviser share.

The tools return text written for the model: every result says where it is
in the paper ("page 2 of 3") and what to call next, and every mistake (an
unknown id, a page out of range, an empty query) comes back as an error
``tool_result`` that names the valid choices, so the model can correct itself
instead of the run failing. Definitions use ``strict: true`` so that tool
inputs always match their schemas.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from precis.agents.paper import PaperView
from precis.models import Document
from precis.text.bm25 import PaperIndex
from precis.util import estimate_tokens


class Passage(Protocol):
    """A search hit, as returned by ``precis.text.bm25.PaperIndex.search``."""

    @property
    def id(self) -> str: ...
    @property
    def section_id(self) -> str: ...
    @property
    def text(self) -> str: ...
    @property
    def score(self) -> float: ...


class PassageIndex(Protocol):
    def search(self, query: str, k: int) -> Sequence[Passage]: ...


def paper_index(doc: Document) -> PassageIndex:
    """BM25 over the paper's passages of about 120 words (captions are not indexed)."""
    return PaperIndex(doc)


READ_SECTION: dict[str, Any] = {
    "name": "read_section",
    "description": (
        'Read one section of the paper by its id from the outline (for example "s3" or '
        '"s3.2"), one page of about 2,500 tokens at a time. The result says "page i of n"; '
        "call again with the next page number to continue. A section's subsections are "
        "separate entries in the outline. Figure and table ids from read_captions (for "
        'example "tab2") also work and return the full caption, including table cells. '
        "Call this for the parts of the paper you need to read in full."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "section_id": {
                "type": "string",
                "description": 'A section id from the outline, or a caption id such as "tab2".',
            },
            "page": {
                "type": "integer",
                "description": "Page to read, starting at 1.",
            },
        },
        "required": ["section_id", "page"],
        "additionalProperties": False,
    },
}

SEARCH_PAPER: dict[str, Any] = {
    "name": "search_paper",
    "description": (
        "Search the paper's body text and return the passages (about 120 words each) that "
        "best match the query, with their section ids. Use it to find a specific number, "
        "dataset, baseline, metric or claim without reading whole sections. Keyword queries "
        'work best, for example "speedup over baseline GPU" or "ImageNet top-1 accuracy".'
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search for."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

READ_CAPTIONS: dict[str, Any] = {
    "name": "read_captions",
    "description": (
        "List every figure and table caption in the paper with its id. Tables often hold "
        "the main quantitative results; when a long caption is shortened here, read_section "
        "with its id returns it in full."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}

READING_TOOLS = (READ_SECTION, SEARCH_PAPER, READ_CAPTIONS)
READING_TOOL_NAMES = frozenset(tool["name"] for tool in READING_TOOLS)

# Below this many tokens per caption, read_captions stops shortening them.
_MIN_CAPTION_TOKENS = 150


@dataclass(frozen=True)
class ToolOutput:
    text: str
    is_error: bool = False


@dataclass
class ReadingLog:
    """Which pages an agent opened, for run statistics."""

    pages: set[tuple[str, int]] = field(default_factory=set)
    tokens_read: int = 0

    def fraction_of(self, doc: Document) -> float:
        """Share of the body's tokens that the agent opened through read_section."""
        return min(1.0, self.tokens_read / doc.n_tokens) if doc.n_tokens else 0.0


class ReadingTools:
    """Runs the reading tools against one paper and logs what was read."""

    def __init__(self, view: PaperView, index: PassageIndex, *, search_results: int) -> None:
        self.view = view
        self.index = index
        self.search_results = search_results
        self.log = ReadingLog()

    def run(self, name: str, tool_input: Mapping[str, Any]) -> ToolOutput:
        if name == "read_section":
            return self._read_section(str(tool_input["section_id"]), int(tool_input["page"]))
        if name == "search_paper":
            return self._search(str(tool_input["query"]))
        if name == "read_captions":
            return self._captions()
        return ToolOutput(f"There is no tool named {name!r}.", is_error=True)

    # -- read_section ------------------------------------------------------

    def _read_section(self, section_id: str, page: int) -> ToolOutput:
        view = self.view
        section_id = section_id.strip()
        if section_id in view.captions:
            caption = view.captions[section_id]
            return ToolOutput(f"[{caption.id}] ({caption.kind})\n\n{caption.text.strip()}")
        if section_id not in view.sections:
            return ToolOutput(self._unknown_id(section_id), is_error=True)
        section = view.sections[section_id]
        pages = view.pages(section_id)
        if not pages:
            children = view.subsections(section_id)
            listing = ", ".join(f"{c.id} {c.title}" for c in children)
            where = f" Read its subsections instead: {listing}." if children else ""
            return ToolOutput(
                f"Section {section_id} ({section.title}) has no text of its own.{where}"
            )
        if not 1 <= page <= len(pages):
            return ToolOutput(
                f"Section {section_id} has {len(pages)} page{'s' if len(pages) > 1 else ''}; "
                f"page {page} does not exist. Pages are numbered from 1.",
                is_error=True,
            )
        if (section_id, page) in self.log.pages:
            return ToolOutput(
                f"You already read page {page} of section {section_id}; it is earlier in this "
                "conversation."
            )
        self.log.pages.add((section_id, page))
        text = pages[page - 1]
        self.log.tokens_read += estimate_tokens(text)
        heading = f"[{section_id}] {section.title} (page {page} of {len(pages)})"
        return ToolOutput(f"{heading}\n\n{text}\n\n{self._after(section_id, page, len(pages))}")

    def _after(self, section_id: str, page: int, n_pages: int) -> str:
        if page < n_pages:
            return f"Continues on page {page + 1} of {n_pages}."
        following = self.view.next_section(section_id)
        if following is None:
            return f"End of section {section_id}, the last section of the body."
        return (
            f"End of section {section_id}. Next in the outline: {following.id} {following.title}."
        )

    def _unknown_id(self, section_id: str) -> str:
        ids = self.view.known_ids()
        close = difflib.get_close_matches(section_id, ids, n=3, cutoff=0.5)
        hint = f" Did you mean {' or '.join(close)}?" if close else ""
        return (
            f"There is no section or caption with id {section_id!r}.{hint} "
            f"Section ids: {', '.join(self.view.sections) or 'none'}. "
            f"Caption ids: {', '.join(self.view.captions) or 'none'}."
        )

    # -- search_paper ------------------------------------------------------

    def _search(self, query: str) -> ToolOutput:
        query = query.strip()
        if not query:
            return ToolOutput("The query is empty; search for a few keywords.", is_error=True)
        hits = self.index.search(query, self.search_results)
        if not hits:
            return ToolOutput(
                f"No passage matches {query!r}. Try other words, or read a section from the "
                "outline."
            )
        blocks = [f"{len(hits)} passages for {query!r}, best first:"]
        for rank, hit in enumerate(hits, 1):
            section = self.view.sections.get(hit.section_id)
            title = f" {section.title}" if section else ""
            blocks.append(f"{rank}. [{hit.section_id}]{title}\n{hit.text.strip()}")
        blocks.append("read_section shows a passage in its context.")
        return ToolOutput("\n\n".join(blocks))

    # -- read_captions -----------------------------------------------------

    def _captions(self) -> ToolOutput:
        captions = self.view.doc.captions
        if not captions:
            return ToolOutput("This paper has no figure or table captions.")
        total = sum(estimate_tokens(c.text) for c in captions)
        limit = None
        if total > self.view.page_tokens:
            limit = max(_MIN_CAPTION_TOKENS, self.view.page_tokens // len(captions))
        lines = []
        for caption in captions:
            text = " ".join(caption.text.split())
            if limit is not None and estimate_tokens(text) > limit:
                text = text[: limit * 4].rstrip() + " [shortened]"
            lines.append(f"[{caption.id}] ({caption.kind}) {text}")
        return ToolOutput("\n\n".join(lines))
