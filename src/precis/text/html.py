"""LaTeXML HTML (arxiv.org/html) to ``Document``.

arXiv renders LaTeX sources to HTML with LaTeXML, which keeps the paper's
structure as nested ``<section class="ltx_section|ltx_subsection|...">``
elements, math as MathML with the LaTeX source in ``alttext``, and floats as
``<figure class="ltx_figure|ltx_table">``. The parser walks that tree:

- sections, subsections and subsubsections become ``Section`` records;
  ``\\paragraph`` blocks stay inside their section with the run-in title in
  front of the text;
- math becomes ``$<LaTeX>$`` (display equations keep their number);
- footnotes are kept inline where they are referenced;
- figure and table floats become ``Caption`` records (tables with their cell
  text) and are left out of the section text;
- the title block, abstract, keywords, acknowledgements and bibliography are
  dropped.

Everything outside the ``ltx_document`` article (arXiv's page chrome, the
table of contents, the license box) is ignored.
"""

from __future__ import annotations

import re
from typing import Literal

from lxml import html as lxml_html
from lxml.etree import ParserError
from lxml.html import HtmlElement

from precis.models import Caption, Document
from precis.text.document import ParseError, SectionDraft, build_document, split_into_parts
from precis.text.normalize import squash
from precis.util import sha256_hex

# Front matter and back matter that never belong in the body text.
_SKIP_CLASSES = frozenset(
    {
        "ltx_abstract",
        "ltx_acknowledgements",
        "ltx_authors",
        "ltx_bibliography",
        "ltx_classification",
        "ltx_dates",
        "ltx_index",
        "ltx_keywords",
        "ltx_pagination",
        "ltx_subtitle",
        "ltx_title_document",
        "ltx_TOC",
    }
)
# Inline material that is markup noise rather than text: unconverted macro
# names (ltx_ERROR) and footnote marks (the footnote text itself is kept).
_DROP_INLINE_CLASSES = frozenset({"ltx_ERROR", "ltx_note_mark", "ltx_note_type"})
_DROP_TAGS = frozenset({"script", "style", "img", "object", "svg", "button", "canvas", "video"})
_RUN_IN_SECTIONS = frozenset({"ltx_paragraph", "ltx_subparagraph"})
_LIST_TAGS = frozenset({"ul", "ol", "dl"})
_BLOCK_TAGS = frozenset({"p", "div", "li", "dd", "dt", "td", "th", "tr", "blockquote", "section"})
_SKIP_TITLES = re.compile(
    r"^(acknowledge?ments?|references|bibliography|abstract)$",
    re.IGNORECASE,
)
_TEX_STYLE = re.compile(r"\\(?:displaystyle|textstyle|scriptstyle)\b\s*")

# arXiv's banner on a conversion that LaTeXML abandoned part-way: the page
# exists but the text after the failure point is missing.
_FATAL_MARKER = b"had a Fatal error"

_LICENSE_LINK = re.compile(rb'<a[^>]*\bid="license-tr"[^>]*>(.*?)</a>', re.DOTALL)
_HREF = re.compile(rb'\bhref="([^"]+)"')
_PARSER = lxml_html.HTMLParser(encoding="utf-8")
_ARXIV_NONEXCLUSIVE = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
_CC_LABEL = re.compile(r"^CC (?P<terms>BY(?:-(?:NC|ND|SA))*) (?P<version>\d\.\d)$")
_CC_ZERO = re.compile(r"^CC (?:Zero|0)(?: 1\.0)?$", re.IGNORECASE)


def html_unusable_reason(data: bytes) -> str | None:
    """Why an arxiv.org/html response is not a usable conversion, or ``None`` if it is.

    arXiv answers some requests for papers it could not convert with an
    ordinary HTML page (a notice, not the paper), and marks conversions that
    died part-way with a "Fatal error" banner; both should fall back to PDF.
    """
    if b"ltx_document" not in data:
        return "the HTML page is a placeholder, not a LaTeXML conversion"
    if _FATAL_MARKER in data:
        return "LaTeXML stopped with a fatal error, so the HTML may be truncated"
    return None


def html_license(data: bytes) -> str | None:
    """The license arXiv shows in the HTML page's info box, as a URL when it has one."""
    match = _LICENSE_LINK.search(data)
    if match is None:
        return None
    label = squash(re.sub(rb"<[^>]+>", b" ", match.group(1)).decode("utf-8", "replace"))
    label = label.removeprefix("License:").strip()
    href = _HREF.search(match.group(0))
    url = href.group(1).decode() if href else ""
    if "creativecommons.org" in url:
        return url
    # The box links to arXiv's help page and names the license in its text
    # ("CC BY 4.0"); the canonical URL is what the PDF metadata carries.
    if cc := _CC_LABEL.match(label):
        return f"https://creativecommons.org/licenses/{cc['terms'].lower()}/{cc['version']}/"
    if _CC_ZERO.match(label):
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    if "non-exclusive" in label.lower():
        return _ARXIV_NONEXCLUSIVE
    return label or None


def parse_html(data: bytes, paper_id: str) -> Document:
    """Parse a LaTeXML HTML page into a ``Document``; raises ``ParseError``."""
    try:
        # arXiv serves UTF-8; without this lxml falls back to Latin-1 when a
        # page has no charset declaration.
        root = lxml_html.document_fromstring(data, parser=_PARSER)
    except (ParserError, ValueError) as exc:
        raise ParseError(f"unreadable HTML: {exc}") from exc
    if not isinstance(root, HtmlElement):
        raise ParseError("unreadable HTML")
    articles = root.find_class("ltx_document")
    if not articles:
        raise ParseError("no LaTeXML document in the HTML page")
    parser = _LatexmlParser()
    drafts = parser.document(articles[0])
    if not drafts:
        raise ParseError("the HTML page has no body text")
    return build_document(paper_id, "html", drafts, parser.captions, sha256_hex(data))


def _classes(el: HtmlElement) -> set[str]:
    return set((el.get("class") or "").split())


def _is_element(el: object) -> bool:
    # Comments and processing instructions share the tree but have no str tag.
    return isinstance(getattr(el, "tag", None), str)


def _clean_tex(tex: str) -> str:
    return squash(_TEX_STYLE.sub("", tex))


class _LatexmlParser:
    """One pass over a LaTeXML article; collects captions as floats are met."""

    def __init__(self) -> None:
        self.captions: list[Caption] = []
        self._counts = {"figure": 0, "table": 0}

    # -- structure ----------------------------------------------------------

    def document(self, article: HtmlElement) -> list[SectionDraft]:
        drafts: list[SectionDraft] = []
        loose: list[str] = []
        for child in article:
            if not _is_element(child) or self._skipped(child):
                continue
            if self._is_sectioning(child):
                appendix = "ltx_appendix" in _classes(child)
                drafts.extend(self._section(child, level=1, appendix=appendix))
            elif not drafts and self._is_loose_text(child):
                loose.extend(self._blocks(child))
        if not drafts:
            # A paper without \section commands: page through it in parts.
            return split_into_parts(loose)
        if loose:
            drafts.insert(0, SectionDraft(title="Untitled section", level=1, paragraphs=loose))
        return drafts

    def _section(self, el: HtmlElement, level: int, appendix: bool) -> list[SectionDraft]:
        title = self._title(el)
        if _SKIP_TITLES.match(title.strip(" .:")):
            return []
        own: list[str] = []
        children: list[SectionDraft] = []
        for child in el:
            if not _is_element(child) or self._is_title(child) or self._skipped(child):
                continue
            if self._is_sectioning(child):
                children.extend(self._section(child, level + 1, appendix))
            else:
                own.extend(self._blocks(child))
        draft = SectionDraft(title=title, level=level, paragraphs=own, appendix=appendix)
        return [draft, *children]

    def _title(self, section: HtmlElement) -> str:
        for child in section:
            if _is_element(child) and self._is_title(child):
                without_tag = self._inline(child, skip_tags=True)
                return squash(without_tag) or squash(self._inline(child))
        return ""

    def _is_loose_text(self, el: HtmlElement) -> bool:
        """Body text outside any section: paragraphs, not front-matter notes or marks."""
        return not self._is_title(el) and el.tag != "span" and "ltx_note" not in _classes(el)

    @staticmethod
    def _is_title(el: HtmlElement) -> bool:
        return el.tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and "ltx_title" in _classes(el)

    @staticmethod
    def _is_sectioning(el: HtmlElement) -> bool:
        return el.tag == "section" and not (_classes(el) & _RUN_IN_SECTIONS)

    @staticmethod
    def _skipped(el: HtmlElement) -> bool:
        return bool(_classes(el) & _SKIP_CLASSES)

    @staticmethod
    def _is_float(el: HtmlElement) -> bool:
        return el.tag == "figure" or "ltx_float" in _classes(el)

    # -- blocks -------------------------------------------------------------

    def _blocks(self, el: HtmlElement) -> list[str]:
        """Paragraph strings for one block-level element of a section body."""
        cls = _classes(el)
        if self._skipped(el) or el.tag in _DROP_TAGS:
            return []
        if self._is_float(el):
            self._add_float(el)
            return []
        if "ltx_para" in cls:
            return self._para(el)
        if el.tag in _LIST_TAGS:
            return self._list(el)
        if el.tag == "p" or el.tag == "table":
            text = squash(self._inline(el))
            return [text] if text else []
        # A container: a \paragraph block, theorem, proof, quote or plain div.
        out: list[str] = []
        title = ""
        if el.text and el.text.strip():
            out.append(el.text)
        for child in el:
            if not _is_element(child):
                continue
            if self._is_title(child):
                title = squash(self._inline(child))
            else:
                out.extend(self._blocks(child))
            if child.tail and child.tail.strip():
                out.append(child.tail)
        out = [p for p in (squash(p) for p in out) if p]
        if title:
            out = [f"{title} {out[0]}", *out[1:]] if out else [title]
        return out

    def _para(self, el: HtmlElement) -> list[str]:
        """An ``ltx_para``: one paragraph, split only where a list interrupts it."""
        out: list[str] = []
        buffer: list[str] = [el.text or ""]

        def flush() -> None:
            text = squash("".join(buffer))
            if text:
                out.append(text)
            buffer.clear()

        for child in el:
            if not _is_element(child):
                buffer.append(child.tail or "")
                continue
            if self._is_float(child):
                self._add_float(child)
            elif child.tag in _LIST_TAGS or "ltx_para" in _classes(child):
                flush()
                out.extend(self._blocks(child))
            else:
                buffer.append(" " + self._inline(child) + " ")
            buffer.append(child.tail or "")
        flush()
        return out

    def _list(self, el: HtmlElement) -> list[str]:
        out: list[str] = []
        term = ""
        for item in el:
            if not _is_element(item):
                continue
            if item.tag == "dt":
                term = squash(self._inline(item))
                continue
            label = ""
            body: list[str] = []
            if item.text and item.text.strip():
                body.append(squash(item.text))
            for child in item:
                if not _is_element(child):
                    continue
                if "ltx_tag" in _classes(child) and not label:
                    label = squash(child.text_content())
                else:
                    body.extend(self._blocks(child))
                if child.tail and child.tail.strip():
                    body.append(squash(child.tail))
            prefix = " ".join(p for p in (term, label) if p)
            term = ""
            if body and prefix:
                body[0] = f"{prefix} {body[0]}"
            elif prefix:
                body = [prefix]
            out.extend(body)
        return out

    # -- inline -------------------------------------------------------------

    def _inline(self, el: HtmlElement, *, skip_tags: bool = False) -> str:
        parts: list[str] = []
        self._collect(el, parts, skip_tags)
        return "".join(parts)

    def _collect(self, el: HtmlElement, parts: list[str], skip_tags: bool) -> None:
        cls = _classes(el)
        tag = el.tag
        if tag == "math":
            parts.append(self._math(el))
            return
        if "ltx_note" in cls:
            parts.append(self._footnote(el))
            return
        if self._is_float(el):
            self._add_float(el)
            return
        if tag in _DROP_TAGS or cls & _DROP_INLINE_CLASSES or (skip_tags and "ltx_tag" in cls):
            return
        if tag == "br":
            parts.append(" ")
            return
        if tag == "table" and cls & {"ltx_equation", "ltx_equationgroup"}:
            parts.append(f" {self._equation(el)} ")
            return
        block = tag in _BLOCK_TAGS
        if block:
            parts.append(" ")
        if el.text:
            parts.append(el.text)
        for child in el:
            if _is_element(child):
                self._collect(child, parts, skip_tags)
            if child.tail:
                parts.append(child.tail)
        if block:
            parts.append(" ")

    @staticmethod
    def _math(el: HtmlElement) -> str:
        tex = el.get("alttext")
        if tex is None:
            annotation = el.xpath(".//*[local-name()='annotation'][@encoding='application/x-tex']")
            tex = annotation[0].text_content() if annotation else el.text_content()
        tex = _clean_tex(tex)
        return f"${tex}$" if tex else ""

    def _equation(self, table: HtmlElement) -> str:
        """A display equation (or aligned group): one ``$...$`` per row, plus its number."""
        rendered: list[str] = []
        for row in table.iter("tr"):
            tex = " ".join(_clean_tex(m.get("alttext") or "") for m in row.iter("math"))
            numbers = [squash(t.text_content()) for t in row.find_class("ltx_tag_equation")]
            pieces = [f"${squash(tex)}$"] if tex.strip() else []
            pieces.extend(n for n in numbers if n)
            if pieces:
                rendered.append(" ".join(pieces))
        return " ".join(rendered)

    def _footnote(self, note: HtmlElement) -> str:
        content = note.find_class("ltx_note_content")
        body = content[0] if content else note
        text = squash(self._inline(body, skip_tags=True))
        return f" (Footnote: {text}) " if text else ""

    # -- floats -------------------------------------------------------------

    def _add_float(self, fig: HtmlElement) -> None:
        kind: Literal["figure", "table"] = "table" if "ltx_table" in _classes(fig) else "figure"
        own: list[str] = []
        panels: list[str] = []
        for caption in fig.iter("figcaption"):
            text = squash(self._inline(caption))
            if not text:
                continue
            owner = next(caption.iterancestors("figure"), None)
            (own if owner is fig else panels).append(text)
        lines = [" ".join([*own, *panels])] if own or panels else []
        if kind == "table":
            lines.extend(self._cells(fig))
        text = "\n".join(line for line in lines if line)
        if not text:
            return
        self._counts[kind] += 1
        prefix = "fig" if kind == "figure" else "tab"
        self.captions.append(Caption(id=f"{prefix}{self._counts[kind]}", kind=kind, text=text))

    def _cells(self, fig: HtmlElement) -> list[str]:
        """Table rows as ``cell | cell`` lines, for the outermost tabulars in a float."""
        rows: list[str] = []
        for table in fig.iter("table"):
            if _classes(table) & {"ltx_equation", "ltx_equationgroup"}:
                continue
            if any(a.tag == "table" for a in table.iterancestors()):
                continue
            for tr in table.iter("tr"):
                if next(tr.iterancestors("table")) is not table:
                    continue
                cells = [
                    squash(self._inline(cell))
                    for cell in tr
                    if _is_element(cell) and cell.tag in {"td", "th"}
                ]
                if any(cells):
                    rows.append(" | ".join(cells))
        return rows
