"""Builders for synthetic test inputs: LaTeXML-style HTML pages and text-only PDFs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "papers"

# One entry per line: the text, or (text, font size).
type PdfLine = str | tuple[str, float]


def make_pdf(
    pages: Sequence[Sequence[PdfLine]],
    *,
    info: Mapping[str, str] | None = None,
) -> bytes:
    """A minimal PDF with one Helvetica text line per entry, top to bottom.

    Each line is its own text object at its own height, which is how pypdf
    sees lines in real papers. Text must be encodable as cp1252.
    """
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",  # the page tree, filled in once the page object numbers are known
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    page_refs: list[int] = []
    for lines in pages:
        commands: list[bytes] = []
        y = 760.0
        for line in lines:
            text, size = (line, 10.0) if isinstance(line, str) else line
            commands.append(b"BT /F1 %.1f Tf 56 %.1f Td (%s) Tj ET" % (size, y, _escape(text)))
            y -= size * 1.5
        stream = b"\n".join(commands)
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        content_ref = len(objects)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content_ref
        )
        page_refs.append(len(objects))
    kids = b" ".join(b"%d 0 R" % ref for ref in page_refs)
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_refs))
    trailer_info = b""
    if info:
        entries = b" ".join(b"/%s (%s)" % (k.encode(), _escape(v)) for k, v in info.items())
        objects.append(b"<< %s >>" % entries)
        trailer_info = b" /Info %d 0 R" % len(objects)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\n" % (len(objects) + 1, trailer_info)
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


def _escape(text: str) -> bytes:
    raw = text.encode("cp1252")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def latexml_page(
    body: str,
    *,
    license_label: str | None = "CC BY 4.0",
    front_matter: bool = True,
) -> bytes:
    """An arxiv.org/html-style page around ``body`` (the article's sections)."""
    infobox = (
        '<div id="infobox"><a id="license-tr" '
        'href="https://info.arxiv.org/help/license/index.html#licenses-available">'
        f"License: {license_label}</a></div>"
        if license_label
        else ""
    )
    front = (
        '<h1 class="ltx_title ltx_title_document">A Paper Title</h1>'
        '<div class="ltx_authors"><span class="ltx_creator ltx_role_author">'
        '<span class="ltx_personname">Ada Author</span></span></div>'
        '<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>'
        '<p class="ltx_p">ABSTRACT-TEXT summarizes the paper.</p></div>'
        '<div class="ltx_keywords">KEYWORD-TEXT</div>'
        if front_matter
        else ""
    )
    return (
        "<!DOCTYPE html><html><head><title>A Paper Title</title></head><body>"
        '<nav class="ltx_page_navbar">NAV-TEXT</nav>'
        f'{infobox}<div class="ltx_page_main"><div class="ltx_page_content">'
        f'<article class="ltx_document">{front}{body}</article>'
        "</div></div></body></html>"
    ).encode()


def section(
    sid: str, number: str, title: str, *content: str, kind: str = "section", heading: str = "h2"
) -> str:
    """A LaTeXML ``<section>`` with a numbered title and the given inner HTML."""
    return (
        f'<section id="{sid}" class="ltx_{kind}">'
        f'<{heading} class="ltx_title ltx_title_{kind}">'
        f'<span class="ltx_tag ltx_tag_{kind}">{number} </span>{title}</{heading}>'
        + "".join(content)
        + "</section>"
    )


def para(*paragraphs: str) -> str:
    """An ``ltx_para`` holding one ``ltx_p`` per argument."""
    inner = "".join(f'<p class="ltx_p">{p}</p>' for p in paragraphs)
    return f'<div class="ltx_para">{inner}</div>'


def atom_entry(
    paper_id: str,
    version: int = 1,
    *,
    published: str = "2025-07-01T10:00:00Z",
    updated: str | None = None,
    categories: Sequence[str] = ("cs.AR",),
    title: str | None = None,
) -> str:
    """One ``<entry>`` as the arXiv API writes it."""
    cats = "".join(
        f'<category term="{c}" scheme="http://arxiv.org/schemas/atom"/>' for c in categories
    )
    versioned = f"{paper_id}v{version}"
    return (
        f"<entry><id>http://arxiv.org/abs/{versioned}</id>"
        f"<title>{title or 'Paper ' + paper_id}\n  continued</title>"
        f"<updated>{updated or published}</updated><published>{published}</published>"
        f'<link href="https://arxiv.org/abs/{versioned}" rel="alternate" type="text/html"/>'
        f'<link href="https://arxiv.org/pdf/{versioned}" rel="related" title="pdf"/>'
        f"<summary>  An abstract\n about {paper_id}.  </summary>{cats}"
        f'<arxiv:primary_category term="{categories[0]}"/>'
        "<author><name>Ada Author</name></author><author><name>Bob Builder</name></author>"
        "</entry>"
    )


def atom_feed(entries: Sequence[str], *, total: int, start: int = 0) -> bytes:
    """A page of API results."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom">'
        "<id>https://arxiv.org/api/x</id><title>query</title>"
        f"<opensearch:totalResults>{total}</opensearch:totalResults>"
        f"<opensearch:startIndex>{start}</opensearch:startIndex>"
        f"<opensearch:itemsPerPage>{len(entries)}</opensearch:itemsPerPage>"
        + "".join(entries)
        + "</feed>"
    ).encode()


def paper_pdf(marker: str = "BODY", **info: str) -> bytes:
    """A small but complete paper as a PDF: passes the quality gate."""
    lines: list[PdfLine] = [("A Synthetic Paper", 16.0), "Abstract", "An abstract."]
    for number, title in enumerate(("Introduction", "Design", "Evaluation", "Conclusion"), 1):
        lines.append(f"{number} {title}")
        for n in range(8):
            lines.append(
                f"{marker} {title.lower()} line {n} explains how the cache keeps its useful"
            )
            lines.append(f"blocks for {title.lower()} number {n} while the memory system is busy.")
    lines += ["References", "[1] A. Smith. Caches. 2020."]
    pages = [lines[i : i + 45] for i in range(0, len(lines), 45)]
    return make_pdf(pages, info={"License": "http://creativecommons.org/licenses/by/4.0/", **info})


def paper_html(marker: str = "BODY") -> bytes:
    """A small but complete LaTeXML page: passes the quality gate."""
    body = "".join(
        section(f"S{n}", str(n), title, para(prose(f"{marker} {title.lower()}", sentences=6)))
        for n, title in enumerate(("Introduction", "Design", "Evaluation", "Conclusion"), 1)
    )
    return latexml_page(body)


def prose(topic: str, sentences: int = 4) -> str:
    """Filler sentences that read as English prose (for length and quality gates)."""
    sentence = (
        f"The {topic} design is evaluated on a set of workloads and it reduces the "
        "latency of the memory system compared with the baseline that we describe."
    )
    return " ".join([sentence] * sentences)
