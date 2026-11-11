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


def prose(topic: str, sentences: int = 4) -> str:
    """Filler sentences that read as English prose (for length and quality gates)."""
    sentence = (
        f"The {topic} design is evaluated on a set of workloads and it reduces the "
        "latency of the memory system compared with the baseline that we describe."
    )
    return " ".join([sentence] * sentences)
