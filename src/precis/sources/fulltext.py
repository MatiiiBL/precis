"""Downloading and parsing a paper's full text: HTML first, PDF as the fallback.

For each paper the fetcher tries ``https://arxiv.org/html/<id>v<n>`` and
falls back to ``https://arxiv.org/pdf/<id>v<n>`` when the HTML does not exist
(404), is a conversion-failure placeholder, or parses into something the
quality gate rejects. Raw files are cached in a ``RawStore``, and a missing
HTML version is remembered there too, so a resumed or repeated run makes no
request for a paper it has seen before. Parsing runs in a worker thread
because a large PDF takes seconds of CPU.
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pypdf import PdfReader

from precis.models import Document, PaperMeta
from precis.sources.http import ArxivHttp, FetchError
from precis.sources.raw import RawStore
from precis.text.document import ParseError
from precis.text.html import html_license, html_unusable_reason, parse_html
from precis.text.pdf import parse_pdf
from precis.text.quality import check_document

logger = logging.getLogger(__name__)

ARXIV = "https://arxiv.org"


@dataclass(frozen=True)
class FullText:
    """A parsed paper and how it was obtained."""

    document: Document
    license: str | None
    fallback_reason: str | None
    """Why the HTML was not used, when the document came from the PDF."""


def html_url(paper_id: str, version: int | None) -> str:
    return f"{ARXIV}/html/{paper_id}" + (f"v{version}" if version is not None else "")


def pdf_url(paper_id: str, version: int | None) -> str:
    return f"{ARXIV}/pdf/{paper_id}" + (f"v{version}" if version is not None else "")


class FullTextFetcher:
    """Turns ``PaperMeta`` into a checked ``Document``; see the module docstring."""

    def __init__(self, http: ArxivHttp, raw: RawStore, prefer: Literal["html", "pdf"]) -> None:
        self.http = http
        self.raw = raw
        self.prefer = prefer

    async def fetch(self, meta: PaperMeta) -> FullText:
        """Raises ``FetchError`` when a download fails and ``ParseError`` when no
        format yields a usable document (the message names every reason)."""
        reasons: list[str] = []
        if self.prefer == "html":
            html = await self._html(meta, reasons)
            if html is not None:
                try:
                    doc = await _parse(parse_html, html, meta.id)
                    return FullText(doc, html_license(html), None)
                except ParseError as exc:
                    reasons.append(f"HTML unusable: {exc}")
        try:
            pdf = await self._pdf(meta)
        except FetchError as exc:
            raise FetchError("; ".join([str(exc), *reasons])) from exc
        try:
            doc = await _parse(parse_pdf, pdf, meta.id)
        except ParseError as exc:
            raise ParseError("; ".join([f"PDF: {exc}", *reasons])) from exc
        license_url = await asyncio.to_thread(_pdf_license, pdf)
        return FullText(doc, license_url, "; ".join(reasons) or None)

    async def _html(self, meta: PaperMeta, reasons: list[str]) -> bytes | None:
        cached = await self.raw.load(meta.id, meta.version, "html")
        if cached is not None:
            return cached
        known_missing = await self.raw.missing_reason(meta.id, meta.version, "html")
        if known_missing is not None:
            reasons.append(known_missing)
            return None
        response = await self.http.get(html_url(meta.id, meta.version))
        if response.status_code == 404:
            reason = "no HTML version (HTTP 404)"
        elif response.status_code != 200:
            # Not remembered: a 403 or similar may not be permanent.
            reasons.append(f"HTML request failed (HTTP {response.status_code})")
            return None
        else:
            reason = html_unusable_reason(response.content) or ""
        if reason:
            await self.raw.mark_missing(meta.id, meta.version, "html", reason)
            reasons.append(reason)
            return None
        await self.raw.save(meta.id, meta.version, "html", response.content)
        return response.content

    async def _pdf(self, meta: PaperMeta) -> bytes:
        cached = await self.raw.load(meta.id, meta.version, "pdf")
        if cached is not None:
            return cached
        response = await self.http.get(pdf_url(meta.id, meta.version))
        if response.status_code == 404:
            raise FetchError("no PDF (HTTP 404; withdrawn versions have no full text)")
        if response.status_code != 200:
            raise FetchError(f"PDF download failed (HTTP {response.status_code})")
        if not response.content.startswith(b"%PDF"):
            raise FetchError("the PDF URL did not return a PDF")
        await self.raw.save(meta.id, meta.version, "pdf", response.content)
        return response.content


async def _parse(parser: Callable[[bytes, str], Document], data: bytes, paper_id: str) -> Document:
    def run() -> Document:
        doc = parser(data, paper_id)
        check_document(doc)
        return doc

    return await asyncio.to_thread(run)


def _pdf_license(data: bytes) -> str | None:
    """arXiv stamps the license URL into the PDF's document info."""
    try:
        metadata = PdfReader(io.BytesIO(data)).metadata
        value = metadata.get("/License") if metadata else None
    except Exception:  # metadata is a nicety; a broken info dict must not fail the paper
        logger.debug("could not read PDF metadata", exc_info=True)
        return None
    return str(value) if value else None
