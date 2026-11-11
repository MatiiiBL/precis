"""The arXiv API's Atom feed to ``PaperMeta``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from lxml import etree

from precis.models import PaperMeta
from precis.text.normalize import squash

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

# New-style ids ("2507.00367v2") and old-style ids ("cs/0112017v1").
_ID = re.compile(r"(?P<id>[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v(?P<version>\d+))?")
_ENTRY_URL = re.compile(r"arxiv\.org/abs/(?P<rest>.+)$")


class ArxivApiError(Exception):
    """The API answered with an error feed or something that is not a feed."""


@dataclass(frozen=True)
class FeedPage:
    """One page of results: ``total`` is the query's full result count."""

    total: int
    start: int
    entries: list[PaperMeta]


def parse_id(text: str) -> tuple[str, int | None]:
    """Split ``2406.01234v2``, ``arXiv:2406.01234`` or an abs/pdf URL into (id, version)."""
    match = _ID.search(text.strip())
    if match is None:
        raise ValueError(f"not an arXiv id: {text!r}")
    version = match["version"]
    return match["id"], int(version) if version else None


def parse_feed(data: bytes) -> FeedPage:
    """Parse one API response; raises ``ArxivApiError`` for error feeds and junk."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ArxivApiError(f"malformed Atom feed: {exc}") from exc
    if root.tag != f"{_ATOM}feed":
        raise ArxivApiError(f"expected an Atom feed, got <{root.tag}>")
    entries: list[PaperMeta] = []
    for entry in root.iter(f"{_ATOM}entry"):
        entry_id = entry.findtext(f"{_ATOM}id") or ""
        if "/api/errors" in entry_id:
            message = squash(entry.findtext(f"{_ATOM}summary") or "unknown error")
            raise ArxivApiError(f"arXiv API error: {message}")
        entries.append(_entry(entry))
    return FeedPage(
        total=int(root.findtext(f"{_OPENSEARCH}totalResults") or 0),
        start=int(root.findtext(f"{_OPENSEARCH}startIndex") or 0),
        entries=entries,
    )


def _entry(entry: etree._Element) -> PaperMeta:
    entry_id = entry.findtext(f"{_ATOM}id") or ""
    url_match = _ENTRY_URL.search(entry_id)
    if url_match is None:
        raise ArxivApiError(f"entry id is not an arXiv abs URL: {entry_id!r}")
    paper_id, version = parse_id(url_match["rest"])
    categories: list[str] = []
    for category in entry.iter(f"{_ATOM}category"):
        term = category.get("term")
        if term and term not in categories:
            categories.append(term)
    primary_el = entry.find(f"{_ARXIV}primary_category")
    primary = primary_el.get("term") if primary_el is not None else None
    primary = primary or (categories[0] if categories else "")
    if primary and primary not in categories:
        categories.insert(0, primary)
    alternate = next(
        (
            link.get("href")
            for link in entry.iter(f"{_ATOM}link")
            if link.get("rel") == "alternate" and link.get("href")
        ),
        None,
    )
    return PaperMeta(
        id=paper_id,
        version=version,
        title=squash(entry.findtext(f"{_ATOM}title") or ""),
        authors=[
            squash(name)
            for author in entry.iter(f"{_ATOM}author")
            if (name := author.findtext(f"{_ATOM}name"))
        ],
        abstract=squash(entry.findtext(f"{_ATOM}summary") or ""),
        categories=categories,
        primary_category=primary,
        published=_date(entry.findtext(f"{_ATOM}published")),
        updated=_date(entry.findtext(f"{_ATOM}updated")),
        license=_license(entry),
        url=alternate or f"https://arxiv.org/abs/{paper_id}",
    )


def _license(entry: etree._Element) -> str | None:
    """The license, when the feed carries one (the current API usually does not)."""
    text = entry.findtext(f"{_ARXIV}license") or entry.findtext(f"{_ATOM}rights")
    if text and text.strip():
        return text.strip()
    for link in entry.iter(f"{_ATOM}link"):
        if link.get("rel") == "license" and link.get("href"):
            return link.get("href")
    return None


def _date(text: str | None) -> date:
    if not text:
        raise ArxivApiError("entry without a date")
    return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).date()
