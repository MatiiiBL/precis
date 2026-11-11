import gzip
from datetime import date
from pathlib import Path

import httpx2 as httpx
import pytest

from precis.config import FetchConfig
from precis.llm.clock import ManualClock
from precis.models import PaperMeta
from precis.sources.fulltext import FullTextFetcher, html_url, pdf_url
from precis.sources.http import ArxivHttp, FetchError
from precis.sources.raw import RawStore
from precis.text.document import ParseError

from .helpers import latexml_page, make_pdf, paper_html, paper_pdf, para, section

META = PaperMeta(
    id="2507.00001",
    version=1,
    title="A Paper",
    authors=["Ada Author"],
    abstract="An abstract.",
    categories=["cs.AR"],
    primary_category="cs.AR",
    published=date(2025, 7, 1),
    url="https://arxiv.org/abs/2507.00001v1",
)
HTML = "/html/2507.00001v1"
PDF = "/pdf/2507.00001v1"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"


class FakeArxiv:
    """arxiv.org with a fixed set of pages; everything else is a 404."""

    def __init__(self, routes: dict[str, tuple[int, bytes]]) -> None:
        self.routes = routes
        self.hits: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.hits.append(request.url.path)
        status, body = self.routes.get(request.url.path, (404, b"Not Found"))
        return httpx.Response(status, content=body)


async def _fetch(site: FakeArxiv, raw_dir: Path, prefer: str = "html", meta: PaperMeta = META):
    config = FetchConfig.model_validate({"min_interval_s": 0, "max_attempts": 2, "prefer": prefer})
    transport = httpx.MockTransport(site.handle)
    async with ArxivHttp(config, transport=transport, clock=ManualClock()) as http:
        return await FullTextFetcher(http, RawStore(raw_dir), config.prefer).fetch(meta)


def _stub_html() -> bytes:
    return latexml_page(section("S1", "1", "Intro", para("Too short to summarize.")))


async def test_html_is_preferred_and_reused_from_disk(tmp_path: Path) -> None:
    site = FakeArxiv({HTML: (200, paper_html())})
    result = await _fetch(site, tmp_path)
    assert result.document.source_format == "html"
    assert result.license == CC_BY
    assert result.fallback_reason is None
    assert site.hits == [HTML]
    assert gzip.decompress((tmp_path / "html" / "2507.00001v1.html.gz").read_bytes())

    offline = FakeArxiv({})
    again = await _fetch(offline, tmp_path)
    assert again.document == result.document
    assert offline.hits == []


async def test_missing_html_falls_back_to_the_pdf_and_is_remembered(tmp_path: Path) -> None:
    site = FakeArxiv({PDF: (200, paper_pdf())})
    result = await _fetch(site, tmp_path)
    assert result.document.source_format == "pdf"
    assert result.fallback_reason == "no HTML version (HTTP 404)"
    assert result.license == "http://creativecommons.org/licenses/by/4.0/"
    assert site.hits == [HTML, PDF]

    offline = FakeArxiv({})
    again = await _fetch(offline, tmp_path)
    assert again.fallback_reason == "no HTML version (HTTP 404)"
    assert offline.hits == []


async def test_a_conversion_placeholder_falls_back_to_the_pdf(tmp_path: Path) -> None:
    placeholder = b"<html><body>No HTML for this paper. Conversion failed.</body></html>"
    site = FakeArxiv({HTML: (200, placeholder), PDF: (200, paper_pdf())})
    result = await _fetch(site, tmp_path)
    assert result.document.source_format == "pdf"
    assert "placeholder" in (result.fallback_reason or "")


async def test_html_that_fails_the_quality_gate_falls_back_to_the_pdf(tmp_path: Path) -> None:
    site = FakeArxiv({HTML: (200, _stub_html()), PDF: (200, paper_pdf())})
    result = await _fetch(site, tmp_path)
    assert result.document.source_format == "pdf"
    assert (result.fallback_reason or "").startswith("HTML unusable: only")


async def test_when_nothing_is_usable_the_error_names_every_reason(tmp_path: Path) -> None:
    stub_pdf = make_pdf([["1 Introduction", "Too short to summarize."]])
    site = FakeArxiv({PDF: (200, stub_pdf)})
    with pytest.raises(ParseError) as error:
        await _fetch(site, tmp_path)
    assert str(error.value).startswith("PDF: only")
    assert "no HTML version (HTTP 404)" in str(error.value)


async def test_a_withdrawn_version_has_no_full_text(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match=r"no PDF \(HTTP 404.*no HTML version"):
        await _fetch(FakeArxiv({}), tmp_path)


async def test_a_pdf_url_that_serves_html_is_an_error(tmp_path: Path) -> None:
    site = FakeArxiv({PDF: (200, b"<html>captcha</html>")})
    with pytest.raises(FetchError, match="did not return a PDF"):
        await _fetch(site, tmp_path, prefer="pdf")


async def test_prefer_pdf_never_asks_for_html(tmp_path: Path) -> None:
    site = FakeArxiv({HTML: (200, paper_html()), PDF: (200, paper_pdf())})
    result = await _fetch(site, tmp_path, prefer="pdf")
    assert result.document.source_format == "pdf"
    assert site.hits == [PDF]


async def test_a_refused_html_request_is_not_remembered(tmp_path: Path) -> None:
    site = FakeArxiv({HTML: (403, b"Forbidden"), PDF: (200, paper_pdf())})
    first = await _fetch(site, tmp_path)
    assert first.fallback_reason == "HTML request failed (HTTP 403)"
    # The PDF is on disk now, but the HTML is asked for again next time.
    retry = FakeArxiv({HTML: (200, paper_html())})
    second = await _fetch(retry, tmp_path)
    assert second.document.source_format == "html"
    assert retry.hits == [HTML]


async def test_unversioned_papers_use_unversioned_urls(tmp_path: Path) -> None:
    meta = META.model_copy(update={"version": None})
    site = FakeArxiv({"/html/2507.00001": (200, paper_html())})
    await _fetch(site, tmp_path, meta=meta)
    assert site.hits == ["/html/2507.00001"]
    assert html_url("cs/0112017", 2) == "https://arxiv.org/html/cs/0112017v2"
    assert pdf_url("2507.00001", None) == "https://arxiv.org/pdf/2507.00001"


async def test_raw_store_round_trip_and_layout(tmp_path: Path) -> None:
    raw = RawStore(tmp_path)
    assert raw.path("cs/0112017", 3, "pdf") == tmp_path / "pdf" / "cs_0112017v3.pdf.gz"
    assert await raw.load("2507.00001", 1, "html") is None
    data = b"<html>" + b"x" * 10_000 + b"</html>"
    await raw.save("2507.00001", 1, "html", data)
    assert await raw.load("2507.00001", 1, "html") == data
    path = raw.path("2507.00001", 1, "html")
    assert path.stat().st_size < len(data) // 10
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temp files left
    await raw.mark_missing("2507.00001", 2, "html", "no HTML version (HTTP 404)")
    assert await raw.missing_reason("2507.00001", 2, "html") == "no HTML version (HTTP 404)"
    assert await raw.missing_reason("2507.00001", 1, "html") is None
