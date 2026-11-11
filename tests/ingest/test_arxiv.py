import re
from datetime import date, datetime

import httpx2 as httpx
import pytest

from precis.config import FetchConfig
from precis.llm.clock import ManualClock
from precis.models import PaperMeta
from precis.sources.arxiv import build_query, fetch_metadata, search
from precis.sources.atom import ArxivApiError, parse_feed, parse_id
from precis.sources.http import ArxivHttp

from .helpers import atom_entry, atom_feed


class FakeApi:
    """The arXiv API over a list of (id, submitted) papers, with paging and windows."""

    def __init__(self, papers: list[tuple[str, datetime]]) -> None:
        self.papers = sorted(papers, key=lambda p: p[1])
        self.requests: list[httpx.Request] = []
        self.empty_pages: dict[int, int] = {}  # start offset -> times to answer empty
        self.duplicate_at: int | None = None
        self.served: list[tuple[int, int]] = []  # (start offset, window total) per search page

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        params = request.url.params
        if "id_list" in params:
            wanted = params["id_list"].split(",")
            entries = [atom_entry(parse_id(i)[0], parse_id(i)[1] or 1) for i in wanted]
            entries = [e for e, i in zip(entries, wanted, strict=True) if "9999" not in i]
            return httpx.Response(200, content=atom_feed(entries, total=len(entries)))
        start, end = _window(params["search_query"])
        matching = [p for p in self.papers if start <= p[1] <= end]
        offset, size = int(params["start"]), int(params["max_results"])
        self.served.append((offset, len(matching)))
        if self.empty_pages.get(offset, 0) > 0:
            self.empty_pages[offset] -= 1
            return httpx.Response(200, content=atom_feed([], total=len(matching), start=offset))
        page = matching[offset : offset + size]
        if self.duplicate_at == offset and offset > 0:
            page = [matching[offset - 1], *page[:-1]]  # an item shifted between pages
        entries = [atom_entry(i, published=f"{d:%Y-%m-%dT%H:%M:00Z}") for i, d in page]
        return httpx.Response(200, content=atom_feed(entries, total=len(matching), start=offset))


def _window(query: str) -> tuple[datetime, datetime]:
    low, high = re.findall(r"\d{12}", query)
    return datetime.strptime(low, "%Y%m%d%H%M"), datetime.strptime(high, "%Y%m%d%H%M")


def _http(api: FakeApi) -> ArxivHttp:
    config = FetchConfig(min_interval_s=0, max_attempts=3)
    return ArxivHttp(config, transport=httpx.MockTransport(api.handle), clock=ManualClock())


async def _search(api: FakeApi, *, page_size: int = 2, cap: int = 100) -> list[PaperMeta]:
    async with _http(api) as http:
        return [
            m
            async for m in search(
                http,
                ["cs.AR", "cs.PF"],
                date(2025, 7, 1),
                date(2025, 7, 31),
                page_size=page_size,
                max_attempts=3,
                result_cap=cap,
            )
        ]


def _papers(n: int) -> list[tuple[str, datetime]]:
    return [(f"2507.{i:05d}", datetime(2025, 7, 1 + i % 30, 10, i % 60)) for i in range(n)]


def test_build_query() -> None:
    start, end = datetime(2025, 7, 1), datetime(2025, 7, 31, 23, 59)
    assert build_query(["cs.AR"], start, end) == (
        "cat:cs.AR AND submittedDate:[202507010000 TO 202507312359]"
    )
    assert build_query(["cs.AR", "cs.PF"], start, end).startswith("(cat:cs.AR OR cat:cs.PF) AND")
    with pytest.raises(ValueError, match="category"):
        build_query([], start, end)


def test_feed_entries_become_paper_meta() -> None:
    entry = atom_entry(
        "2507.00367",
        3,
        published="2025-07-01T01:48:28Z",
        updated="2025-09-02T00:00:00Z",
        categories=["cs.AR", "cs.CR"],
        title="Presto: Hardware",
    )
    page = parse_feed(atom_feed([entry, atom_entry("cs/0112017")], total=2))
    first, old = page.entries
    assert first == PaperMeta(
        id="2507.00367",
        version=3,
        title="Presto: Hardware continued",
        authors=["Ada Author", "Bob Builder"],
        abstract="An abstract about 2507.00367.",
        categories=["cs.AR", "cs.CR"],
        primary_category="cs.AR",
        published=date(2025, 7, 1),
        updated=date(2025, 9, 2),
        license=None,
        url="https://arxiv.org/abs/2507.00367v3",
    )
    assert (old.id, old.version) == ("cs/0112017", 1)
    assert page.total == 2


def test_error_feeds_and_junk_raise() -> None:
    error = (
        "<entry><id>http://arxiv.org/api/errors#incorrect_id_format</id>"
        "<title>Error</title><summary>incorrect id format for 1234</summary></entry>"
    )
    with pytest.raises(ArxivApiError, match="incorrect id format"):
        parse_feed(atom_feed([error], total=1))
    with pytest.raises(ArxivApiError, match="malformed"):
        parse_feed(b"<feed><unclosed>")
    with pytest.raises(ArxivApiError, match="Atom feed"):
        parse_feed(b"<html/>")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2406.01234", ("2406.01234", None)),
        ("2406.01234v2", ("2406.01234", 2)),
        ("arXiv:2406.01234", ("2406.01234", None)),
        ("https://arxiv.org/abs/2406.01234v3", ("2406.01234", 3)),
        ("https://arxiv.org/pdf/2406.01234", ("2406.01234", None)),
        ("hep-th/9901001v1", ("hep-th/9901001", 1)),
    ],
)
def test_parse_id(text: str, expected: tuple[str, int | None]) -> None:
    assert parse_id(text) == expected


def test_parse_id_rejects_junk() -> None:
    with pytest.raises(ValueError, match="not an arXiv id"):
        parse_id("hello")


async def test_search_pages_through_every_result_in_order() -> None:
    api = FakeApi(_papers(5))
    found = await _search(api)
    assert [m.id for m in found] == [p[0] for p in api.papers]
    assert [r.url.params["start"] for r in api.requests] == ["0", "2", "4"]
    params = api.requests[0].url.params
    assert (params["sortBy"], params["sortOrder"], params["max_results"]) == (
        "submittedDate",
        "ascending",
        "2",
    )


async def test_an_empty_page_is_asked_for_again() -> None:
    api = FakeApi(_papers(5))
    api.empty_pages = {2: 2}
    found = await _search(api)
    assert len(found) == 5
    assert [r.url.params["start"] for r in api.requests] == ["0", "2", "2", "2", "4"]


async def test_a_page_that_stays_empty_is_an_error() -> None:
    api = FakeApi(_papers(5))
    api.empty_pages = {2: 99}
    with pytest.raises(ArxivApiError, match="empty page at offset 2"):
        await _search(api)


async def test_zero_results_are_confirmed_once() -> None:
    api = FakeApi([])
    assert await _search(api) == []
    assert len(api.requests) == 2


async def test_large_ranges_are_split_into_windows_under_the_cap() -> None:
    api = FakeApi(_papers(40))
    found = await _search(api, page_size=5, cap=12)
    assert sorted(m.id for m in found) == sorted(p[0] for p in api.papers)
    assert len({m.id for m in found}) == 40
    assert api.served[0] == (0, 40)  # the whole month was probed, then split
    # Pages past the first are only ever read from windows under the cap.
    later_pages = [total for start, total in api.served if start > 0]
    assert later_pages
    assert max(later_pages) <= 12


async def test_papers_shifted_between_pages_are_yielded_once() -> None:
    api = FakeApi(_papers(6))
    api.duplicate_at = 2
    found = await _search(api)
    assert len(found) == len({m.id for m in found})


async def test_fetch_metadata_by_id_keeps_the_given_order() -> None:
    api = FakeApi([])
    ids = [f"2507.{i:05d}" for i in range(150, 0, -1)] + ["2406.01234v2"]
    async with _http(api) as http:
        metas = await fetch_metadata(http, ids, max_attempts=3)
    assert [m.id for m in metas] == [parse_id(i)[0] for i in ids]
    assert metas[-1].version == 2
    assert len(api.requests) == 2  # 100 ids per request


async def test_fetch_metadata_names_the_missing_ids() -> None:
    api = FakeApi([])
    async with _http(api) as http:
        with pytest.raises(ArxivApiError, match=r"2507\.09999"):
            await fetch_metadata(http, ["2507.00001", "2507.09999"], max_attempts=1)
