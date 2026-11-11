"""Searching the arXiv API by category and submission date, and fetching metadata by id.

A search is ``(cat:A OR cat:B) AND submittedDate:[start TO end]``, sorted by
submission date and read in pages of ``fetch.page_size``. Two properties of
the API shape the code:

- Deep paging is capped and unreliable, so a date range whose result count
  exceeds ``result_cap`` is split in half (recursively, down to the minute)
  and each half is paged separately.
- The API sometimes answers a valid request with an empty page. An empty
  page before the reported end of the results is retried as a fresh
  request; a window that reports zero results is checked once more before
  it is believed.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from datetime import date, datetime, time, timedelta

from precis.models import PaperMeta
from precis.sources.atom import ArxivApiError, FeedPage, parse_feed, parse_id
from precis.sources.http import ArxivHttp

logger = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
# Results per query before a date window is split. The API documents a hard
# limit of 30,000, but paging past ~10,000 is slow and drops pages.
RESULT_CAP = 10_000
IDS_PER_REQUEST = 100
_ZERO_RESULT_ATTEMPTS = 2


def build_query(categories: Sequence[str], start: datetime, end: datetime) -> str:
    """The ``search_query`` for papers in any of ``categories`` submitted in [start, end]."""
    if not categories:
        raise ValueError("at least one category is required")
    cats = " OR ".join(f"cat:{c}" for c in categories)
    if len(categories) > 1:
        cats = f"({cats})"
    return f"{cats} AND submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"


async def search(
    http: ArxivHttp,
    categories: Sequence[str],
    since: date,
    until: date,
    *,
    page_size: int,
    max_attempts: int,
    result_cap: int = RESULT_CAP,
) -> AsyncIterator[PaperMeta]:
    """Yield papers submitted from ``since`` to ``until`` (inclusive), oldest first.

    Each paper is yielded once even when paging shifts it between pages.
    Stop iterating to stop requesting: pages are fetched lazily.
    """
    start = datetime.combine(since, time.min)
    end = datetime.combine(until, time(23, 59))
    seen: set[str] = set()
    pager = _Pager(http, page_size=page_size, max_attempts=max_attempts, result_cap=result_cap)
    async for meta in pager.window(categories, start, end):
        if meta.id not in seen:
            seen.add(meta.id)
            yield meta


async def fetch_metadata(
    http: ArxivHttp, ids: Sequence[str], *, max_attempts: int
) -> list[PaperMeta]:
    """Metadata for ``ids`` (with or without versions), in the order given.

    Raises ``ArxivApiError`` naming the ids the API did not return.
    """
    wanted = [parse_id(i) for i in ids]
    found: dict[str, PaperMeta] = {}
    for n in range(0, len(wanted), IDS_PER_REQUEST):
        chunk = wanted[n : n + IDS_PER_REQUEST]
        id_list = ",".join(f"{i}v{v}" if v else i for i, v in chunk)
        params: dict[str, str | int] = {"id_list": id_list, "max_results": len(chunk)}
        page = await _request(http, params, max_attempts=max_attempts, expect=len(chunk))
        found.update((m.id, m) for m in page.entries)
    missing = [i for i, _ in wanted if i not in found]
    if missing:
        raise ArxivApiError(f"the API returned no entry for {', '.join(missing)}")
    return [found[i] for i, _ in wanted]


class _Pager:
    def __init__(self, http: ArxivHttp, *, page_size: int, max_attempts: int, result_cap: int):
        self.http = http
        self.page_size = page_size
        self.max_attempts = max_attempts
        self.result_cap = result_cap

    async def window(
        self, categories: Sequence[str], start: datetime, end: datetime
    ) -> AsyncIterator[PaperMeta]:
        params: dict[str, str | int] = {
            "search_query": build_query(categories, start, end),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
            "max_results": self.page_size,
        }
        page = await self._page(params, 0)
        if page.total > self.result_cap:
            if end - start > timedelta(minutes=1):
                middle = start + (end - start) / 2
                middle = middle.replace(second=0, microsecond=0)
                logger.info(
                    "%s..%s has %d results; splitting at %s", start, end, page.total, middle
                )
                async for meta in self.window(categories, start, middle):
                    yield meta
                async for meta in self.window(categories, middle + timedelta(minutes=1), end):
                    yield meta
                return
            logger.warning(
                "%s has %d results, more than the cap of %d; the rest are skipped",
                start,
                page.total,
                self.result_cap,
            )
        offset = 0
        limit = min(page.total, self.result_cap)
        while True:
            for meta in page.entries:
                yield meta
            # Advance by what arrived: the API sometimes returns short pages.
            offset += len(page.entries)
            if offset >= limit:
                return
            page = await self._page(params, offset)

    async def _page(self, params: dict[str, str | int], start: int) -> FeedPage:
        return await _request(self.http, {**params, "start": start}, max_attempts=self.max_attempts)


async def _request(
    http: ArxivHttp,
    params: dict[str, str | int],
    *,
    max_attempts: int,
    expect: int | None = None,
) -> FeedPage:
    """One API request, retried while the page comes back empty when it should not.

    ``expect`` is how many entries the request must return (for id lists);
    for searches it is what the feed's own total says is left.
    """
    start = int(params.get("start", 0))
    remaining = 0
    for attempt in range(1, max_attempts + 1):
        response = await http.get(API_URL, params=params)
        if response.status_code != 200:
            raise ArxivApiError(f"arXiv API returned HTTP {response.status_code}")
        page = parse_feed(response.content)
        if page.entries:
            return page
        remaining = expect if expect is not None else page.total - start
        # A zero total is usually true but is sometimes the same transient
        # failure, so it is believed only on the second answer.
        if remaining <= 0 and attempt >= min(_ZERO_RESULT_ATTEMPTS, max_attempts):
            return page
        logger.info("empty page at offset %d (%d results expected); asking again", start, remaining)
    raise ArxivApiError(
        f"the API returned an empty page at offset {start} ({remaining} results expected) "
        f"{max_attempts} times"
    )
