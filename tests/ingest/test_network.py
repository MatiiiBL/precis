"""Smoke tests against the real arXiv; run with ``pytest -m network``.

They respect the 3-second interval, so they take about ten seconds.
"""

from datetime import date
from pathlib import Path

import pytest

from precis.config import FetchConfig
from precis.sources.arxiv import fetch_metadata, search
from precis.sources.fulltext import FullTextFetcher
from precis.sources.http import ArxivHttp
from precis.sources.raw import RawStore

pytestmark = pytest.mark.network

PAPER = "2508.21267v1"  # CC BY 4.0, has an HTML version (also a test fixture)


async def test_search_and_fetch_one_paper(tmp_path: Path) -> None:
    config = FetchConfig(raw_dir=tmp_path)
    async with ArxivHttp(config) as http:
        day = [
            m
            async for m in search(
                http,
                ["cs.AR"],
                date(2025, 7, 1),
                date(2025, 7, 1),
                page_size=50,
                max_attempts=config.max_attempts,
            )
        ]
        assert day
        assert all(m.published == date(2025, 7, 1) for m in day)
        assert all("cs.AR" in m.categories for m in day)

        (meta,) = await fetch_metadata(http, [PAPER], max_attempts=config.max_attempts)
        assert meta.title.startswith("Catwalk")
        full = await FullTextFetcher(http, RawStore(tmp_path), "html").fetch(meta)
    assert full.document.source_format == "html"
    assert full.license == "https://creativecommons.org/licenses/by/4.0/"
    assert full.document.n_tokens > 3000
