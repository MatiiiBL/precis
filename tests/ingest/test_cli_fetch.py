from pathlib import Path

import httpx2 as httpx
import pytest
from typer.testing import CliRunner

from precis.cli import app
from precis.cli import fetch as fetch_cli
from precis.config import FetchConfig
from precis.llm.clock import ManualClock
from precis.models import Stage, StageStatus
from precis.sources.http import ArxivHttp
from precis.store import Store

from .helpers import atom_entry, atom_feed, paper_html, paper_pdf

IDS = ["2507.00001", "2507.00002", "2507.00003"]


class FakeArxiv:
    """export.arxiv.org and arxiv.org in one: search, id lookup, HTML and PDF."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.api_down = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path == "/api/query":
            if self.api_down:
                return httpx.Response(400, text="bad query")
            params = request.url.params
            if "id_list" in params:
                entries = [atom_entry(i) for i in params["id_list"].split(",") if i in IDS]
                return httpx.Response(200, content=atom_feed(entries, total=len(entries)))
            start, size = int(params["start"]), int(params["max_results"])
            page = [atom_entry(i) for i in IDS[start : start + size]]
            return httpx.Response(200, content=atom_feed(page, total=len(IDS), start=start))
        if path == "/html/2507.00001v1":
            return httpx.Response(200, content=paper_html())
        if path in ("/pdf/2507.00002v1", "/pdf/2507.00001v1"):
            return httpx.Response(200, content=paper_pdf())
        return httpx.Response(404, text="Not Found")


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch) -> FakeArxiv:
    fake = FakeArxiv()

    def open_http(config: FetchConfig) -> ArxivHttp:
        transport = httpx.MockTransport(fake.handle)
        return ArxivHttp(config, transport=transport, clock=ManualClock())

    monkeypatch.setattr(fetch_cli, "open_http", open_http)
    return fake


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "precis.toml"
    path.write_text(
        f'store_path = "{tmp_path / "precis.db"}"\n'
        f'[fetch]\nmin_interval_s = 0\nmax_attempts = 2\nraw_dir = "{tmp_path / "raw"}"\n'
    )
    return path


def _run(config: Path, *args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["--config", str(config), "fetch", *args])
    return result.exit_code, result.output


def _store(tmp_path: Path) -> Store:
    return Store(tmp_path / "precis.db")


def test_fetch_arxiv_stores_metadata_text_and_failures(
    site: FakeArxiv, config: Path, tmp_path: Path
) -> None:
    code, output = _run(config, "arxiv", "--category", "cs.AR", "--since", "2025-07-01")
    assert code == 0, output
    assert "parsed from HTML" in output
    with _store(tmp_path) as store:
        assert sorted(p.id for p in store.list_papers()) == IDS
        formats = {i: d.source_format for i in IDS if (d := store.get_document(i))}
        assert formats == {"2507.00001": "html", "2507.00002": "pdf"}
        failed = store.get_stage("2507.00003", Stage.FETCH)
        assert failed is not None
        assert failed.status is StageStatus.FAILED
        assert "no PDF (HTTP 404" in (failed.error or "")
        (run,) = store.runs()
        assert run["config"]["command"] == "fetch arxiv"
        assert run["stats"]["counts"] == {"html": 1, "pdf": 1, "failed": 1}

    site.paths.clear()
    code, output = _run(config, "arxiv", "--category", "cs.AR", "--since", "2025-07-01")
    assert code == 0, output
    # Done papers come from the store; the failed one is tried again, and only
    # its PDF, since the missing HTML was remembered.
    assert [p for p in site.paths if not p.startswith("/api")] == ["/pdf/2507.00003v1"]
    assert "already done" in output


def test_metadata_only_and_limit(site: FakeArxiv, config: Path, tmp_path: Path) -> None:
    code, output = _run(
        config, "arxiv", "--category", "cs.AR", "--since", "2025-07-01", "--limit", "2",
        "--metadata-only",
    )  # fmt: skip
    assert code == 0, output
    assert all(p.startswith("/api") for p in site.paths)
    with _store(tmp_path) as store:
        assert sorted(p.id for p in store.list_papers()) == IDS[:2]
        assert store.papers_with_status(Stage.FETCH, StageStatus.PENDING) == IDS[:2]
        assert store.get_document(IDS[0]) is None


def test_fetch_ids(site: FakeArxiv, config: Path, tmp_path: Path) -> None:
    code, output = _run(config, "ids", "2507.00001")
    assert code == 0, output
    with _store(tmp_path) as store:
        assert store.get_document("2507.00001") is not None
    code, output = _run(config, "ids", "2507.00001", "2507.09999")
    assert code == 1
    assert "2507.09999" in output


def test_a_failing_search_exits_with_an_error(site: FakeArxiv, config: Path) -> None:
    site.api_down = True
    code, output = _run(config, "arxiv", "--category", "cs.AR", "--since", "2025-07-01")
    assert code == 1
    assert "looking up papers failed" in output


def test_fetch_local(config: Path, tmp_path: Path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "one.pdf").write_bytes(paper_pdf())
    code, output = _run(config, "local", str(papers))
    assert code == 0, output
    assert "parsed from PDF" in output
    with _store(tmp_path) as store:
        (paper,) = store.list_papers()
        assert paper.primary_category == "local"
    code, output = _run(config, "local", str(tmp_path / "missing"))
    assert code != 0
