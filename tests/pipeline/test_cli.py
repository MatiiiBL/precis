"""``precis run``, ``precis status`` and ``precis export`` with the offline client."""

import json
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from precis.cli import app
from precis.models import Stage, StageStatus
from precis.store import Store

from .helpers import add_papers, paper


@pytest.fixture
def store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # no precis.toml here
    path = tmp_path / "data" / "precis.db"
    store = Store(path)
    add_papers(store, 4)
    store.close()
    return path


def precis(store_path: Path, *args: str) -> Result:
    # Wide enough that rich does not wrap the lines the tests look for.
    return CliRunner(env={"COLUMNS": "200"}).invoke(app, ["--store", str(store_path), *args])


def test_offline_run_then_status_and_export(store_path: Path, tmp_path: Path) -> None:
    result = precis(store_path, "run", "--offline")
    assert result.exit_code == 0, result.output
    assert "4 of 4 papers finished, 0 blocked" in result.output
    assert "Final summaries of 4 papers: 100.0% of " in result.output
    assert "Revised 1 summaries: " in result.output

    status = precis(store_path, "status")
    assert status.exit_code == 0, status.output
    assert "4 papers" in status.output
    assert "Recent runs" in status.output
    assert "claude-opus-5-5" in status.output

    out = tmp_path / "out" / "summaries.jsonl"
    export = precis(store_path, "export", "--out", str(out))
    assert export.exit_code == 0, export.output
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 4
    assert {line["paper"]["id"] for line in lines} == {f"2501.{i:05d}" for i in range(4)}
    revised = [line for line in lines if line["revised"]]
    assert len(revised) == 1
    assert revised[0]["summary"]["revision"] == 1
    assert all(line["supported_fraction"] is not None for line in lines)


def test_rerun_is_served_without_requests(store_path: Path) -> None:
    assert precis(store_path, "run", "--offline").exit_code == 0
    again = precis(store_path, "run", "--offline")
    assert again.exit_code == 0
    assert "0 requests" in again.output


def test_ids_limit_stages_and_strategy(store_path: Path) -> None:
    result = precis(
        store_path,
        "run",
        "--offline",
        "--stages",
        "read",
        "--strategy",
        "fulltext",
        "--ids",
        "2501.00001,2501.00002",
        "--ids",
        "2501.00003",
        "--limit",
        "2",
    )
    assert result.exit_code == 0, result.output
    assert "2 of 2 papers finished" in result.output
    store = Store(store_path)
    assert store.papers_with_status(Stage.READ, StageStatus.DONE) == ["2501.00001", "2501.00002"]
    assert store.papers_with_status(Stage.VERIFY, StageStatus.DONE) == []
    store.close()


def test_batch_mode_runs_offline_and_costs_less(store_path: Path) -> None:
    online_store = store_path.with_name("online.db")
    batch = precis(store_path, "run", "--offline", "--mode", "batch")
    assert batch.exit_code == 0, batch.output
    assert "4 of 4 papers finished" in batch.output
    store = Store(store_path)
    calls = store.calls()
    assert calls
    assert all(c.batch for c in calls)
    batch_cost = sum(c.cost_usd for c in calls)
    store.close()

    copy = Store(online_store)
    add_papers(copy, 4)
    copy.close()
    online = precis(online_store, "run", "--offline")
    assert online.exit_code == 0, online.output
    store = Store(online_store)
    online_cost = sum(c.cost_usd for c in store.calls())
    store.close()
    # Half the token price, but fewer cache hits between turns minutes apart.
    assert online_cost / 2 < batch_cost < online_cost


def test_budget_flag(store_path: Path) -> None:
    result = precis(store_path, "run", "--offline", "--max-cost", "0.05")
    assert result.exit_code == 0, result.output
    assert "The budget was reached" in result.output


def test_bad_input(store_path: Path, tmp_path: Path) -> None:
    bad = precis(store_path, "run", "--offline", "--stages", "read,fetch")
    assert bad.exit_code == 2
    assert "choose from read, verify, revise" in bad.output
    empty = precis(tmp_path / "empty.db", "run", "--offline")
    assert empty.exit_code == 1
    assert "No fetched papers in the store" in empty.output


def test_limit_picks_fetched_papers_and_blocked_reasons_are_counted(store_path: Path) -> None:
    store = Store(store_path)
    # Newer than the fetched papers, but its text has not been downloaded yet.
    meta, _ = paper(9)
    store.upsert_paper(meta.model_copy(update={"published": date(2026, 1, 1)}))
    store.close()

    picked = precis(store_path, "run", "--offline", "--stages", "read", "--limit", "2")
    assert picked.exit_code == 0, picked.output
    assert "2 of 2 papers finished, 0 blocked" in picked.output

    named = precis(store_path, "run", "--offline", "--ids", f"{meta.id},2501.00000")
    assert named.exit_code == 0, named.output
    assert "1 of 2 papers finished, 1 blocked" in named.output
    assert "Blocked: 1 no document; run precis fetch." in named.output
