from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus.factories import fill_store, themed_corpus
from precis.cli import app
from precis.cli.corpus import load_report, load_topics
from precis.models import LLMCall
from precis.store import Store

runner = CliRunner()


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    path = tmp_path / "precis.db"
    papers, _ = themed_corpus(8)
    with Store(path) as store:
        fill_store(store, papers)
    return path


def test_topics_report_and_site_offline(store_path: Path, tmp_path: Path) -> None:
    base = ["--store", str(store_path)]
    result = runner.invoke(app, [*base, "topics", "--offline", "--k", "4"])
    assert result.exit_code == 0, result.output
    assert "4 topics" in result.output

    report_path = tmp_path / "out" / "report.md"
    result = runner.invoke(app, [*base, "report", "--offline", "--out", str(report_path)])
    assert result.exit_code == 0, result.output
    text = report_path.read_text()
    assert text.startswith("# Recent Work in cs.AR: 32 Papers in 4 Topics")
    assert text.count("\n## ") == 1 + 1 + 4 + 1

    site = tmp_path / "site"
    result = runner.invoke(app, [*base, "site", "--out", str(site)])
    assert result.exit_code == 0, result.output
    assert "No report" not in result.output
    assert len(list((site / "papers").glob("*.html"))) == 32

    with Store(store_path) as store:
        model = load_topics(store)
        assert model is not None
        assert load_report(store, model) is not None
        commands = [run["config"]["command"] for run in store.runs()]
        assert commands == ["topics", "report"]
        assert {row.stage for row in store.usage()} == {"label", "synthesize", "introduce"}


def test_rerunning_topics_is_served_from_the_cache(store_path: Path) -> None:
    base = ["--store", str(store_path)]
    for _ in range(2):
        assert runner.invoke(app, [*base, "topics", "--offline", "--k", "3"]).exit_code == 0
    with Store(store_path) as store:
        second = store.runs()[-1]["run_id"]
        calls = store.calls(run_id=second)
    assert calls
    assert all(c.status == "cached" for c in calls)


def test_site_warns_when_the_report_is_for_other_topics(store_path: Path, tmp_path: Path) -> None:
    base = ["--store", str(store_path)]
    runner.invoke(app, [*base, "topics", "--offline", "--k", "4"])
    runner.invoke(app, [*base, "report", "--offline", "--out", str(tmp_path / "r.md")])
    runner.invoke(app, [*base, "topics", "--offline", "--k", "3"])
    result = runner.invoke(app, [*base, "site", "--out", str(tmp_path / "site")])
    assert result.exit_code == 0
    assert "No report for the current topics" in result.output


def test_commands_explain_missing_inputs(tmp_path: Path) -> None:
    empty = ["--store", str(tmp_path / "empty.db")]
    result = runner.invoke(app, [*empty, "topics", "--offline"])
    assert result.exit_code == 1
    assert "precis run" in result.output

    path = tmp_path / "some.db"
    with Store(path) as store:
        fill_store(store, themed_corpus(2)[0])
    for command in (["report", "--offline"], ["site"]):
        result = runner.invoke(app, ["--store", str(path), *command])
        assert result.exit_code == 1
        assert "precis topics" in result.output


def test_offline_corpus_commands_refuse_a_store_with_paid_results(store_path: Path) -> None:
    base = ["--store", str(store_path)]
    assert runner.invoke(app, [*base, "topics", "--offline", "--k", "4"]).exit_code == 0
    with Store(store_path) as store:
        store.record_call(
            LLMCall(
                call_id="c1",
                run_id="real",
                paper_id=None,
                stage="label",
                model="claude-opus-5-5",
                request_hash="h",
                status="ok",
                cost_usd=0.1,
                started_at=datetime.now(UTC),
            )
        )
        before = load_topics(store)
    for command in ("topics", "report"):
        result = runner.invoke(app, [*base, command, "--offline"])
        assert result.exit_code == 1, result.output
        assert "paid for with the API" in result.output
    with Store(store_path) as store:
        assert load_topics(store) == before
