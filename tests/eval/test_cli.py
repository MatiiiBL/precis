import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus.factories import fill_store, make_view
from precis.cli import app
from precis.eval.bench import populate
from precis.models import Document, Section, Verdict
from precis.store import Store

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path) -> list[str]:
    """A store of synthetic papers with full text, and a config writing into tmp_path."""
    with Store(tmp_path / "precis.db") as store:
        populate(store, 10, seed=0)
    config = tmp_path / "precis.toml"
    config.write_text(
        f'[eval]\nresults_dir = "{tmp_path / "results"}"\nfigures_dir = "{tmp_path / "img"}"\n'
    )
    return ["--config", str(config), "--store", str(tmp_path / "precis.db")]


def test_eval_abstract_offline(workspace: list[str], tmp_path: Path) -> None:
    result = runner.invoke(app, [*workspace, "eval", "abstract", "--offline", "--n", "4"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "results" / "abstract.json").read_text())
    assert data["papers"] == 4
    assert data["strategy"] == "agentic"
    assert 0 < data["recall"]["mean"] <= 1
    assert data["rouge"]["rouge1"] > data["rouge_mismatched"]["rouge1"]


def test_eval_ablation_offline(workspace: list[str], tmp_path: Path) -> None:
    result = runner.invoke(app, [*workspace, "eval", "ablation", "--offline", "--n", "3"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "results" / "ablation.json").read_text())
    strategies = {s["strategy"]: s for s in data["strategies"]}
    assert set(strategies) == {"agentic", "fulltext", "mapreduce"}
    assert strategies["fulltext"]["requests_per_paper"] == 1
    for row in strategies.values():
        assert row["papers"] == 3
        assert row["cost_per_paper_usd"] > 0
        assert row["latency_measured"] == 3  # a fresh store: nothing came from the cache
        assert row["latency_p50_s"] > 0


def test_rerun_reports_the_same_cost_from_the_cache(workspace: list[str], tmp_path: Path) -> None:
    command = [*workspace, "eval", "ablation", "--offline", "--n", "2"]
    costs = []
    for _ in range(2):
        assert runner.invoke(app, command).exit_code == 0
        data = json.loads((tmp_path / "results" / "ablation.json").read_text())
        costs.append([s["cost_per_paper_usd"] for s in data["strategies"]])
        latencies = [s["latency_measured"] for s in data["strategies"]]
    assert costs[0] == costs[1]
    assert latencies == [0, 0, 0]


def test_eval_needs_papers_with_text(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--store", str(tmp_path / "empty.db"), "eval", "abstract"])
    assert result.exit_code == 1
    assert "precis fetch" in result.output


def test_bench_throughput_writes_results_and_a_figure(workspace: list[str], tmp_path: Path) -> None:
    args = ["bench", "throughput", "--papers", "8", "--concurrency", "8", "--concurrency", "2"]
    result = runner.invoke(app, [*workspace, *args])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "results" / "bench-throughput.json").read_text())
    assert [p["concurrency"] for p in data["points"]] == [2, 8]
    assert (tmp_path / "img" / "bench-throughput.png").is_file()


def test_eval_faithfulness_writes_results(tmp_path: Path) -> None:
    store_path = tmp_path / "precis.db"
    verdicts = [Verdict.CONTRADICTED, Verdict.SUPPORTED, Verdict.SUPPORTED]
    papers = [
        make_view("p1", ["caches", "prefetching"], draft_verdicts=verdicts, revised=True),
        make_view("p2", ["caches", "prefetching"]),
    ]
    with Store(store_path) as store:
        fill_store(store, papers)
        store.put_document(
            Document(
                paper_id="p1",
                source_format="html",
                sections=[Section(id="s4", title="Results", level=1, text="a quote", n_tokens=2)],
                captions=[],
                n_tokens=2,
                sha256="0" * 64,
            )
        )
    config = tmp_path / "precis.toml"
    config.write_text(f'[eval]\nresults_dir = "{tmp_path / "results"}"\n')
    result = runner.invoke(
        app, ["--config", str(config), "--store", str(store_path), "eval", "faithfulness"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "results" / "faithfulness.json").read_text())
    assert (data["papers"], data["revised"]) == (2, 1)
    assert data["quotes"]["checks"] == 6
    assert data["quotes"]["not_found"] == 0
    assert "Faithfulness of the summaries" in (tmp_path / "results" / "faithfulness.md").read_text()
