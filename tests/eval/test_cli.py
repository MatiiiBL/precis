import json
from pathlib import Path

from typer.testing import CliRunner

from corpus.factories import fill_store, make_view
from precis.cli import app
from precis.models import Document, Section, Verdict
from precis.store import Store

runner = CliRunner()


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
