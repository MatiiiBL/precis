"""``precis eval`` and ``precis bench``: quality evaluation and throughput benchmarks.

Every command writes ``<name>.json`` and ``<name>.md`` to ``eval.results_dir``
(``results/`` by default), so the numbers in the docs can be regenerated.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from precis.cli.corpus import config_of, require_papers
from precis.eval.faithfulness import evaluate_faithfulness, faithfulness_markdown
from precis.eval.results import write_results
from precis.models import Document
from precis.store import Store

console = Console()

eval_app = typer.Typer(help="Evaluate the quality of the summaries.", no_args_is_help=True)


def _show(markdown: str, paths: Sequence[Path]) -> None:
    console.print(Markdown(markdown))
    console.print("Wrote " + ", ".join(str(p) for p in paths))


@eval_app.command("faithfulness")
def faithfulness_command(ctx: typer.Context) -> None:
    """Verdict rates before and after revision, and an audit of the evidence quotes."""
    config = config_of(ctx)
    with Store(config.store_path) as store:
        papers = require_papers(store)
        documents: dict[str, Document] = {}
        for paper in papers:
            doc = store.get_document(paper.id)
            if doc is not None:
                documents[paper.id] = doc
    result = evaluate_faithfulness(papers, documents)
    markdown = faithfulness_markdown(result)
    _show(markdown, write_results(config.eval.results_dir, "faithfulness", result, markdown))


def register(app: typer.Typer) -> None:
    """Add the evaluation and benchmark commands to ``app``."""
    app.add_typer(eval_app, name="eval")
