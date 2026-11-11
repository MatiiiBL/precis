"""Write evaluation and benchmark results: JSON for machines, Markdown for people."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

# Pre-formatted numbers ("$0.039", "16%", "1,200", "n/a") align like numbers.
_NUMBER_TEXT = re.compile(r"^(?:[$-]?[\d,]+(?:\.\d+)?%?|n/a)$")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A GitHub Markdown table; columns whose cells are all numbers align right.

    Floats get three decimals below 10 and one above; callers that want a
    fixed precision pass formatted strings, which still align as numbers.
    """

    def is_number(value: object) -> bool:
        if isinstance(value, int | float):
            return True
        return isinstance(value, str) and bool(_NUMBER_TEXT.match(value))

    def numeric(column: int) -> bool:
        return bool(rows) and all(is_number(row[column]) for row in rows)

    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:,.3f}" if abs(value) < 10 else f"{value:,.1f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value).replace("|", "\\|")

    align = ["---:" if numeric(i) else "---" for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_results(directory: Path, name: str, result: BaseModel, markdown: str) -> list[Path]:
    """Write ``<name>.json`` and ``<name>.md`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{name}.json"
    md_path = directory / f"{name}.md"
    json_path.write_text(result.model_dump_json(indent=2) + "\n")
    md_path.write_text(markdown if markdown.endswith("\n") else markdown + "\n")
    return [json_path, md_path]
