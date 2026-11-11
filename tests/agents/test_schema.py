"""The tool and output schemas must be ones strict mode accepts."""

from typing import Any

import pytest

from precis.agents.critic import CHECKS_SCHEMA
from precis.agents.prompts import load_prompt
from precis.agents.submissions import SUMMARY_SCHEMA
from precis.agents.tools import READING_TOOLS
from precis.models import Verdict

_UNSUPPORTED = {
    "$ref",
    "$defs",
    "title",
    "default",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
}


def walk(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, list):
        return [n for item in node for n in walk(item)]
    if not isinstance(node, dict):
        return []
    found = [node]
    for value in node.values():
        found += walk(value)
    return found


@pytest.mark.parametrize(
    "schema",
    [SUMMARY_SCHEMA, CHECKS_SCHEMA, *(t["input_schema"] for t in READING_TOOLS)],
    ids=["summary", "checks", *(t["name"] for t in READING_TOOLS)],
)
def test_schema_is_strict_compatible(schema: dict[str, Any]) -> None:
    for node in walk(schema):
        assert not _UNSUPPORTED & node.keys(), node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])


def test_reading_tools_are_strict_and_described() -> None:
    for tool in READING_TOOLS:
        assert tool["strict"] is True
        assert len(tool["description"]) > 80


def test_model_facing_descriptions_replace_developer_ones() -> None:
    claim = SUMMARY_SCHEMA["properties"]["findings"]["items"]["properties"]["claim"]
    assert "paper's own numbers" in claim["description"]
    target = CHECKS_SCHEMA["properties"]["checks"]["items"]["properties"]["target"]
    assert "PaperSummary" not in target["description"]
    verdict = CHECKS_SCHEMA["properties"]["checks"]["items"]["properties"]["verdict"]
    # Every stored verdict, plus "unchecked", which never reaches a stored record.
    assert verdict["enum"] == [*(v.value for v in Verdict), "unchecked"]
    section = CHECKS_SCHEMA["properties"]["checks"]["items"]["properties"]["section_id"]
    assert [o["type"] for o in section["anyOf"]] == ["string", "null"]
    assert "description" not in section["anyOf"][0]


@pytest.mark.parametrize(
    "name",
    ["reader", "critic", "reviser", "fulltext", "mapreduce_notes", "mapreduce_combine"],
)
def test_prompts_load_with_includes_expanded(name: Any) -> None:
    prompt = load_prompt(name)
    assert "<!-- include" not in prompt.text
    assert len(prompt.version) == 12
    assert prompt.version == load_prompt(name).version


def test_summary_style_is_shared_by_every_summarizing_prompt() -> None:
    style = "Numbers come only from the paper."
    for name in ("reader", "reviser", "fulltext", "mapreduce_combine"):
        assert style in load_prompt(name).text
    assert style not in load_prompt("critic").text
    versions = {load_prompt(n).version for n in ("reader", "critic", "reviser", "fulltext")}
    assert len(versions) == 4
