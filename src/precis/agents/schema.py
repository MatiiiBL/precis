"""JSON schemas that strict tool use and structured outputs accept.

Strict mode supports a subset of JSON Schema: every object must set
``additionalProperties: false``, and numeric, string-length and array-size
constraints are not allowed. Pydantic's schemas also carry titles, defaults
and ``$ref``s, and their descriptions are written for developers. This module
derives a strict schema from a record, inlines the references, and lets the
caller replace field descriptions with ones written for the model. The record
itself stays the validator: everything the schema cannot express (word
limits, list sizes, ids that must exist) is checked in code afterwards.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

# Keywords strict mode rejects or that only matter to Pydantic. Constraints
# dropped here are still enforced when the record validates the input.
_DROPPED = frozenset(
    {
        "title",
        "default",
        "examples",
        "format",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def strict_schema(
    model: type[BaseModel], *, descriptions: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Return a strict-mode schema for ``model``.

    ``descriptions`` maps dotted field paths (``"findings.claim"``; list items
    share their list's path) to the description the model should see. Every
    property is required; optional fields stay nullable through ``anyOf``.
    The record's own docstring is dropped because the tool description or the
    prompt already says what the object is.
    """
    raw = model.model_json_schema()
    defs: dict[str, Any] = raw.pop("$defs", {})
    schema = _convert(raw, defs, "", dict(descriptions or {}))
    schema.pop("description", None)
    return schema


def _convert(
    node: Mapping[str, Any], defs: Mapping[str, Any], path: str, descriptions: Mapping[str, str]
) -> dict[str, Any]:
    if "$ref" in node:
        target = defs[node["$ref"].rsplit("/", 1)[-1]]
        merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
        return _convert(merged, defs, path, descriptions)
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED:
            continue
        if key == "properties":
            out[key] = {
                name: _convert(sub, defs, f"{path}.{name}".lstrip("."), descriptions)
                for name, sub in value.items()
            }
        elif key == "items":
            out[key] = _convert(value, defs, path, descriptions)
        elif key in ("anyOf", "allOf", "oneOf"):
            # The description belongs on the union, not on each option.
            out[key] = [_convert(option, defs, path, {}) for option in value]
        else:
            out[key] = value
    if out.get("type") == "object":
        out["additionalProperties"] = False
        out["required"] = list(out.get("properties", {}))
    if path in descriptions and "properties" not in out:
        out["description"] = descriptions[path]
    return out
