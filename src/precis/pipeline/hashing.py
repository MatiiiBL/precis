"""The ``config_hash`` of each per-paper stage.

A stage is skipped when a ``done`` record with the same hash exists, so the
hash covers everything that determines the stage's output: the model
settings of every role the stage runs, the versions of the prompts and tool
definitions it sends, ``REQUEST_FORMAT_VERSION`` for the request text written
in code, the options that shape its requests, and the inputs it reads (the
document, or the summary and verification of the stage before).
Including the inputs means that a changed summary re-runs its verification
even when no setting changed.
"""

from __future__ import annotations

from typing import Any

from precis.agents import messages
from precis.agents.critic import CHECKS_SCHEMA
from precis.agents.prompts import PromptName, load_prompt
from precis.agents.submissions import SUMMARY_SCHEMA
from precis.agents.tools import READING_TOOLS
from precis.config import AgentConfig, AgentsConfig
from precis.models import Document, PaperMeta, PaperSummary, Verification
from precis.util import canonical_json, sha256_hex

_STRATEGY_PROMPTS: dict[str, tuple[PromptName, ...]] = {
    "agentic": ("reader",),
    "fulltext": ("fulltext",),
    "mapreduce": ("mapreduce_notes", "mapreduce_combine"),
}

# Tool descriptions and schemas are prompts too.
_TOOLS_VERSION = sha256_hex(canonical_json([READING_TOOLS, SUMMARY_SCHEMA, CHECKS_SCHEMA]))[:12]


def _hash(parts: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(parts))[:16]


def _stage_hash(parts: dict[str, Any]) -> str:
    """A stage hash also covers the request text written in code (briefs, notices, rendering)."""
    return _hash({**parts, "format": messages.REQUEST_FORMAT_VERSION})


def _role(config: AgentConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def _prompts(*names: PromptName) -> dict[str, str]:
    return {name: load_prompt(name).version for name in names}


def _reading_options(config: AgentsConfig) -> dict[str, Any]:
    return {
        "page_tokens": config.page_tokens,
        "search_results": config.search_results,
        "tools": _TOOLS_VERSION,
    }


def document_digest(meta: PaperMeta, doc: Document, *, include_abstract: bool) -> str:
    """What the reader is shown of a paper: its parsed text and the metadata in the brief."""
    shown = meta.model_dump(
        mode="json", include={"title", "authors", "categories", "primary_category"}
    )
    if include_abstract:
        shown["abstract"] = meta.abstract
    return _hash({"meta": shown, "document": doc.model_dump(mode="json")})


def read_hash(config: AgentsConfig, meta: PaperMeta, doc: Document) -> str:
    options: dict[str, Any] = {"include_abstract": config.include_abstract}
    if config.strategy == "agentic":
        options |= _reading_options(config)
    elif config.strategy == "mapreduce":
        options |= {"page_tokens": config.page_tokens, "chunk_tokens": config.chunk_tokens}
    return _stage_hash(
        {
            "stage": "read",
            "strategy": config.strategy,
            "reader": _role(config.reader),
            "prompts": _prompts(*_STRATEGY_PROMPTS[config.strategy]),
            "options": options,
            "input": document_digest(meta, doc, include_abstract=config.include_abstract),
        }
    )


def verify_hash(config: AgentsConfig, meta: PaperMeta, doc: Document, summary: PaperSummary) -> str:
    return _stage_hash(
        {
            "stage": "verify",
            "critic": _role(config.critic),
            "prompts": _prompts("critic"),
            "options": _reading_options(config),
            "input": [
                document_digest(meta, doc, include_abstract=False),
                summary.model_dump(mode="json"),
            ],
        }
    )


def revise_hash(
    config: AgentsConfig,
    meta: PaperMeta,
    doc: Document,
    summary: PaperSummary,
    verification: Verification,
) -> str:
    """The revise stage also re-verifies, so the critic's settings count too."""
    return _stage_hash(
        {
            "stage": "revise",
            "enabled": config.revise,
            "reviser": _role(config.reviser),
            "critic": _role(config.critic),
            "prompts": _prompts("reviser", "critic"),
            "options": _reading_options(config),
            "input": [
                document_digest(meta, doc, include_abstract=False),
                summary.model_dump(mode="json"),
                verification.model_dump(mode="json"),
            ],
        }
    )
