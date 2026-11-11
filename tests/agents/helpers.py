"""A sample paper and a scripted gateway for the agent tests."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from anthropic.types import Message

from precis.config import AgentsConfig
from precis.models import Caption, Document, PaperMeta, Section
from precis.util import estimate_tokens

INTRO = (
    "Last-level caches (LLCs) lose much of their capacity to lines that are never reused. "
    "Replacement policies such as LRU cannot tell these lines apart from useful ones.\n\n"
    "We propose ReuseNet, a lightweight reuse predictor that guides LLC replacement. "
    "ReuseNet reduces LLC misses by 18.4% over LRU on SPEC CPU2017."
)
POLICIES = (
    "LRU evicts the least recently used line. Hawkeye learns from Belady's optimal policy "
    "and needs 28 KB of state per core."
)
DESIGN_1 = (
    "ReuseNet hashes the program counter of each load into a table of 2048 saturating counters. "
    "A counter is trained up on reuse and down on eviction without reuse."
)
DESIGN_2 = (
    "On a miss, ReuseNet inserts lines predicted dead at the lowest priority. "
    "The table and its tags take 12.5 KB of storage in total."
)
SETUP = (
    "We simulate a 2 MB 16-way LLC in ChampSim. "
    "The workloads are 29 SPEC CPU2017 benchmarks, each run for 500 million instructions."
)
RESULTS = (
    "ReuseNet reduces LLC misses by 18.4% on average over LRU. "
    "It improves IPC by 6.2% over LRU and by 1.9% over Hawkeye. "
    "The largest gain, 41.3% fewer misses, is on mcf."
)
CONCLUSION = (
    "ReuseNet makes LLC replacement cheaper to learn. However, it was evaluated only in "
    "simulation, and multicore interference is left to future work."
)


def section(sid: str, title: str, text: str, level: int = 1) -> Section:
    return Section(id=sid, title=title, level=level, text=text, n_tokens=estimate_tokens(text))


def make_meta(paper_id: str = "2406.01234", title: str | None = None) -> PaperMeta:
    return PaperMeta(
        id=paper_id,
        title=title or "ReuseNet: Learned Reuse Prediction for Last-Level Caches",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="We present ReuseNet, a reuse predictor for last-level cache replacement.",
        categories=["cs.AR", "cs.LG"],
        primary_category="cs.AR",
        published=date(2024, 6, 3),
        url=f"https://arxiv.org/abs/{paper_id}",
    )


def make_doc(paper_id: str = "2406.01234") -> Document:
    sections = [
        section("s1", "Introduction", INTRO),
        section("s2", "Background", ""),
        section("s2.1", "Replacement policies", POLICIES, level=2),
        section("s3", "Design of ReuseNet", f"{DESIGN_1}\n\n{DESIGN_2}"),
        section("s4", "Evaluation", ""),
        section("s4.1", "Setup", SETUP, level=2),
        section("s4.2", "Results", RESULTS, level=2),
        section("s5", "Conclusion", CONCLUSION),
    ]
    captions = [
        Caption(
            id="fig1", kind="figure", text="Miss reduction over LRU per SPEC CPU2017 benchmark."
        ),
        Caption(id="tab1", kind="table", text="Storage: ReuseNet 12.5 KB; Hawkeye 28 KB."),
    ]
    return Document(
        paper_id=paper_id,
        source_format="html",
        sections=sections,
        captions=captions,
        n_tokens=sum(s.n_tokens for s in sections),
        sha256="0" * 64,
    )


def agents_config(**overrides: Any) -> AgentsConfig:
    """Small pages, so that the design section spans two of them."""
    return AgentsConfig.model_validate({"page_tokens": 40, **overrides})


# -- replies -----------------------------------------------------------------


def message(
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    *,
    usage: tuple[int, int] = (100, 20),
    stop_details: dict[str, Any] | None = None,
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        }
    )


def thinking() -> dict[str, Any]:
    return {"type": "thinking", "thinking": "", "signature": "sig-1"}


def call(name: str, tool_input: dict[str, Any], call_id: str = "toolu_1") -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}


def tool_turn(*calls: dict[str, Any], usage: tuple[int, int] = (100, 20)) -> Message:
    return message([thinking(), *calls], "tool_use", usage=usage)


def text_turn(text: str, stop_reason: str = "end_turn") -> Message:
    return message([thinking(), {"type": "text", "text": text}], stop_reason)


def valid_draft(**overrides: Any) -> dict[str, Any]:
    """A summary of the sample paper that passes validation."""
    draft: dict[str, Any] = {
        "tldr": "ReuseNet predicts line reuse to guide LLC replacement, reducing misses by "
        "18.4% over LRU on SPEC CPU2017.",
        "problem": "LLCs waste capacity on lines that are never reused.",
        "approach": "A table of saturating counters indexed by load PC predicts dead lines, "
        "which are inserted at the lowest priority.",
        "findings": [
            {
                "claim": "ReuseNet reduces LLC misses by 18.4% on average over LRU.",
                "section_ids": ["s4.2"],
            },
            {"claim": "ReuseNet improves IPC by 6.2% over LRU.", "section_ids": ["s4.2"]},
        ],
        "contributions": ["ReuseNet, a reuse predictor for LLC replacement."],
        "limitations": ["Evaluated only in simulation."],
        "methods": ["ReuseNet", "LRU", "Hawkeye"],
        "datasets": ["SPEC CPU2017"],
        "keywords": ["Cache Replacement", "last-level cache", "reuse prediction"],
    }
    draft.update(overrides)
    return draft


# -- gateway -----------------------------------------------------------------

Step = Message | Callable[[dict[str, Any]], Message] | BaseException


@dataclass
class ScriptedGateway:
    """A ``ModelGateway`` that replays a script and records every request."""

    script: list[Step]
    requests: list[dict[str, Any]] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)

    async def create(
        self, params: dict[str, Any], *, stage: str, paper_id: str | None = None
    ) -> Message:
        self.requests.append(copy.deepcopy(params))
        self.stages.append(stage)
        if not self.script:
            raise AssertionError("the script ran out of replies")
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step(params) if callable(step) else step


def last_user_blocks(params: dict[str, Any]) -> Sequence[dict[str, Any]]:
    """The content blocks of the last user message of a request."""
    user = [m for m in params["messages"] if m["role"] == "user"][-1]
    content = user["content"]
    return [{"type": "text", "text": content}] if isinstance(content, str) else content
