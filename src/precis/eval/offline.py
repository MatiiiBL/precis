"""Offline stand-ins for the judge: key-point extraction and coverage.

Key points are the abstract's first sentences; a key point counts as
covered when most of its content words appear in the summary. That is a
crude lexical judge, but it is deterministic and responds to the summary,
so the evaluation commands produce meaningful-looking numbers offline and
their plumbing is tested end to end.
"""

from __future__ import annotations

import json
import re
from typing import Any

from precis.eval.judge import COVERAGE_SYSTEM, EXTRACT_SYSTEM
from precis.llm.fake import FakeReply, Responder

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]{3,}")


def _user(params: dict[str, Any]) -> str:
    content = params["messages"][0]["content"]
    return content if isinstance(content, str) else ""


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def extract(params: dict[str, Any]) -> FakeReply | None:
    if params.get("system") != EXTRACT_SYSTEM:
        return None
    abstract = _user(params).split("Abstract:\n", 1)[-1]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.split()) >= 4]
    return FakeReply.text(json.dumps({"key_points": sentences[:5]}))


def judge_coverage(params: dict[str, Any]) -> FakeReply | None:
    if params.get("system") != COVERAGE_SYSTEM:
        return None
    prompt = _user(params)
    points_part, _, summary = prompt.partition("\n\nSummary:\n")
    points = re.findall(r"^(\d+)\. (.*)$", points_part, flags=re.M)
    summary_words = _words(summary)
    judgments = []
    for number, point in points:
        words = _words(point)
        overlap = len(words & summary_words) / len(words) if words else 0.0
        judgments.append(
            {
                "point": int(number),
                "covered": overlap >= 0.5,
                "reason": f"{100 * overlap:.0f}% of the point's content words appear.",
            }
        )
    return FakeReply.text(json.dumps({"judgments": judgments}))


RESPONDERS: tuple[Responder, ...] = (extract, judge_coverage)
